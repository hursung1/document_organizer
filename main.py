from __future__ import annotations

import asyncio
import json
import random
import re
import time
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from redis.asyncio import Redis
from redis.exceptions import RedisError

from doc_organizer.arxiv_fetcher import ArxivFetcherService
from doc_organizer.arxiv_ingest import ArxivMilvusIngestionService
from doc_organizer.ingest import DocumentIngestionService
from doc_organizer.qa_service import DocumentQAService
from doc_organizer.settings import IngestionSettings

app = FastAPI(title="doc-organizer", version="0.1.0")

settings = IngestionSettings.from_env()
service: DocumentIngestionService | None = None
qa_service: DocumentQAService | None = None
arxiv_service: ArxivFetcherService | None = None
arxiv_milvus_service: ArxivMilvusIngestionService | None = None
redis_client: Redis | None = None
ingest_lock = asyncio.Lock()
arxiv_lock = asyncio.Lock()
periodic_task: asyncio.Task | None = None
arxiv_task: asyncio.Task | None = None
runtime_state: dict[str, Any] = {
    "running": False,
    "last_run_at": None,
    "last_summary": None,
    "last_error": None,
    "last_qa_error": None,
    "arxiv_running": False,
    "arxiv_last_run_at": None,
    "arxiv_last_summary": None,
    "arxiv_last_error": None,
    "arxiv_last_milvus_summary": None,
}


class StarterDoc(BaseModel):
    id: str
    title: str
    summary: str


class StarterDocsResponse(BaseModel):
    documents: list[StarterDoc]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    reasoning: str | None = None
    suggested_questions: list[str]


class ConversationSummary(BaseModel):
    conversation_id: str
    title: str
    created_at: str
    updated_at: str


class ConversationsResponse(BaseModel):
    conversations: list[ConversationSummary]


class ConversationMessage(BaseModel):
    role: str
    text: str
    reasoning: str | None = None
    created_at: str


class ConversationMessagesResponse(BaseModel):
    conversation_id: str
    messages: list[ConversationMessage]


CHAT_KEY_PREFIX = "doc_organizer:chat"
CHAT_INDEX_KEY = f"{CHAT_KEY_PREFIX}:index"
STAGE_LABELS = {
    "analyze_query": "질의 분석 중",
    "retrieve_docs": "문서 검색 중",
    "enrich_with_arxiv_pdf": "arXiv 원문 보강 중",
    "generate_answer": "답변 생성 중",
}
SUMMARY_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "using",
    "based",
    "study",
    "paper",
    "method",
    "model",
    "data",
    "approach",
    "results",
    "문서",
    "내용",
    "대한",
    "관련",
    "통해",
    "기반",
    "연구",
    "방법",
    "모델",
    "결과",
}


def _chat_meta_key(conversation_id: str) -> str:
    return f"{CHAT_KEY_PREFIX}:{conversation_id}:meta"


def _chat_messages_key(conversation_id: str) -> str:
    return f"{CHAT_KEY_PREFIX}:{conversation_id}:messages"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _sse_event(event: str, payload: dict[str, Any]) -> str:
    body = json.dumps(payload, ensure_ascii=False)
    return f"event: {event}\ndata: {body}\n\n"


def _title_from_message(message: str) -> str:
    normalized = _normalize_space(message)
    if not normalized:
        return "새 대화"
    if len(normalized) <= 40:
        return normalized
    return f"{normalized[:40]}..."


async def _resolve_redis_client() -> Redis:
    global redis_client
    if redis_client is not None:
        return redis_client
    redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
    await redis_client.ping()
    return redis_client


async def _create_or_touch_conversation(
    conversation_id: str,
    first_message: str | None = None,
) -> None:
    client = await _resolve_redis_client()
    now_iso = _utc_now_iso()
    meta_key = _chat_meta_key(conversation_id)

    existing = await client.hgetall(meta_key)
    if not existing:
        title = _title_from_message(first_message or "")
        await client.hset(
            meta_key,
            mapping={
                "conversation_id": conversation_id,
                "title": title,
                "created_at": now_iso,
                "updated_at": now_iso,
            },
        )
    else:
        await client.hset(meta_key, mapping={"updated_at": now_iso})
        if first_message and not existing.get("title"):
            await client.hset(meta_key, mapping={"title": _title_from_message(first_message)})

    await client.zadd(CHAT_INDEX_KEY, {conversation_id: time.time()})


async def _append_message(
    conversation_id: str,
    role: str,
    text: str,
    reasoning: str | None = None,
) -> None:
    client = await _resolve_redis_client()
    payload = {
        "role": role,
        "text": text,
        "reasoning": reasoning,
        "created_at": _utc_now_iso(),
    }
    await client.rpush(_chat_messages_key(conversation_id), json.dumps(payload, ensure_ascii=False))
    await _create_or_touch_conversation(
        conversation_id,
        first_message=text if role == "user" else None,
    )


async def _list_conversations(limit: int = 200) -> list[ConversationSummary]:
    client = await _resolve_redis_client()
    ids = await client.zrevrange(CHAT_INDEX_KEY, 0, max(0, limit - 1))
    out: list[ConversationSummary] = []
    for conversation_id in ids:
        meta = await client.hgetall(_chat_meta_key(conversation_id))
        if not meta:
            continue
        out.append(
            ConversationSummary(
                conversation_id=conversation_id,
                title=meta.get("title", "새 대화"),
                created_at=meta.get("created_at", _utc_now_iso()),
                updated_at=meta.get("updated_at", _utc_now_iso()),
            )
        )
    return out


async def _get_conversation_messages(conversation_id: str) -> list[ConversationMessage]:
    client = await _resolve_redis_client()
    raw_items = await client.lrange(_chat_messages_key(conversation_id), 0, -1)
    messages: list[ConversationMessage] = []
    for raw in raw_items:
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            continue
        messages.append(
            ConversationMessage(
                role=str(item.get("role", "assistant")),
                text=str(item.get("text", "")),
                reasoning=(
                    str(item.get("reasoning", "")).strip() or None
                    if item.get("reasoning") is not None
                    else None
                ),
                created_at=str(item.get("created_at", _utc_now_iso())),
            )
        )
    return messages


async def _delete_conversation(conversation_id: str) -> bool:
    client = await _resolve_redis_client()
    removed_index = await client.zrem(CHAT_INDEX_KEY, conversation_id)
    deleted_keys = await client.delete(
        _chat_meta_key(conversation_id),
        _chat_messages_key(conversation_id),
    )
    return bool(removed_index or deleted_keys)


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\|[-: ]+\|", " ", text)
    return _normalize_space(text)


def _split_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"[\r\n]+", " ", text)
    raw = re.split(r"(?<=[.!?])\s+|(?<=[다요])\s+", cleaned)
    return [_normalize_space(s) for s in raw if _normalize_space(s)]


def _first_sentence(text: str, fallback: str) -> str:
    for sentence in _split_sentences(text):
        if len(sentence) < 30:
            continue
        if not sentence.endswith((".", "!", "?", "다", "요")):
            sentence = f"{sentence}."
        return sentence
    return fallback


def _extract_summary_keywords(text: str, limit: int = 3) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{1,}|[가-힣]{2,}", text)
    normalized_tokens = [token.strip() for token in tokens if token.strip()]
    filtered = [
        token
        for token in normalized_tokens
        if token.lower() not in SUMMARY_STOPWORDS and len(token) >= 2
    ]
    if not filtered:
        return []
    ranked = Counter(token.lower() for token in filtered).most_common(limit)
    return [token for token, _ in ranked]


def _korean_one_sentence_summary(title: str, body: str) -> str:
    clean_title = _normalize_space(title) or "이 문서"
    keywords = _extract_summary_keywords(_normalize_space(f"{title} {body}"))
    if keywords:
        keyword_text = ", ".join(keywords)
        return f"{clean_title} 문서는 {keyword_text}을 중심으로 핵심 내용과 의미를 한 문장으로 요약한다."
    return f"{clean_title} 문서는 핵심 내용과 의의를 간결하게 정리한 문서다."


def _build_doc_title(path: Path) -> str:
    return path.stem.replace("_", " ").strip() or path.name


def _load_documents() -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    md_paths = sorted(settings.results_dir.glob("**/*.md"))

    seen_titles: set[str] = set()
    for path in md_paths:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        cleaned = _strip_html(raw)
        title = _build_doc_title(path)
        seen_titles.add(title)
        docs.append(
            {
                "id": str(path.resolve()),
                "title": title,
                "content": cleaned,
                "summary": _korean_one_sentence_summary(title=title, body=cleaned),
                "source_path": str(path),
            }
        )

    pdf_candidates = sorted(settings.docs_dir.glob("**/*.pdf")) + sorted(Path(".").glob("*.pdf"))
    for pdf in pdf_candidates:
        title = _build_doc_title(pdf)
        if title in seen_titles:
            continue
        docs.append(
            {
                "id": str(pdf.resolve()),
                "title": title,
                "content": "",
                "summary": f"{title} 문서는 아직 OCR 결과가 없어 상세 텍스트를 준비 중입니다.",
                "source_path": str(pdf),
            }
        )
        seen_titles.add(title)

    return docs


def _pick_starter_docs(documents: list[dict[str, str]], count: int = 3) -> list[dict[str, str]]:
    if not documents:
        return []
    ranked = sorted(
        documents,
        key=lambda d: (len(d.get("content", "")), d.get("title", "")),
        reverse=True,
    )
    return ranked[:count]


def _load_arxiv_starter_docs(count: int = 3) -> list[dict[str, str]]:
    arxiv_dir = settings.results_dir / "arxiv"
    if not arxiv_dir.exists():
        return []

    candidates: list[dict[str, str]] = []
    for json_file in sorted(arxiv_dir.glob("*.json")):
        if json_file.name.startswith("_"):
            continue
        try:
            raw = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(raw, list):
            continue

        for item in raw:
            if not isinstance(item, dict):
                continue
            title = _normalize_space(str(item.get("title", "")))
            summary = _normalize_space(str(item.get("summary", "")))
            if not title or not summary:
                continue
            doc_id = str(
                item.get("arxiv_id")
                or item.get("source_url")
                or item.get("pdf_url")
                or f"{json_file.stem}:{title}"
            )
            candidates.append(
                {
                    "id": doc_id,
                    "title": title,
                    "summary": _korean_one_sentence_summary(title=title, body=summary),
                }
            )

    if not candidates:
        return []
    if len(candidates) <= count:
        random.shuffle(candidates)
        return candidates
    return random.sample(candidates, count)


def _resolve_qa_service() -> DocumentQAService:
    global qa_service
    if qa_service is not None:
        return qa_service
    qa_service = DocumentQAService(settings)
    return qa_service


async def _run_ingestion(trigger: str) -> dict[str, Any]:
    if service is None:
        raise HTTPException(status_code=503, detail="Ingestion service is not ready.")
    if ingest_lock.locked():
        raise HTTPException(status_code=409, detail="Ingestion is already running.")

    async with ingest_lock:
        runtime_state["running"] = True
        runtime_state["last_error"] = None
        try:
            summary = await asyncio.to_thread(service.run_once)
            runtime_state["last_run_at"] = datetime.now(UTC).isoformat()
            runtime_state["last_summary"] = summary.to_dict()
            return {
                "trigger": trigger,
                "run_at": runtime_state["last_run_at"],
                "summary": runtime_state["last_summary"],
            }
        except Exception as exc:
            runtime_state["last_error"] = str(exc)
            raise
        finally:
            runtime_state["running"] = False


async def _periodic_ingestion_loop() -> None:
    global service
    while True:
        try:
            if service is None:
                service = DocumentIngestionService(settings)
                runtime_state["last_error"] = None
            await _run_ingestion(trigger="scheduler")
        except Exception as exc:
            runtime_state["running"] = False
            runtime_state["last_error"] = str(exc)
        await asyncio.sleep(settings.interval_seconds)


async def _run_arxiv_update(trigger: str) -> dict[str, Any]:
    if arxiv_service is None:
        raise HTTPException(status_code=503, detail="ArXiv service is not ready.")
    if arxiv_milvus_service is None:
        raise HTTPException(status_code=503, detail="ArXiv Milvus service is not ready.")
    if arxiv_lock.locked():
        raise HTTPException(status_code=409, detail="ArXiv update is already running.")

    async with arxiv_lock:
        runtime_state["arxiv_running"] = True
        runtime_state["arxiv_last_error"] = None
        try:
            summary = await asyncio.to_thread(arxiv_service.run_once)
            milvus_summary = await asyncio.to_thread(
                arxiv_milvus_service.upsert_from_file,
                Path(summary.output_file),
            )
            runtime_state["arxiv_last_run_at"] = datetime.now(UTC).isoformat()
            runtime_state["arxiv_last_summary"] = summary.to_dict() | {
                "milvus": milvus_summary.to_dict()
            }
            runtime_state["arxiv_last_milvus_summary"] = milvus_summary.to_dict()
            return {
                "trigger": trigger,
                "run_at": runtime_state["arxiv_last_run_at"],
                "summary": runtime_state["arxiv_last_summary"],
            }
        except Exception as exc:
            runtime_state["arxiv_last_error"] = str(exc)
            raise
        finally:
            runtime_state["arxiv_running"] = False


async def _arxiv_scheduler_loop() -> None:
    global arxiv_service
    global arxiv_milvus_service
    while True:
        try:
            if arxiv_service is None:
                arxiv_service = ArxivFetcherService(settings)
            if arxiv_milvus_service is None:
                arxiv_milvus_service = ArxivMilvusIngestionService(settings)
            now = datetime.now().astimezone()
            if arxiv_service.should_run(now):
                await _run_arxiv_update(trigger="scheduler")
        except Exception as exc:
            runtime_state["arxiv_running"] = False
            runtime_state["arxiv_last_error"] = str(exc)
        await asyncio.sleep(60)


@app.on_event("startup")
async def on_startup() -> None:
    global service
    global qa_service
    global arxiv_service
    global arxiv_milvus_service
    global periodic_task
    global arxiv_task
    settings.ensure_directories()
    try:
        service = DocumentIngestionService(settings)
        qa_service = DocumentQAService(settings)
        arxiv_service = ArxivFetcherService(settings)
        arxiv_milvus_service = ArxivMilvusIngestionService(settings)
    except Exception as exc:
        runtime_state["last_error"] = str(exc)
    if settings.auto_ingest_enabled:
        periodic_task = asyncio.create_task(_periodic_ingestion_loop())
    arxiv_task = asyncio.create_task(_arxiv_scheduler_loop())


@app.on_event("shutdown")
async def on_shutdown() -> None:
    global redis_client
    global qa_service
    global periodic_task
    global arxiv_task
    if periodic_task:
        periodic_task.cancel()
        try:
            await periodic_task
        except asyncio.CancelledError:
            pass
        periodic_task = None
    if arxiv_task:
        arxiv_task.cancel()
        try:
            await arxiv_task
        except asyncio.CancelledError:
            pass
        arxiv_task = None
    if qa_service is not None:
        await qa_service.aclose()
        qa_service = None
    if redis_client is not None:
        try:
            await redis_client.aclose()
        except Exception:
            pass
        redis_client = None


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "running": runtime_state["running"],
        "interval_seconds": settings.interval_seconds,
        "auto_ingest_enabled": settings.auto_ingest_enabled,
        "last_run_at": runtime_state["last_run_at"],
        "last_summary": runtime_state["last_summary"],
        "last_error": runtime_state["last_error"],
        "last_qa_error": runtime_state["last_qa_error"],
        "arxiv_running": runtime_state["arxiv_running"],
        "arxiv_last_run_at": runtime_state["arxiv_last_run_at"],
        "arxiv_last_summary": runtime_state["arxiv_last_summary"],
        "arxiv_last_error": runtime_state["arxiv_last_error"],
        "arxiv_last_milvus_summary": runtime_state["arxiv_last_milvus_summary"],
        "arxiv_schedule_hour": settings.arxiv_schedule_hour,
        "arxiv_collection": settings.arxiv_collection_name,
    }


@app.post("/ingest/run")
async def run_ingest_now() -> dict[str, Any]:
    return await _run_ingestion(trigger="manual")


@app.post("/arxiv/run")
async def run_arxiv_now() -> dict[str, Any]:
    return await _run_arxiv_update(trigger="manual")


@app.get("/", response_class=FileResponse)
async def index() -> FileResponse:
    return FileResponse("static/index.html")


@app.get("/api/starter-docs", response_model=StarterDocsResponse)
async def starter_docs() -> StarterDocsResponse:
    documents = await asyncio.to_thread(_load_arxiv_starter_docs, 3)
    if not documents:
        docs = await asyncio.to_thread(_load_documents)
        documents = _pick_starter_docs(docs, count=3)
    return StarterDocsResponse(
        documents=[
            StarterDoc(id=doc["id"], title=doc["title"], summary=doc["summary"])
            for doc in documents
        ]
    )


@app.get("/api/conversations", response_model=ConversationsResponse)
async def conversations() -> ConversationsResponse:
    try:
        return ConversationsResponse(conversations=await _list_conversations())
    except RedisError as exc:
        raise HTTPException(status_code=503, detail=f"Redis is not available: {exc}") from exc


@app.get("/api/conversations/{conversation_id}/messages", response_model=ConversationMessagesResponse)
async def conversation_messages(conversation_id: str) -> ConversationMessagesResponse:
    try:
        return ConversationMessagesResponse(
            conversation_id=conversation_id,
            messages=await _get_conversation_messages(conversation_id),
        )
    except RedisError as exc:
        raise HTTPException(status_code=503, detail=f"Redis is not available: {exc}") from exc


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str) -> dict[str, Any]:
    try:
        deleted = await _delete_conversation(conversation_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        return {"deleted": True, "conversation_id": conversation_id}
    except RedisError as exc:
        raise HTTPException(status_code=503, detail=f"Redis is not available: {exc}") from exc


@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is empty.")
    conversation_id = (payload.conversation_id or "").strip() or uuid.uuid4().hex

    try:
        await _append_message(conversation_id, role="user", text=message)
        history_messages = await _get_conversation_messages(conversation_id)
        history_payload = [
            {"role": item.role, "text": item.text}
            for item in history_messages
        ]
        qa = _resolve_qa_service()
        rag_result = await qa.answer(
            message,
            conversation_id,
            history_payload,
        )
        runtime_state["last_qa_error"] = None
        answer = rag_result.answer
        reasoning = rag_result.reasoning
        suggested_questions = await qa.suggest_follow_up_questions(
            message=message,
            retrieved=rag_result.retrieved,
            answer=answer,
            limit=2,
        )
        await _append_message(
            conversation_id,
            role="assistant",
            text=answer,
            reasoning=reasoning,
        )
    except Exception as exc:
        runtime_state["last_qa_error"] = str(exc)
        raise HTTPException(status_code=500, detail=f"Failed to answer with RAG: {exc}") from exc

    return ChatResponse(
        conversation_id=conversation_id,
        answer=answer,
        reasoning=reasoning,
        suggested_questions=suggested_questions,
    )


@app.post("/api/chat/stream")
async def chat_stream(payload: ChatRequest) -> StreamingResponse:
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is empty.")
    conversation_id = (payload.conversation_id or "").strip() or uuid.uuid4().hex

    async def event_generator():
        try:
            qa = _resolve_qa_service()
            await _append_message(conversation_id, role="user", text=message)
            history_messages = await _get_conversation_messages(conversation_id)
            history_payload = [{"role": item.role, "text": item.text} for item in history_messages]

            final_answer: str | None = None
            final_reasoning: str | None = None
            final_suggested_questions: list[str] = []
            async for progress in qa.answer_with_progress(
                message=message,
                conversation_id=conversation_id,
                chat_history=history_payload,
            ):
                if progress.kind == "stage":
                    stage = (progress.stage or "").strip()
                    label = STAGE_LABELS.get(stage, "답변 생성 중")
                    yield _sse_event("stage", {"stage": stage, "label": label})
                    continue

                if progress.kind == "final" and progress.response is not None:
                    final_answer = progress.response.answer
                    final_reasoning = progress.response.reasoning
                    final_suggested_questions = await qa.suggest_follow_up_questions(
                        message=message,
                        retrieved=progress.response.retrieved,
                        answer=final_answer,
                        limit=2,
                    )
                    await _append_message(
                        conversation_id,
                        role="assistant",
                        text=final_answer,
                        reasoning=final_reasoning,
                    )
                    runtime_state["last_qa_error"] = None
                    yield _sse_event(
                        "final",
                        {
                            "conversation_id": conversation_id,
                            "answer": final_answer,
                            "reasoning": final_reasoning,
                            "suggested_questions": final_suggested_questions,
                        },
                    )
                    break

            if final_answer is None:
                raise RuntimeError("No final response from QA pipeline.")

            yield _sse_event("done", {"ok": True})
        except Exception as exc:
            runtime_state["last_qa_error"] = str(exc)
            yield _sse_event("error", {"detail": f"Failed to answer with RAG: {exc}"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


app.mount("/static", StaticFiles(directory="static"), name="static")
