from __future__ import annotations

import asyncio
import json
import random
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from redis import Redis
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
    created_at: str


class ConversationMessagesResponse(BaseModel):
    conversation_id: str
    messages: list[ConversationMessage]


CHAT_KEY_PREFIX = "doc_organizer:chat"
CHAT_INDEX_KEY = f"{CHAT_KEY_PREFIX}:index"


def _chat_meta_key(conversation_id: str) -> str:
    return f"{CHAT_KEY_PREFIX}:{conversation_id}:meta"


def _chat_messages_key(conversation_id: str) -> str:
    return f"{CHAT_KEY_PREFIX}:{conversation_id}:messages"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _title_from_message(message: str) -> str:
    normalized = _normalize_space(message)
    if not normalized:
        return "새 대화"
    if len(normalized) <= 40:
        return normalized
    return f"{normalized[:40]}..."


def _resolve_redis_client() -> Redis:
    global redis_client
    if redis_client is not None:
        return redis_client
    redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
    redis_client.ping()
    return redis_client


def _create_or_touch_conversation(conversation_id: str, first_message: str | None = None) -> None:
    client = _resolve_redis_client()
    now_iso = _utc_now_iso()
    meta_key = _chat_meta_key(conversation_id)

    existing = client.hgetall(meta_key)
    if not existing:
        title = _title_from_message(first_message or "")
        client.hset(
            meta_key,
            mapping={
                "conversation_id": conversation_id,
                "title": title,
                "created_at": now_iso,
                "updated_at": now_iso,
            },
        )
    else:
        client.hset(meta_key, mapping={"updated_at": now_iso})
        if first_message and not existing.get("title"):
            client.hset(meta_key, mapping={"title": _title_from_message(first_message)})

    client.zadd(CHAT_INDEX_KEY, {conversation_id: time.time()})


def _append_message(conversation_id: str, role: str, text: str) -> None:
    client = _resolve_redis_client()
    payload = {
        "role": role,
        "text": text,
        "created_at": _utc_now_iso(),
    }
    client.rpush(_chat_messages_key(conversation_id), json.dumps(payload, ensure_ascii=False))
    _create_or_touch_conversation(conversation_id, first_message=text if role == "user" else None)


def _list_conversations(limit: int = 200) -> list[ConversationSummary]:
    client = _resolve_redis_client()
    ids = client.zrevrange(CHAT_INDEX_KEY, 0, max(0, limit - 1))
    out: list[ConversationSummary] = []
    for conversation_id in ids:
        meta = client.hgetall(_chat_meta_key(conversation_id))
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


def _get_conversation_messages(conversation_id: str) -> list[ConversationMessage]:
    client = _resolve_redis_client()
    raw_items = client.lrange(_chat_messages_key(conversation_id), 0, -1)
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
                created_at=str(item.get("created_at", _utc_now_iso())),
            )
        )
    return messages


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
                "summary": _first_sentence(
                    cleaned,
                    f"{title} 문서는 주요 내용을 요약할 수 있을 만큼 텍스트가 충분하지 않습니다.",
                ),
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
                    "summary": summary,
                }
            )

    if not candidates:
        return []
    if len(candidates) <= count:
        random.shuffle(candidates)
        return candidates
    return random.sample(candidates, count)


def _suggest_questions(message: str) -> list[str]:
    topic = re.sub(r"\s+", " ", message).strip()
    if len(topic) > 30:
        topic = f"{topic[:30]}..."
    return [
        f"방금 답변을 3줄로 요약해줘.",
        f"'{topic}' 관련해서 실무 체크리스트를 만들어줘.",
        f"근거가 된 문서 조각을 더 자세히 보여줘.",
    ]


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
        qa_service.close()
        qa_service = None
    if redis_client is not None:
        try:
            redis_client.close()
        except Exception:
            pass
        redis_client = None


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "running": runtime_state["running"],
        "interval_seconds": settings.interval_seconds,
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
    documents = _load_arxiv_starter_docs(count=3)
    if not documents:
        documents = _pick_starter_docs(_load_documents(), count=3)
    return StarterDocsResponse(
        documents=[
            StarterDoc(id=doc["id"], title=doc["title"], summary=doc["summary"])
            for doc in documents
        ]
    )


@app.get("/api/conversations", response_model=ConversationsResponse)
async def conversations() -> ConversationsResponse:
    try:
        return ConversationsResponse(conversations=_list_conversations())
    except RedisError as exc:
        raise HTTPException(status_code=503, detail=f"Redis is not available: {exc}") from exc


@app.get("/api/conversations/{conversation_id}/messages", response_model=ConversationMessagesResponse)
async def conversation_messages(conversation_id: str) -> ConversationMessagesResponse:
    try:
        return ConversationMessagesResponse(
            conversation_id=conversation_id,
            messages=_get_conversation_messages(conversation_id),
        )
    except RedisError as exc:
        raise HTTPException(status_code=503, detail=f"Redis is not available: {exc}") from exc


@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is empty.")
    conversation_id = (payload.conversation_id or "").strip() or uuid.uuid4().hex

    try:
        _append_message(conversation_id, role="user", text=message)
        rag_result = await asyncio.to_thread(_resolve_qa_service().answer, message)
        runtime_state["last_qa_error"] = None
        answer = rag_result.answer
        _append_message(conversation_id, role="assistant", text=answer)
    except Exception as exc:
        runtime_state["last_qa_error"] = str(exc)
        raise HTTPException(status_code=500, detail=f"Failed to answer with RAG: {exc}") from exc

    return ChatResponse(
        conversation_id=conversation_id,
        answer=answer,
        suggested_questions=_suggest_questions(message),
    )


app.mount("/static", StaticFiles(directory="static"), name="static")
