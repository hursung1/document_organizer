from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from doc_organizer.ingest import DocumentIngestionService
from doc_organizer.qa_service import DocumentQAService
from doc_organizer.settings import IngestionSettings

app = FastAPI(title="doc-organizer", version="0.1.0")

settings = IngestionSettings.from_env()
service: DocumentIngestionService | None = None
qa_service: DocumentQAService | None = None
ingest_lock = asyncio.Lock()
periodic_task: asyncio.Task | None = None
runtime_state: dict[str, Any] = {
    "running": False,
    "last_run_at": None,
    "last_summary": None,
    "last_error": None,
    "last_qa_error": None,
}


class StarterDoc(BaseModel):
    id: str
    title: str
    summary: str


class StarterDocsResponse(BaseModel):
    documents: list[StarterDoc]


class ChatStartRequest(BaseModel):
    document_id: str = Field(min_length=1)


class ChatRequest(BaseModel):
    document_id: str = Field(min_length=1)
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    document_id: str
    answer: str
    suggested_questions: list[str]


class RagChatRequest(BaseModel):
    message: str = Field(min_length=1)


class RagRetrievedChunk(BaseModel):
    id: str | int
    score: float
    source: str
    chunk_id: int
    text: str


class RagChatResponse(BaseModel):
    analyzed_query: dict[str, Any]
    retrieved: list[RagRetrievedChunk]
    answer: str


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


def _extract_context_snippets(content: str, question: str, top_k: int = 3) -> list[str]:
    if not content.strip():
        return []
    paragraphs = [p.strip() for p in content.split("\n") if p.strip()]
    if not paragraphs:
        paragraphs = re.split(r"(?<=[.!?다요])\s+", content)

    query_tokens = set(re.findall(r"[A-Za-z0-9가-힣]{2,}", question.lower()))
    scored: list[tuple[int, str]] = []
    for paragraph in paragraphs:
        p_tokens = set(re.findall(r"[A-Za-z0-9가-힣]{2,}", paragraph.lower()))
        score = len(query_tokens & p_tokens)
        scored.append((score, paragraph))
    scored.sort(key=lambda item: (item[0], len(item[1])), reverse=True)

    best = [text for score, text in scored if score > 0][:top_k]
    if best:
        return best
    return [text for _, text in scored[:top_k]]


def _suggest_questions(title: str) -> list[str]:
    return [
        f"{title} 문서의 핵심 목적을 한 번 더 정리해줘.",
        f"{title} 내용을 실제 업무에 적용할 때 주의할 점은 뭐야?",
        f"{title}에서 가장 중요한 항목 3가지를 알려줘.",
    ]


def _start_answer(document: dict[str, str]) -> str:
    title = document["title"]
    content = document.get("content", "").strip()
    if not content:
        return (
            f"{title} 문서는 현재 원본 파일만 확인되고 OCR 결과 텍스트가 충분하지 않아, "
            "문서 적재가 완료되면 상세 설명을 제공할 수 있습니다."
        )

    snippets = _extract_context_snippets(content, question=title, top_k=3)
    lines = [
        f"{title} 문서 설명:",
        f"- 요약: {_first_sentence(content, f'{title} 문서의 핵심 문장을 찾지 못했습니다.')}",
    ]
    for index, snippet in enumerate(snippets, start=1):
        shortened = snippet[:280] + ("..." if len(snippet) > 280 else "")
        lines.append(f"- 핵심 {index}: {shortened}")
    return "\n".join(lines)


def _chat_answer(document: dict[str, str], message: str) -> str:
    title = document["title"]
    content = document.get("content", "").strip()
    if not content:
        return (
            f"{title} 문서는 아직 본문 텍스트가 충분하지 않아 질문에 대한 근거 답변이 어렵습니다. "
            "인제스트 완료 후 다시 질문해 주세요."
        )

    snippets = _extract_context_snippets(content, question=message, top_k=3)
    lines = [f"질문: {message}", f"{title} 문서를 기준으로 설명합니다."]
    for idx, snippet in enumerate(snippets, start=1):
        clipped = snippet[:320] + ("..." if len(snippet) > 320 else "")
        lines.append(f"{idx}. {clipped}")
    lines.append("필요하면 특정 항목(예: 목적, 절차, 금액, 일정)만 따로 더 자세히 풀어드리겠습니다.")
    return "\n".join(lines)


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


@app.on_event("startup")
async def on_startup() -> None:
    global service
    global qa_service
    global periodic_task
    settings.ensure_directories()
    try:
        service = DocumentIngestionService(settings)
        qa_service = DocumentQAService(settings)
    except Exception as exc:
        runtime_state["last_error"] = str(exc)
    periodic_task = asyncio.create_task(_periodic_ingestion_loop())


@app.on_event("shutdown")
async def on_shutdown() -> None:
    global periodic_task
    if periodic_task:
        periodic_task.cancel()
        try:
            await periodic_task
        except asyncio.CancelledError:
            pass
        periodic_task = None


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
    }


@app.post("/ingest/run")
async def run_ingest_now() -> dict[str, Any]:
    return await _run_ingestion(trigger="manual")


@app.get("/", response_class=FileResponse)
async def index() -> FileResponse:
    return FileResponse("static/index.html")


@app.get("/api/starter-docs", response_model=StarterDocsResponse)
async def starter_docs() -> StarterDocsResponse:
    documents = _pick_starter_docs(_load_documents(), count=3)
    return StarterDocsResponse(
        documents=[
            StarterDoc(id=doc["id"], title=doc["title"], summary=doc["summary"])
            for doc in documents
        ]
    )


def _find_document(document_id: str) -> dict[str, str]:
    for doc in _load_documents():
        if doc["id"] == document_id:
            return doc
    raise HTTPException(status_code=404, detail="Document not found.")


@app.post("/api/chat/start", response_model=ChatResponse)
async def start_chat(payload: ChatStartRequest) -> ChatResponse:
    document = _find_document(payload.document_id)
    return ChatResponse(
        document_id=document["id"],
        answer=_start_answer(document),
        suggested_questions=_suggest_questions(document["title"]),
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    document = _find_document(payload.document_id)
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is empty.")

    try:
        rag_result = await asyncio.to_thread(_resolve_qa_service().answer, message)
        runtime_state["last_qa_error"] = None
        answer = rag_result.answer
    except Exception as exc:
        runtime_state["last_qa_error"] = str(exc)
        raise HTTPException(status_code=500, detail="Failed to answer with RAG.") from exc

    return ChatResponse(
        document_id=document["id"],
        answer=answer,
        suggested_questions=_suggest_questions(document["title"]),
    )


@app.post("/api/rag/chat", response_model=RagChatResponse)
async def rag_chat(payload: RagChatRequest) -> RagChatResponse:
    global qa_service
    if qa_service is None:
        try:
            qa_service = DocumentQAService(settings)
            runtime_state["last_qa_error"] = None
        except Exception as exc:
            runtime_state["last_qa_error"] = str(exc)
            raise HTTPException(status_code=503, detail="RAG service is not ready.")

    try:
        result = await asyncio.to_thread(qa_service.answer, payload.message.strip())
        return RagChatResponse(
            analyzed_query=result.to_dict()["analyzed_query"],
            retrieved=[RagRetrievedChunk(**item) for item in result.to_dict()["retrieved"]],
            answer=result.answer,
        )
    except Exception as exc:
        runtime_state["last_qa_error"] = str(exc)
        raise HTTPException(status_code=500, detail="Failed to answer with RAG.") from exc


app.mount("/static", StaticFiles(directory="static"), name="static")
