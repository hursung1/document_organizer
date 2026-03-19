from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Query
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
logger = logging.getLogger(__name__)

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
starter_summary_llm: Any | None = None
starter_summary_llm_initialized = False
ARXIV_MANUAL_LOOKBACK_DAYS_DEFAULT = 10
STARTER_SUMMARY_SENTENCE_COUNT = 3
STARTER_SUMMARY_MAX_CHARS = 1200
STARTER_SUMMARY_FIRST_TOKEN_TIMEOUT_SEC = 15.0
STARTER_SUMMARY_STREAM_IDLE_TIMEOUT_SEC = 20.0
STARTER_SUMMARY_SYNC_TIMEOUT_SEC = 20.0


class StarterDoc(BaseModel):
    id: str
    title: str
    summary: str
    raw_summary: str | None = None
    summary_source: str | None = None


class StarterDocsResponse(BaseModel):
    documents: list[StarterDoc]


class StarterSummaryRequest(BaseModel):
    id: str
    title: str
    summary: str = ""
    raw_summary: str | None = None
    summary_source: str | None = None


class StarterSummaryResponse(BaseModel):
    id: str
    summary: str


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


def _coerce_llm_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_coerce_llm_content(item) for item in value]
        return "\n".join([part for part in parts if part]).strip()
    if isinstance(value, dict):
        preferred_keys = ("text", "content", "summary", "output_text", "message", "answer")
        parts: list[str] = []
        for key in preferred_keys:
            if key in value:
                text = _coerce_llm_content(value.get(key))
                if text:
                    parts.append(text)
        if parts:
            return "\n".join(parts).strip()
        nested = [_coerce_llm_content(item) for item in value.values()]
        return "\n".join([part for part in nested if part]).strip()
    return str(value).strip()


def _ensure_sentence_end(text: str) -> str:
    trimmed = _normalize_space(text)
    if not trimmed:
        return ""
    if trimmed.endswith((".", "!", "?", "다", "요")):
        return trimmed
    return f"{trimmed}."


def _pick_summary_sentences(text: str, limit: int = STARTER_SUMMARY_SENTENCE_COUNT) -> list[str]:
    sentences: list[str] = []
    for sentence in _split_sentences(text):
        cleaned = re.sub(r"^[\-\*\d\.\)\s]+", "", sentence)
        cleaned = _ensure_sentence_end(cleaned)
        if not cleaned:
            continue
        sentences.append(cleaned)
        if len(sentences) >= limit:
            break
    return sentences


def _fallback_starter_summary(title: str, summary: str) -> str:
    base = _pick_summary_sentences(summary, STARTER_SUMMARY_SENTENCE_COUNT)
    generic = [
        f"{_normalize_space(title) or '이 논문'}은 핵심 문제와 해결 아이디어를 간결하게 제시한다.",
        "핵심 방법과 실험 결과를 함께 보면 성능의 근거를 빠르게 파악할 수 있다.",
        "실제 적용 가능성은 데이터 조건과 계산 비용을 함께 검토해 판단하는 것이 좋다.",
    ]
    for sentence in generic:
        if len(base) >= STARTER_SUMMARY_SENTENCE_COUNT:
            break
        base.append(_ensure_sentence_end(sentence))
    return " ".join(base[:STARTER_SUMMARY_SENTENCE_COUNT])


def _build_starter_chat_llm() -> Any | None:
    provider = (settings.llm_provider or "ollama").strip().lower()
    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except Exception:
            return None
        return ChatOllama(
            model=settings.qa_model,
            base_url=settings.ollama_host,
            temperature=0,
            reasoning=settings.ollama_reasoning,
        )
    if provider == "gemini":
        if not settings.gemini_api_key:
            return None
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except Exception:
            return None
        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0,
        )
    return None


def _resolve_starter_chat_llm() -> Any | None:
    global starter_summary_llm
    global starter_summary_llm_initialized
    if starter_summary_llm_initialized:
        return starter_summary_llm
    starter_summary_llm_initialized = True
    starter_summary_llm = _build_starter_chat_llm()
    logger.info(
        "starter_summary_llm_initialized provider=%s model=%s ollama_host=%s has_client=%s",
        (settings.llm_provider or "").strip().lower(),
        (settings.qa_model if (settings.llm_provider or "").strip().lower() == "ollama" else settings.gemini_model),
        settings.ollama_host,
        bool(starter_summary_llm),
    )
    return starter_summary_llm


async def _invoke_with_timeout(
    llm: Any,
    messages: list[tuple[str, str]],
    timeout_sec: float,
) -> Any:
    async def _invoke() -> Any:
        ainvoke = getattr(llm, "ainvoke", None)
        if callable(ainvoke):
            return await ainvoke(messages)
        return await asyncio.to_thread(llm.invoke, messages)

    try:
        return await asyncio.wait_for(_invoke(), timeout=timeout_sec)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"Starter summary LLM timeout after {timeout_sec:.1f}s") from exc


def _starter_summary_prompts(title: str, source_text: str) -> tuple[str, str]:
    system_prompt = (
        "너는 논문 요약 편집자다. 입력된 summary를 한국어 자연문 3문장으로 다시 요약해라.\n"
        "규칙:\n"
        "1) 정확히 3문장\n"
        "2) 각 문장 20~55자 내외\n"
        "3) 불릿/번호/마크다운 금지\n"
        "4) 논문 제목을 반복하지 말고 핵심 문제, 방법, 결과를 담을 것"
    )
    user_prompt = (
        f"논문 제목:\n{title}\n\n"
        f"원본 summary:\n{source_text}\n\n"
        "한국어 3문장 요약:"
    )
    return system_prompt, user_prompt


def _finalize_streamed_summary(generated_text: str) -> str | None:
    candidates = _pick_summary_sentences(generated_text, STARTER_SUMMARY_SENTENCE_COUNT)
    if len(candidates) < STARTER_SUMMARY_SENTENCE_COUNT:
        return None
    return " ".join(candidates[:STARTER_SUMMARY_SENTENCE_COUNT])


def _split_stream_tokens(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"\S+\s*|\s+", text)


_REASONING_KEYS = ("reasoning_content", "thinking", "reasoning", "thought", "think")
_REASONING_BLOCK_TYPES = {"reasoning", "thinking", "reasoning_content", "thought", "think"}


def _coerce_visible_stream_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                item_type = _normalize_space(str(item.get("type", ""))).lower()
                if item_type in _REASONING_BLOCK_TYPES:
                    continue
            text = _coerce_visible_stream_content(item)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    if isinstance(value, dict):
        item_type = _normalize_space(str(value.get("type", ""))).lower()
        if item_type in _REASONING_BLOCK_TYPES:
            return ""
    return _coerce_llm_content(value)


def _has_reasoning_marker(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, list):
        return any(_has_reasoning_marker(item) for item in value)
    if isinstance(value, dict):
        item_type = _normalize_space(str(value.get("type", ""))).lower()
        if item_type in _REASONING_BLOCK_TYPES:
            return bool(_coerce_llm_content(value))
        for key in _REASONING_KEYS:
            if key not in value:
                continue
            if _coerce_llm_content(value.get(key)):
                return True
        return any(_has_reasoning_marker(item) for item in value.values())
    return False


def _extract_stream_chunk_piece(chunk: Any) -> tuple[str, bool]:
    reasoning_detected = False

    raw_content = getattr(chunk, "content", None)
    visible_piece = _coerce_visible_stream_content(raw_content)
    if _has_reasoning_marker(raw_content):
        reasoning_detected = True

    for container_name in ("additional_kwargs", "response_metadata"):
        container = getattr(chunk, container_name, None)
        if not isinstance(container, dict):
            continue
        for key in _REASONING_KEYS:
            text = _coerce_llm_content(container.get(key))
            if not text:
                continue
            reasoning_detected = True

    return visible_piece, reasoning_detected


async def _iter_llm_summary_tokens(
    llm: Any,
    messages: list[tuple[str, str]],
    *,
    first_token_timeout_sec: float,
    idle_timeout_sec: float,
) -> AsyncGenerator[tuple[str, bool], None]:
    astream = getattr(llm, "astream", None)
    if callable(astream):
        accumulated = ""
        received_any_token = False
        stream_iter = astream(messages).__aiter__()

        while True:
            timeout_sec = (
                first_token_timeout_sec
                if not received_any_token
                else idle_timeout_sec
            )
            try:
                chunk = await asyncio.wait_for(stream_iter.__anext__(), timeout=timeout_sec)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError as exc:
                if not received_any_token:
                    raise TimeoutError(
                        f"Starter summary stream timeout before first token ({first_token_timeout_sec:.1f}s)"
                    ) from exc
                raise TimeoutError(
                    f"Starter summary stream stalled after token ({idle_timeout_sec:.1f}s idle)"
                ) from exc

            raw_piece, from_reasoning = _extract_stream_chunk_piece(chunk)
            if from_reasoning and not received_any_token:
                received_any_token = True
            if not raw_piece:
                continue
            if raw_piece.startswith(accumulated):
                delta = raw_piece[len(accumulated):]
                accumulated = raw_piece
            else:
                delta = raw_piece
                accumulated += raw_piece
            if delta:
                received_any_token = True
                yield delta, from_reasoning

        if not received_any_token:
            raise TimeoutError(
                f"Starter summary stream ended without tokens ({first_token_timeout_sec:.1f}s first-token budget)"
            )
        return

    response = await _invoke_with_timeout(
        llm,
        messages,
        timeout_sec=STARTER_SUMMARY_SYNC_TIMEOUT_SEC,
    )
    generated = _normalize_space(_coerce_llm_content(getattr(response, "content", "")))
    for piece in _split_stream_tokens(generated):
        if piece:
            yield piece, False


async def _summarize_arxiv_summary(doc_id: str, title: str, summary: str) -> str:
    clean_summary = _normalize_space(summary)
    if not clean_summary:
        return _fallback_starter_summary(title=title, summary=summary)
    llm = _resolve_starter_chat_llm()
    if llm is None:
        return _fallback_starter_summary(title=title, summary=clean_summary)

    source_text = clean_summary[:STARTER_SUMMARY_MAX_CHARS]
    system_prompt, user_prompt = _starter_summary_prompts(title=title, source_text=source_text)

    started = time.perf_counter()
    provider = (settings.llm_provider or "").strip().lower() or "unknown"
    model_name = settings.qa_model if provider == "ollama" else settings.gemini_model
    logger.info(
        "starter_summary_llm_call_start doc_id=%s provider=%s model=%s source_chars=%d",
        doc_id[:120],
        provider,
        model_name,
        len(source_text),
    )
    try:
        response = await _invoke_with_timeout(
            llm,
            [("system", system_prompt), ("human", user_prompt)],
            timeout_sec=STARTER_SUMMARY_SYNC_TIMEOUT_SEC,
        )
        generated = _normalize_space(_coerce_llm_content(getattr(response, "content", "")))
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "starter_summary_llm_call_success doc_id=%s provider=%s model=%s elapsed_ms=%d generated_chars=%d",
            doc_id[:120],
            provider,
            model_name,
            elapsed_ms,
            len(generated),
        )
    except Exception:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.warning(
            "starter_summary_llm_call_fallback doc_id=%s provider=%s model=%s elapsed_ms=%d",
            doc_id[:120],
            provider,
            model_name,
            elapsed_ms,
            exc_info=True,
        )
        return _fallback_starter_summary(title=title, summary=source_text)

    finalized = _finalize_streamed_summary(generated)
    if finalized is None:
        return _fallback_starter_summary(title=title, summary=source_text)
    return finalized


async def _generate_starter_summary(
    *,
    doc_id: str,
    title: str,
    summary: str,
    raw_summary: str,
    summary_source: str,
) -> str:
    clean_title = _normalize_space(title) or "이 논문"
    clean_summary = _normalize_space(summary)
    clean_raw_summary = _normalize_space(raw_summary)
    clean_source = _normalize_space(summary_source).lower()

    if clean_source == "arxiv" and clean_raw_summary:
        return await _summarize_arxiv_summary(
            doc_id=doc_id,
            title=clean_title,
            summary=clean_raw_summary,
        )

    base_text = clean_raw_summary or clean_summary
    if not base_text:
        return _fallback_starter_summary(title=clean_title, summary=clean_summary)

    picked = _pick_summary_sentences(base_text, STARTER_SUMMARY_SENTENCE_COUNT)
    if len(picked) >= STARTER_SUMMARY_SENTENCE_COUNT:
        return " ".join(picked[:STARTER_SUMMARY_SENTENCE_COUNT])
    return _fallback_starter_summary(title=clean_title, summary=base_text)


async def _stream_arxiv_starter_summary_events(
    *,
    doc_id: str,
    title: str,
    summary: str,
) -> AsyncGenerator[str, None]:
    clean_id = _normalize_space(doc_id)
    clean_title = _normalize_space(title) or "이 논문"
    clean_summary = _normalize_space(summary)
    source_text = clean_summary[:STARTER_SUMMARY_MAX_CHARS]

    llm = _resolve_starter_chat_llm()
    provider = (settings.llm_provider or "").strip().lower() or "unknown"
    model_name = settings.qa_model if provider == "ollama" else settings.gemini_model
    if llm is None or not source_text:
        logger.warning(
            "starter_summary_stream_unavailable doc_id=%s provider=%s model=%s has_client=%s",
            clean_id[:120],
            provider,
            model_name,
            bool(llm),
        )
        yield _sse_event("error", {"id": clean_id, "message": "요약 생성 실패"})
        yield _sse_event("done", {"id": clean_id})
        return

    system_prompt, user_prompt = _starter_summary_prompts(
        title=clean_title,
        source_text=source_text,
    )
    messages = [("system", system_prompt), ("human", user_prompt)]

    for attempt in range(1, 3):
        started = time.perf_counter()
        chunk_parts: list[str] = []
        first_token_logged = False
        yield _sse_event("start", {"id": clean_id, "attempt": attempt})
        logger.info(
            "starter_summary_stream_call_start doc_id=%s provider=%s model=%s attempt=%d source_chars=%d",
            clean_id[:120],
            provider,
            model_name,
            attempt,
            len(source_text),
        )

        try:
            async for piece, from_reasoning in _iter_llm_summary_tokens(
                llm,
                messages,
                first_token_timeout_sec=STARTER_SUMMARY_FIRST_TOKEN_TIMEOUT_SEC,
                idle_timeout_sec=STARTER_SUMMARY_STREAM_IDLE_TIMEOUT_SEC,
            ):
                if not piece:
                    continue
                if not first_token_logged:
                    first_token_logged = True
                    first_token_ms = int((time.perf_counter() - started) * 1000)
                    logger.info(
                        "starter_summary_stream_first_token doc_id=%s provider=%s model=%s attempt=%d first_token_ms=%d from_reasoning=%s",
                        clean_id[:120],
                        provider,
                        model_name,
                        attempt,
                        first_token_ms,
                        from_reasoning,
                    )
                chunk_parts.append(piece)
                yield _sse_event("token", {"id": clean_id, "token": piece})

            generated = _normalize_space("".join(chunk_parts))
            finalized = _finalize_streamed_summary(generated)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            if finalized:
                logger.info(
                    "starter_summary_stream_call_success doc_id=%s provider=%s model=%s attempt=%d elapsed_ms=%d generated_chars=%d",
                    clean_id[:120],
                    provider,
                    model_name,
                    attempt,
                    elapsed_ms,
                    len(generated),
                )
                yield _sse_event("complete", {"id": clean_id, "summary": finalized})
                yield _sse_event("done", {"id": clean_id})
                return

            raise ValueError("summary output does not contain 3 valid sentences")
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            logger.warning(
                "starter_summary_stream_call_retry doc_id=%s provider=%s model=%s attempt=%d elapsed_ms=%d err=%s",
                clean_id[:120],
                provider,
                model_name,
                attempt,
                elapsed_ms,
                exc,
            )
            if attempt < 2:
                yield _sse_event("retry", {"id": clean_id, "attempt": attempt + 1})
                continue
            yield _sse_event("error", {"id": clean_id, "message": "요약 생성 실패"})
            yield _sse_event("done", {"id": clean_id})
            return


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
                    "summary": summary,
                    "raw_summary": summary,
                    "summary_source": "arxiv",
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


async def _run_arxiv_update(
    trigger: str,
    *,
    lookback_days: int = ARXIV_MANUAL_LOOKBACK_DAYS_DEFAULT,
) -> dict[str, Any]:
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
            if trigger == "manual":
                summary = await asyncio.to_thread(
                    arxiv_service.run_last_month,
                    None,
                    lookback_days,
                )
                output_files = [Path(path) for path in summary.output_files]
            else:
                summary = await asyncio.to_thread(arxiv_service.run_once)
                output_files = [Path(summary.output_file)]

            milvus_runs: list[dict[str, Any]] = []
            milvus_aggregated = {
                "files_processed": len(output_files),
                "total_papers": 0,
                "embedded_papers": 0,
                "upserted_papers": 0,
                "failed_papers": 0,
                "collection_name": settings.arxiv_collection_name,
            }
            for output_file in output_files:
                milvus_summary = await asyncio.to_thread(
                    arxiv_milvus_service.upsert_from_file,
                    output_file,
                )
                run_item = milvus_summary.to_dict() | {"output_file": str(output_file)}
                milvus_runs.append(run_item)
                milvus_aggregated["total_papers"] += milvus_summary.total_papers
                milvus_aggregated["embedded_papers"] += milvus_summary.embedded_papers
                milvus_aggregated["upserted_papers"] += milvus_summary.upserted_papers
                milvus_aggregated["failed_papers"] += milvus_summary.failed_papers

            runtime_state["arxiv_last_run_at"] = datetime.now(UTC).isoformat()
            runtime_state["arxiv_last_summary"] = summary.to_dict() | {
                "milvus": milvus_aggregated,
            }
            runtime_state["arxiv_last_milvus_summary"] = milvus_aggregated | {
                "runs": milvus_runs,
            }
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
        "llm_provider": settings.llm_provider,
        "qa_model": settings.qa_model,
        "ollama_reasoning": settings.ollama_reasoning,
        "starter_summary_first_token_timeout_sec": STARTER_SUMMARY_FIRST_TOKEN_TIMEOUT_SEC,
        "starter_summary_stream_idle_timeout_sec": STARTER_SUMMARY_STREAM_IDLE_TIMEOUT_SEC,
        "starter_summary_sync_timeout_sec": STARTER_SUMMARY_SYNC_TIMEOUT_SEC,
    }


@app.post("/ingest/run")
async def run_ingest_now() -> dict[str, Any]:
    return await _run_ingestion(trigger="manual")


@app.post("/arxiv/run")
async def run_arxiv_now(
    lookback_days: int = Query(default=ARXIV_MANUAL_LOOKBACK_DAYS_DEFAULT, ge=1),
) -> dict[str, Any]:
    return await _run_arxiv_update(trigger="manual", lookback_days=lookback_days)


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
            StarterDoc(
                id=doc["id"],
                title=doc["title"],
                summary=doc["summary"],
                raw_summary=doc.get("raw_summary"),
                summary_source=doc.get("summary_source"),
            )
            for doc in documents
        ]
    )


@app.post("/api/starter-docs/summary", response_model=StarterSummaryResponse)
async def starter_doc_summary(payload: StarterSummaryRequest) -> StarterSummaryResponse:
    clean_id = _normalize_space(payload.id)
    clean_title = _normalize_space(payload.title)
    clean_summary = _normalize_space(payload.summary)
    clean_raw_summary = _normalize_space(payload.raw_summary or "")
    clean_source = _normalize_space(payload.summary_source or "")

    generated = await _generate_starter_summary(
        doc_id=clean_id,
        title=clean_title,
        summary=clean_summary,
        raw_summary=clean_raw_summary,
        summary_source=clean_source,
    )
    return StarterSummaryResponse(id=clean_id, summary=generated)


@app.post("/api/starter-docs/summary/stream")
async def starter_doc_summary_stream(payload: StarterSummaryRequest) -> StreamingResponse:
    clean_id = _normalize_space(payload.id)
    clean_title = _normalize_space(payload.title)
    clean_summary = _normalize_space(payload.summary)
    clean_raw_summary = _normalize_space(payload.raw_summary or "")
    clean_source = _normalize_space(payload.summary_source or "").lower()

    async def event_stream() -> AsyncGenerator[str, None]:
        if clean_source != "arxiv":
            yield _sse_event("start", {"id": clean_id, "attempt": 1})
            generated = await _generate_starter_summary(
                doc_id=clean_id,
                title=clean_title,
                summary=clean_summary,
                raw_summary=clean_raw_summary,
                summary_source=clean_source,
            )
            for piece in _split_stream_tokens(generated):
                yield _sse_event("token", {"id": clean_id, "token": piece})
            yield _sse_event("complete", {"id": clean_id, "summary": generated})
            yield _sse_event("done", {"id": clean_id})
            return

        async for event in _stream_arxiv_starter_summary_events(
            doc_id=clean_id,
            title=clean_title,
            summary=clean_raw_summary or clean_summary,
        ):
            yield event

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
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
