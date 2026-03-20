from __future__ import annotations

import asyncio
from contextlib import AbstractAsyncContextManager
import json
import logging
import re
from time import perf_counter
import uuid
from dataclasses import asdict, dataclass
from typing import Any, AsyncGenerator, Awaitable, Callable, TypedDict

from .arxiv_pdf_enricher import ArxivPdfEnricher, PdfEvidence
from .query_analyzer import AnalyzedQuery, analyze_user_query, parse_llm_analysis
from .settings import IngestionSettings

logger = logging.getLogger(__name__)

ENRICH_TRIGGER_SIGNALS = (
    "원문",
    "본문",
    "상세",
    "자세히",
    "구체",
    "깊게",
    "디테일",
    "세부",
    "수식",
    "실험",
    "ablation",
    "증명",
    "algorithm",
)

NEW_ID_PATTERN = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b")
OLD_ID_PATTERN = re.compile(r"\b[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?\b")
STREAM_TOKEN_PATTERN = re.compile(r"\S+\s*|\s+")


class RAGState(TypedDict, total=False):
    message: str
    chat_history: list[dict[str, Any]]
    analyzed_query: AnalyzedQuery
    tool_plan: "ToolPlan"
    memory_candidates: list[dict[str, str]]
    retrieved: list["RetrievalResult"]
    pdf_evidence: list[PdfEvidence]
    answer: str
    reasoning: str
    error: str
    pdf_enrich_error: str
    pdf_enrich_attempted: bool
    pdf_enrich_used: int
    pdf_cache_hit_count: int
    pdf_download_fail_count: int


@dataclass(slots=True)
class RetrievalResult:
    id: int | str
    score: float
    source: str
    chunk_id: int
    text: str
    arxiv_id: str | None = None
    pdf_url: str | None = None
    source_url: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class QAResponse:
    analyzed_query: AnalyzedQuery
    retrieved: list[RetrievalResult]
    answer: str
    reasoning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "analyzed_query": asdict(self.analyzed_query),
            "retrieved": [asdict(item) for item in self.retrieved],
            "answer": self.answer,
            "reasoning": self.reasoning,
        }


@dataclass(slots=True)
class QAProgressEvent:
    kind: str
    stage: str | None = None
    token: str | None = None
    plan: dict[str, Any] | None = None
    response: QAResponse | None = None


@dataclass(slots=True)
class ToolPlan:
    use_vdb: bool
    use_pdf: bool
    reasons: list[str]
    candidate_arxiv_ids: list[str]


class DocumentQAService:
    def __init__(self, settings: IngestionSettings) -> None:
        from .embeddings import OllamaEmbeddingClient
        from .milvus_store import MilvusStore

        self.settings = settings
        self._checkpointer_acm: AbstractAsyncContextManager | None = None
        self._checkpointer: Any | None = None
        self._graph: Any | None = None
        self._init_lock = asyncio.Lock()
        self.embedder = OllamaEmbeddingClient(
            host=settings.ollama_host,
            model=settings.embedding_model,
        )
        self.store = MilvusStore(
            collection_name=settings.arxiv_collection_name,
            uri=settings.milvus_uri,
            user=settings.milvus_user,
            password=settings.milvus_password,
        )
        self.store.connect()
        self.chat_llm = self._build_chat_llm()
        self.pdf_enricher = ArxivPdfEnricher(settings)

    def _build_chat_llm(self):
        provider = (self.settings.llm_provider or "ollama").strip().lower()
        if provider == "ollama":
            try:
                from langchain_ollama import ChatOllama
            except Exception as exc:
                raise RuntimeError(
                    "Ollama provider import 실패: langchain-ollama 설치 필요"
                ) from exc
            return ChatOllama(
                model=self.settings.qa_model,
                base_url=self.settings.ollama_host,
                temperature=0,
                reasoning=self.settings.qa_ollama_reasoning,
            )
        if provider == "gemini":
            if not self.settings.gemini_api_key:
                raise RuntimeError(
                    "LLM_PROVIDER=gemini 이지만 GEMINI_API_KEY 가 설정되지 않았습니다."
                )
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except Exception as exc:
                raise RuntimeError(
                    "Gemini provider import 실패: langchain-google-genai 설치 필요"
                ) from exc
            return ChatGoogleGenerativeAI(
                model=self.settings.gemini_model,
                google_api_key=self.settings.gemini_api_key,
                temperature=0,
                streaming=True,
            )
        raise RuntimeError(f"지원하지 않는 LLM_PROVIDER: {self.settings.llm_provider}")

    async def answer(
        self,
        message: str,
        conversation_id: str | None = None,
        chat_history: list[dict[str, Any]] | None = None,
    ) -> QAResponse:
        final_response: QAResponse | None = None
        async for event in self.answer_with_progress(
            message=message,
            conversation_id=conversation_id,
            chat_history=chat_history,
        ):
            if event.kind == "final" and event.response is not None:
                final_response = event.response
        if final_response is None:
            fallback_analyzed = analyze_user_query(message)
            return QAResponse(
                analyzed_query=fallback_analyzed,
                retrieved=[],
                answer="답변 생성에 실패했습니다.",
            )
        return final_response

    async def answer_with_progress(
        self,
        message: str,
        conversation_id: str | None = None,
        chat_history: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[QAProgressEvent, None]:
        await self._ensure_graph_initialized()
        fallback_analyzed = analyze_user_query(message)
        if not fallback_analyzed.normalized:
            yield QAProgressEvent(
                kind="final",
                response=QAResponse(
                    analyzed_query=fallback_analyzed,
                    retrieved=[],
                    answer="질문을 입력해 주세요.",
                ),
            )
            return

        state: RAGState = {"message": message}
        if chat_history:
            state["chat_history"] = chat_history[-8:]

        yield QAProgressEvent(kind="stage", stage="planning")
        planning_out = await self._timed_node("planning", self._node_planning)(state)
        state.update(planning_out)
        plan = state.get("tool_plan") or ToolPlan(
            use_vdb=True,
            use_pdf=False,
            reasons=["planning 결과가 없어 기본 검색 경로를 사용합니다."],
            candidate_arxiv_ids=[],
        )
        yield QAProgressEvent(
            kind="plan",
            plan={
                "use_vdb": plan.use_vdb,
                "use_pdf": plan.use_pdf,
                "reasons": plan.reasons,
                "candidate_arxiv_ids": plan.candidate_arxiv_ids,
            },
        )

        if plan.use_vdb:
            yield QAProgressEvent(kind="stage", stage="retrieve_docs")
            retrieve_out = await self._timed_node("retrieve_docs", self._node_retrieve_docs)(state)
            state.update(retrieve_out)
        else:
            state.update({"retrieved": []})

        if plan.use_pdf:
            yield QAProgressEvent(kind="stage", stage="enrich_with_arxiv_pdf")
            enrich_out = await self._timed_node(
                "enrich_with_arxiv_pdf",
                self._node_enrich_with_arxiv_pdf,
            )(state)
            state.update(enrich_out)
        else:
            state.update(
                {
                    "pdf_evidence": [],
                    "pdf_enrich_attempted": False,
                    "pdf_enrich_used": 0,
                    "pdf_cache_hit_count": 0,
                    "pdf_download_fail_count": 0,
                }
            )

        route = self._conditional_edge_doc_base_or_not(state)
        yield QAProgressEvent(kind="stage", stage="generate_answer")
        token_queue: asyncio.Queue[str] = asyncio.Queue()

        async def _on_token(token: str) -> None:
            if token:
                await token_queue.put(token)

        if route == "generate_answer_doc_base":
            async def _generate_doc_base(with_state: RAGState) -> RAGState:
                return await self._node_generate_answer_doc_base(with_state, token_callback=_on_token)

            generate_task = asyncio.create_task(
                self._timed_node("generate_answer_doc_base", _generate_doc_base)(state)
            )
        else:
            async def _generate_llm(with_state: RAGState) -> RAGState:
                return await self._node_generate_answer_llm(with_state, token_callback=_on_token)

            generate_task = asyncio.create_task(
                self._timed_node("generate_answer_llm", _generate_llm)(state)
            )

        while True:
            if generate_task.done() and token_queue.empty():
                break
            try:
                token = await asyncio.wait_for(token_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            yield QAProgressEvent(kind="token", token=token)

        generate_out = await generate_task
        state.update(generate_out)

        analyzed = state.get("analyzed_query") or fallback_analyzed
        retrieved = state.get("retrieved") or []
        answer = state.get("answer") or state.get("error") or "답변 생성에 실패했습니다."
        reasoning = state.get("reasoning")
        yield QAProgressEvent(
            kind="final",
            response=QAResponse(
                analyzed_query=analyzed,
                retrieved=retrieved,
                answer=answer,
                reasoning=reasoning,
            ),
        )

    async def suggest_follow_up_questions(
        self,
        message: str,
        retrieved: list[RetrievalResult],
        answer: str = "",
        limit: int = 2,
    ) -> list[str]:
        target_count = max(1, min(limit, 4))
        fallback = self._fallback_follow_up_questions(message, target_count)

        context_lines: list[str] = []
        for idx, item in enumerate(retrieved[:4], start=1):
            source = item.source.rsplit("/", 1)[-1] if item.source else "unknown"
            snippet = " ".join(item.text.split())
            if not snippet:
                continue
            context_lines.append(
                f"[{idx}] source={source}, score={item.score:.4f}\n{snippet[:700]}"
            )
        context_text = "\n\n".join(context_lines) if context_lines else "(검색 문서 내용 없음)"

        system_prompt = (
            "너는 사용자가 다음에 물어볼 만한 후속 질문을 생성하는 도우미다.\n"
            "반드시 JSON으로만 답하고, 형식은 {\"questions\":[\"...\", \"...\"]} 이어야 한다.\n"
            "규칙:\n"
            "1) 질문은 한국어\n"
            "2) 현재 사용자 질문과 검색된 문서 내용에 직접 연결될 것\n"
            "3) 중복/표현만 다른 질문 금지\n"
            "4) 개수는 정확히 요청 개수"
        )
        user_prompt = (
            f"사용자 질문:\n{message}\n\n"
            f"검색된 문서:\n{context_text}\n\n"
            f"현재 답변:\n{answer}\n\n"
            f"질문 {target_count}개를 생성해라."
        )

        try:
            response = await self.chat_llm.ainvoke(
                [
                    ("system", system_prompt),
                    ("human", user_prompt),
                ]
            )
            raw = str(getattr(response, "content", "") or "").strip()
            parsed = self._parse_follow_up_questions(raw, target_count)
            if parsed:
                return parsed
            return fallback
        except Exception:
            return fallback

    @staticmethod
    def _parse_follow_up_questions(raw_text: str, limit: int) -> list[str]:
        if not raw_text.strip():
            return []

        candidates = [raw_text.strip()]
        fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, flags=re.DOTALL)
        candidates.extend(fenced)
        brace_match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if brace_match:
            candidates.append(brace_match.group(0))

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            raw_questions = payload.get("questions")
            if not isinstance(raw_questions, list):
                continue
            normalized: list[str] = []
            seen: set[str] = set()
            for item in raw_questions:
                if not isinstance(item, str):
                    continue
                text = re.sub(r"\s+", " ", item).strip()
                if len(text) < 4:
                    continue
                dedup_key = text.lower()
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                normalized.append(text)
                if len(normalized) >= limit:
                    break
            if len(normalized) >= limit:
                return normalized[:limit]
        return []

    @staticmethod
    def _fallback_follow_up_questions(message: str, limit: int) -> list[str]:
        topic = re.sub(r"\s+", " ", message).strip()
        if len(topic) > 50:
            topic = f"{topic[:50]}..."
        candidates = [
            f"'{topic}'와 직접 연결되는 핵심 근거를 문서 기준으로 2개만 더 설명해줘.",
            f"이 주제에서 문서가 말하는 한계나 반례는 무엇인지 알려줘.",
            f"이 내용을 실무에 적용할 때 체크해야 할 조건을 문서 근거로 정리해줘.",
        ]
        return candidates[: max(1, limit)]

    @staticmethod
    def _to_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (int, float, bool)):
            return str(value).strip()
        if isinstance(value, list):
            parts = [DocumentQAService._to_text(item) for item in value]
            return "\n".join([part for part in parts if part]).strip()
        if isinstance(value, dict):
            preferred_keys = (
                "text",
                "content",
                "reasoning_content",
                "thinking",
                "reasoning",
                "summary",
                "output_text",
            )
            parts: list[str] = []
            for key in preferred_keys:
                if key in value:
                    text = DocumentQAService._to_text(value.get(key))
                    if text:
                        parts.append(text)
            if parts:
                return "\n".join(parts).strip()
            nested_parts = [DocumentQAService._to_text(item) for item in value.values()]
            return "\n".join([part for part in nested_parts if part]).strip()
        return str(value).strip()

    @staticmethod
    def _to_stream_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, list):
            return "".join(DocumentQAService._to_stream_text(item) for item in value)
        if isinstance(value, dict):
            preferred_keys = (
                "text",
                "content",
                "output_text",
                "message",
                "final",
                "answer",
            )
            parts: list[str] = []
            for key in preferred_keys:
                if key not in value:
                    continue
                text = DocumentQAService._to_stream_text(value.get(key))
                if text:
                    parts.append(text)
            if parts:
                return "".join(parts)
            return "".join(DocumentQAService._to_stream_text(item) for item in value.values())
        return str(value)

    @staticmethod
    def _extract_content_list_parts(content: list[Any]) -> tuple[str, str | None]:
        answer_parts: list[str] = []
        reasoning_parts: list[str] = []

        reasoning_types = {
            "reasoning",
            "thinking",
            "reasoning_content",
            "thought",
            "think",
            "analysis",
        }
        answer_types = {
            "text",
            "output_text",
            "final",
            "answer",
            "message",
        }

        for item in content:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    answer_parts.append(text)
                continue
            if not isinstance(item, dict):
                text = DocumentQAService._to_text(item)
                if text:
                    answer_parts.append(text)
                continue

            item_type = str(item.get("type", "")).strip().lower()
            if item_type in reasoning_types:
                text = DocumentQAService._to_text(item)
                if text:
                    reasoning_parts.append(text)
                continue
            if item_type in answer_types:
                text = DocumentQAService._to_text(item.get("text") or item.get("content") or item)
                if text:
                    answer_parts.append(text)
                continue

            text = DocumentQAService._to_text(item.get("text") or item.get("content"))
            if text:
                answer_parts.append(text)

            extra_reasoning = DocumentQAService._to_text(
                item.get("reasoning")
                or item.get("reasoning_content")
                or item.get("thinking")
                or item.get("thought")
                or item.get("think")
            )
            if extra_reasoning:
                reasoning_parts.append(extra_reasoning)

        answer = "\n".join([part for part in answer_parts if part]).strip()
        reasoning = "\n".join([part for part in reasoning_parts if part]).strip() or None
        return answer, reasoning

    @staticmethod
    def _extract_answer_and_reasoning(response: Any) -> tuple[str, str | None]:
        content = getattr(response, "content", "")
        additional_kwargs = getattr(response, "additional_kwargs", {}) or {}
        response_metadata = getattr(response, "response_metadata", {}) or {}

        reasoning: str | None = None
        for key in ("reasoning_content", "thinking", "reasoning", "thought", "think"):
            value = additional_kwargs.get(key) or response_metadata.get(key)
            text = DocumentQAService._to_text(value)
            if text:
                reasoning = text
                break

        if reasoning is None:
            direct_reasoning = DocumentQAService._to_text(getattr(response, "reasoning_content", None))
            if direct_reasoning:
                reasoning = direct_reasoning

        if isinstance(content, str):
            answer = content.strip()
        elif isinstance(content, list):
            answer, content_reasoning = DocumentQAService._extract_content_list_parts(content)
            if content_reasoning and not reasoning:
                reasoning = content_reasoning
        else:
            answer = str(content).strip()

        think_match = re.search(r"<think>(.*?)</think>", answer, flags=re.IGNORECASE | re.DOTALL)
        if think_match:
            think_text = think_match.group(1).strip()
            if think_text:
                reasoning = reasoning or think_text
            answer = re.sub(
                r"<think>.*?</think>\s*",
                "",
                answer,
                flags=re.IGNORECASE | re.DOTALL,
            ).strip()

        return answer, reasoning

    async def _invoke_chat_llm(
        self,
        messages: list[tuple[str, str]],
        token_callback: Callable[[str], Awaitable[None]] | None = None,
    ) -> tuple[str, str | None]:
        if token_callback is None:
            response = await self.chat_llm.ainvoke(messages)
            return self._extract_answer_and_reasoning(response)

        async def _emit_stream_tokens(text: str) -> int:
            emitted_count = 0
            for piece in self._split_stream_tokens(text):
                if not piece:
                    continue
                await token_callback(piece)
                emitted_count += 1
                await asyncio.sleep(0)
            return emitted_count

        astream = getattr(self.chat_llm, "astream", None)
        if callable(astream):
            emitted_text = ""
            answer_parts: list[str] = []
            reasoning_parts: list[str] = []
            emitted_token_count = 0
            async for chunk in astream(messages):
                piece, reasoning_piece = self._extract_stream_chunk_piece(
                    chunk,
                    emitted_prefix=emitted_text,
                )
                if piece:
                    emitted_text += piece
                    answer_parts.append(piece)
                    emitted_token_count += await _emit_stream_tokens(piece)
                if reasoning_piece:
                    reasoning_parts.append(reasoning_piece)
            answer = "".join(answer_parts).strip()
            reasoning = "\n".join(part for part in reasoning_parts if part).strip() or None
            if answer:
                logger.info(
                    "qa_stream_emit_summary mode=astream token_chunks=%d answer_len=%d",
                    emitted_token_count,
                    len(answer),
                )
                return answer, reasoning

        response = await self.chat_llm.ainvoke(messages)
        answer, reasoning = self._extract_answer_and_reasoning(response)
        if answer:
            emitted = await _emit_stream_tokens(answer)
            logger.info(
                "qa_stream_emit_summary mode=ainvoke_fallback token_chunks=%d answer_len=%d",
                emitted,
                len(answer),
            )
        return answer, reasoning

    @classmethod
    def _split_stream_tokens(cls, text: str) -> list[str]:
        if not text:
            return []
        return STREAM_TOKEN_PATTERN.findall(text)

    @classmethod
    def _extract_stream_chunk_piece(
        cls,
        chunk: Any,
        *,
        emitted_prefix: str,
    ) -> tuple[str, str]:
        visible_raw = ""
        text_accessor = getattr(chunk, "text", None)
        if text_accessor is not None and not callable(text_accessor):
            try:
                visible_raw = str(text_accessor)
            except Exception:
                visible_raw = ""

        content = getattr(chunk, "content", "")
        if isinstance(content, str):
            if not visible_raw:
                visible_raw = content
            reasoning_raw = ""
        elif isinstance(content, list):
            visible_parts: list[str] = []
            reasoning_parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    item_type = str(item.get("type", "")).strip().lower()
                    if item_type in {"reasoning", "thinking", "reasoning_content", "thought", "analysis"}:
                        text = cls._to_stream_text(item.get("text") or item.get("content") or item)
                        if text:
                            reasoning_parts.append(text)
                        continue
                    text = cls._to_stream_text(item.get("text") or item.get("content") or item)
                    if text:
                        visible_parts.append(text)
                    continue
                text = cls._to_stream_text(item)
                if text:
                    visible_parts.append(text)
            if not visible_raw:
                visible_raw = "".join(visible_parts)
            reasoning_raw = "\n".join(reasoning_parts)
        else:
            if not visible_raw:
                visible_raw = cls._to_stream_text(content)
            reasoning_raw = ""

        for container_name in ("additional_kwargs", "response_metadata"):
            container = getattr(chunk, container_name, None)
            if not isinstance(container, dict):
                continue
            for key in ("reasoning_content", "thinking", "reasoning", "thought", "think"):
                text = cls._to_stream_text(container.get(key))
                if text:
                    reasoning_raw = f"{reasoning_raw}\n{text}".strip()

        visible = visible_raw or ""
        if visible.startswith(emitted_prefix):
            delta = visible[len(emitted_prefix):]
        else:
            delta = visible
        return delta, reasoning_raw.strip()

    async def _ensure_graph_initialized(self) -> None:
        if self._graph is not None:
            return
        async with self._init_lock:
            if self._graph is not None:
                return
            checkpointer = await self._ensure_async_redis_checkpointer()
            self._graph = self._build_graph(checkpointer)

    def _build_graph(self, checkpointer: Any):
        from langgraph.graph import END, START, StateGraph

        workflow = StateGraph(RAGState)
        workflow.add_node(
            "planning",
            self._timed_node("planning", self._node_planning),
        )
        workflow.add_node(
            "retrieve_docs",
            self._timed_node("retrieve_docs", self._node_retrieve_docs),
        )
        workflow.add_node(
            "enrich_with_arxiv_pdf",
            self._timed_node("enrich_with_arxiv_pdf", self._node_enrich_with_arxiv_pdf),
        )
        workflow.add_node(
            "generate_answer_doc_base",
            self._timed_node("generate_answer_doc_base", self._node_generate_answer_doc_base),
        )
        workflow.add_node(
            "generate_answer_llm",
            self._timed_node("generate_answer_llm", self._node_generate_answer_llm),
        )
        workflow.add_edge(START, "planning")
        workflow.add_edge("planning", "retrieve_docs")
        workflow.add_edge("retrieve_docs", "enrich_with_arxiv_pdf")
        workflow.add_conditional_edges(
            "enrich_with_arxiv_pdf",
            self._conditional_edge_doc_base_or_not,
        )
        workflow.add_edge("generate_answer_doc_base", END)
        workflow.add_edge("generate_answer_llm", END)
        return workflow.compile(checkpointer=checkpointer)

    async def _ensure_async_redis_checkpointer(self):
        if self._checkpointer is not None:
            return self._checkpointer
        try:
            from langgraph.checkpoint.redis.aio import AsyncRedisSaver
        except Exception as exc:
            raise RuntimeError(
                "Async Redis checkpointer import failed. Install langgraph-checkpoint-redis."
            ) from exc

        try:
            if hasattr(AsyncRedisSaver, "from_conn_string"):
                acm = AsyncRedisSaver.from_conn_string(self.settings.redis_url)
                saver = await acm.__aenter__()
                await saver.asetup()
                self._checkpointer_acm = acm
                self._checkpointer = saver
                return saver

            saver = AsyncRedisSaver(redis_url=self.settings.redis_url)
            await saver.asetup()
            self._checkpointer = saver
            return saver
        except Exception as exc:
            if self._checkpointer_acm is not None:
                await self._checkpointer_acm.__aexit__(type(exc), exc, exc.__traceback__)
                self._checkpointer_acm = None
            logger.warning(
                "Redis checkpointer unavailable; falling back to in-memory saver: %s",
                exc,
            )
            from langgraph.checkpoint.memory import InMemorySaver

            self._checkpointer = InMemorySaver()
            return self._checkpointer

    async def aclose(self) -> None:
        if self._checkpointer_acm is not None:
            await self._checkpointer_acm.__aexit__(None, None, None)
            self._checkpointer_acm = None
        self._checkpointer = None
        self._graph = None

    def _timed_node(
        self,
        node_name: str,
        func: Callable[[RAGState], Awaitable[RAGState]],
    ) -> Callable[[RAGState], Awaitable[RAGState]]:
        async def wrapped(state: RAGState) -> RAGState:
            started = perf_counter()
            try:
                result = await func(state)
                elapsed_ms = (perf_counter() - started) * 1000
                self._log_node(node_name=node_name, elapsed_ms=elapsed_ms, result=result)
                return result
            except Exception as exc:
                elapsed_ms = (perf_counter() - started) * 1000
                self._log_node(
                    node_name=node_name,
                    elapsed_ms=elapsed_ms,
                    result={"error": str(exc)},
                    level="error",
                )
                raise

        return wrapped

    def _log_node(
        self,
        node_name: str,
        elapsed_ms: float,
        result: RAGState,
        level: str = "info",
    ) -> None:
        payload = {
            "event": "rag_node_completed",
            "node": node_name,
            "duration_ms": round(elapsed_ms, 2),
            "result": self._summarize_result(result),
        }
        message = json.dumps(payload, ensure_ascii=False)
        if level == "error":
            logger.error(message)
        else:
            logger.info(message)
        print(message, flush=True)

    @staticmethod
    def _summarize_result(result: RAGState) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        if "error" in result and result.get("error"):
            summary["error"] = str(result.get("error"))
        analyzed = result.get("analyzed_query")
        if analyzed:
            summary["search_query"] = analyzed.search_query[:200]
            summary["keywords"] = analyzed.keywords[:8]
        tool_plan = result.get("tool_plan")
        if tool_plan:
            summary["tool_plan"] = {
                "use_vdb": bool(tool_plan.use_vdb),
                "use_pdf": bool(tool_plan.use_pdf),
                "reasons": tool_plan.reasons[:4],
                "candidate_arxiv_ids": tool_plan.candidate_arxiv_ids[:6],
            }
        if "retrieved" in result:
            retrieved = result.get("retrieved") or []
            summary["retrieved_count"] = len(retrieved)
            top_sources: list[str] = []
            for item in retrieved[:3]:
                source = item.source.rsplit("/", 1)[-1] if item.source else "unknown"
                top_sources.append(source)
            summary["top_sources"] = top_sources
        if "pdf_evidence" in result:
            summary["pdf_evidence_count"] = len(result.get("pdf_evidence") or [])
        if "pdf_enrich_attempted" in result:
            summary["pdf_enrich_attempted"] = bool(result.get("pdf_enrich_attempted"))
        if "pdf_enrich_used" in result:
            summary["pdf_enrich_used"] = int(result.get("pdf_enrich_used") or 0)
        if "pdf_cache_hit_count" in result:
            summary["pdf_cache_hit_count"] = int(result.get("pdf_cache_hit_count") or 0)
        if "pdf_download_fail_count" in result:
            summary["pdf_download_fail_count"] = int(result.get("pdf_download_fail_count") or 0)
        if "answer" in result and result.get("answer"):
            answer = str(result.get("answer"))
            summary["answer_preview"] = answer[:240]
            summary["answer_len"] = len(answer)
        return summary

    async def _invoke_json_analysis(self, message: str, chat_history_text: str = "") -> str:
        analysis_prompt = (
            "사용자 질문을 검색용 질의로 변환해 JSON으로만 답하라.\n"
            '형식: {"normalized":"...", "keywords":["..."], "search_query":"..."}\n'
            "규칙:\n"
            "1) normalized는 질문 의도를 보존한 정규화 문장\n"
            "2) keywords는 핵심 키워드 3~8개\n"
            "3) search_query는 벡터/BM25 검색에 유리한 확장 질의\n"
            "4) JSON 외 다른 텍스트 금지"
        )
        user_content = message
        if chat_history_text:
            user_content = f"대화 맥락:\n{chat_history_text}\n\n현재 질문:\n{message}"
        response = await self.chat_llm.ainvoke(
            [
                ("system", analysis_prompt),
                ("human", user_content),
            ]
        )
        return str(response.content or "").strip()

    async def _node_planning(self, state: RAGState) -> RAGState:
        message = (state.get("message") or "").strip()
        chat_history = state.get("chat_history") or []
        chat_history_text = self._history_to_text(chat_history)
        fallback = analyze_user_query(message)
        if not fallback.normalized:
            return {
                "analyzed_query": fallback,
                "tool_plan": ToolPlan(
                    use_vdb=False,
                    use_pdf=False,
                    reasons=["질문이 비어 있어 tool 실행을 생략합니다."],
                    candidate_arxiv_ids=[],
                ),
                "error": "질문이 비어 있습니다.",
            }

        memory_candidates = self._extract_memory_arxiv_candidates(chat_history)
        message_arxiv_id = self._extract_arxiv_id(message)
        if message_arxiv_id and not any(
            candidate.get("arxiv_id", "") == message_arxiv_id
            for candidate in memory_candidates
        ):
            memory_candidates.insert(
                0,
                {
                    "arxiv_id": message_arxiv_id,
                    "title": "",
                    "pdf_url": self._build_pdf_url(message_arxiv_id),
                    "source_url": f"https://arxiv.org/abs/{message_arxiv_id}",
                },
            )

        plan = self._build_tool_plan(
            message=message,
            memory_candidates=memory_candidates,
            has_message_arxiv_id=bool(message_arxiv_id),
        )
        try:
            llm_text = await self._invoke_json_analysis(
                message,
                chat_history_text=chat_history_text,
            )
            analyzed = parse_llm_analysis(message, llm_text)
            return {
                "analyzed_query": analyzed,
                "tool_plan": plan,
                "memory_candidates": memory_candidates,
            }
        except Exception as exc:
            return {
                "analyzed_query": fallback,
                "tool_plan": plan,
                "memory_candidates": memory_candidates,
                "error": f"질의 분석 LLM 오류: {exc}",
            }

    async def _node_analyze_query(self, state: RAGState) -> RAGState:
        return await self._node_planning(state)

    def _build_tool_plan(
        self,
        message: str,
        memory_candidates: list[dict[str, str]],
        has_message_arxiv_id: bool,
    ) -> ToolPlan:
        reasons: list[str] = []
        detail_question = self._is_detail_question(message)
        has_memory_candidates = bool(memory_candidates)

        use_vdb = True
        use_pdf = False

        if detail_question:
            use_pdf = True
            reasons.append("질문이 원문/상세형이어서 PDF enrichment를 사용합니다.")
        if has_message_arxiv_id:
            use_pdf = True
            reasons.append("질문에 arXiv ID가 명시되어 PDF enrichment를 사용합니다.")

        if use_pdf and has_memory_candidates:
            use_vdb = False
            reasons.append("대화 메모리에서 arXiv 후보를 확보해 VDB 검색을 생략합니다.")
        elif use_pdf:
            use_vdb = True
            reasons.append("PDF 후보 arXiv ID 확보를 위해 VDB 검색을 선행합니다.")
        else:
            reasons.append("기본 문서 근거 확보를 위해 VDB 검색을 사용합니다.")

        candidate_arxiv_ids = [
            candidate.get("arxiv_id", "").strip()
            for candidate in memory_candidates
            if candidate.get("arxiv_id", "").strip()
        ]
        return ToolPlan(
            use_vdb=use_vdb,
            use_pdf=use_pdf,
            reasons=reasons,
            candidate_arxiv_ids=candidate_arxiv_ids,
        )

    @classmethod
    def _extract_memory_arxiv_candidates(
        cls,
        chat_history: list[dict[str, Any]],
        limit: int = 20,
    ) -> list[dict[str, str]]:
        candidates: list[dict[str, str]] = []
        seen: set[str] = set()

        for turn in reversed(chat_history):
            if len(candidates) >= limit:
                break
            metadata = turn.get("metadata")
            if isinstance(metadata, dict):
                memory_items = metadata.get("arxiv_memory")
                if isinstance(memory_items, list):
                    for item in memory_items:
                        if not isinstance(item, dict):
                            continue
                        arxiv_id = str(item.get("arxiv_id", "")).strip()
                        if not arxiv_id or arxiv_id in seen:
                            continue
                        seen.add(arxiv_id)
                        candidates.append(
                            {
                                "arxiv_id": arxiv_id,
                                "title": str(item.get("title", "")).strip(),
                                "pdf_url": str(item.get("pdf_url", "")).strip(),
                                "source_url": str(item.get("source_url", "")).strip(),
                            }
                        )
                        if len(candidates) >= limit:
                            break
            if len(candidates) >= limit:
                break

            text = str(turn.get("text", "")).strip()
            if not text:
                continue
            for arxiv_id in cls._extract_arxiv_ids_from_text(text):
                if arxiv_id in seen:
                    continue
                seen.add(arxiv_id)
                candidates.append(
                    {
                        "arxiv_id": arxiv_id,
                        "title": "",
                        "pdf_url": cls._build_pdf_url(arxiv_id),
                        "source_url": f"https://arxiv.org/abs/{arxiv_id}",
                    }
                )
                if len(candidates) >= limit:
                    break
        return candidates

    @classmethod
    def _extract_arxiv_ids_from_text(cls, text: str) -> list[str]:
        raw = (text or "").strip()
        if not raw:
            return []
        found: list[str] = []
        seen: set[str] = set()
        for pattern in (NEW_ID_PATTERN, OLD_ID_PATTERN):
            for match in pattern.findall(raw):
                normalized = str(match).strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                found.append(normalized)
        return found

    @staticmethod
    def _safe_parse_metadata(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            value = raw.strip()
            if not value:
                return {}
            try:
                loaded = json.loads(value)
            except Exception:
                return {}
            if isinstance(loaded, dict):
                return loaded
        return {}

    @classmethod
    def _extract_arxiv_id(cls, source: str) -> str:
        text = (source or "").strip()
        if not text:
            return ""
        matched = NEW_ID_PATTERN.search(text)
        if matched:
            return matched.group(0)
        matched = OLD_ID_PATTERN.search(text.lower())
        if matched:
            return matched.group(0)
        return ""

    @staticmethod
    def _build_pdf_url(arxiv_id: str) -> str:
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    async def _node_retrieve_docs(self, state: RAGState) -> RAGState:
        analyzed = state.get("analyzed_query")
        if not analyzed:
            return {"retrieved": [], "error": "질의 분석 결과가 없습니다."}
        try:
            query_vector = await asyncio.to_thread(self.embedder.embed_query, analyzed.search_query)
            hits = await asyncio.to_thread(
                self.store.hybrid_search,
                analyzed.search_query,
                query_vector,
                self.settings.retrieval_top_k,
                self.settings.dense_weight,
                self.settings.sparse_weight,
            )
            retrieved = self._build_retrieval_results(hits)
            if not retrieved:
                fallback = await asyncio.to_thread(
                    self._fallback_retrieve_from_arxiv_files,
                    analyzed.search_query,
                    self.settings.retrieval_top_k,
                )
                if fallback:
                    return {"retrieved": fallback}
            return {"retrieved": retrieved}
        except Exception as exc:
            fallback = await asyncio.to_thread(
                self._fallback_retrieve_from_arxiv_files,
                analyzed.search_query,
                self.settings.retrieval_top_k,
            )
            if fallback:
                return {
                    "retrieved": fallback,
                    "error": f"Milvus 검색 실패로 로컬 arXiv 파일 검색으로 대체했습니다. ({exc})",
                }
            return {
                "retrieved": [],
                "error": (
                    "검색 단계에서 오류가 발생했습니다. "
                    f"Milvus/임베딩 연결 상태를 확인해 주세요. ({exc})"
                ),
            }

    def _build_retrieval_results(self, hits: list[dict[str, Any]]) -> list[RetrievalResult]:
        results: list[RetrievalResult] = []
        for item in hits:
            metadata = self._safe_parse_metadata(item.get("metadata"))
            source = str(item.get("source") or "")
            arxiv_id = str(metadata.get("arxiv_id", "")).strip() or self._extract_arxiv_id(source)
            pdf_url = str(metadata.get("pdf_url", "")).strip() or (
                self._build_pdf_url(arxiv_id) if arxiv_id else None
            )
            source_url = str(metadata.get("source_url", "")).strip() or None
            results.append(
                RetrievalResult(
                    id=item["id"],
                    score=float(item.get("score") or 0),
                    source=source,
                    chunk_id=int(item.get("chunk_id") or 0),
                    text=str(item.get("text") or ""),
                    arxiv_id=arxiv_id or None,
                    pdf_url=pdf_url or None,
                    source_url=source_url,
                    metadata=metadata or None,
                )
            )
        return results

    @classmethod
    def _needs_pdf_enrichment(
        cls,
        message: str,
        retrieved: list[RetrievalResult],
        top_k: int,
        min_score: float,
    ) -> bool:
        if not retrieved:
            return False
        has_arxiv = any((item.arxiv_id or "").strip() for item in retrieved)
        if not has_arxiv:
            return False
        if cls._is_detail_question(message):
            return True
        if len(retrieved) < top_k:
            return True
        top_score = float(retrieved[0].score)
        if top_score < min_score:
            return True
        lowered = (message or "").lower()
        return any(signal.lower() in lowered for signal in ENRICH_TRIGGER_SIGNALS)

    @staticmethod
    def _is_detail_question(message: str) -> bool:
        lowered = (message or "").lower()
        return any(signal.lower() in lowered for signal in ENRICH_TRIGGER_SIGNALS)

    async def _node_enrich_with_arxiv_pdf(self, state: RAGState) -> RAGState:
        retrieved = state.get("retrieved") or []
        message = str(state.get("message") or "")
        analyzed = state.get("analyzed_query")
        memory_candidates = state.get("memory_candidates") or []
        retrieved_for_enrich = self._build_retrieval_for_pdf_enrichment(
            retrieved=retrieved,
            message=message,
            memory_candidates=memory_candidates,
        )

        if not self.settings.arxiv_pdf_enrich_enabled:
            return {
                "pdf_evidence": [],
                "pdf_enrich_attempted": False,
                "pdf_enrich_used": 0,
                "pdf_cache_hit_count": 0,
                "pdf_download_fail_count": 0,
            }
        if not analyzed:
            return {"pdf_evidence": [], "pdf_enrich_attempted": False, "pdf_enrich_used": 0}
        if not retrieved_for_enrich:
            return {
                "pdf_evidence": [],
                "pdf_enrich_attempted": False,
                "pdf_enrich_used": 0,
                "pdf_cache_hit_count": 0,
                "pdf_download_fail_count": 0,
            }

        try:
            evidences = await self.pdf_enricher.enrich(
                query=message,
                analyzed_query=analyzed,
                retrieved_docs=retrieved_for_enrich,
            )
            return {
                "pdf_evidence": evidences,
                "pdf_enrich_attempted": True,
                "pdf_enrich_used": len(evidences),
                "pdf_cache_hit_count": self.pdf_enricher.last_metrics.cache_hit_count,
                "pdf_download_fail_count": self.pdf_enricher.last_metrics.download_fail_count,
            }
        except Exception as exc:
            return {
                "pdf_evidence": [],
                "pdf_enrich_attempted": True,
                "pdf_enrich_used": 0,
                "pdf_cache_hit_count": self.pdf_enricher.last_metrics.cache_hit_count,
                "pdf_download_fail_count": self.pdf_enricher.last_metrics.download_fail_count,
                "pdf_enrich_error": str(exc),
            }

    def _build_retrieval_for_pdf_enrichment(
        self,
        retrieved: list[RetrievalResult],
        message: str,
        memory_candidates: list[dict[str, str]],
    ) -> list[RetrievalResult]:
        combined: list[RetrievalResult] = list(retrieved)
        seen_ids = {
            (item.arxiv_id or "").strip()
            for item in combined
            if (item.arxiv_id or "").strip()
        }

        message_arxiv_id = self._extract_arxiv_id(message)
        if message_arxiv_id and message_arxiv_id not in seen_ids:
            seen_ids.add(message_arxiv_id)
            combined.insert(
                0,
                RetrievalResult(
                    id=f"message:{message_arxiv_id}",
                    score=1.0,
                    source=f"arXiv:{message_arxiv_id}",
                    chunk_id=0,
                    text="",
                    arxiv_id=message_arxiv_id,
                    pdf_url=self._build_pdf_url(message_arxiv_id),
                    source_url=f"https://arxiv.org/abs/{message_arxiv_id}",
                    metadata={"arxiv_id": message_arxiv_id},
                ),
            )

        for candidate in memory_candidates:
            arxiv_id = str(candidate.get("arxiv_id", "")).strip()
            if not arxiv_id or arxiv_id in seen_ids:
                continue
            seen_ids.add(arxiv_id)
            title = str(candidate.get("title", "")).strip()
            summary_text = f"Title: {title}" if title else ""
            combined.append(
                RetrievalResult(
                    id=f"memory:{arxiv_id}",
                    score=0.95,
                    source=f"arXiv:{arxiv_id}",
                    chunk_id=0,
                    text=summary_text,
                    arxiv_id=arxiv_id,
                    pdf_url=str(candidate.get("pdf_url", "")).strip() or self._build_pdf_url(arxiv_id),
                    source_url=str(candidate.get("source_url", "")).strip() or f"https://arxiv.org/abs/{arxiv_id}",
                    metadata={"arxiv_id": arxiv_id, "title": title},
                )
            )
        return combined

    def _fallback_retrieve_from_arxiv_files(
        self,
        query: str,
        limit: int = 5,
    ) -> list[RetrievalResult]:
        arxiv_dir = self.settings.arxiv_storage_dir
        if not arxiv_dir.exists():
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        candidates: list[tuple[float, RetrievalResult]] = []
        for file_path in sorted(arxiv_dir.glob("*.json"), reverse=True):
            if file_path.name.startswith("_"):
                continue
            try:
                raw = json.loads(file_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(raw, list):
                continue

            for idx, item in enumerate(raw):
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title", "")).strip()
                summary = str(item.get("summary", "")).strip()
                if not title and not summary:
                    continue
                text = f"Title: {title}\nSummary: {summary}".strip()
                text_tokens = self._tokenize(text)
                if not text_tokens:
                    continue

                overlap = len(query_tokens & text_tokens)
                if overlap == 0:
                    continue
                score = overlap / max(len(query_tokens), 1)

                arxiv_id = str(item.get("arxiv_id", "")).strip() or self._extract_arxiv_id(
                    str(item.get("source_url", "")).strip()
                )
                pdf_url = str(item.get("pdf_url", "")).strip() or (
                    self._build_pdf_url(arxiv_id) if arxiv_id else None
                )
                source_url = str(item.get("source_url", "")).strip() or None
                source = arxiv_id or source_url or file_path.name
                metadata: dict[str, Any] = {
                    "arxiv_id": arxiv_id,
                    "pdf_url": pdf_url,
                    "source_url": source_url,
                }
                candidates.append(
                    (
                        score,
                        RetrievalResult(
                            id=f"{source}:{idx}",
                            score=float(score),
                            source=source,
                            chunk_id=0,
                            text=text,
                            arxiv_id=arxiv_id or None,
                            pdf_url=pdf_url,
                            source_url=source_url,
                            metadata=metadata,
                        ),
                    )
                )

        candidates.sort(key=lambda row: row[0], reverse=True)
        return [row[1] for row in candidates[:limit]]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[A-Za-z0-9가-힣]{2,}", text.lower()))

    async def _node_generate_answer_doc_base(
        self,
        state: RAGState,
        token_callback: Callable[[str], Awaitable[None]] | None = None,
    ) -> RAGState:
        user_message = (state.get("message") or "").strip()
        chat_history = state.get("chat_history") or []
        chat_history_text = self._history_to_text(chat_history)
        retrieved = state.get("retrieved") or []
        pdf_evidence = state.get("pdf_evidence") or []

        if not retrieved and not pdf_evidence:
            prior_error = state.get("error")
            if prior_error:
                return {"answer": prior_error}
            return {"answer": "관련 문서를 찾지 못했습니다. 질문을 더 구체적으로 입력해 주세요."}

        context_lines: list[str] = []
        context_sources: dict[int, tuple[str, str | None]] = {}
        dedup_keys: set[str] = set()
        context_idx = 1

        for item in retrieved:
            source = item.source.rsplit("/", 1)[-1] if item.source else "unknown"
            snippet = " ".join(item.text.split())
            key = self._normalize_text_key(snippet[:800])
            if key in dedup_keys:
                continue
            dedup_keys.add(key)
            arxiv_id = (item.arxiv_id or "").strip() or self._extract_arxiv_id(item.source)
            if arxiv_id:
                source_label = f"arXiv:{arxiv_id}"
                source_url = (item.pdf_url or "").strip() or self._build_pdf_url(arxiv_id)
            else:
                source_label = source
                source_url = (item.source_url or "").strip() or None
            context_sources[context_idx] = (source_label, source_url)
            context_lines.append(
                f"[{context_idx}] source={source}, chunk={item.chunk_id}, score={item.score:.4f}\n{snippet[:1000]}"
            )
            context_idx += 1

        for ev in pdf_evidence:
            snippet = " ".join(ev.snippet.split())
            key = self._normalize_text_key(snippet[:800])
            if key in dedup_keys:
                continue
            dedup_keys.add(key)
            context_sources[context_idx] = (
                f"arXiv:{ev.arxiv_id}",
                (ev.pdf_url or "").strip() or self._build_pdf_url(ev.arxiv_id),
            )
            context_lines.append(
                f"[{context_idx}] source=arXiv:{ev.arxiv_id}, chunk=pdf, score={ev.score:.4f}\n{snippet[:1000]}"
            )
            context_idx += 1

        context = "\n\n".join(context_lines)

        system_prompt = (
            "당신은 문서 기반 질의응답 도우미다. 반드시 제공된 context 근거로만 답변하고, "
            "근거가 부족하면 모른다고 답해라. 답변 마지막에 사용한 근거 번호를 [1], [2] 형식으로 표시해라. "
            "PDF 근거를 사용한 경우 출처(arXiv 링크)를 포함해라."
        )
        user_prompt = (
            f"conversation_history:\n{chat_history_text}\n\n"
            f"question:\n{user_message}\n\n"
            f"context:\n{context}\n\n"
            "한국어로 간결하고 정확하게 답변해라."
        )
        try:
            answer, reasoning = await self._invoke_chat_llm(
                [
                    ("system", system_prompt),
                    ("human", user_prompt),
                ],
                token_callback=token_callback,
            )
            if not answer:
                raise RuntimeError("Empty response from LLM.")
            cited_indexes = self._extract_cited_context_indexes(answer)
            cited_source_lines = self._collect_cited_source_lines(
                cited_indexes=cited_indexes,
                context_sources=context_sources,
                max_items=3,
            )
            if not cited_source_lines:
                cited_source_lines = self._fallback_source_lines(
                    context_sources=context_sources,
                    max_items=3,
                )
            answer_with_sources = self._append_source_lines(answer, cited_source_lines)
            result: RAGState = {"answer": answer_with_sources}
            if reasoning:
                result["reasoning"] = reasoning
            return result
        except Exception as exc:
            fallback_snippet = ""
            if retrieved:
                fallback_snippet = " ".join(retrieved[0].text.split())[:500]
            elif pdf_evidence:
                fallback_snippet = " ".join(pdf_evidence[0].snippet.split())[:500]
            answer = (
                f"LLM 응답 생성 오류: {exc}\n"
                "상위 검색 결과를 기반으로 대체 답변을 제공합니다.\n"
                f"- 핵심 근거: {fallback_snippet}\n"
                "[1]"
            )
            cited_indexes = self._extract_cited_context_indexes(answer)
            cited_source_lines = self._collect_cited_source_lines(
                cited_indexes=cited_indexes,
                context_sources=context_sources,
                max_items=3,
            )
            if not cited_source_lines:
                cited_source_lines = self._fallback_source_lines(
                    context_sources=context_sources,
                    max_items=3,
                )
            return {"answer": self._append_source_lines(answer, cited_source_lines)}

    async def _node_generate_answer_llm(
        self,
        state: RAGState,
        token_callback: Callable[[str], Awaitable[None]] | None = None,
    ) -> RAGState:
        user_message = state.get("message")
        chat_history = state.get("chat_history") or []
        chat_history_text = self._history_to_text(chat_history)

        system_prompt = (
            "당신은 질의응답 도우미다. 사용자가 특정 사항에 관한 요구사항이 있다면 친절히 답변해라."
            "모르는 내용에 대해서 절대 답변하지 마라."
        )
        user_prompt = (
            f"conversation_history:\n{chat_history_text}\n\n"
            f"question:\n{user_message}\n\n"
            "한국어로 간결하고 정확하게 답변해라."
        )
        try:
            answer, reasoning = await self._invoke_chat_llm(
                [
                    ("system", system_prompt),
                    ("human", user_prompt),
                ],
                token_callback=token_callback,
            )
            if not answer:
                raise RuntimeError("Empty response from LLM.")
            result: RAGState = {"answer": self._append_source_lines(answer, [])}
            if reasoning:
                result["reasoning"] = reasoning
            return result
        except Exception as exc:
            return {"answer": self._append_source_lines(f"LLM 응답 생성 오류: {exc}", [])}

    def _conditional_edge_doc_base_or_not(self, state: RAGState) -> str:
        retrieved = state.get("retrieved", [])
        pdf_evidence = state.get("pdf_evidence", [])
        route = "generate_answer_doc_base" if (retrieved or pdf_evidence) else "generate_answer_llm"
        payload = {
            "event": "rag_route_selected",
            "route": route,
            "retrieved_count": len(retrieved),
            "pdf_evidence_count": len(pdf_evidence),
        }
        message = json.dumps(payload, ensure_ascii=False)
        logger.info(message)
        print(message, flush=True)
        return route

    @staticmethod
    def _normalize_text_key(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()

    @staticmethod
    def _extract_cited_context_indexes(answer: str) -> list[int]:
        seen: set[int] = set()
        out: list[int] = []
        for match in re.findall(r"\[(\d+)\]", answer or ""):
            try:
                idx = int(match)
            except ValueError:
                continue
            if idx <= 0 or idx in seen:
                continue
            seen.add(idx)
            out.append(idx)
        return out

    @staticmethod
    def _collect_cited_source_lines(
        cited_indexes: list[int],
        context_sources: dict[int, tuple[str, str | None]],
        max_items: int = 3,
    ) -> list[str]:
        lines: list[str] = []
        seen: set[str] = set()
        for idx in cited_indexes:
            source_info = context_sources.get(idx)
            if source_info is None:
                continue
            label, url = source_info
            if url:
                line = f"- [{label}]({url})"
            else:
                line = f"- {label}"
            dedup_key = line.lower()
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            lines.append(line)
            if len(lines) >= max_items:
                break
        return lines

    @staticmethod
    def _fallback_source_lines(
        context_sources: dict[int, tuple[str, str | None]],
        max_items: int = 3,
    ) -> list[str]:
        lines: list[str] = []
        seen: set[str] = set()
        for idx in sorted(context_sources.keys()):
            label, url = context_sources[idx]
            if url:
                line = f"- [{label}]({url})"
            else:
                line = f"- {label}"
            dedup_key = line.lower()
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            lines.append(line)
            if len(lines) >= max_items:
                break
        return lines

    @staticmethod
    def _strip_existing_sources_section(answer: str) -> str:
        text = (answer or "").rstrip()
        return re.sub(
            r"\n+Sources:\n(?:- .*(?:\n|$))*\s*$",
            "",
            text,
            flags=re.IGNORECASE,
        ).rstrip()

    @staticmethod
    def _append_source_lines(answer: str, source_lines: list[str]) -> str:
        base_answer = DocumentQAService._strip_existing_sources_section(answer)
        lines = [base_answer, "", "Sources:"]
        if source_lines:
            lines.extend(source_lines)
        else:
            lines.append("- (사용된 문서 출처를 식별하지 못했습니다.)")
        return "\n".join(lines)

    @staticmethod
    def _history_to_text(chat_history: list[dict[str, Any]]) -> str:
        if not chat_history:
            return "(없음)"
        lines: list[str] = []
        for turn in chat_history[-6:]:
            role = str(turn.get("role", "user")).strip() or "user"
            text = str(turn.get("text", "")).strip()
            if not text:
                continue
            lines.append(f"{role}: {text}")
        return "\n".join(lines) if lines else "(없음)"
