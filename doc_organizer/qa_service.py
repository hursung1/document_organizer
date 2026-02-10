from __future__ import annotations

from contextlib import AbstractContextManager
import uuid
from dataclasses import asdict, dataclass
from typing import Any, TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from .embeddings import OllamaEmbeddingClient
from .milvus_store import MilvusStore
from .query_analyzer import AnalyzedQuery, analyze_user_query, parse_llm_analysis
from .settings import IngestionSettings


class RAGState(TypedDict, total=False):
    message: str
    analyzed_query: AnalyzedQuery
    retrieved: list["RetrievalResult"]
    answer: str
    error: str


@dataclass(slots=True)
class RetrievalResult:
    id: int | str
    score: float
    source: str
    chunk_id: int
    text: str


@dataclass(slots=True)
class QAResponse:
    analyzed_query: AnalyzedQuery
    retrieved: list[RetrievalResult]
    answer: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "analyzed_query": asdict(self.analyzed_query),
            "retrieved": [asdict(item) for item in self.retrieved],
            "answer": self.answer,
        }


class DocumentQAService:
    def __init__(self, settings: IngestionSettings) -> None:
        self.settings = settings
        self._checkpointer_cm: AbstractContextManager | None = None
        self._checkpointer: Any | None = None
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
        self.chat_llm = ChatOllama(
            model=settings.qa_model,
            base_url=settings.ollama_host,
            temperature=0,
        )
        self.graph = self._build_graph()

    def answer(self, message: str) -> QAResponse:
        fallback_analyzed = analyze_user_query(message)
        if not fallback_analyzed.normalized:
            return QAResponse(
                analyzed_query=fallback_analyzed,
                retrieved=[],
                answer="질문을 입력해 주세요.",
            )

        thread_id = uuid.uuid4().hex
        config = {"configurable": {"thread_id": thread_id}}
        out = self.graph.invoke({"message": message}, config=config)

        analyzed = out.get("analyzed_query") or fallback_analyzed
        retrieved = out.get("retrieved") or []
        answer = out.get("answer") or out.get("error") or "답변 생성에 실패했습니다."
        return QAResponse(analyzed_query=analyzed, retrieved=retrieved, answer=answer)

    def _build_graph(self):
        workflow = StateGraph(RAGState)
        workflow.add_node("analyze_query", self._node_analyze_query)
        workflow.add_node("retrieve_docs", self._node_retrieve_docs)
        workflow.add_node("generate_answer_doc_base", self._node_generate_answer_doc_base)
        workflow.add_node("generate_answer_llm", self._node_generate_answer_llm)
        workflow.add_edge(START, "analyze_query")
        workflow.add_edge("analyze_query", "retrieve_docs")
        workflow.add_conditional_edges("retrieve_docs", self._conditional_edge_doc_base_or_not)
        workflow.add_edge("generate_answer_doc_base", END)
        workflow.add_edge("generate_answer_llm", END)
        return workflow.compile(checkpointer=self._ensure_redis_checkpointer())

    def _ensure_redis_checkpointer(self):
        if self._checkpointer is not None:
            return self._checkpointer
        try:
            from langgraph.checkpoint.redis import RedisSaver
        except Exception as exc:
            raise RuntimeError(
                "Redis checkpointer import failed. Install langgraph-checkpoint-redis."
            ) from exc

        if hasattr(RedisSaver, "from_conn_string"):
            cm = RedisSaver.from_conn_string(self.settings.redis_url)
            saver = cm.__enter__()
            saver.setup()
            self._checkpointer_cm = cm
            self._checkpointer = saver
            return saver

        if hasattr(RedisSaver, "from_url"):
            saver = RedisSaver.from_url(self.settings.redis_url)
            saver.setup()
            self._checkpointer = saver
            return saver

        saver = RedisSaver(self.settings.redis_url)
        saver.setup()
        self._checkpointer = saver
        return saver

    def close(self) -> None:
        if self._checkpointer_cm is not None:
            self._checkpointer_cm.__exit__(None, None, None)
            self._checkpointer_cm = None
        self._checkpointer = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _invoke_json_analysis(self, message: str) -> str:
        analysis_prompt = (
            "사용자 질문을 검색용 질의로 변환해 JSON으로만 답하라.\n"
            '형식: {"normalized":"...", "keywords":["..."], "search_query":"..."}\n'
            "규칙:\n"
            "1) normalized는 질문 의도를 보존한 정규화 문장\n"
            "2) keywords는 핵심 키워드 3~8개\n"
            "3) search_query는 벡터/BM25 검색에 유리한 확장 질의\n"
            "4) JSON 외 다른 텍스트 금지"
        )
        response = self.chat_llm.invoke(
            [
                ("system", analysis_prompt),
                ("human", message),
            ]
        )
        return str(response.content or "").strip()

    def _node_analyze_query(self, state: RAGState) -> RAGState:
        message = (state.get("message") or "").strip()
        fallback = analyze_user_query(message)
        if not fallback.normalized:
            return {
                "analyzed_query": fallback,
                "error": "질문이 비어 있습니다.",
            }
        try:
            llm_text = self._invoke_json_analysis(message)
            analyzed = parse_llm_analysis(message, llm_text)
            return {"analyzed_query": analyzed}
        except Exception as exc:
            return {
                "analyzed_query": fallback,
                "error": f"질의 분석 LLM 오류: {exc}",
            }

    def _node_retrieve_docs(self, state: RAGState) -> RAGState:
        analyzed = state.get("analyzed_query")
        if not analyzed:
            return {"retrieved": [], "error": "질의 분석 결과가 없습니다."}
        try:
            query_vector = self.embedder.embed_query(analyzed.search_query)
            hits = self.store.hybrid_search(
                query_text=analyzed.search_query,
                query_vector=query_vector,
                limit=self.settings.retrieval_top_k,
                dense_weight=self.settings.dense_weight,
                sparse_weight=self.settings.sparse_weight,
            )
            retrieved = [
                RetrievalResult(
                    id=item["id"],
                    score=item["score"],
                    source=item.get("source") or "",
                    chunk_id=int(item.get("chunk_id") or 0),
                    text=item.get("text") or "",
                )
                for item in hits
            ]
            return {"retrieved": retrieved}
        except Exception as exc:
            return {
                "retrieved": [],
                "error": (
                    "검색 단계에서 오류가 발생했습니다. "
                    f"Milvus/임베딩 연결 상태를 확인해 주세요. ({exc})"
                ),
            }

    def _node_generate_answer_doc_base(self, state: RAGState) -> RAGState:
        user_message = (state.get("message") or "").strip()
        retrieved = state.get("retrieved") or []
        if not retrieved:
            prior_error = state.get("error")
            if prior_error:
                return {"answer": prior_error}
            return {"answer": "관련 문서를 찾지 못했습니다. 질문을 더 구체적으로 입력해 주세요."}

        context_lines: list[str] = []
        for index, item in enumerate(retrieved, start=1):
            source = item.source.rsplit("/", 1)[-1] if item.source else "unknown"
            snippet = " ".join(item.text.split())
            context_lines.append(
                f"[{index}] source={source}, chunk={item.chunk_id}, score={item.score:.4f}\n{snippet[:1000]}"
            )
        context = "\n\n".join(context_lines)

        system_prompt = (
            "당신은 문서 기반 질의응답 도우미다. 반드시 제공된 context 근거로만 답변하고, "
            "근거가 부족하면 모른다고 답해라. 답변 마지막에 사용한 근거 번호를 [1], [2] 형식으로 표시해라."
        )
        user_prompt = (
            f"question:\n{user_message}\n\n"
            f"context:\n{context}\n\n"
            "한국어로 간결하고 정확하게 답변해라."
        )
        try:
            response = self.chat_llm.invoke(
                [
                    ("system", system_prompt),
                    ("human", user_prompt),
                ]
            )
            answer = str(response.content or "").strip()
            if not answer:
                raise RuntimeError("Empty response from LLM.")
            return {"answer": answer}
        except Exception as exc:
            top = retrieved[0]
            snippet = " ".join(top.text.split())[:500]
            return {
                "answer": (
                    f"LLM 응답 생성 오류: {exc}\n"
                    "상위 검색 결과를 기반으로 대체 답변을 제공합니다.\n"
                    f"- 핵심 근거: {snippet}\n"
                    "[1]"
                )
            }
        
    def _node_generate_answer_llm(self, state: RAGState) -> RAGState:
        user_message = state.get("message")

        system_prompt = (
            "당신은 질의응답 도우미다. 사용자가 특정 사항에 관한 요구사항이 있다면 친절히 답변해라."
            "모르는 내용에 대해서 절대 답변하지 마라."
        )
        user_prompt = (
            f"question:\n{user_message}\n\n"
            "한국어로 간결하고 정확하게 답변해라."
        )
        try:
            response = self.chat_llm.invoke(
                [
                    ("system", system_prompt),
                    ("human", user_prompt),
                ]
            )
            answer = str(response.content or "").strip()
            if not answer:
                raise RuntimeError("Empty response from LLM.")
            return {"answer": answer}
        except Exception as exc:
            return {
                "answer": (
                    f"LLM 응답 생성 오류: {exc}\n"
                )
            }

    def _conditional_edge_doc_base_or_not(self, state: RAGState) -> str:
        retreived = state.get("retrieved", [])
        return "generate_answer_doc_base" if retreived else "generate_answer_llm"
