from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ollama import Client

from .embeddings import OllamaEmbeddingClient
from .milvus_store import MilvusStore
from .query_analyzer import AnalyzedQuery, analyze_user_query
from .settings import IngestionSettings


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
        self.embedder = OllamaEmbeddingClient(
            host=settings.ollama_host,
            model=settings.embedding_model,
        )
        self.store = MilvusStore(
            collection_name=settings.collection_name,
            uri=settings.milvus_uri,
            user=settings.milvus_user,
            password=settings.milvus_password,
        )
        self.store.connect()
        self.llm_client = Client(host=settings.ollama_host)

    def answer(self, message: str) -> QAResponse:
        analyzed = analyze_user_query(message)
        if not analyzed.normalized:
            return QAResponse(analyzed_query=analyzed, retrieved=[], answer="질문을 입력해 주세요.")

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
        answer = self._generate_answer(analyzed.normalized, retrieved)
        return QAResponse(analyzed_query=analyzed, retrieved=retrieved, answer=answer)

    def _generate_answer(self, user_message: str, retrieved: list[RetrievalResult]) -> str:
        if not retrieved:
            return "관련 문서를 찾지 못했습니다. 질문을 더 구체적으로 입력해 주세요."

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
            response = self.llm_client.chat(
                model=self.settings.qa_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            message = response["message"]["content"]
            return message.strip()
        except Exception:
            top = retrieved[0]
            snippet = " ".join(top.text.split())[:500]
            return (
                "LLM 응답 생성에 실패하여 상위 검색 결과를 기반으로 답변합니다.\n"
                f"- 핵심 근거: {snippet}\n"
                "[1]"
            )

