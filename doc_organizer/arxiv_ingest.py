from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .embeddings import OllamaEmbeddingClient
from .milvus_store import MilvusStore
from .settings import IngestionSettings


@dataclass(slots=True)
class ArxivMilvusIngestionSummary:
    total_papers: int
    embedded_papers: int
    upserted_papers: int
    failed_papers: int
    collection_name: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ArxivMilvusIngestionService:
    def __init__(self, settings: IngestionSettings) -> None:
        self.settings = settings
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

    def upsert_from_file(self, file_path: Path) -> ArxivMilvusIngestionSummary:
        raw = self._load_raw(file_path)
        records = [item for item in raw if isinstance(item, dict)]
        total = len(records)
        if total == 0:
            return ArxivMilvusIngestionSummary(
                total_papers=0,
                embedded_papers=0,
                upserted_papers=0,
                failed_papers=0,
                collection_name=self.settings.arxiv_collection_name,
            )

        texts: list[str] = []
        valid_records: list[dict[str, Any]] = []
        for item in records:
            text = self._build_embedding_text(
                title=str(item.get("title", "")),
                summary=str(item.get("summary", "")),
            )
            if not text:
                continue
            texts.append(text)
            valid_records.append(item)

        if not texts:
            return ArxivMilvusIngestionSummary(
                total_papers=total,
                embedded_papers=0,
                upserted_papers=0,
                failed_papers=total,
                collection_name=self.settings.arxiv_collection_name,
            )

        vectors = self.embedder.embed_texts(texts)

        upserted = 0
        failed = total - len(valid_records)
        for item, text, vector in zip(valid_records, texts, vectors, strict=False):
            arxiv_id = str(item.get("arxiv_id", "")).strip()
            if not arxiv_id:
                failed += 1
                continue

            metadata = {
                "arxiv_id": arxiv_id,
                "published": str(item.get("published", "")),
                "updated": str(item.get("updated", "")),
                "authors": [str(author) for author in item.get("authors", [])],
                "categories": [str(category) for category in item.get("categories", [])],
                "pdf_url": str(item.get("pdf_url", "")),
                "source_url": str(item.get("source_url", "")),
                "matched_topics": [str(topic) for topic in item.get("matched_topics", [])],
                "ingested_at": datetime.now(UTC).isoformat(),
            }
            source_path = self.settings.arxiv_storage_dir / "milvus_source" / f"{arxiv_id}.json"
            try:
                self.store.replace_document(
                    source_path=source_path,
                    chunks=[text],
                    vectors=[vector],
                    metadata=metadata,
                )
                upserted += 1
            except Exception:
                failed += 1

        self.store.flush()
        return ArxivMilvusIngestionSummary(
            total_papers=total,
            embedded_papers=len(valid_records),
            upserted_papers=upserted,
            failed_papers=failed,
            collection_name=self.settings.arxiv_collection_name,
        )

    @staticmethod
    def _build_embedding_text(title: str, summary: str) -> str:
        clean_title = " ".join(title.split())
        clean_summary = " ".join(summary.split())
        if not clean_title and not clean_summary:
            return ""
        return f"Title: {clean_title}\nSummary: {clean_summary}".strip()

    @staticmethod
    def _load_raw(file_path: Path) -> list[dict[str, Any]]:
        if not file_path.exists():
            return []
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, dict)]
