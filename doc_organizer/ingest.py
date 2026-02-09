from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from glmocr import GlmOcr

from .chunking import chunk_text
from .embeddings import OllamaEmbeddingClient
from .milvus_store import MilvusStore
from .ocr import iter_pdf_files, parse_pdf
from .settings import IngestionSettings


@dataclass(slots=True)
class IngestionSummary:
    total_files: int = 0
    processed_files: int = 0
    skipped_empty_files: int = 0
    inserted_chunks: int = 0
    failed_files: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class DocumentIngestionService:
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

    def run_once(self) -> IngestionSummary:
        self.settings.ensure_directories()
        pdf_files = iter_pdf_files(self.settings.docs_dir)
        summary = IngestionSummary(total_files=len(pdf_files))
        if not pdf_files:
            return summary

        with GlmOcr(config_path=self.settings.ocr_config_path) as parser:
            for pdf_path in pdf_files:
                try:
                    parsed = parse_pdf(parser, pdf_path, self.settings.results_dir)
                    chunks = chunk_text(
                        parsed.text,
                        chunk_size=self.settings.chunk_size,
                        overlap=self.settings.chunk_overlap,
                    )
                    if not chunks:
                        summary.skipped_empty_files += 1
                        continue

                    vectors = self.embedder.embed_texts(chunks)
                    inserted = self.store.replace_document(
                        source_path=pdf_path,
                        chunks=chunks,
                        vectors=vectors,
                        metadata={
                            "file_name": pdf_path.name,
                            "ingested_at": datetime.now(UTC).isoformat(),
                        },
                    )
                    summary.inserted_chunks += inserted
                    summary.processed_files += 1
                except Exception:
                    summary.failed_files += 1
                    continue

        self.store.flush()
        return summary

