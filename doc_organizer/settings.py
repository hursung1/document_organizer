from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class IngestionSettings:
    docs_dir: Path = Path("src_docs")
    results_dir: Path = Path("results")
    ocr_config_path: str = "config.yaml"
    collection_name: str = "doc_chunks"
    milvus_uri: str = "tcp://localhost:19530"
    milvus_user: str | None = None
    milvus_password: str | None = None
    ollama_host: str = "http://localhost:11434"
    embedding_model: str = "embeddinggemma:latest"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    interval_seconds: int = 21600
    auto_ingest_enabled: bool = False
    retrieval_top_k: int = 5
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    llm_provider: str = "gemini"
    qa_model: str = "qwen3:latest"
    ollama_reasoning: bool | str | None = None
    gemini_model: str = "gemini-2.0-flash"
    gemini_api_key: str | None = None
    redis_url: str = "redis://localhost:6379/0"
    arxiv_storage_dir: Path = Path("results/arxiv")
    arxiv_collection_name: str = "arxiv_papers"
    arxiv_schedule_hour: int = 9
    arxiv_max_results_per_topic: int = 50
    arxiv_topics: tuple[str, ...] = (
        "computer science",
        "artificial intelligence",
        "large language model",
        "LLM",
    )
    arxiv_pdf_enrich_enabled: bool = True
    arxiv_pdf_cache_dir: Path = Path("results/arxiv/pdf_cache")
    arxiv_pdf_cache_ttl_hours: int = 168
    arxiv_pdf_max_docs_per_query: int = 2
    arxiv_pdf_max_snippets_per_doc: int = 3
    arxiv_pdf_min_score: float = 0.35
    arxiv_pdf_download_timeout_sec: int = 30
    arxiv_pdf_extract_max_chars: int = 120000

    @classmethod
    def from_env(cls) -> IngestionSettings:
        qa_model = os.getenv("QA_MODEL", "qwen3:latest")
        ollama_reasoning = cls._parse_ollama_reasoning(
            os.getenv("OLLAMA_REASONING"),
            qa_model=qa_model,
        )
        return cls(
            docs_dir=Path(os.getenv("DOCS_DIR", "src_docs")),
            results_dir=Path(os.getenv("RESULTS_DIR", "results")),
            ocr_config_path=os.getenv("OCR_CONFIG_PATH", "config.yaml"),
            collection_name=os.getenv("MILVUS_COLLECTION", "doc_chunks"),
            milvus_uri=os.getenv("MILVUS_URI", "tcp://localhost:19530"),
            milvus_user=os.getenv("MILVUS_USER"),
            milvus_password=os.getenv("MILVUS_PASSWORD"),
            ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            embedding_model=os.getenv("EMBED_MODEL", "embeddinggemma:latest"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            interval_seconds=int(os.getenv("INGEST_INTERVAL_SECONDS", "21600")),
            auto_ingest_enabled=os.getenv("AUTO_INGEST_ENABLED", "false").lower()
            in {"1", "true", "yes", "on"},
            retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "5")),
            dense_weight=float(os.getenv("DENSE_WEIGHT", "0.7")),
            sparse_weight=float(os.getenv("SPARSE_WEIGHT", "0.3")),
            llm_provider=os.getenv("LLM_PROVIDER", "gemini"),
            qa_model=qa_model,
            ollama_reasoning=ollama_reasoning,
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            arxiv_storage_dir=Path(os.getenv("ARXIV_STORAGE_DIR", "results/arxiv")),
            arxiv_collection_name=os.getenv("ARXIV_COLLECTION", "arxiv_papers"),
            arxiv_schedule_hour=int(os.getenv("ARXIV_SCHEDULE_HOUR", "9")),
            arxiv_max_results_per_topic=int(os.getenv("ARXIV_MAX_RESULTS_PER_TOPIC", "50")),
            arxiv_topics=tuple(
                topic.strip()
                for topic in os.getenv(
                    "ARXIV_TOPICS",
                    "computer science,artificial intelligence,large language model,LLM",
                ).split(",")
                if topic.strip()
            ),
            arxiv_pdf_enrich_enabled=os.getenv("ARXIV_PDF_ENRICH_ENABLED", "true").lower()
            in {"1", "true", "yes", "on"},
            arxiv_pdf_cache_dir=Path(
                os.getenv("ARXIV_PDF_CACHE_DIR", "results/arxiv/pdf_cache")
            ),
            arxiv_pdf_cache_ttl_hours=int(os.getenv("ARXIV_PDF_CACHE_TTL_HOURS", "168")),
            arxiv_pdf_max_docs_per_query=int(os.getenv("ARXIV_PDF_MAX_DOCS_PER_QUERY", "2")),
            arxiv_pdf_max_snippets_per_doc=int(
                os.getenv("ARXIV_PDF_MAX_SNIPPETS_PER_DOC", "3")
            ),
            arxiv_pdf_min_score=float(os.getenv("ARXIV_PDF_MIN_SCORE", "0.35")),
            arxiv_pdf_download_timeout_sec=int(
                os.getenv("ARXIV_PDF_DOWNLOAD_TIMEOUT_SEC", "30")
            ),
            arxiv_pdf_extract_max_chars=int(
                os.getenv("ARXIV_PDF_EXTRACT_MAX_CHARS", "120000")
            ),
        )

    def ensure_directories(self) -> None:
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.arxiv_storage_dir.mkdir(parents=True, exist_ok=True)
        self.arxiv_pdf_cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _parse_ollama_reasoning(
        raw_value: str | None,
        *,
        qa_model: str,
    ) -> bool | str | None:
        if raw_value is None:
            model = (qa_model or "").strip().lower()
            if model.startswith("gpt-oss"):
                return "high"
            return None

        normalized = raw_value.strip().lower()
        if normalized in {"", "none", "null", "auto", "default"}:
            return None
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return raw_value.strip()
