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
    milvus_uri: str = "http://localhost:19530"
    milvus_user: str | None = None
    milvus_password: str | None = None
    ollama_host: str = "http://localhost:11434"
    embedding_model: str = "embeddinggemma:latest"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    interval_seconds: int = 21600
    retrieval_top_k: int = 5
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    qa_model: str = "qwen3:latest"
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

    @classmethod
    def from_env(cls) -> IngestionSettings:
        return cls(
            docs_dir=Path(os.getenv("DOCS_DIR", "src_docs")),
            results_dir=Path(os.getenv("RESULTS_DIR", "results")),
            ocr_config_path=os.getenv("OCR_CONFIG_PATH", "config.yaml"),
            collection_name=os.getenv("MILVUS_COLLECTION", "doc_chunks"),
            milvus_uri=os.getenv("MILVUS_URI", "http://localhost:19530"),
            milvus_user=os.getenv("MILVUS_USER"),
            milvus_password=os.getenv("MILVUS_PASSWORD"),
            ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            embedding_model=os.getenv("EMBED_MODEL", "embeddinggemma:latest"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            interval_seconds=int(os.getenv("INGEST_INTERVAL_SECONDS", "21600")),
            retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "5")),
            dense_weight=float(os.getenv("DENSE_WEIGHT", "0.7")),
            sparse_weight=float(os.getenv("SPARSE_WEIGHT", "0.3")),
            qa_model=os.getenv("QA_MODEL", "qwen3:latest"),
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
        )

    def ensure_directories(self) -> None:
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.arxiv_storage_dir.mkdir(parents=True, exist_ok=True)
