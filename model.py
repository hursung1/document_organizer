from __future__ import annotations

import argparse
from pathlib import Path

from doc_organizer.ingest import DocumentIngestionService
from doc_organizer.settings import IngestionSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OCR -> Milvus ingestion once.")
    parser.add_argument("--docs-dir", default="src_docs")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--config-path", default="config.yaml")
    parser.add_argument("--collection", default="doc_chunks")
    parser.add_argument("--milvus-uri", default="http://localhost:19530")
    parser.add_argument("--milvus-user", default=None)
    parser.add_argument("--milvus-password", default=None)
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--embed-model", default="embeddinggemma:latest")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = IngestionSettings(
        docs_dir=Path(args.docs_dir),
        results_dir=Path(args.results_dir),
        ocr_config_path=args.config_path,
        collection_name=args.collection,
        milvus_uri=args.milvus_uri,
        milvus_user=args.milvus_user,
        milvus_password=args.milvus_password,
        ollama_host=args.ollama_host,
        embedding_model=args.embed_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    service = DocumentIngestionService(settings)
    summary = service.run_once()
    print(summary.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
