from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pymilvus import (
    AnnSearchRequest,
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    WeightedRanker,
    connections,
    utility,
)

TEXT_FIELD = "text"
DENSE_VECTOR_FIELD = "vector"
SPARSE_VECTOR_FIELD = "sparse_vector"
BM25_FUNCTION_NAME = "text_bm25_fn"


class MilvusStore:
    def __init__(
        self,
        collection_name: str,
        uri: str,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        self._collection_name = collection_name
        self._uri = uri
        self._user = user
        self._password = password
        self._collection: Collection | None = None

    def connect(self) -> None:
        connections.connect(
            alias="default",
            uri=self._uri,
            user=self._user,
            password=self._password,
        )

    def ensure_collection(self, dim: int) -> Collection:
        if self._collection is not None:
            return self._collection

        if utility.has_collection(self._collection_name):
            collection = Collection(self._collection_name)
            vector_field = next(f for f in collection.schema.fields if f.name == DENSE_VECTOR_FIELD)
            existing_dim = vector_field.params.get("dim")
            if existing_dim != dim:
                raise RuntimeError(
                    f"Milvus dim mismatch: collection={existing_dim}, embedding={dim}"
                )
            field_names = {f.name for f in collection.schema.fields}
            required_fields = {TEXT_FIELD, DENSE_VECTOR_FIELD, SPARSE_VECTOR_FIELD}
            if not required_fields.issubset(field_names):
                raise RuntimeError(
                    "Existing Milvus collection schema is incompatible. "
                    "Expected text/vector/sparse_vector fields for hybrid search."
                )
            text_field = next(f for f in collection.schema.fields if f.name == TEXT_FIELD)
            analyzer_enabled = bool(text_field.params.get("enable_analyzer"))
            if not analyzer_enabled:
                raise RuntimeError(
                    "Existing Milvus collection is missing text analyzer "
                    "(text.enable_analyzer=true required for BM25). "
                    "Drop/recreate the collection or use a new collection name."
                )
            self._collection = collection
            return collection

        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="chunk_id", dtype=DataType.INT64),
            FieldSchema(
                name=TEXT_FIELD,
                dtype=DataType.VARCHAR,
                max_length=65535,
                enable_analyzer=True,
            ),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name=DENSE_VECTOR_FIELD, dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name=SPARSE_VECTOR_FIELD, dtype=DataType.SPARSE_FLOAT_VECTOR),
        ]
        bm25_function = Function(
            name=BM25_FUNCTION_NAME,
            function_type=FunctionType.BM25,
            input_field_names=[TEXT_FIELD],
            output_field_names=[SPARSE_VECTOR_FIELD],
        )
        schema = CollectionSchema(
            fields=fields,
            description="OCR document chunks",
            functions=[bm25_function],
        )
        collection = Collection(name=self._collection_name, schema=schema)
        collection.create_index(
            field_name=DENSE_VECTOR_FIELD,
            index_params={
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 200},
            },
        )
        collection.create_index(
            field_name=SPARSE_VECTOR_FIELD,
            index_params={
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "BM25",
                "params": {},
            },
        )
        self._collection = collection
        return collection

    def replace_document(
        self,
        source_path: Path,
        chunks: list[str],
        vectors: list[list[float]],
        metadata: dict[str, Any],
    ) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors size mismatch")
        if not chunks:
            return 0

        collection = self.ensure_collection(dim=len(vectors[0]))
        source = str(source_path.resolve())
        escaped_source = source.replace("\\", "\\\\").replace('"', '\\"')
        collection.delete(expr=f'source == "{escaped_source}"')

        metadata_json = json.dumps(metadata, ensure_ascii=False)
        entities = [
            [source] * len(chunks),
            list(range(len(chunks))),
            chunks,
            [metadata_json] * len(chunks),
            vectors,
        ]
        collection.insert(entities)
        return len(chunks)

    def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float],
        limit: int = 5,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> list[dict[str, Any]]:
        if not query_text.strip():
            return []
        collection = self.ensure_collection(dim=len(query_vector))
        collection.load()

        search_limit = max(20, limit * 4)
        dense_req = AnnSearchRequest(
            data=[query_vector],
            anns_field=DENSE_VECTOR_FIELD,
            param={
                "metric_type": "COSINE",
                "params": {"ef": 128},
            },
            limit=search_limit,
        )
        sparse_req = AnnSearchRequest(
            data=[query_text],
            anns_field=SPARSE_VECTOR_FIELD,
            param={
                "metric_type": "BM25",
                "params": {"drop_ratio_search": 0.2},
            },
            limit=search_limit,
        )

        bm25_ranker = Function(
            name=BM25_FUNCTION_NAME,
            function_type=FunctionType.BM25,
            input_field_names=[TEXT_FIELD],
            output_field_names=[SPARSE_VECTOR_FIELD],
        )
        reranker = WeightedRanker(dense_weight, sparse_weight)
        search_res = collection.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=reranker,
            limit=limit,
            output_fields=["source", "chunk_id", TEXT_FIELD, "metadata"],
            ranker=bm25_ranker,
        )

        hits = search_res[0] if search_res else []
        records: list[dict[str, Any]] = []
        for hit in hits:
            entity = hit.entity
            records.append(
                {
                    "id": hit.id,
                    "score": float(hit.score),
                    "source": entity.get("source"),
                    "chunk_id": entity.get("chunk_id"),
                    "text": entity.get(TEXT_FIELD, ""),
                    "metadata": entity.get("metadata"),
                }
            )
        return records

    def flush(self) -> None:
        if self._collection is not None:
            self._collection.flush()
