from __future__ import annotations

from typing import Any

from ollama import Client


class OllamaEmbeddingClient:
    def __init__(self, host: str, model: str) -> None:
        self._client = Client(host=host)
        self._model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = self._client.embed(model=self._model, input=texts)
        embeddings = self._extract_embeddings(response)
        if not embeddings:
            raise RuntimeError("Ollama returned empty embeddings.")
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed_texts([text])
        if not vectors:
            raise RuntimeError("Failed to create query embedding.")
        return vectors[0]

    @staticmethod
    def _extract_embeddings(response: Any) -> list[list[float]]:
        embeddings = getattr(response, "embeddings", None)
        if embeddings:
            return embeddings
        if isinstance(response, dict):
            data = response.get("embeddings")
            if data:
                return data
        raise RuntimeError("Failed to parse embeddings from Ollama response.")
