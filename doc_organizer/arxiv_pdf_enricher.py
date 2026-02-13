from __future__ import annotations

import asyncio
import json
import logging
import re
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .query_analyzer import AnalyzedQuery
from .settings import IngestionSettings

logger = logging.getLogger(__name__)

NEW_ID_PATTERN = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b")
OLD_ID_PATTERN = re.compile(r"\b[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?\b")


@dataclass(slots=True)
class PdfEvidence:
    arxiv_id: str
    pdf_url: str
    source_url: str | None
    snippet: str
    score: float


@dataclass(slots=True)
class PdfEnrichMetrics:
    attempted: bool = False
    cache_hit_count: int = 0
    download_fail_count: int = 0


class ArxivPdfEnricher:
    def __init__(self, settings: IngestionSettings) -> None:
        self.settings = settings
        self.cache_dir = settings.arxiv_pdf_cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()
        self.last_metrics = PdfEnrichMetrics()

    async def enrich(
        self,
        query: str,
        analyzed_query: AnalyzedQuery,
        retrieved_docs: list[Any],
    ) -> list[PdfEvidence]:
        self.last_metrics = PdfEnrichMetrics()
        if not self.settings.arxiv_pdf_enrich_enabled:
            return []

        candidates = self._select_candidates(retrieved_docs)
        if not candidates:
            return []
        self.last_metrics.attempted = True

        query_tokens = self._tokenize(f"{query} {analyzed_query.search_query}".strip())
        keyword_tokens = {k.lower() for k in analyzed_query.keywords if k.strip()}

        evidences: list[PdfEvidence] = []
        for candidate in candidates:
            arxiv_id = str(candidate.get("arxiv_id", "")).strip()
            if not arxiv_id:
                continue
            pdf_url = str(candidate.get("pdf_url", "")).strip() or self._build_pdf_url(arxiv_id)
            source_url = str(candidate.get("source_url", "")).strip() or None

            text = await self._get_or_build_text(arxiv_id=arxiv_id, pdf_url=pdf_url)
            if not text:
                continue

            snippets = self._select_relevant_snippets(
                text=text,
                query_tokens=query_tokens,
                keyword_tokens=keyword_tokens,
                limit=self.settings.arxiv_pdf_max_snippets_per_doc,
            )
            for snippet, score in snippets:
                evidences.append(
                    PdfEvidence(
                        arxiv_id=arxiv_id,
                        pdf_url=pdf_url,
                        source_url=source_url,
                        snippet=snippet,
                        score=score,
                    )
                )

        evidences.sort(key=lambda item: item.score, reverse=True)
        return evidences

    @staticmethod
    def _build_pdf_url(arxiv_id: str) -> str:
        clean_id = arxiv_id.strip()
        return f"https://arxiv.org/pdf/{clean_id}.pdf"

    def _select_candidates(self, retrieved_docs: list[Any]) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        seen_ids: set[str] = set()
        for doc in retrieved_docs:
            arxiv_id = self._pick_first_text(
                self._candidate_get(doc, "arxiv_id"),
                self._extract_arxiv_id(self._candidate_get(doc, "source")),
            )
            if not arxiv_id or arxiv_id in seen_ids:
                continue
            seen_ids.add(arxiv_id)
            out.append(
                {
                    "arxiv_id": arxiv_id,
                    "pdf_url": self._pick_first_text(
                        self._candidate_get(doc, "pdf_url"),
                        self._build_pdf_url(arxiv_id),
                    ),
                    "source_url": self._candidate_get(doc, "source_url"),
                }
            )
            if len(out) >= self.settings.arxiv_pdf_max_docs_per_query:
                break
        return out

    @staticmethod
    def _candidate_get(doc: Any, key: str) -> str:
        value: Any = None
        if isinstance(doc, dict):
            value = doc.get(key)
        else:
            value = getattr(doc, key, None)
        return str(value).strip() if value is not None else ""

    @staticmethod
    def _pick_first_text(*values: str) -> str:
        for value in values:
            normalized = str(value).strip()
            if normalized:
                return normalized
        return ""

    @staticmethod
    def _extract_arxiv_id(value: str) -> str:
        source = (value or "").strip()
        if not source:
            return ""
        matched = NEW_ID_PATTERN.search(source)
        if matched:
            return matched.group(0)
        matched = OLD_ID_PATTERN.search(source.lower())
        if matched:
            return matched.group(0)
        return ""

    async def _get_or_build_text(self, arxiv_id: str, pdf_url: str) -> str:
        lock = await self._lock_for(arxiv_id)
        async with lock:
            cached_text = await asyncio.to_thread(self._read_cache_if_valid, arxiv_id)
            if cached_text:
                self.last_metrics.cache_hit_count += 1
                return cached_text

            pdf_path = self._pdf_path(arxiv_id)
            try:
                await asyncio.to_thread(
                    self._download_pdf_sync,
                    pdf_url,
                    pdf_path,
                    self.settings.arxiv_pdf_download_timeout_sec,
                )
            except Exception as exc:
                self.last_metrics.download_fail_count += 1
                logger.warning("arxiv_pdf_download_failed id=%s url=%s err=%s", arxiv_id, pdf_url, exc)
                return ""

            try:
                text = await asyncio.to_thread(self._extract_text_from_pdf_sync, pdf_path)
            except Exception as exc:
                logger.warning("arxiv_pdf_ocr_failed id=%s path=%s err=%s", arxiv_id, pdf_path, exc)
                return ""
            if not text:
                return ""

            trimmed = text[: self.settings.arxiv_pdf_extract_max_chars]
            await asyncio.to_thread(self._write_cache_sync, arxiv_id, pdf_url, trimmed)
            return trimmed

    async def _lock_for(self, arxiv_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            lock = self._locks.get(arxiv_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[arxiv_id] = lock
            return lock

    def _read_cache_if_valid(self, arxiv_id: str) -> str:
        txt_path = self._txt_path(arxiv_id)
        meta_path = self._meta_path(arxiv_id)
        if not txt_path.exists() or not meta_path.exists():
            return ""
        try:
            raw_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return ""
        if not isinstance(raw_meta, dict):
            return ""

        expires_at_str = str(raw_meta.get("expires_at", "")).strip()
        if not expires_at_str:
            return ""
        try:
            expires_at = datetime.fromisoformat(expires_at_str)
        except ValueError:
            return ""
        now = datetime.now(UTC)
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        if now >= expires_at:
            return ""
        try:
            return txt_path.read_text(encoding="utf-8")
        except Exception:
            return ""

    def _write_cache_sync(self, arxiv_id: str, pdf_url: str, text: str) -> None:
        pdf_path = self._pdf_path(arxiv_id)
        txt_path = self._txt_path(arxiv_id)
        meta_path = self._meta_path(arxiv_id)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(text, encoding="utf-8")
        now = datetime.now(UTC)
        meta = {
            "fetched_at": now.isoformat(),
            "parsed_at": now.isoformat(),
            "pdf_url": pdf_url,
            "expires_at": (now + timedelta(hours=self.settings.arxiv_pdf_cache_ttl_hours)).isoformat(),
        }
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _download_pdf_sync(pdf_url: str, pdf_path: Path, timeout_sec: int) -> None:
        request = urllib.request.Request(
            url=pdf_url,
            headers={"User-Agent": "doc-organizer/0.1"},
        )
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            body = response.read()
        if not body:
            raise RuntimeError("Downloaded PDF body is empty.")
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(body)

    def _extract_text_from_pdf_sync(self, pdf_path: Path) -> str:
        try:
            from glmocr import GlmOcr
        except Exception as exc:
            raise RuntimeError("glmocr import failed. Install glmocr to enable PDF enrichment.") from exc
        with GlmOcr(config_path=self.settings.ocr_config_path) as parser:
            result = parser.parse(str(pdf_path))
            text = self._extract_text(result.json_result, result.markdown_result)
        return text

    @staticmethod
    def _extract_text(ocr_json: Any, markdown_result: str | None) -> str:
        if markdown_result and markdown_result.strip():
            return markdown_result.strip()

        texts: list[str] = []

        def walk(value: Any) -> None:
            if isinstance(value, dict):
                content = value.get("content")
                if isinstance(content, str) and content.strip():
                    texts.append(content.strip())
                for item in value.values():
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        walk(ocr_json)
        return "\n".join(texts).strip()

    @classmethod
    def _tokenize(cls, text: str) -> set[str]:
        return set(re.findall(r"[A-Za-z0-9가-힣]{2,}", (text or "").lower()))

    def _select_relevant_snippets(
        self,
        text: str,
        query_tokens: set[str],
        keyword_tokens: set[str],
        limit: int,
    ) -> list[tuple[str, float]]:
        segments = self._split_text(text)
        scored: list[tuple[float, str]] = []
        seen: set[str] = set()
        for segment in segments:
            key = re.sub(r"\s+", " ", segment).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            segment_tokens = self._tokenize(segment)
            if not segment_tokens:
                continue
            overlap = len(query_tokens & segment_tokens)
            keyword_hits = len(keyword_tokens & segment_tokens)
            score = float(overlap + (keyword_hits * 1.5))
            if score <= 0:
                continue
            scored.append((score, segment))

        scored.sort(key=lambda row: row[0], reverse=True)
        return [(segment, score) for score, segment in scored[: max(1, limit)]]

    @staticmethod
    def _split_text(text: str) -> list[str]:
        if not text.strip():
            return []
        normalized = re.sub(r"[ \t]+", " ", text)
        paragraphs = [
            re.sub(r"\s+", " ", item).strip()
            for item in re.split(r"\n{2,}", normalized)
            if re.sub(r"\s+", " ", item).strip()
        ]
        if not paragraphs:
            paragraphs = [re.sub(r"\s+", " ", normalized).strip()]

        segments: list[str] = []
        max_chars = 1000
        for para in paragraphs:
            if len(para) <= max_chars:
                segments.append(para)
                continue
            sentences = re.split(r"(?<=[.!?])\s+", para)
            chunk = ""
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if len(chunk) + len(sentence) + 1 <= max_chars:
                    chunk = f"{chunk} {sentence}".strip()
                else:
                    if chunk:
                        segments.append(chunk)
                    chunk = sentence
            if chunk:
                segments.append(chunk)
        return segments

    def _cache_stem(self, arxiv_id: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", arxiv_id.strip())

    def _pdf_path(self, arxiv_id: str) -> Path:
        return self.cache_dir / f"{self._cache_stem(arxiv_id)}.pdf"

    def _txt_path(self, arxiv_id: str) -> Path:
        return self.cache_dir / f"{self._cache_stem(arxiv_id)}.txt"

    def _meta_path(self, arxiv_id: str) -> Path:
        return self.cache_dir / f"{self._cache_stem(arxiv_id)}.meta.json"
