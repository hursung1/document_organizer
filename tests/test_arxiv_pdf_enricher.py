from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from doc_organizer.arxiv_pdf_enricher import ArxivPdfEnricher
from doc_organizer.settings import IngestionSettings


class ArxivPdfEnricherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.settings = IngestionSettings(
            arxiv_pdf_cache_dir=Path(self.temp_dir.name),
            arxiv_pdf_max_docs_per_query=2,
            arxiv_pdf_max_snippets_per_doc=2,
        )
        self.enricher = ArxivPdfEnricher(self.settings)

    def test_select_candidates_dedup_and_limit(self) -> None:
        docs = [
            {"arxiv_id": "2501.00001", "pdf_url": "", "source_url": ""},
            {"arxiv_id": "2501.00001", "pdf_url": "", "source_url": ""},
            {"arxiv_id": "2501.00002", "pdf_url": "", "source_url": ""},
            {"arxiv_id": "2501.00003", "pdf_url": "", "source_url": ""},
        ]

        candidates = self.enricher._select_candidates(docs)

        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0]["arxiv_id"], "2501.00001")
        self.assertEqual(candidates[1]["arxiv_id"], "2501.00002")

    def test_cache_read_write_with_ttl(self) -> None:
        arxiv_id = "2501.01234"
        text = "cached content"
        self.enricher._write_cache_sync(arxiv_id=arxiv_id, pdf_url="https://arxiv.org/pdf/2501.01234.pdf", text=text)

        loaded = self.enricher._read_cache_if_valid(arxiv_id)
        self.assertEqual(loaded, text)

    def test_cache_expired_returns_empty(self) -> None:
        arxiv_id = "2501.02222"
        self.enricher.settings.arxiv_pdf_cache_ttl_hours = -1
        self.enricher._write_cache_sync(
            arxiv_id=arxiv_id,
            pdf_url="https://arxiv.org/pdf/2501.02222.pdf",
            text="old content",
        )

        loaded = self.enricher._read_cache_if_valid(arxiv_id)
        self.assertEqual(loaded, "")

    def test_snippet_selection_prefers_query_overlap(self) -> None:
        text = (
            "This paragraph is unrelated to the question.\n\n"
            "The algorithm uses ablation studies and detailed experimental setup.\n\n"
            "Another short unrelated sentence."
        )
        query_tokens = self.enricher._tokenize("algorithm ablation experiment")
        keyword_tokens = {"algorithm", "ablation"}

        snippets = self.enricher._select_relevant_snippets(
            text=text,
            query_tokens=query_tokens,
            keyword_tokens=keyword_tokens,
            limit=2,
        )

        self.assertTrue(snippets)
        self.assertIn("algorithm", snippets[0][0].lower())


if __name__ == "__main__":
    unittest.main()
