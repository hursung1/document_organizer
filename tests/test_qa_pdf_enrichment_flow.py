from __future__ import annotations

import unittest

from doc_organizer.arxiv_pdf_enricher import PdfEvidence
from doc_organizer.qa_service import DocumentQAService, RetrievalResult


class QaPdfEnrichmentFlowTests(unittest.TestCase):
    def test_safe_parse_metadata(self) -> None:
        valid = DocumentQAService._safe_parse_metadata('{"arxiv_id":"2501.01234","pdf_url":"u"}')
        invalid = DocumentQAService._safe_parse_metadata("not json")

        self.assertEqual(valid.get("arxiv_id"), "2501.01234")
        self.assertEqual(invalid, {})

    def test_needs_pdf_enrichment_on_low_score(self) -> None:
        retrieved = [
            RetrievalResult(
                id="1",
                score=0.1,
                source="2501.01234",
                chunk_id=0,
                text="t",
                arxiv_id="2501.01234",
            )
        ]

        needed = DocumentQAService._needs_pdf_enrichment(
            message="논문 원문 기준으로 설명해줘",
            retrieved=retrieved,
            top_k=5,
            min_score=0.35,
        )
        self.assertTrue(needed)

    def test_needs_pdf_enrichment_false_when_high_confidence(self) -> None:
        retrieved = [
            RetrievalResult(
                id=str(idx),
                score=0.9,
                source=f"2501.0123{idx}",
                chunk_id=idx,
                text="content",
                arxiv_id=f"2501.0123{idx}",
            )
            for idx in range(5)
        ]

        needed = DocumentQAService._needs_pdf_enrichment(
            message="요약해줘",
            retrieved=retrieved,
            top_k=5,
            min_score=0.35,
        )
        self.assertFalse(needed)

    def test_route_doc_base_when_pdf_evidence_exists(self) -> None:
        service = DocumentQAService.__new__(DocumentQAService)
        route = service._conditional_edge_doc_base_or_not(
            {
                "retrieved": [],
                "pdf_evidence": [
                    PdfEvidence(
                        arxiv_id="2501.01234",
                        pdf_url="https://arxiv.org/pdf/2501.01234.pdf",
                        source_url=None,
                        snippet="evidence",
                        score=2.0,
                    )
                ],
            }
        )
        self.assertEqual(route, "generate_answer_doc_base")

    def test_append_source_lines(self) -> None:
        answer = "답변 본문"
        out = DocumentQAService._append_source_lines(
            answer,
            ["- [arXiv:2501.01234](https://arxiv.org/pdf/2501.01234.pdf)"],
        )
        self.assertIn("Sources:", out)
        self.assertIn("2501.01234", out)


if __name__ == "__main__":
    unittest.main()
