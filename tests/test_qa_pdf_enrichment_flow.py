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

    def test_needs_pdf_enrichment_true_for_detail_question(self) -> None:
        retrieved = [
            RetrievalResult(
                id=str(idx),
                score=0.95,
                source=f"2501.0123{idx}",
                chunk_id=idx,
                text="content",
                arxiv_id=f"2501.0123{idx}",
            )
            for idx in range(5)
        ]

        needed = DocumentQAService._needs_pdf_enrichment(
            message="이 논문 내용을 상세하게, 본문 기준으로 자세히 설명해줘",
            retrieved=retrieved,
            top_k=5,
            min_score=0.35,
        )
        self.assertTrue(needed)

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

    def test_extract_memory_arxiv_candidates_from_metadata(self) -> None:
        history = [
            {
                "role": "assistant",
                "text": "이전 답변",
                "metadata": {
                    "arxiv_memory": [
                        {
                            "arxiv_id": "2501.01234",
                            "title": "Memory Title",
                            "pdf_url": "https://arxiv.org/pdf/2501.01234.pdf",
                            "source_url": "https://arxiv.org/abs/2501.01234",
                        }
                    ]
                },
            }
        ]
        candidates = DocumentQAService._extract_memory_arxiv_candidates(history)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["arxiv_id"], "2501.01234")
        self.assertEqual(candidates[0]["title"], "Memory Title")

    def test_build_tool_plan_skips_vdb_when_detail_and_memory_exists(self) -> None:
        service = DocumentQAService.__new__(DocumentQAService)
        plan = service._build_tool_plan(
            message="이 논문 원문 기준으로 자세히 설명해줘",
            memory_candidates=[
                {
                    "arxiv_id": "2501.01234",
                    "title": "T",
                    "pdf_url": "https://arxiv.org/pdf/2501.01234.pdf",
                    "source_url": "https://arxiv.org/abs/2501.01234",
                }
            ],
            has_message_arxiv_id=False,
        )
        self.assertFalse(plan.use_vdb)
        self.assertTrue(plan.use_pdf)


class QaStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_invoke_chat_llm_streams_tokens_for_ainvoke_fallback(self) -> None:
        class _FakeResponse:
            content = "alpha beta"

        class _FakeLlm:
            async def ainvoke(self, _messages):
                return _FakeResponse()

        service = DocumentQAService.__new__(DocumentQAService)
        service.chat_llm = _FakeLlm()
        emitted: list[str] = []

        answer, reasoning = await service._invoke_chat_llm(
            [("human", "hello")],
            token_callback=self._collect(emitted),
        )

        self.assertEqual(answer, "alpha beta")
        self.assertIsNone(reasoning)
        self.assertGreater(len(emitted), 1)
        self.assertEqual("".join(emitted), "alpha beta")

    async def test_invoke_chat_llm_splits_single_stream_chunk(self) -> None:
        class _Chunk:
            def __init__(self, text: str) -> None:
                self.content = text
                self.additional_kwargs = {}
                self.response_metadata = {}

        class _FakeLlm:
            async def astream(self, _messages):
                yield _Chunk("hello world")

            async def ainvoke(self, _messages):
                raise AssertionError("ainvoke should not be called when streaming succeeds")

        service = DocumentQAService.__new__(DocumentQAService)
        service.chat_llm = _FakeLlm()
        emitted: list[str] = []

        answer, reasoning = await service._invoke_chat_llm(
            [("human", "hello")],
            token_callback=self._collect(emitted),
        )

        self.assertEqual(answer, "hello world")
        self.assertIsNone(reasoning)
        self.assertGreater(len(emitted), 1)
        self.assertEqual("".join(emitted), "hello world")

    @staticmethod
    def _collect(bucket: list[str]):
        async def _callback(piece: str) -> None:
            bucket.append(piece)

        return _callback


if __name__ == "__main__":
    unittest.main()
