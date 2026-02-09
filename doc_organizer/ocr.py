from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from glmocr import GlmOcr

SUPPORTED_EXTENSIONS = {".pdf"}


@dataclass(slots=True)
class ParsedDocument:
    source_path: Path
    text: str


def iter_pdf_files(docs_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in docs_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _extract_text(ocr_json: Any, markdown_result: str | None) -> str:
    if markdown_result and markdown_result.strip():
        return markdown_result.strip()

    texts: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            content = value.get("content")
            if isinstance(content, str) and content.strip():
                texts.append(content.strip())
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(ocr_json)
    return "\n".join(texts).strip()


def parse_pdf(parser: GlmOcr, pdf_path: Path, results_dir: Path) -> ParsedDocument:
    result = parser.parse(str(pdf_path))
    result.save(output_dir=results_dir)
    text = _extract_text(result.json_result, result.markdown_result)
    return ParsedDocument(source_path=pdf_path, text=text)

