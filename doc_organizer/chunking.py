from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    clean = " ".join(text.split())
    if not clean:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    text_len = len(clean)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(clean[start:end])
        if end == text_len:
            break
        start = end - overlap
    return chunks

