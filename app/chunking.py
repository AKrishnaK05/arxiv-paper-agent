from typing import Any, Dict, List


def chunk_pages(
    pages: List[Dict[str, Any]],
    paper_id: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[Dict[str, Any]]:

    words: List[tuple] = []

    for page in pages:
        page_num = page.get("page_number") or page.get("page", 1)
        page_words = page["text"].split()

        for word in page_words:
            words.append((word, page_num))

    chunks: List[Dict[str, Any]] = []
    start = 0
    chunk_id = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]

        text = " ".join(item[0] for item in chunk_words)
        pages_in_chunk = sorted(set(item[1] for item in chunk_words))

        chunks.append({
            "chunk_id": chunk_id,
            "paper_id": paper_id,
            "text": text,
            "pages": pages_in_chunk,
            "start_page": pages_in_chunk[0] if pages_in_chunk else 1,
            "end_page": pages_in_chunk[-1] if pages_in_chunk else 1,
        })

        chunk_id += 1

        if end == len(words):
            break

        start = end - overlap

    return chunks