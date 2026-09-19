from app.pdf_service import parse_pdf
from app.chunking import chunk_pages


pdf_path = "data/papers/test_paper.pdf"

pages = parse_pdf(pdf_path)

chunks = chunk_pages(
    pages,
    paper_id="test-paper"
)

print("Number of chunks:", len(chunks))

for chunk in chunks[:3]:
    print("\n" + "=" * 60)
    print("Chunk ID:", chunk["chunk_id"])
    print("Pages:", chunk["pages"])
    print("Words:", len(chunk["text"].split()))
    print("Text:")
    print(chunk["text"][:500])