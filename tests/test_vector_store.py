import sys
sys.stdout.reconfigure(encoding="utf-8")

from app.pdf_service import parse_pdf
from app.chunking import chunk_pages
from app.vector_store import VectorStore


pdf_path = "data/papers/test_paper.pdf"

pages = parse_pdf(pdf_path)

chunks = chunk_pages(
    pages,
    paper_id="test-paper"
)

store = VectorStore()

store.add_chunks(chunks)

results = store.search(
    "What is the main method proposed in this paper?",
    top_k=3
)

print("Retrieved chunks:")

for document in results["documents"][0]:
    print("\n" + "=" * 60)
    print(document[:1000])