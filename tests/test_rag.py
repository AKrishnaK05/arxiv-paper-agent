from app.rag_service import RAGService
from app.output_formatter import (
    format_json,
    format_markdown
)


rag = RAGService()

result = rag.answer(
    question="What method does the paper propose?",
    paper_id="test-paper",
    top_k=5
)

print("\nMARKDOWN\n")
print(
    format_markdown(
        result["answer"],
        result["sources"]
    )
)

print("\nJSON\n")
print(
    format_json(
        result["answer"],
        result["sources"]
    )
)