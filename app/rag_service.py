from typing import Any, Dict, List, Optional
from app.vector_store import VectorStore
from app.llm_service import LLMService
from app.output_formatter import clean_sources


class RAGService:

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        llm: Optional[LLMService] = None
    ):
        self.vector_store = vector_store or VectorStore()
        self.llm = llm or LLMService()

    def answer(
        self,
        question: str,
        paper_id: str,
        top_k: int = 5,
        output_format: str = "markdown"
    ) -> Dict[str, Any]:

        results = self.vector_store.search(
            query=question,
            paper_id=paper_id,
            top_k=top_k
        )

        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]

        if not documents:
            return {
                "answer": "I could not find relevant information in the paper.",
                "sources": []
            }

        context_parts = []

        for document, metadata in zip(
            documents,
            metadatas
        ):
            meta = metadata or {}
            context_parts.append(
                f"""
Source:
Paper ID: {meta.get("paper_id")}
Pages: {meta.get("pages")}

Content:
{document}
"""
            )

        context = "\n\n".join(context_parts)

        prompt = f"""
You are a research paper question-answering assistant.

Answer the user's question using ONLY the provided
paper context.

If the context does not contain enough information to
answer the question, say so clearly.

Do not invent facts.

User question:
{question}

Paper context:
{context}

Provide a concise, factual answer.
"""

        answer = self.llm.generate(prompt)

        raw_sources = [
            {
                "paper_id": (meta or {}).get("paper_id"),
                "pages": (meta or {}).get("pages")
            }
            for meta in metadatas
            if meta
        ]

        sources = clean_sources(raw_sources)

        return {
            "answer": answer,
            "sources": sources
        }