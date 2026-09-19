from typing import Any, Dict, List, Optional
from app.vector_store import VectorStore
from app.llm_service import LLMService
from app.output_formatter import clean_sources


class BriefingService:

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        llm: Optional[LLMService] = None
    ):
        self.vector_store = vector_store or VectorStore()
        self.llm = llm or LLMService()

    def generate(
        self,
        paper_id: str,
        top_k: int = 10,
        paper_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        queries = [
            "What problem does this paper address?",
            "What method or approach does the paper propose?",
            "What are the main experimental results?",
            "What are the limitations of the approach?",
            "What future work is discussed?"
        ]

        retrieved_documents = []
        retrieved_metadata = []

        for query in queries:
            results = self.vector_store.search(
                query=query,
                paper_id=paper_id,
                top_k=top_k
            )

            documents = (results.get("documents") or [[]])[0]
            metadata = (results.get("metadatas") or [[]])[0]

            retrieved_documents.extend(documents)
            retrieved_metadata.extend(metadata)

        # Remove duplicate chunks
        unique_chunks = []
        seen = set()

        for document, metadata in zip(
            retrieved_documents,
            retrieved_metadata
        ):
            chunk_id = metadata.get("chunk_id")

            if chunk_id in seen:
                continue

            seen.add(chunk_id)

            unique_chunks.append(
                {
                    "text": document,
                    "metadata": metadata
                }
            )

        context_parts = []

        for chunk in unique_chunks:
            metadata = chunk["metadata"]

            context_parts.append(
                f"""
Paper ID: {metadata.get("paper_id")}
Pages: {metadata.get("pages")}

Content:
{chunk["text"]}
"""
            )

        context = "\n\n".join(context_parts)

        # Build paper metadata header if available
        meta_context = ""
        meta_header = f"# Executive Briefing: {paper_id}\n"
        if paper_metadata:
            title = paper_metadata.get("title") or paper_id
            authors_list = paper_metadata.get("authors") or []
            authors = ", ".join(authors_list) if isinstance(authors_list, list) else str(authors_list)
            pub_date = paper_metadata.get("published") or "N/A"
            link = paper_metadata.get("pdf_url") or f"https://arxiv.org/abs/{paper_id}"

            meta_header = (
                f"# Executive Briefing: {title}\n\n"
                f"**Authors:** {authors} | **arXiv ID:** `{paper_id}` | **Published:** {pub_date} | **Link:** {link}\n"
            )
            meta_context = (
                f"Known Paper Metadata:\n"
                f"- Title: {title}\n"
                f"- Authors: {authors}\n"
                f"- arXiv ID: {paper_id}\n"
                f"- Published Date: {pub_date}\n"
                f"- Link: {link}\n\n"
            )

        prompt = f"""
You are an academic research assistant.

Create a structured executive briefing for the research paper using ONLY the supplied paper context and metadata.

Do not invent information. If any section cannot be supported by the available context, state that clearly.

{meta_context}Produce the briefing using this exact Markdown structure:

{meta_header}
## Why This Paper Matters
A concise 1-paragraph plain-English summary explaining the significance of the paper and why a researcher should care.

## Problem Statement
Explain the core problem or research challenge addressed by the paper.

## Method & Approach
Detail the proposed method, architecture, or approach using concise bullet points.

## Key Results & Claims
Summarize the main experimental results, findings, and quantitative/qualitative claims.

## Limitations
Explicitly describe limitations stated or clearly supported by the paper context. Do not skip this section.

## Future Work
Describe future work or research directions mentioned in the paper.

## Suggested Follow-up Questions
List 3-4 insightful questions a reader or researcher might ask to probe deeper into this work.

Paper context:

{context}
"""

        answer = self.llm.generate(prompt)

        raw_sources = [
            {
                "paper_id": (chunk.get("metadata") or {}).get("paper_id"),
                "pages": (chunk.get("metadata") or {}).get("pages")
            }
            for chunk in unique_chunks
        ]

        sources = clean_sources(raw_sources)

        return {
            "briefing": answer,
            "sources": sources
        }