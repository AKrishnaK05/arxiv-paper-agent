import os
from typing import Any, Dict, Optional, Union
from app.query import understand_query
from app.retrieval import retrieve_papers
from app.pdf_service import download_pdf, parse_pdf
from app.chunking import chunk_pages
from app.vector_store import VectorStore
from app.rag_service import RAGService
from app.output_formatter import format_json, format_markdown
from app.briefing_service import BriefingService

class ResearchPipeline:

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        rag: Optional[RAGService] = None,
        briefing_service: Optional[BriefingService] = None
    ):
        self.vector_store = vector_store or VectorStore()
        self.rag = rag or RAGService(vector_store=self.vector_store)
        self.briefing_service = briefing_service or BriefingService(vector_store=self.vector_store)
        self.briefing = self.briefing_service
        self.current_paper: Optional[Dict[str, Any]] = None
        self.current_paper_id: Optional[str] = None

    def reset(self) -> None:
        self.current_paper = None
        self.current_paper_id = None

    def run(
        self,
        user_input: str,
        paper_id: Optional[str] = None,
        intent: Optional[str] = None,
        output_format: str = "markdown"
    ) -> Union[str, Dict[str, Any]]:

        # Step 1: Validate input
        if not user_input or not user_input.strip():
            return {
                "error": "User input cannot be empty."
            }

        if output_format not in {"markdown", "json"}:
            return {
                "error": f"Unsupported output format: {output_format}. "
                         "Use 'markdown' or 'json'."
            }

        # 1. Understand query (always classify intent and detect query type)
        state = {
            "user_input": user_input
        }
        state = understand_query(state)

        # Allow caller to explicitly specify intent ("briefing" or "qa")
        if intent in {"briefing", "qa"}:
            state["intent"] = intent

        # If the input is a malformed ID and no explicit paper_id was provided, fail fast
        if state.get("query_type") == "invalid_paper_id" and not paper_id:
            return {
                "error": state.get(
                    "error",
                    f"Invalid arXiv paper ID: '{user_input}'."
                )
            }

        # If paper_id is explicitly passed as a parameter, bind it while preserving classified intent
        if paper_id:
            state["paper_id"] = paper_id
            state["query_type"] = "paper_id"
            if self.current_paper and self.current_paper.get("arxiv_id") == paper_id:
                state["selected_paper"] = self.current_paper
                state["papers"] = [self.current_paper]
        else:
            # If query does not contain an explicit arXiv ID, but an active paper session exists
            # and the query is a question, anchor the follow-up to the active paper
            if state.get("query_type") != "paper_id" and self.current_paper_id and state.get("intent") == "qa":
                state["paper_id"] = self.current_paper_id
                state["query_type"] = "paper_id"
                if self.current_paper:
                    state["selected_paper"] = self.current_paper
                    state["papers"] = [self.current_paper]

        # 2. Retrieve paper(s) if not already resolved from active session
        if not state.get("papers"):
            state = retrieve_papers(state)

        if not state.get("papers"):
            if state.get("query_type") == "paper_id":
                return {
                    "error": f"Could not find arXiv paper: "
                             f"{state.get('paper_id')}"
                }

            return {
                "error": "No relevant papers were found for the query."
            }

        # 3. Select paper
        paper = state.get("selected_paper")

        if not paper:
            paper = state["papers"][0]

        paper_id = paper["arxiv_id"]
        self.current_paper = paper
        self.current_paper_id = paper_id

        # 4. Download PDF (if not already downloaded)
        pdf_path = f"data/papers/{paper_id.replace('/', '_')}.pdf"

        if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) == 0:
            try:
                download_pdf(
                    paper["pdf_url"],
                    pdf_path
                )
            except Exception as e:
                if os.path.exists(pdf_path) and os.path.getsize(pdf_path) == 0:
                    try:
                        os.remove(pdf_path)
                    except OSError:
                        pass
                return {
                    "error": f"Failed to download the paper PDF: {str(e)}"
                }

        # 5-7. Parse, chunk and index if not already in vector store
        if not self.vector_store.has_paper(paper_id):
            try:
                pages = parse_pdf(pdf_path)

                if not pages:
                    return {
                        "error": "The PDF was downloaded but no text "
                                 "could be extracted."
                    }

                chunks = chunk_pages(
                    pages,
                    paper_id=paper_id
                )

                if not chunks:
                    return {
                        "error": "No usable chunks could be created "
                                 "from the paper."
                    }

                self.vector_store.add_chunks(chunks)

            except Exception as e:
                return {
                    "error": f"Failed to process the paper: {str(e)}"
                }

        # 8. Route request based on intent
        if state.get("intent") == "briefing":
            try:
                result = self.briefing_service.generate(
                    paper_id=paper_id,
                    paper_metadata=paper
                )
            except Exception as e:
                return {
                    "error": f"Failed to generate the briefing: {str(e)}"
                }

            if output_format == "json":
                return format_json(
                    result["briefing"],
                    result["sources"]
                )

            elif output_format == "markdown":
                return format_markdown(
                    result["briefing"],
                    result["sources"]
                )

            return result

        else:
            try:
                result = self.rag.answer(
                    question=user_input,
                    paper_id=paper_id,
                    top_k=5
                )
            except Exception as e:
                return {
                    "error": f"Failed to generate the answer: {str(e)}"
                }

            if output_format == "json":
                return format_json(
                    result["answer"],
                    result["sources"]
                )

            elif output_format == "markdown":
                return format_markdown(
                    result["answer"],
                    result["sources"]
                )

            return result