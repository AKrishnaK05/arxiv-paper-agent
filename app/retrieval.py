from app.state import AgentState
from app.arxiv_service import search_papers, get_paper


def retrieve_papers(state: AgentState) -> AgentState:
    query_type = state.get("query_type", "topic")

    if query_type == "topic":
        papers = search_papers(
            state["user_input"],
            max_results=5
        )
        state["selected_paper"] = None

    elif query_type == "paper_id":
        paper_id = state.get("paper_id") or state.get("user_input")
        paper = get_paper(paper_id)

        papers = [paper] if paper else []
        state["selected_paper"] = paper

    else:
        papers = []
        state["selected_paper"] = None

    state["papers"] = papers

    return state