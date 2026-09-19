import re
from app.state import AgentState


def understand_query(state: AgentState):
    user_input = state["user_input"].strip()
    lowered = user_input.lower()

    # Strip common conversational briefing wrappers to isolate target if present
    target = re.sub(
        r"^(give me an? (executive )?briefing (for|of)?|summarize|summary of)\s+",
        "",
        user_input,
        flags=re.IGNORECASE
    ).strip()

    arxiv_id_pattern = r"^\d{4}\.\d{4,5}(v\d+)?$"
    arxiv_url_pattern = r"arxiv\.org/(abs|pdf)/(\d{4}\.\d{4,5}(v\d+)?)"
    embedded_id_pattern = r"\b(\d{4}\.\d{4,5}(v\d+)?)\b"

    # Malformed patterns: look like an arXiv identifier or URL but fail validation
    malformed_id_pattern = r"^\d{1,6}\.[0-9a-zA-Z]+$"
    malformed_url_pattern = r"arxiv\.org/(abs|pdf)/(\S+)"

    url_match = re.search(arxiv_url_pattern, user_input)
    embedded_id_match = re.search(embedded_id_pattern, user_input)

    # Paper identification
    if re.match(arxiv_id_pattern, target):
        query_type = "paper_id"
        state["paper_id"] = target

    elif url_match:
        query_type = "paper_id"
        state["paper_id"] = url_match.group(2)

    elif embedded_id_match:
        query_type = "paper_id"
        state["paper_id"] = embedded_id_match.group(1)

    elif re.match(malformed_id_pattern, target):
        query_type = "invalid_paper_id"
        state["paper_id"] = None
        state["error"] = (
            f"Invalid arXiv paper ID: '{target}'. "
            "Expected format like '2109.05633' or '2109.05633v1'."
        )

    elif re.search(malformed_url_pattern, user_input):
        query_type = "invalid_paper_id"
        state["paper_id"] = None
        state["error"] = (
            f"Invalid arXiv URL: '{user_input}'. "
            "Expected format like 'https://arxiv.org/abs/2109.05633'."
        )

    else:
        query_type = "topic"
        state["paper_id"] = None

    state["query_type"] = query_type

    # Intent detection

    briefing_keywords = [
        "executive briefing",
        "briefing",
        "give me a summary",
        "summarize this paper",
        "summarise this paper",
        "summarize the paper",
        "summarise the paper",
        "overview of this paper",
        "overview of the paper",
        "research briefing",
        "research summary",
    ]

    if any(keyword in lowered for keyword in briefing_keywords):
        state["intent"] = "briefing"
    else:
        state["intent"] = "qa"

    return state