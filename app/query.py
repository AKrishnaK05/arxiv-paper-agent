import re
from app.state import AgentState


def understand_query(state: AgentState):
    user_input = state["user_input"].strip()
    lowered = user_input.lower()

    # Strip common conversational briefing wrappers and arxiv prefixes
    target = re.sub(
        r"^(give me an? (executive )?briefing (for|of)?|summarize|summary of)\s+",
        "",
        user_input,
        flags=re.IGNORECASE
    ).strip()
    target = re.sub(r"^(arxiv:\s*|arXiv:\s*)", "", target, flags=re.IGNORECASE).strip()

    # Valid arXiv IDs: YYMM.NNNN or YYMM.NNNNN (month must be 01-12)
    arxiv_id_pattern = r"^\d{2}(0[1-9]|1[0-2])\.\d{4,5}(v\d+)?$"
    arxiv_url_pattern = r"arxiv\.org/(abs|pdf)/(\d{2}(0[1-9]|1[0-2])\.\d{4,5}(v\d+)?)"
    embedded_id_pattern = r"\b(\d{2}(0[1-9]|1[0-2])\.\d{4,5}(v\d+)?)\b"

    # Malformed patterns: look like an arXiv identifier or URL but fail validation
    malformed_id_pattern = r"^(\d{1,6}\.[0-9a-zA-Z]+|\d{4}\.\d{4,5}(v\d+)?)$"
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

    # Intent detection: differentiate full executive briefings from specific QA
    explicit_briefing_phrases = [
        "executive briefing",
        "full briefing",
        "entire paper briefing",
        "generate briefing",
        "create briefing",
    ]

    general_briefing_words = [
        "briefing",
        "digest",
        "give me a summary",
        "summarize",
        "summarise",
        "summary",
        "overview",
    ]

    qa_indicators = [
        "what", "how", "why", "who", "where", "when", "which", "is", "are",
        "can", "could", "does", "do", "explain", "tell me", "compare", "?"
    ]

    is_explicit_briefing = any(p in lowered for p in explicit_briefing_phrases)
    has_qa_indicator = any(w in lowered for w in qa_indicators) or lowered.endswith("?")
    has_general_briefing = any(w in lowered for w in general_briefing_words)

    if is_explicit_briefing:
        state["intent"] = "briefing"
    elif has_qa_indicator:
        state["intent"] = "qa"
    elif has_general_briefing:
        state["intent"] = "briefing"
    elif query_type in {"paper_id", "topic"}:
        state["intent"] = "briefing"
    else:
        state["intent"] = "qa"

    return state