from typing import TypedDict, Optional, List, Dict, Any

class AgentState(TypedDict, total=False):
    user_input: str
    query_type: str
    intent: Optional[str]
    paper_id: Optional[str]
    papers: List[Dict[str, Any]]
    selected_paper: Optional[Dict[str, Any]]
    pdf_path: Optional[str]
    pages: Optional[List[Dict[str, Any]]]
    chunks: List[Dict[str, Any]]
    briefing: Optional[str]
    output_format: Optional[str]
    error: Optional[str]
    conversation_history: List[Dict[str, str]]
