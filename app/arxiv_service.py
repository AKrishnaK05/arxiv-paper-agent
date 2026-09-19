import re
from typing import Any, Dict, List, Optional
import arxiv

client = arxiv.Client(
    page_size=10,
    delay_seconds=3.0,
    num_retries=3
)

def _clean_arxiv_id(arxiv_id: str) -> str:
    arxiv_id = arxiv_id.strip()
    match = re.search(r"(\d{4}\.\d{4,5}(v\d+)?)", arxiv_id)
    return match.group(1) if match else arxiv_id

def _format_paper(result: arxiv.Result) -> Dict[str, Any]:
    return {
        "arxiv_id": result.get_short_id(),
        "title": result.title.replace("\n", " ").strip(),
        "authors": [author.name for author in result.authors],
        "abstract": result.summary.replace("\n", " ").strip(),
        "published": result.published.isoformat(),
        "pdf_url": result.pdf_url,
        "categories": result.categories
    }

def search_papers(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    try:
        return [_format_paper(result) for result in client.results(search)]
    except Exception:
        return []

def get_paper(arxiv_id: str) -> Optional[Dict[str, Any]]:
    clean_id = _clean_arxiv_id(arxiv_id)
    search = arxiv.Search(id_list=[clean_id])

    try:
        result = next(client.results(search), None)
    except Exception:
        return None

    if not result:
        return None

    return _format_paper(result)