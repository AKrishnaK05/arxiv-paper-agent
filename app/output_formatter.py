import json
import re
from typing import Any, Dict, List, Optional, Union


def parse_pages(pages_val: Any) -> List[int]:
    """Extract a clean, sorted list of integer page numbers from varied representations."""
    if isinstance(pages_val, list):
        parsed = []
        for item in pages_val:
            try:
                parsed.append(int(item))
            except (ValueError, TypeError):
                pass
        return sorted(list(set(parsed)))

    if isinstance(pages_val, int):
        return [pages_val]

    if isinstance(pages_val, str):
        numbers = re.findall(r"\d+", pages_val)
        return sorted(list(set(int(n) for n in numbers)))

    return []


def clean_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate and normalize sources with integer page lists."""
    cleaned = []
    seen = set()

    for item in sources or []:
        if not isinstance(item, dict):
            continue

        paper_id = str(item.get("paper_id") or "").strip()
        pages = parse_pages(item.get("pages"))

        key = (paper_id, tuple(pages))
        if key not in seen:
            seen.add(key)
            cleaned.append({
                "paper_id": paper_id,
                "pages": pages
            })

    return cleaned


def parse_markdown_sections(text: str) -> Dict[str, str]:
    """Parse Markdown headings (## Section) into a structured dictionary."""
    sections: Dict[str, str] = {}
    current_section: Optional[str] = None
    current_lines: List[str] = []

    for line in text.splitlines():
        heading_match = re.match(r"^##\s+(.+)$", line)
        if heading_match:
            if current_section is not None:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = heading_match.group(1).strip()
            current_lines = []
        elif current_section is not None:
            current_lines.append(line)

    if current_section is not None and current_lines:
        sections[current_section] = "\n".join(current_lines).strip()

    return sections


def extract_markdown_title(text: str) -> Optional[str]:
    """Extract top-level H1 title from Markdown text if present."""
    for line in text.splitlines():
        title_match = re.match(r"^#\s+(.+)$", line)
        if title_match:
            return title_match.group(1).strip()
    return None


def extract_metadata_line(text: str) -> Optional[Dict[str, str]]:
    """Extract authors, arxiv_id, published date, link from header line if present."""
    match = re.search(
        r"\*\*Authors:\*\*\s*(.*?)\s*\|\s*\*\*arXiv ID:\*\*\s*`?(.*?)`?\s*\|\s*\*\*Published:\*\*\s*(.*?)\s*\|\s*\*\*Link:\*\*\s*(\S+)",
        text
    )
    if match:
        return {
            "authors": match.group(1).strip(),
            "arxiv_id": match.group(2).strip(),
            "published": match.group(3).strip(),
            "link": match.group(4).strip()
        }
    return None


def format_json(
    content: Union[str, Dict[str, Any]],
    sources: List[Dict[str, Any]]
) -> str:
    """Format answer or briefing and sources into a structured JSON string."""
    normalized_sources = clean_sources(sources)

    if isinstance(content, dict):
        result = dict(content)
        result["sources"] = normalized_sources
        return json.dumps(result, indent=2, ensure_ascii=False)

    text = str(content).strip()
    sections = parse_markdown_sections(text)

    # If the text has multiple ## sections (e.g. Executive Briefing), structure it
    if len(sections) >= 2:
        title = extract_markdown_title(text) or "Research Summary"
        metadata = extract_metadata_line(text)
        sections.pop("Sources", None)
        result = {
            "title": title,
            "sections": sections,
            "sources": normalized_sources
        }
        if metadata:
            result["metadata"] = metadata
    else:
        result = {
            "answer": text,
            "sources": normalized_sources
        }

    return json.dumps(result, indent=2, ensure_ascii=False)


def format_markdown(content: str, sources: List[Dict[str, Any]]) -> str:
    """Format answer or briefing with clean, human-readable citations."""
    normalized_sources = clean_sources(sources)
    text = str(content).strip()

    # Strip any model-hallucinated Sources heading to ensure only verified citations are displayed
    clean_text = re.sub(r"\n##\s*Sources.*$", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    lines = []
    if not clean_text.startswith("#"):
        lines.append(f"# Answer\n\n{clean_text}\n")
    else:
        lines.append(f"{clean_text}\n")

    if normalized_sources:
        lines.append("## Sources\n")
        for source in normalized_sources:
            paper_id = source["paper_id"]
            pages = source["pages"]

            if len(pages) == 1:
                pages_str = f"Page: {pages[0]}"
            elif len(pages) > 1:
                pages_str = f"Pages: {', '.join(str(p) for p in pages)}"
            else:
                pages_str = "Pages: N/A"

            lines.append(f"- Paper: `{paper_id}` | {pages_str}")

    return "\n".join(lines).strip() + "\n"