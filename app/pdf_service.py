import os
import re
from typing import Any, Dict, List
import pymupdf
import requests

def download_pdf(pdf_url: str, output_path: str) -> str:
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 (arxiv-paper-agent)"
    }
    response = requests.get(pdf_url, headers=headers, timeout=30)
    response.raise_for_status()

    if not response.content.startswith(b"%PDF"):
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        raise ValueError(f"Downloaded content from {pdf_url} is not a valid PDF file.")

    with open(output_path, "wb") as file:
        file.write(response.content)

    return output_path


def parse_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) == 0:
        return []

    document = pymupdf.open(pdf_path)
    total_pages = len(document)

    pages = []
    in_references = False

    for page_number, page in enumerate(document, start=1):
        text = page.get_text()

        lines = []

        for raw_line in text.splitlines():
            line = raw_line.replace("\r", "").strip()

            if not line:
                continue

            # Only trigger reference detection after page 2 or after first 30% of document
            # to prevent premature truncation on Table of Contents / outlines
            is_deep_enough = page_number > 2 or (total_pages > 1 and page_number > total_pages * 0.3)
            if is_deep_enough and re.match(r"^(\d+\.?\s*)?(references|bibliography)$", line, re.IGNORECASE):
                in_references = True
                continue

            # Resume parsing if an appendix or supplementary section starts
            if in_references and re.match(r"^(\d+\.?\s*|[A-Z]\.?\s*)?(appendix|supplementary|supplemental)", line, re.IGNORECASE):
                in_references = False

            if in_references:
                continue

            if re.fullmatch(r"\d+", line):
                continue

            lines.append(line)

        cleaned_text = "\n".join(lines)

        if cleaned_text:
            pages.append({
                "page_number": page_number,
                "page": page_number,
                "text": cleaned_text
            })

    document.close()

    return pages