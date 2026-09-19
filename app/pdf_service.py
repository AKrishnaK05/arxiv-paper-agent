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

    with open(output_path, "wb") as file:
        file.write(response.content)

    return output_path


def parse_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    document = pymupdf.open(pdf_path)

    pages = []
    references_started = False

    for page_number, page in enumerate(document, start=1):
        text = page.get_text()

        lines = []

        for line in text.splitlines():
            line = line.strip()

            if not line:
                continue

            if re.match(r"^(references|bibliography)$", line, re.IGNORECASE):
                references_started = True
                break

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

        if references_started:
            break

    document.close()

    return pages