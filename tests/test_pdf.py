import sys
sys.stdout.reconfigure(encoding="utf-8")

from app.pdf_service import download_pdf, parse_pdf


pdf_url = "https://arxiv.org/pdf/2605.09649v1"
pdf_path = "data/papers/test_paper.pdf"

download_pdf(pdf_url, pdf_path)

pages = parse_pdf(pdf_path)

print("Pages extracted:", len(pages))
for page in pages[:2]:
    print("\nPAGE:", page["page_number"])
    print(page["text"][:1000])