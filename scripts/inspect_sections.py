from app.pdf_service import parse_pdf

pdf_path = "data/papers/test_paper.pdf"

pages = parse_pdf(pdf_path)

for page in pages:
    print("\n")
    print("=" * 80)
    print(f"PAGE {page['page_number']}")
    print("=" * 80)

    for line in page["text"].splitlines():
        line = line.strip()

        if line:
            print(line)