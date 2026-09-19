from app.arxiv_service import search_papers


papers = search_papers("KV cache compression LLM", max_results=3)

for paper in papers:
    print("=" * 60)
    print("ID:", paper["arxiv_id"])
    print("TITLE:", paper["title"])
    print("AUTHORS:", paper["authors"])
    print("PUBLISHED:", paper["published"])
    print("PDF:", paper["pdf_url"])