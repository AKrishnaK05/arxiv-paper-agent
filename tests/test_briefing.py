from app.briefing_service import BriefingService
from app.vector_store import VectorStore


vector_store = VectorStore()

briefing_service = BriefingService(
    vector_store=vector_store
)

result = briefing_service.generate(
    paper_id="2109.05633v1"
)

print("\n" + "=" * 60)
print("EXECUTIVE BRIEFING")
print("=" * 60)

print(result["briefing"])

print("\n" + "=" * 60)
print("SOURCES")
print("=" * 60)

for source in result["sources"]:
    print(source)