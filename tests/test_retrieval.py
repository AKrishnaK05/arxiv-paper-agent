from app.query import understand_query
from app.retrieval import retrieve_papers


state = {
    "user_input": "KV cache compression LLM"
}

state = understand_query(state)
state = retrieve_papers(state)

print("Query type:", state["query_type"])
print("Number of papers:", len(state["papers"]))

for paper in state["papers"]:
    print("\n", paper["title"])