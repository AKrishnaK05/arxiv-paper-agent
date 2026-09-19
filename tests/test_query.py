from app.query import understand_query


def test_query(user_input):
    state = {
        "user_input": user_input
    }

    result = understand_query(state)

    print(f"Input: {user_input}")
    print(f"Type: {result['query_type']}")
    print("-" * 40)


test_query("2401.12345")
test_query("https://arxiv.org/abs/2401.12345")
test_query("Recent research on RAG systems")