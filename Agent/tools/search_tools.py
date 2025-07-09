from Agent.vectors.embedding import get_embedding
from Agent.search_client.opensearch_client import get_opensearch_client

INDEX_NAME = "patent_chunks"
HOST = "localhost"
PORT = 9200


def keyword_search(query_text, top_k=20):
    """
    Perform keyword search using OpenSearch.
    """
    client = get_opensearch_client(HOST, PORT)

    try:
        search_query = {
            "size": top_k,
            "query": {"match": {"text": query_text}},
            "_source": ["source_file", "text", "chunk_index"],
        }

        response = client.search(index=INDEX_NAME, body=search_query)
        return response["hits"]["hits"]
    except Exception as e:
        print(f"Keyword search error: {e}")
        return []


def semantic_search(query_text, top_k=20):
    """
    Perform semantic (vector) search using embeddings.
    """
    client = get_opensearch_client(HOST, PORT)

    try:
        query_embedding = get_embedding(query_text)

        search_query = {
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": top_k,
                    }
                }
            },
            "_source": ["source_file", "text", "chunk_index"],
        }

        response = client.search(index=INDEX_NAME, body=search_query)
        return response["hits"]["hits"]
    except Exception as e:
        print(f"Semantic search error: {e}")
        return []


def hybrid_search(query_text, top_k=20):
    """
    Perform hybrid search: semantic + keyword.
    """
    client = get_opensearch_client(HOST, PORT)

    try:
        query_embedding = get_embedding(query_text)

        search_query = {
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_embedding,
                                    "k": top_k
                                }
                            }
                        },
                        {"match": {"text": query_text}},
                    ]
                }
            },
            "_source": ["source_file", "text", "chunk_index"],
        }

        response = client.search(index=INDEX_NAME, body=search_query)
        return response["hits"]["hits"]
    except Exception as e:
        print(f"Hybrid search error: {e}")
        return keyword_search(query_text, top_k)


def iterative_search(query_text, refinement_steps=3, top_k=20):
    """
    Perform iterative keyword search with query refinement.
    """
    client = get_opensearch_client(HOST, PORT)
    all_results = []
    current_query = query_text

    for step in range(refinement_steps):
        try:
            search_query = {
                "size": top_k,
                "query": {"match": {"text": current_query}},
                "_source": ["source_file", "text", "chunk_index"],
            }

            response = client.search(index=INDEX_NAME, body=search_query)
            results = response["hits"]["hits"]

            for result in results:
                if result not in all_results:
                    all_results.append(result)

            if not results:
                break

            top_text = results[0]["_source"]["text"]
            current_query += " " + top_text.split(".")[0]

        except Exception as e:
            print(f"Iterative search error at step {step}: {e}")
            break

    return all_results


if __name__ == "__main__":
    query = input("Enter your search query: ")

    print("\n🔍 Hybrid Search Results:")
    hybrid_results = hybrid_search(query)

    if not hybrid_results:
        print("❌ No results found.")
    else:
        for res in hybrid_results:
            print(f"\n📄 Source File: {res['_source']['source_file']}")
            print(f"🔢 Chunk Index: {res['_source']['chunk_index']}")
            print(f"📝 Text:\n{res['_source']['text']}\n")
