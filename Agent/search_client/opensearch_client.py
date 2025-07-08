from opensearchpy import OpenSearch

def get_opensearch_client(host: str, port: int):
    """
    Connect to OpenSearch instance running at the given host and port.
    """
    client = OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_compress=True,
        timeout=30,
        max_retries=3,
        retry_on_timeout=True,
    )

    if client.ping():
        print("Connected to OpenSearch!")
        info = client.info()
        print(f"Cluster: {info['cluster_name']} | Version: {info['version']['number']}")
    else:
        raise ConnectionError(" Failed to connect to OpenSearch.")

    return client


def create_index_if_not_exists(client, index_name: str):
    """
    Create index for PDF chunks + embeddings if it doesn't exist.
    """
    if client.indices.exists(index=index_name):
        print(f" Index '{index_name}' exists. Deleting and recreating for fresh start...")
        client.indices.delete(index=index_name)

    # Get embedding dimension dynamically from nomic-embed-text
    from Agent.vectors.embedding import get_embedding
    dummy_embedding = get_embedding("This is a sample chunk of a PDF.")
    dim = len(dummy_embedding)
    print(f"Embedding dimension detected: {dim}")

    # Create OpenSearch index with knn_vector mapping
    mapping = {
        "mappings": {
            "properties": {
                "pdf_name": {"type": "keyword"},
                "chunk": {"type": "text"},
                "embedding": {"type": "knn_vector", "dimension": dim},
                "source_page": {"type": "integer"},
                "token_count": {"type": "integer"}
            }
        },
        "settings": {
            "index": {
                "knn": True,
                "knn.space_type": "cosinesimil"
            }
        }
    }

    try:
        client.indices.create(index=index_name, body=mapping)
        print(f" Created index '{index_name}' with vector search mappings.")
    except Exception as e:
        print(f" Error creating index: {e}")
        raise


if __name__ == "__main__":
    host = "localhost"
    port = 9200

    client = get_opensearch_client(host, port)

    index_name = "pdf_chunks"
    create_index_if_not_exists(client, index_name)

    print("\n Available Indices:")
    for idx in client.cat.indices(format="json"):
        print(f"  - {idx['index']}")
