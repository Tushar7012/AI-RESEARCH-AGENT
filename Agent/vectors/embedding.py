import requests

def get_embedding(text: str) -> list[float]:
    """
    Get embedding from local Ollama model.
    Model: nomic-embed-text:latest
    """
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": text
        }
    )
    response.raise_for_status()
    return response.json()["embedding"]
