import os
import requests
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    """
    Generate embeddings for input text using a local Ollama model.

    Args:
        text (str): The text input to be embedded.
        model (str): The name of the Ollama model to use (default: nomic-embed-text).

    Returns:
        list: Embedding vector (list of floats).
    """
    url = "http://localhost:11434/api/embeddings"
    payload = {
        "model": model,
        "prompt": text
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        return response.json()["embedding"]
    else:
        raise Exception(f"Failed to get embedding: {response.status_code}, {response.text}")


if __name__ == "__main__":
    # Test it with a sample string
    sample_text = "The transformer model revolutionized NLP."
    try:
        embedding = get_embedding(sample_text)
        print(" Embedding dimension:", len(embedding))
        print(" First 5 values:", embedding[:5])
    except Exception as e:
        print(" Error:", e)
