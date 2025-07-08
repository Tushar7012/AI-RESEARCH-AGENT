import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from Agent.vectors.embedding import get_embedding
from Agent.search_client.opensearch_client import create_index_if_not_exists, get_opensearch_client


def load_chunks_from_pdfs(pdf_dir):
    """
    Loads and chunks patent PDFs, then generates embeddings.

    Args:
        pdf_dir (str): Path to directory containing PDF files.

    Returns:
        list: A list of chunk dictionaries with text and embedding.
    """
    if not os.path.exists(pdf_dir):
        raise FileNotFoundError(f"'{pdf_dir}' does not exist.")

    chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_dir, filename)
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            split_docs = text_splitter.split_documents(docs)

            for idx, chunk in enumerate(split_docs):
                text = chunk.page_content.strip()
                if not text:
                    continue

                try:
                    embedding = get_embedding(text)
                except Exception as e:
                    print(f" Skipping chunk due to embedding error: {e}")
                    continue

                chunks.append({
                    "source_file": filename,
                    "chunk_index": idx,
                    "text": text,
                    "embedding": embedding
                })

    return chunks


def index_chunks(client, index_name, chunks):
    """
    Index the PDF chunks into OpenSearch.

    Args:
        client: OpenSearch client.
        index_name (str): Name of the OpenSearch index.
        chunks (list): List of chunk dictionaries.
    """
    for chunk in chunks:
        client.index(index=index_name, body=chunk)
    print(f" Indexed {len(chunks)} chunks into '{index_name}' index.")


if __name__ == "__main__":
    pdf_dir = "D:/LangGraph/GenAI-2/AI-RESEARCH-AGENT/data/patent_pdfs"  
    index_name = "patent_chunks"
    host = "localhost"
    port = 9200

    try:
        client = get_opensearch_client(host, port)
        create_index_if_not_exists(client, index_name)

        chunks = load_chunks_from_pdfs(pdf_dir)
        print(f" Loaded and embedded {len(chunks)} chunks from PDFs in '{pdf_dir}'")

        index_chunks(client, index_name, chunks)
    except Exception as e:
        print(f" Error: {e}")
