import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def load_pdf_documents(pdf_folder: str) -> list[Document]:
    """
    Load and split PDF files from the given folder using PyPDFLoader and LangChain.

    Args:
        pdf_folder (str): Path to the folder containing PDF files.

    Returns:
        list[Document]: A list of chunked document objects.
    """
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    if not os.path.exists(pdf_folder):
        raise FileNotFoundError(f"Folder '{pdf_folder}' not found.")

    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

    if not pdf_files:
        raise ValueError(f"No PDF files found in '{pdf_folder}'.")

    for filename in pdf_files:
        file_path = os.path.join(pdf_folder, filename)
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            chunks = splitter.split_documents(documents)
            all_chunks.extend(chunks)
            print(f" Loaded and split '{filename}' into {len(chunks)} chunks.")
        except Exception as e:
            print(f" Failed to process '{filename}': {e}")

    return all_chunks
