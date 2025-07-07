import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pdf_documents(folder: str):
    """
    Load and split PDF files from the given folder using PyPDFLoader and LangChain.

    Args:
        pdf_folder (str): Path to the folder containing PDF files.

    Returns:
        list[Document]: A list of chunked document objects.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = []
    for f in os.listdir(folder):
        if f.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder, f))
            docs.extend(splitter.split_documents(loader.load()))
    return docs
