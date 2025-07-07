from Agent.data_ingestion.ingestion import load_pdf_documents

pdf_folder = "D:/LangGraph/GenAI-2/AI-RESEARCH-AGENT/data/patent_pdfs"
chunks = load_pdf_documents(pdf_folder)

print(f"Loaded {len(chunks)} chunks from 2 PDFs.")