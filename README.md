# Patent Innovation Predictor — Local RAG Agent
This project is a local Research Agent for analyzing patents stored as PDFs, using:

LangChain, OpenSearch, Ollama, and CrewAI

Your own local embeddings

No paid API keys — runs fully on your machine

# ⚙️ How It Works
1️⃣ You load PDF patents

Store your PDF files in a folder (e.g. ./data/patent_pdfs)

Use ingestion.py to:

Load each PDF

Split into text chunks

Generate embeddings using nomic-embed-text (or your Ollama embedding model)

Store all chunks + embeddings in OpenSearch

2️⃣ You query the PDFs

Use search_tools.py to run:

🔍 Keyword Search — match plain text

🧲 Semantic Search — vector similarity

🧩 Hybrid Search — both together

🔁 Iterative Search — refine and explore

3️⃣ You run an Agent

Uses CrewAI with:

Multiple agents (analyst, reviewer, trend summarizer)

Uses your Ollama model for LLM tasks (llama3, mistral, deepseek-llm, etc.)

Plans research, finds chunks, summarizes insights

4️⃣ Optional Frontend

You can build a simple Streamlit or Flask app to:

Upload new PDFs

Run searches

Show results in your browser

📦 Project Structure
bash
Copy
Edit
