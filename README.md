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

# 📦 Project Structure
    .
    ├── data/patent_pdfs/          # Your PDF files
    ├── Agent/
    │   ├── ingestion.py           # Loads + splits + embeds
    │   ├── vectors/embedding.py   # Embedding logic
    │   ├── search_client/         # OpenSearch config
    │   ├── search_tools.py        # Keyword + semantic + hybrid
    │   ├── crew_ai/               # CrewAI multi-agent workflow
    │   └── main_menu.py           # Main CLI menu
    ├── docker-compose.yml         # Runs OpenSearch + Dashboard

# How To Run
Step 1: Start OpenSearch

    docker compose -f docker-compose.yml up

Step 2: Run the PDF ingestion

    python Agent/ingestion.py

Step 3: Run the agent or search tools

    python Agent/crew_ai/patent_crew.py

Step 4: For embeddings — run ollama serve and pull a model:

    ollama pull llama3
    ollama serve

# 🚀 Tips
    Make sure OpenSearch is running (http://localhost:9200)

    Make sure your Ollama model is running (ollama list)

    Use hybrid search for best results!

    Add more PDFs anytime — re‑ingest to update.

# TechStack
    LangChain
    OpenSearch
    Ollama
    CrewAI
    Embeddings