import os
from datetime import datetime
import requests
from dotenv import load_dotenv

from Agent.search_client.opensearch_client import get_opensearch_client
from Agent.crew_ai.patent_crew import run_patent_analysis
from Agent.tools.search_tools import hybrid_search, iterative_search, semantic_search, keyword_search


def display_menu():
    print("\n" + "=" * 60)
    print("  PATENT INTELLIGENCE MENU - CHATBOT & HEALTHCARE AI  ")
    print("=" * 60)
    print("1. Run complete chatbot patent analysis with CrewAI")
    print("2. Search chunks using keyword/semantic/hybrid methods")
    print("3. Iterative query refinement search")
    print("4. System diagnostic check (OpenSearch, Ollama, Embeddings)")
    print("5. Exit")
    print("-" * 60)
    return input("Select an option (1-5): ")


def run_complete_analysis():
    print("\n🚀 Running full patent crew analysis (chatbot/assistant PDFs)...")
    research_area = input("Enter research area (default: Chatbots): ") or "Chatbots"
    model_name = input("Enter Ollama model to use (default: llama3): ") or "llama3"
    try:
        result = run_patent_analysis(research_area, model_name)
        if not isinstance(result, str):
            result = str(result)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chatbot_analysis_{timestamp}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(result)

        print(f"\n✅ Analysis complete! Results saved to {filename}")
        print("\n" + "=" * 60)
        print("🔎 ANALYSIS PREVIEW")
        print("-" * 60)
        print(result[:500] + "...\n")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")


def search_patents():
    print("\n🔍 PATENT CHUNK SEARCH")
    query = input("Enter query text: ")
    if not query:
        print("Search query cannot be empty.")
        return

    search_type = input("Choose search type (1=Keyword, 2=Semantic, 3=Hybrid) [3]: ") or "3"

    try:
        if search_type == "1":
            results = keyword_search(query)
        elif search_type == "2":
            results = semantic_search(query)
        else:
            results = hybrid_search(query)

        print(f"\nResults for '{query}': {len(results)} found")
        for i, hit in enumerate(results):
            src = hit.get("_source", {})
            print(f"{i+1}. File: {src.get('source_file', 'N/A')} | Chunk: {src.get('chunk_index', 'N/A')}")
            print(f"   Text: {src.get('text', '')[:200]}...")
            print("-" * 50)

    except Exception as e:
        print(f"❌ Search error: {e}")


def iterative_exploration():
    print("\n🔁 ITERATIVE EXPLORATION")
    query = input("Enter base query: ")
    if not query:
        print("Query cannot be empty.")
        return
    steps = input("Number of refinement steps [3]: ")
    try:
        steps = int(steps or 3)
    except:
        steps = 3

    try:
        results = iterative_search(query, refinement_steps=steps)
        print(f"\n🔄 Total results: {len(results)}")
        for i, hit in enumerate(results):
            src = hit.get("_source", {})
            print(f"{i+1}. Chunk: {src.get('chunk_index', 'N/A')} | File: {src.get('source_file', 'N/A')}")
            print(f"   Text: {src.get('text', '')[:180]}...")
            print("-" * 50)
    except Exception as e:
        print(f"❌ Iterative search failed: {e}")


def check_system_status():
    print("\n🛠 SYSTEM STATUS CHECK")

    try:
        client = get_opensearch_client("localhost", 9200)
        indices = client.cat.indices(format="json")
        print("✅ OpenSearch OK | Indices:")
        for idx in indices:
            print(f"  - {idx['index']}: {idx['docs.count']} docs")
    except Exception as e:
        print(f"❌ OpenSearch failed: {e}")

    try:
        res = requests.get("http://localhost:11434/api/tags")
        if res.status_code == 200:
            models = res.json().get("models", [])
            print(f"✅ Ollama OK | Models: {', '.join(m['name'] for m in models)}")
        else:
            print(f"❌ Ollama failed with status: {res.status_code}")
    except Exception as e:
        print(f"❌ Ollama check failed: {e}")

    try:
        from Agent.vectors.embedding import get_embedding
        vec = get_embedding("ping test")
        print(f"✅ Embedding OK | dim = {len(vec)}")
    except Exception as e:
        print(f"❌ Embedding error: {e}")


def main():
    load_dotenv()
    while True:
        choice = display_menu()
        if choice == "1":
            run_complete_analysis()
        elif choice == "2":
            search_patents()
        elif choice == "3":
            iterative_exploration()
        elif choice == "4":
            check_system_status()
        elif choice == "5":
            print("\n👋 Exiting. Stay innovative!")
            break
        else:
            print("\n⚠️ Invalid input. Please choose 1-5.")
        input("\nPress Enter to return to menu...")


if __name__ == "__main__":
    main()
