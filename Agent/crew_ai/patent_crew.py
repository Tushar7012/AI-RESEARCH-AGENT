import os
from datetime import datetime
import requests

from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from Agent.search_client.opensearch_client import get_opensearch_client


def check_ollama_availability():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        return [model.get("name") for model in models if model.get("name")]
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return []


def test_model(model_name):
    try:
        llm = OllamaLLM(model=model_name, temperature=0.2)
        prompt = ChatPromptTemplate.from_template("Say hello!")
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({})
        return bool(result)
    except Exception as e:
        print(f"Error testing model {model_name}: {e}")
        return False


class SearchPatentChunksTool(BaseTool):
    name: str = "search_patent_chunks"
    description: str = "Search for relevant chatbot/healthcare patent PDF chunks"

    def _run(self, query: str, top_k: int = 20) -> str:
        client = get_opensearch_client("localhost", 9200)
        index_name = "patent_chunks"

        search_query = {
            "size": top_k,
            "query": {"match": {"text": query}},
            "_source": ["source_file", "chunk_index", "text"],
        }

        try:
            response = client.search(index=index_name, body=search_query)
            results = response["hits"]["hits"]

            formatted = []
            for i, hit in enumerate(results):
                source = hit["_source"]
                formatted.append(
                    f"{i+1}. File: {source.get('source_file')} | Chunk: {source.get('chunk_index')}\n"
                    f"   Text: {source.get('text', '')[:300]}...\n"
                )
            return "\n".join(formatted)
        except Exception as e:
            return f"Error searching chunks: {str(e)}"


class SummarizeChunkTrendsTool(BaseTool):
    name: str = "summarize_patent_chunks"
    description: str = "Summarize patterns and innovations in chatbot/healthcare patent chunks"

    def _run(self, data: str) -> str:
        return f"Chunk-based insight summary:\n\n{data[:1000]}..."


def create_patent_analysis_crew(model_name="llama3"):
    available_models = check_ollama_availability()
    if not available_models:
        raise RuntimeError("Ollama is not running or no models found.")

    if not test_model(model_name):
        raise RuntimeError(f"Model {model_name} failed basic test.")

    if not model_name.startswith("ollama/"):
        model_name = f"ollama/{model_name}"

    llm = OllamaLLM(model=model_name, temperature=0.2)

    tools = [SearchPatentChunksTool(), SummarizeChunkTrendsTool()]

    # Updated agent roles for chatbot and virtual assistant patents
    lead_analyst = Agent(
        role="Lead Patent Analyst",
        goal="Coordinate chatbot/virtual assistant patent research",
        backstory="You lead AI patent reviews for intelligent assistants and healthcare bots.",
        verbose=True,
        allow_delegation=True,
        llm=llm,
        tools=tools,
    )

    document_reviewer = Agent(
        role="Patent Chunk Reviewer",
        goal="Extract and group relevant PDF chunks from OpenSearch",
        backstory="You specialize in digging out meaningful ideas from structured document chunks.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=tools,
    )

    trend_analyst = Agent(
        role="Innovation Trend Analyst",
        goal="Summarize themes and innovations in chatbot-related patents",
        backstory="You analyze conversational AI documents and identify rising patterns in tech claims.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=tools,
    )

    task1 = Task(
        description="""
        Plan a chatbot and virtual assistant research strategy using patent chunks.
        Identify 2–3 key functional areas (e.g., personalization, healthcare intake).
        Propose chunk filtering criteria and comparison focus.
        """,
        expected_output="A research plan with analysis goals and chunk grouping strategy.",
        agent=lead_analyst,
    )

    task2 = Task(
        description="""
        Use the search_patent_chunks tool to gather relevant chunks.
        Group by functionality (e.g., chatbot dialogue, virtual care logic).
        Identify which files they came from and any overlaps in ideas.
        """,
        expected_output="List of grouped chunks with brief taglines and source file info.",
        agent=document_reviewer,
        dependencies=[task1],
    )

    task3 = Task(
        description="""
        Summarize the chunk group insights to identify trends:
        - Repeated architectural choices
        - Domain overlaps (e.g., chatbot logic used in healthcare)
        - Technical innovation highlights
        - Recommendations for R&D
        """,
        expected_output="Innovation pattern summary with 3–5 bullet point trends.",
        agent=trend_analyst,
        dependencies=[task2],
    )

    return Crew(
        agents=[lead_analyst, document_reviewer, trend_analyst],
        tasks=[task1, task2, task3],
        verbose=True,
        process=Process.sequential,
        cache=False,
    )


def run_patent_analysis(research_area="Chatbots", model_name="llama3"):
    try:
        crew = create_patent_analysis_crew(model_name)
        result = crew.kickoff(inputs={"research_area": research_area})

        if hasattr(result, "output"):
            return result.output
        elif hasattr(result, "result"):
            return result.result
        return str(result)

    except Exception as e:
        return f"❌ Analysis failed: {str(e)}\nMake sure Ollama is running and PDFs were ingested."


if __name__ == "__main__":
    research_area = input("Enter research area (default: Chatbots): ") or "Chatbots"
    model_name = input("Enter Ollama model to use(defaults: llama3) : ") or "llama3"

    result = run_patent_analysis(research_area, model_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chatbot_patent_analysis_{timestamp}.txt"

    with open(filename, "w") as f:
        f.write(result)

    print(f"\n📄 Analysis complete! Results saved to {filename}")
