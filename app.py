import streamlit as st
from datetime import datetime
from Agent.crew_ai.patent_crew import run_patent_analysis
from Agent.tools.search_tools import hybrid_search, iterative_search, semantic_search, keyword_search

st.set_page_config(page_title="Patent Innovation Explorer", layout="centered")
st.title("🧠 Patent Innovation Explorer")

st.sidebar.title("🔍 Search Settings")
mode = st.sidebar.radio("Choose mode", ["Run Analysis", "Search Patents", "Iterative Exploration"])

if mode == "Run Analysis":
    st.subheader("📊 Run Full Analysis")
    research_area = st.text_input("Research Area", "Chatbots")
    model_name = st.text_input("Ollama Model", "llama3")

    if st.button("Run Analysis"):
        with st.spinner("Running CrewAI analysis..."):
            result = run_patent_analysis(research_area, model_name)
            st.success("Analysis Complete!")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_{timestamp}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(result)

            st.download_button("📥 Download Report", result, file_name=filename)
            st.text_area("🧾 Analysis Preview", result[:1000] + "...", height=300)

elif mode == "Search Patents":
    st.subheader("🔎 Search Patent Chunks")
    query = st.text_input("Enter Search Query")
    search_type = st.selectbox("Search Type", ["Hybrid", "Keyword", "Semantic"])

    if st.button("Search") and query:
        if search_type == "Keyword":
            results = keyword_search(query)
        elif search_type == "Semantic":
            results = semantic_search(query)
        else:
            results = hybrid_search(query)

        st.markdown(f"**Results Found:** {len(results)}")
        for i, hit in enumerate(results):
            source = hit.get("_source", {})
            st.write(f"{i+1}. File: {source.get('source_file')} | Chunk: {source.get('chunk_index')}")
            st.text(source.get("text", "")[:300] + "...")
            st.markdown("---")

elif mode == "Iterative Exploration":
    st.subheader("🔁 Iterative Search")
    query = st.text_input("Initial Query")
    steps = st.slider("Refinement Steps", 1, 10, 3)

    if st.button("Explore") and query:
        results = iterative_search(query, refinement_steps=steps)
        st.markdown(f"**Total Chunks Found:** {len(results)}")
        for i, hit in enumerate(results):
            source = hit.get("_source", {})
            st.write(f"{i+1}. File: {source.get('source_file')} | Chunk: {source.get('chunk_index')}")
            st.text(source.get("text", "")[:300] + "...")
            st.markdown("---")
