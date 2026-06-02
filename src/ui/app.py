import streamlit as st
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.rag.rag_pipeline import (
    RAGPipeline
)

from src.utils.bootstrap import (
    ensure_vector_db_exists
)


st.set_page_config(
    page_title="RAG Chatbot",
    layout="wide"
)


@st.cache_resource
def get_rag():

    ensure_vector_db_exists()

    return RAGPipeline()


st.title(
    "PDF RAG Chatbot"
)

st.caption(
    "Ask questions from your indexed PDF documents."
)

query = st.text_input(
    "Ask a question"
)

if st.button("Submit"):

    if not query.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Preparing vector database if needed..."
        ):

            rag = get_rag()

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            response = rag.ask(
                query
            )

        st.subheader(
            "Answer"
        )

        st.write(
            response["answer"]
        )

        st.subheader(
            "Sources"
        )

        for source in response[
            "sources"
        ]:

            st.write(
                f"{source['source']} "
                f"(Page {source['page']})"
            )
