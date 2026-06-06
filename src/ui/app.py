import streamlit as st
import sys
import os
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.rag.rag_pipeline import RAGPipeline
from src.utils.bootstrap import ensure_vector_db_exists
from src.config.settings import CLAUDE_MODEL

# --- Page Configuration ---
st.set_page_config(
    page_title="Citizen Services - Smart RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Premium Custom Styling (CSS) ---
st.markdown("""
    <style>
    /* Dark mode adjustments and font family */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Title and Header customization */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .caption-text {
        font-size: 1.05rem;
        color: #9ca3af;
        margin-bottom: 2rem;
    }
    
    /* Metrics panel */
    .metric-card {
        background-color: #1e1b4b;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #3730a3;
        margin-bottom: 1rem;
    }
    
    /* Citation pills */
    .source-badge {
        display: inline-block;
        background-color: #312e81;
        color: #e0e7ff;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 8px;
        border: 1px solid #4338ca;
        text-decoration: none;
    }
    
    /* Answer box styling */
    .answer-box {
        background-color: #111827;
        border-left: 5px solid #6366f1;
        padding: 1.5rem;
        border-radius: 0px 12px 12px 0px;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar premium feel */
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #818cf8;
        margin-bottom: 1rem;
    }
    
    /* Info box styling */
    .custom-info-box {
        background-color: #1e293b;
        color: #94a3b8;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #334155;
        font-size: 0.85rem;
        margin-top: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Lazy RAG Initialization ---
@st.cache_resource
def get_rag():
    ensure_vector_db_exists()
    return RAGPipeline()

# --- Load Database Stats for Sidebar ---
try:
    # Set dummy environment variables to allow import and initial load
    if not os.getenv("INFISICAL_CLIENT_ID"):
        os.environ["INFISICAL_CLIENT_ID"] = "dummy"
        os.environ["INFISICAL_CLIENT_SECRET"] = "dummy"
        os.environ["INFISICAL_PROJECT_ID"] = "dummy"
    
    rag = get_rag()
    db_docs = rag.retriever.vectordb.get()
    unique_files = sorted(list(set(m.get("source", "Unknown") for m in db_docs["metadatas"])))
    total_chunks = len(db_docs["documents"])
except Exception:
    unique_files = []
    total_chunks = 0

# --- Sidebar Content ---
with st.sidebar:
    st.markdown('<div class="sidebar-header">🛠️ System Panel</div>', unsafe_allow_html=True)
    
    # Database Metrics
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("📚 Corpus Status")
    st.metric(label="Indexed PDFs", value=len(unique_files))
    st.metric(label="Total Text Chunks", value=total_chunks)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # List of files
    if unique_files:
        st.subheader("📂 Document List")
        for f in unique_files:
            st.markdown(f"- `{f}`")
    
    # Secrets notice for deployment
    st.markdown("""
        <div class="custom-info-box">
            <strong>Streamlit Cloud Secrets</strong><br>
            To deploy to Streamlit Cloud, add the following to your App Secrets (Advanced settings):<br><br>
            <code>INFISICAL_CLIENT_ID = "..."</code><br>
            <code>INFISICAL_CLIENT_SECRET = "..."</code><br>
            <code>INFISICAL_PROJECT_ID = "..."</code>
        </div>
    """, unsafe_allow_html=True)

# --- Main Page Content ---
st.markdown('<div class="main-title">🤖 Citizen Services AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="caption-text">Ask complex research questions from your indexed AI & LLM literature corpus.</div>', unsafe_allow_html=True)

# Suggested Questions
st.subheader("💡 Suggested Questions")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("What is Low-Rank Adaptation (LoRA)?"):
        st.session_state.query_input = "What is Low-Rank Adaptation (LoRA) and how does it save memory?"
with col2:
    if st.button("Explain the Transformer architecture"):
        st.session_state.query_input = "Explain the main components of the Transformer architecture from the Attention paper."
with col3:
    if st.button("What is Retrieval-Augmented Generation (RAG)?"):
        st.session_state.query_input = "What is Retrieval-Augmented Generation (RAG) and what are its benefits?"

# Text Input
if 'query_input' not in st.session_state:
    st.session_state.query_input = ""

query = st.text_input("Ask a question about the papers:", value=st.session_state.query_input)

if st.button("Submit Query", type="primary"):
    if not query.strip():
        st.warning("Please enter a question to query the corpus.")
    else:
        # Check if dummy environment variables are still present
        # Streamlit Cloud needs real variables set in secrets
        is_configured = True
        if os.getenv("INFISICAL_CLIENT_ID") == "dummy" or not os.getenv("INFISICAL_CLIENT_ID"):
            # Try to fetch from streamlit secrets if available
            try:
                os.environ["INFISICAL_CLIENT_ID"] = st.secrets["INFISICAL_CLIENT_ID"]
                os.environ["INFISICAL_CLIENT_SECRET"] = st.secrets["INFISICAL_CLIENT_SECRET"]
                os.environ["INFISICAL_PROJECT_ID"] = st.secrets["INFISICAL_PROJECT_ID"]
            except Exception:
                is_configured = False
                
        if not is_configured:
            st.error("""
                🛑 **Secrets Configuration Missing**
                
                Universal Auth client ID or secret has not been configured in the system environment variables or Streamlit secrets.
                Please configure the following environment variables:
                - `INFISICAL_CLIENT_ID`
                - `INFISICAL_CLIENT_SECRET`
                - `INFISICAL_PROJECT_ID`
            """)
        else:
            with st.spinner("Retrieving document context and generating answer..."):
                try:
                    rag_instance = get_rag()
                    response = rag_instance.ask(query)
                    
                    st.success("Query processed successfully!")
                    
                    # Styled Answer Box
                    st.markdown("### 📝 Generated Answer")
                    st.markdown(f'<div class="answer-box">{response["answer"]}</div>', unsafe_allow_html=True)
                    
                    # Citations
                    st.markdown("### 📚 Source Citations")
                    if response.get("sources"):
                        st.subheader("Sources")
                        for source in response["sources"]:
                            index_str = f"[{source['index']}] " if "index" in source else ""
                            st.write(f"{index_str}{source['source']} (Page {source['page']})")
                    else:
                        st.info("No explicit source citations were returned for this response.")
                        
                except Exception as e:
                    error_msg = str(e)
                    if "404" in error_msg or "not_found_error" in error_msg:
                        st.error(f"""
                            🛑 **API Model Error (404 Not Found)**
                            
                            The Anthropic API returned a model not found error. This typically means:
                            1. The model configured (`{CLAUDE_MODEL}`) is not enabled or available for your API key.
                            2. Your Anthropic API credits have expired or the billing is inactive.
                            
                            *System Error Message:* `{error_msg}`
                        """)
                    else:
                        st.error(f"An unexpected error occurred during processing: `{error_msg}`")