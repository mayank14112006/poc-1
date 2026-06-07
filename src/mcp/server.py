import os
import sys
from pathlib import Path

# Force CPU-only mode and limit threads to avoid hangs and scheduling overhead in sandboxed/containerized environments
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from mcp.server.fastmcp import FastMCP

from src.mcp.tools import (
    search_documents_logic,
    summarise_document_logic,
    get_retriever,
    get_rag,
)


mcp = FastMCP(
    "rag-mcp-server"
)


@mcp.tool()
def search_documents(
    query: str,
    k: int = 5
):
    """Search the PDF knowledge base for chunks relevant to the query and return top-k results with source, page, and content.

    Use this tool to find information about any topic covered in the indexed PDFs.
    The knowledge base contains the following documents:
    - Attention Is All You Need.pdf  (Transformer architecture, self-attention mechanism, multi-head attention)
    - BERT.pdf                        (Bidirectional encoders, masked language modelling)
    - Chain-of-Thought.pdf            (Chain-of-thought prompting, reasoning traces)
    - GPT-3.pdf                       (Large language models, few-shot learning)
    - InstructGPT Paper.pdf           (RLHF, instruction following)
    - LORA.pdf                        (Low-rank adaptation, parameter-efficient fine-tuning)
    - Llama 2.pdf                     (Open LLM, chat fine-tuning)
    - RAG.pdf                         (Retrieval-augmented generation)
    - ReAct Paper.pdf                 (Reasoning and acting, agent frameworks)
    - Sentence-BERT Paper.pdf         (Sentence embeddings, semantic similarity)

    Args:
        query: Natural language search query (e.g. "self-attention mechanism", "few-shot learning")
        k: Number of chunks to return (default 5, max 12)
    """
    return search_documents_logic(query, k)


@mcp.tool()
def summarise_document(
    doc_id: str
):
    """Summarise all content chunks from a specific PDF document in the knowledge base.

    IMPORTANT: doc_id must be the EXACT filename (including .pdf extension) from the list below.
    Available documents (use these exact strings as doc_id):
    - "Attention Is All You Need.pdf"  → Transformer, self-attention, multi-head attention
    - "BERT.pdf"                        → BERT, masked LM, bidirectional transformers
    - "Chain-of-Thought.pdf"            → Chain-of-thought prompting, reasoning
    - "GPT-3.pdf"                       → GPT-3, few-shot learning, large LMs
    - "InstructGPT Paper.pdf"           → InstructGPT, RLHF, alignment
    - "LORA.pdf"                        → LoRA, low-rank adaptation, fine-tuning
    - "Llama 2.pdf"                     → Llama 2, open-source LLM, chat models
    - "RAG.pdf"                         → Retrieval-augmented generation
    - "ReAct Paper.pdf"                 → ReAct, reasoning + acting agents
    - "Sentence-BERT Paper.pdf"         → Sentence-BERT, embeddings, semantic similarity

    Args:
        doc_id: Exact filename of the document to summarise (e.g. "Attention Is All You Need.pdf")
    """
    return summarise_document_logic(doc_id)


if __name__ == "__main__":
    import time
    import threading

    # Start background thread to initialize the retriever (loading weights / Chroma DB)
    # after a 5-second delay to ensure FastMCP completes its handshake first
    def pre_load_retriever():
        print("[server] Background pre-load thread: sleeping 5s to let FastMCP finish handshake...", file=sys.stderr, flush=True)
        time.sleep(5)
        print("[server] Background pre-load thread: starting initialization...", file=sys.stderr, flush=True)
        t0 = time.time()
        try:
            get_retriever(prevent_block=False)
            get_rag(prevent_block=False)
            print(f"[server] Retriever and RAG pre-loaded successfully in background in {time.time()-t0:.2f}s.", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[server] Error pre-loading retriever in background: {e}", file=sys.stderr, flush=True)

    print("[server] Starting background pre-loading thread...", file=sys.stderr, flush=True)
    loading_thread = threading.Thread(target=pre_load_retriever, daemon=True)
    loading_thread.start()

    print("[server] Starting FastMCP server...", file=sys.stderr, flush=True)
    mcp.run()