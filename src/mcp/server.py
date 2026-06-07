import os
import sys
import threading
from pathlib import Path

# Force CPU-only mode to avoid CUDA hangs in sandboxed environments (UWP / Claude Desktop)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
    # ── STRATEGY ─────────────────────────────────────────────────────────────
    # Claude Desktop's initialize handshake times out after 60 seconds.
    # Pre-warming synchronously before mcp.run() can exceed that limit.
    #
    # Solution:
    #   1. Kick off pre-warm in a background THREAD immediately.
    #   2. Call mcp.run() right away → initialize responds in <1 second.
    #   3. Tool call functions in tools.py wait on _init_event (up to 5 min).
    #      The 5-min wait is safe: Claude Desktop's per-tool timeout is ~4 min,
    #      but pre-warm completes in ~30-60 seconds, long before that.
    # ─────────────────────────────────────────────────────────────────────────
    def _prewarm():
        """Pre-warm ONLY local components (no network calls).

        get_retriever() loads SentenceTransformer + ChromaDB — fully local, ~30s.
        get_rag() requires Infisical (network) — skipped here to avoid hangs.
        RAGPipeline is loaded lazily on the first summarise_document call instead.
        """
        print("[server] Background pre-warm: loading local retriever...", file=sys.stderr, flush=True)
        try:
            get_retriever()
            print("[server] Pre-warm complete — search_documents is ready.", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[server] Pre-warm error (retriever): {e}", file=sys.stderr, flush=True)

    threading.Thread(target=_prewarm, daemon=True).start()
    print("[server] MCP server starting (pre-warm running in background)...", file=sys.stderr, flush=True)
    mcp.run()