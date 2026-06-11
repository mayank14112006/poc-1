import os
import threading

# Force CPU-only mode — prevents CUDA driver scans that can hang for minutes
os.environ["CUDA_VISIBLE_DEVICES"] = ""

_retriever = None
_rag = None

_retriever_lock = threading.Lock()
_rag_lock = threading.Lock()

_load_status = "Initializing background pre-loader..."


def get_load_status():
    global _load_status
    return _load_status


def set_load_status(status):
    global _load_status
    import sys
    print(f"[tools] Load status update: {status}", file=sys.stderr, flush=True)
    _load_status = status


def get_retriever(prevent_block=False):
    global _retriever
    import sys
    if _retriever is not None:
        return _retriever

    if prevent_block:
        # Fast path check without lock to prevent tool call timeout
        return None

    print("[tools] get_retriever start...", file=sys.stderr, flush=True)
    print("[tools] get_retriever: acquiring lock...", file=sys.stderr, flush=True)
    with _retriever_lock:
        print("[tools] get_retriever: lock acquired...", file=sys.stderr, flush=True)
        if _retriever is None:
            set_load_status("Importing Retriever and vector DB libraries (Step 1 of 3)...")
            print("[tools] get_retriever: importing Retriever...", file=sys.stderr, flush=True)
            from src.retrieval.retriever import Retriever
            set_load_status("Initializing Retriever (Step 2 of 3)...")
            print("[tools] get_retriever: Retriever imported. Initializing...", file=sys.stderr, flush=True)
            _retriever = Retriever()
            print("[tools] get_retriever: Retriever initialized.", file=sys.stderr, flush=True)
    print("[tools] get_retriever end.", file=sys.stderr, flush=True)
    return _retriever


def get_rag(prevent_block=False):
    global _rag
    if _rag is not None:
        return _rag

    if prevent_block:
        return None

    with _rag_lock:
        if _rag is None:
            set_load_status("Initializing RAG pipeline (connecting retriever to generator)...")
            retriever = get_retriever(prevent_block=False)
            from src.rag.rag_pipeline import RAGPipeline
            _rag = RAGPipeline(retriever=retriever)
    return _rag



def search_documents_logic(
    query: str,
    k: int = 5
):
    import sys
    print(f"[tools] search_documents_logic start: query='{query}', k={k}", file=sys.stderr, flush=True)
    retriever = get_retriever(prevent_block=True)
    if retriever is None:
        print("[tools] search_documents_logic: retriever not loaded yet, returning loading message.", file=sys.stderr, flush=True)
        status = get_load_status()
        return [
            {
                "source": "System Notification",
                "page": 0,
                "content": f"The knowledge base is currently initializing: {status}\n\nOn Windows/UWP environments, child processes are suspended when the Claude Desktop app is minimized or placed in the background. To ensure the initialization completes, please keep this Claude window open and active in the foreground for 60-90 seconds, then try your query again."
            }
        ]

    docs = retriever.search(
        query=query,
        k=k
    )

    results = []

    for doc in docs:

        results.append(
            {
                "source": doc.metadata.get(
                    "source",
                    "Unknown"
                ),
                "page": doc.metadata.get(
                    "page",
                    "N/A"
                ),
                "content": doc.page_content
            }
        )

    return results


def ask_documents_logic(
    query: str
):

    rag = get_rag()

    return rag.ask(
        query
    )


def summarise_document_logic(
    doc_id: str
):
    import sys
    print(f"[tools] summarise_document_logic start: doc_id='{doc_id}'", file=sys.stderr, flush=True)
    retriever = get_retriever(prevent_block=True)
    rag = get_rag(prevent_block=True)
    if retriever is None or rag is None:
        print("[tools] summarise_document_logic: retriever or rag not loaded yet, returning loading message.", file=sys.stderr, flush=True)
        status = get_load_status()
        return {
            "error": f"The knowledge base is currently initializing: {status}\n\nOn Windows/UWP environments, child processes are suspended when the Claude Desktop app is minimized or placed in the background. To ensure the initialization completes, please keep this Claude window open and active in the foreground for 60-90 seconds, then try again."
        }

    # Query Chroma DB directly using native metadata filtering
    docs = retriever.vectordb.get(where={"source": doc_id})

    if not docs or not docs["documents"]:
        return {
            "error": f"No document found with source: {doc_id}"
        }

    # Sort chunks by page number to guarantee chronological sequence
    sorted_chunks = sorted(
        zip(docs["metadatas"], docs["documents"]),
        key=lambda x: x[0].get("page", 0)
    )

    combined_text = "\n".join(doc for _, doc in sorted_chunks)

    return rag.generator.generate_answer(
        query=f"Summarise the document {doc_id}",
        context=combined_text[:30000]
    )