import os
import threading

# Force CPU-only mode — prevents CUDA driver scans that can hang for minutes
os.environ["CUDA_VISIBLE_DEVICES"] = ""

_retriever = None
_rag = None

# Set by server.py's background pre-warm thread when models are loaded.
# Tool functions wait on this before executing so they never time out.
_init_event = threading.Event()


def get_retriever():
    global _retriever
    if _retriever is None:
        from src.retrieval.retriever import Retriever
        _retriever = Retriever()
    return _retriever


def get_rag():
    global _rag
    if _rag is None:
        from src.rag.rag_pipeline import RAGPipeline
        _rag = RAGPipeline()
    return _rag



def search_documents_logic(
    query: str,
    k: int = 5
):
    # Wait up to 5 minutes for the background pre-warm thread to finish.
    # Pre-warm takes ~30-60s; Claude Desktop tool timeout is ~4 min — safe.
    _init_event.wait(timeout=300.0)

    retriever = get_retriever()

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
    # Wait up to 5 minutes for the background pre-warm thread to finish.
    _init_event.wait(timeout=300.0)

    retriever = get_retriever()
    rag = get_rag()

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