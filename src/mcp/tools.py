import os

# Force CPU-only mode to avoid CUDA hangs in sandboxed environments (UWP / Claude Desktop)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

_retriever = None
_rag = None


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