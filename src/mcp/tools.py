from src.retrieval.retriever import Retriever
from src.rag.rag_pipeline import RAGPipeline


_retriever = None
_rag = None


def get_retriever():

    global _retriever

    if _retriever is None:

        _retriever = Retriever()

    return _retriever


def get_rag():

    global _rag

    if _rag is None:

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

    docs = retriever.vectordb.get()

    all_text = []

    for metadata, document in zip(
        docs["metadatas"],
        docs["documents"]
    ):

        if metadata.get("source") == doc_id:

            all_text.append(
                document
            )

    if not all_text:

        return {
            "error": f"No document found with doc_id/source: {doc_id}"
        }

    combined_text = "\n".join(
        all_text
    )

    return rag.generator.generate_answer(
        query=f"Summarise the document {doc_id}",
        context=combined_text[:15000]
    )