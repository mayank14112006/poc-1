from src.retrieval.retriever import (
    Retriever
)

from src.generation.claude_generator import (
    ClaudeGenerator
)

from src.config.settings import (
    TOP_K
)


class RAGPipeline:

    def __init__(self, retriever=None):

        if retriever is not None:
            self.retriever = retriever
        else:
            self.retriever = Retriever()

        self.generator = (
            ClaudeGenerator()
        )

    def ask(
        self,
        query,
        k=TOP_K
    ):

        docs = (
            self.retriever.search(
                query,
                k
            )
        )

        print(
            "DOC COUNT:",
            len(docs)
        )

        for doc in docs:
            print(
                "METADATA:",
                doc.metadata
            )

        context_parts = []
        import os

        for i, doc in enumerate(docs):

            metadata = doc.metadata or {}

            source = (
                metadata.get("source")
                or metadata.get("file_path")
                or metadata.get("filename")
                or "Unknown document"
            )
            # Ensure base filename only
            source = os.path.basename(source)

            page = (
                metadata.get("page")
                or metadata.get("page_number")
                or "N/A"
            )

            context_parts.append(
                f"""
Source {i + 1}
Document: {source}
Page: {page}

Content:
{doc.page_content}
"""
            )

        context = "\n\n".join(
            context_parts
        )

        answer = (
            self.generator.generate_answer(
                query,
                context
            )
        )

        sources = []

        for i, doc in enumerate(docs):

            metadata = doc.metadata or {}

            source = (
                metadata.get("source")
                or metadata.get("file_path")
                or metadata.get("filename")
                or "Unknown document"
            )
            source = os.path.basename(source)

            page = (
                metadata.get("page")
                or metadata.get("page_number")
                or "N/A"
            )

            sources.append(
                {
                    "index": i + 1,
                    "source": source,
                    "page": page,
                    "content_preview": doc.page_content[:150].strip() + "..."
                }
            )

        if not sources and docs:

            sources.append(
                {
                    "index": 1,
                    "source": "Retrieved document chunks found, but metadata missing",
                    "page": "N/A",
                    "content_preview": ""
                }
            )

        return {
            "answer": answer,
            "sources": sources
        }