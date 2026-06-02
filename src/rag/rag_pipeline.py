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

    def __init__(self):

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

        context_parts = []

        for i, doc in enumerate(docs):

            source = doc.metadata.get(
                "source",
                "Unknown"
            )

            page = doc.metadata.get(
                "page",
                "N/A"
            )

            context_parts.append(
                f"""
Source {i+1}
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

        seen = set()

        for doc in docs:

            source = doc.metadata.get(
                "source",
                "Unknown"
            )

            page = doc.metadata.get(
                "page",
                "N/A"
            )

            key = (
                source,
                page
            )

            if key not in seen:

                seen.add(key)

                sources.append(
                    {
                        "source": source,
                        "page": page
                    }
                )

        return {
            "answer": answer,
            "sources": sources
        }