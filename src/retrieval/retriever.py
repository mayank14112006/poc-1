from src.embeddings.embedding_generator import (
    EmbeddingGenerator
)

from src.vectordb.chroma_manager import (
    ChromaManager
)


class Retriever:

    def __init__(self):

        embedding_model = (
            EmbeddingGenerator()
            .get_model()
        )

        self.vectordb = (
            ChromaManager(
                embedding_model
            ).load_vector_store()
        )

    def search(
        self,
        query,
        k=12
    ):

        # Step 1: MMR gives diverse results
        mmr_retriever = self.vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": 30
            }
        )

        mmr_docs = mmr_retriever.invoke(
            query
        )

        # Step 2: Similarity search gives closest semantic matches
        similarity_docs = (
            self.vectordb.similarity_search(
                query,
                k=k
            )
        )

        # Step 3: Combine and remove duplicates
        final_docs = []
        seen = set()

        for doc in mmr_docs + similarity_docs:

            source = doc.metadata.get(
                "source",
                "Unknown"
            )

            page = doc.metadata.get(
                "page",
                "N/A"
            )

            content_key = (
                source,
                page,
                doc.page_content[:100]
            )

            if content_key not in seen:

                seen.add(
                    content_key
                )

                final_docs.append(
                    doc
                )

        return final_docs[:k]