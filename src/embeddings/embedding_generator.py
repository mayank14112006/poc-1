from sentence_transformers import (
    SentenceTransformer
)


class SentenceTransformerEmbeddingWrapper:

    def __init__(
        self,
        model
    ):

        self.model = model

    def embed_documents(
        self,
        texts
    ):

        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        return embeddings.tolist()

    def embed_query(
        self,
        text
    ):

        embedding = self.model.encode(
            text,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        return embedding.tolist()


class EmbeddingGenerator:

    def __init__(self):

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def get_model(
        self
    ):

        return (
            SentenceTransformerEmbeddingWrapper(
                self.model
            )
        )
