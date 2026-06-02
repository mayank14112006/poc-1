from sentence_transformers import (
    SentenceTransformer
)


class SentenceTransformerEmbeddingWrapper:

    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )

    def embed_query(self, text):
        encoded = self.model.encode(
            [text],
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return encoded[0]


class EmbeddingGenerator:

    def __init__(self):

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def get_model(self):

        return SentenceTransformerEmbeddingWrapper(self.model)