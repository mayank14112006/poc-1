from sentence_transformers import SentenceTransformer


class LocalSentenceTransformerEmbeddings:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):

        texts = [str(text) for text in texts]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        return embeddings.tolist()

    def embed_query(self, text):

        embedding = self.model.encode(
            str(text),
            convert_to_numpy=True
        )

        return embedding.tolist()


class EmbeddingGenerator:

    def __init__(self):
        self._model = None

    def get_model(self):

        if self._model is None:

            self._model = LocalSentenceTransformerEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )

        return self._model
