from sentence_transformers import SentenceTransformer


class LocalSentenceTransformerEmbeddings:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name, local_files_only=True)

    def embed_documents(self, texts):
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        return embeddings.tolist()

    def embed_query(self, text):
        embedding = self.model.encode(
            text,
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