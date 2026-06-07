from sentence_transformers import SentenceTransformer


class LocalSentenceTransformerEmbeddings:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        import sys
        print(f"[embedding_generator] LocalSentenceTransformerEmbeddings __init__ start: model_name='{model_name}'...", file=sys.stderr, flush=True)
        import os
        from src.config.settings import BASE_DIR
        local_model_path = os.path.join(BASE_DIR, "src", "embeddings", "all-MiniLM-L6-v2")
        if os.path.exists(local_model_path) and os.listdir(local_model_path):
            self.model = SentenceTransformer(local_model_path)
        else:
            self.model = SentenceTransformer(model_name, local_files_only=True)
        print(f"[embedding_generator] LocalSentenceTransformerEmbeddings __init__ end.", file=sys.stderr, flush=True)

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
        import sys
        print("[embedding_generator] get_model start...", file=sys.stderr, flush=True)
        if self._model is None:
            print("[embedding_generator] get_model: initializing LocalSentenceTransformerEmbeddings...", file=sys.stderr, flush=True)
            self._model = LocalSentenceTransformerEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
            print("[embedding_generator] get_model: LocalSentenceTransformerEmbeddings initialized.", file=sys.stderr, flush=True)
        print("[embedding_generator] get_model end.", file=sys.stderr, flush=True)
        return self._model