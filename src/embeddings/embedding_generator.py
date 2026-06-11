import os
import sys
import time

# Force CPU-only mode and limit threads to avoid hangs and scheduling overhead in sandboxed/containerized environments
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


class LocalSentenceTransformerEmbeddings:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print(f"[embedding_generator] LocalSentenceTransformerEmbeddings __init__ start: model_name='{model_name}'...", file=sys.stderr, flush=True)
        import torch
        # Optimize CPU threads to prevent scheduling thrashing in containerized/sandboxed environments
        torch.set_num_threads(1)

        print("[embedding_generator] Importing sentence_transformers...", file=sys.stderr, flush=True)
        t_import = time.time()
        from sentence_transformers import SentenceTransformer
        print(f"[embedding_generator] sentence_transformers imported in {time.time()-t_import:.2f}s.", file=sys.stderr, flush=True)

        from src.config.settings import BASE_DIR
        local_model_path = os.path.join(BASE_DIR, "src", "embeddings", "all-MiniLM-L6-v2")
        
        print(f"[embedding_generator] Loading SentenceTransformer model from {local_model_path}...", file=sys.stderr, flush=True)
        t_model = time.time()
        if os.path.exists(local_model_path) and os.listdir(local_model_path):
            self.model = SentenceTransformer(local_model_path)
        else:
            self.model = SentenceTransformer(model_name, local_files_only=True)
        print(f"[embedding_generator] SentenceTransformer model loaded in {time.time()-t_model:.2f}s.", file=sys.stderr, flush=True)
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