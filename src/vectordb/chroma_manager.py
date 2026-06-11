import os
import sys
import time

from src.config.settings import (
    CHROMA_DIR
)


class ChromaManager:

    def __init__(
        self,
        embedding_model
    ):

        self.embedding_model = (
            embedding_model
        )

    def create_vector_store(
        self,
        chunks
    ):
        print("[chroma_manager] Importing langchain_chroma for create...", file=sys.stderr, flush=True)
        t_import = time.time()
        from langchain_chroma import Chroma
        print(f"[chroma_manager] langchain_chroma imported in {time.time()-t_import:.2f}s.", file=sys.stderr, flush=True)

        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_model,
            persist_directory=CHROMA_DIR
        )

        return vectordb

    def load_vector_store(
        self
    ):
        print("[chroma_manager] Importing langchain_chroma for load...", file=sys.stderr, flush=True)
        t_import = time.time()
        from langchain_chroma import Chroma
        print(f"[chroma_manager] langchain_chroma imported in {time.time()-t_import:.2f}s.", file=sys.stderr, flush=True)

        vectordb = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=self.embedding_model
        )

        return vectordb