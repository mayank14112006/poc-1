from langchain_chroma import (
    Chroma
)

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

        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_model,
            persist_directory=CHROMA_DIR
        )

        return vectordb

    def load_vector_store(
        self
    ):

        vectordb = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=self.embedding_model
        )

        return vectordb