from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from src.config.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


class ChunkProcessor:

    def __init__(self):

        self.splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
        )

    def split_documents(
        self,
        documents
    ):

        chunks = self.splitter.split_documents(
            documents
        )

        return chunks