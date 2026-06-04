import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)
from src.loaders.pdf_loader import PDFLoader
from src.chunking.text_splitter import ChunkProcessor
from src.embeddings.embedding_generator import (
    EmbeddingGenerator
)
from src.vectordb.chroma_manager import (
    ChromaManager
)

from src.config.settings import PDF_DIR


def main():

    print("Loading PDFs...")

    loader = PDFLoader()

    documents = loader.load_directory(
        PDF_DIR
    )

    print(
        f"Loaded {len(documents)} pages"
    )

    # Normalize source metadata to store only the filename
    for doc in documents:
        if "source" in doc.metadata:
            doc.metadata["source"] = os.path.basename(doc.metadata["source"])

    print("Chunking...")

    chunker = ChunkProcessor()

    chunks = chunker.split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks"
    )

    print("Generating embeddings...")

    embedding_model = (
        EmbeddingGenerator()
        .get_model()
    )

    chroma = ChromaManager(
        embedding_model
    )

    chroma.create_vector_store(
        chunks
    )

    print("Indexing complete")


if __name__ == "__main__":
    main()