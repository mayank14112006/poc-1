import os

from src.config.settings import (
    CHROMA_DIR,
    PDF_DIR
)

from src.loaders.pdf_loader import (
    PDFLoader
)

from src.chunking.text_splitter import (
    ChunkProcessor
)

from src.embeddings.embedding_generator import (
    EmbeddingGenerator
)

from src.vectordb.chroma_manager import (
    ChromaManager
)


def ensure_vector_db_exists():

    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        return

    print(
        "ChromaDB not found. Creating vector database..."
    )

    loader = PDFLoader()

    documents = loader.load_directory(
        PDF_DIR
    )

    if not documents:

        raise ValueError(
            "No PDFs found in data/raw_pdfs."
        )

    # Normalize source metadata to store only the filename
    for doc in documents:
        if "source" in doc.metadata:
            doc.metadata["source"] = os.path.basename(doc.metadata["source"])

    chunker = ChunkProcessor()

    chunks = chunker.split_documents(
        documents
    )

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

    print(
        "ChromaDB created successfully."
    )