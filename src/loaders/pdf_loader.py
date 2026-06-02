from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader
)


class PDFLoader:

    def load_single_pdf(
        self,
        pdf_path: str
    ):

        loader = PyPDFLoader(pdf_path)

        docs = loader.load()

        return docs

    def load_directory(
        self,
        directory_path: str
    ):

        all_docs = []

        pdf_files = Path(
            directory_path
        ).glob("*.pdf")

        for pdf in pdf_files:

            loader = PyPDFLoader(
                str(pdf)
            )

            docs = loader.load()

            all_docs.extend(docs)

        return all_docs