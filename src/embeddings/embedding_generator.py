from langchain_voyageai import VoyageAIEmbeddings
from src.config.settings import EMBEDDING_MODEL, _get_secret


def _get_voyage_key() -> str:
    return _get_secret("VOYAGE_API_KEY")


class EmbeddingGenerator:

    def __init__(self):
        self._model = None

    def get_model(self):

        if self._model is None:

            voyage_key = _get_voyage_key()

            self._model = VoyageAIEmbeddings(
                voyage_api_key=voyage_key,
                model=EMBEDDING_MODEL
            )

        return self._model
