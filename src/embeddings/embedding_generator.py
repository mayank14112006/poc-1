import os
from langchain_voyageai import VoyageAIEmbeddings  # type: ignore[import]
from src.config.settings import EMBEDDING_MODEL


def _get_voyage_key() -> str:
    """Pull Voyage API key from Infisical via the already-loaded settings env."""
    from infisical_sdk import InfisicalSDKClient

    token = os.getenv("INFISICAL_TOKEN")
    project_id = os.getenv("INFISICAL_PROJECT_ID")

    if not token or not project_id:
        raise EnvironmentError(
            "INFISICAL_TOKEN and INFISICAL_PROJECT_ID must be set."
        )

    client = InfisicalSDKClient(token=token)

    secret = client.getSecret(
        secret_name="VOYAGE_API_KEY",
        project_id=project_id,
        environment_slug="dev",
        secret_path="/"
    )

    return secret.secret_value


class EmbeddingGenerator:

    def __init__(self):
        self._model = None

    def get_model(self):

        if self._model is None:

            voyage_key = _get_voyage_key()

            self._model = VoyageAIEmbeddings(
                voyage_api_key=voyage_key,
                model=EMBEDDING_MODEL      # "voyage-3-lite"
            )

        return self._model
