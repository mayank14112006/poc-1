import os
from infisical_sdk import InfisicalSDKClient

# ---------- Paths ----------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

PDF_DIR = os.path.join(
    BASE_DIR,
    "data",
    "raw_pdfs"
)

CHROMA_DIR = os.path.join(
    BASE_DIR,
    "chroma_db"
)



def _get_secret(name: str) -> str:
    client_id = os.getenv("INFISICAL_CLIENT_ID")
    client_secret = os.getenv("INFISICAL_CLIENT_SECRET")
    project_id = os.getenv("INFISICAL_PROJECT_ID")

    if not client_id or not client_secret or not project_id:
        raise EnvironmentError(
            "INFISICAL_CLIENT_ID, INFISICAL_CLIENT_SECRET, "
            "and INFISICAL_PROJECT_ID must be set."
        )

    client = InfisicalSDKClient(
        host="https://app.infisical.com"
    )

    client.auth.universal_auth.login(
        client_id=client_id,
        client_secret=client_secret
    )

    secret = client.secrets.get_secret_by_name(
        secret_name=name,
        project_id=project_id,
        environment_slug="dev",
        secret_path="/"
    )

    return secret.secretValue



ANTHROPIC_API_KEY = _get_secret("ANTHROPIC_API_KEY")

# ---------- Models ----------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CLAUDE_MODEL = "claude-haiku-4-5-20251001"

MAX_OUTPUT_TOKENS = 1200

# ---------- Retrieval ----------

CHUNK_SIZE = 1000

CHUNK_OVERLAP = 200

TOP_K = 12
