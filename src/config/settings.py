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

# ---------- Infisical ----------
# Only two env vars ever need to be set manually:
#   INFISICAL_TOKEN   — your machine identity / service token
#   INFISICAL_PROJECT_ID — your project ID (not a secret, just an ID)

def _get_secret(name: str) -> str:
    token = os.getenv("INFISICAL_TOKEN")
    project_id = os.getenv("INFISICAL_PROJECT_ID")

    if not token or not project_id:
        raise EnvironmentError(
            f"INFISICAL_TOKEN and INFISICAL_PROJECT_ID must be set to fetch secret: {name}"
        )

    client = InfisicalSDKClient(token=token)

    secret = client.getSecret(
        secret_name=name,
        project_id=project_id,
        environment_slug="dev",
        secret_path="/"
    )

    return secret.secret_value


ANTHROPIC_API_KEY = _get_secret("ANTHROPIC_API_KEY")

# ---------- Models ----------

EMBEDDING_MODEL = "voyage-3-lite"   # Voyage AI free tier

CLAUDE_MODEL = "claude-haiku-4-5-20251001"

MAX_OUTPUT_TOKENS = 1200

# ---------- Retrieval ----------

CHUNK_SIZE = 1000

CHUNK_OVERLAP = 200

TOP_K = 12
