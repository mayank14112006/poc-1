import os

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

# ---------- Models ----------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CLAUDE_MODEL = (
    "claude-haiku-4-5-20251001"
)

MAX_OUTPUT_TOKENS = 1200

# ---------- Secrets ----------

# Environment variable name expected: ANTHROPIC_API_KEY
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# ---------- Retrieval ----------

CHUNK_SIZE = 1000

CHUNK_OVERLAP = 200

TOP_K = 12