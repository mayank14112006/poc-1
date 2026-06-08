# PDF RAG Chatbot & MCP Server — POC 01

A specification-compliant **Retrieval-Augmented Generation (RAG)** chatbot and **Model Context Protocol (MCP)** server built over a curated corpus of 10 seminal ML/LLM research papers.

This project implements **POC 01** of the AI Engineering Intern assessment.

🚀 **Live Streamlit Demo:** [https://5hjfvsgxrf5zsk9bvpn3mu.streamlit.app/](https://5hjfvsgxrf5zsk9bvpn3mu.streamlit.app/)

---

## Architecture Overview

The system operates in two independent phases: **Offline Indexing** and **Runtime Query Processing**.

```
╔══════════════════════════════════════════════════════════════════════╗
║                    PHASE 1 — OFFLINE INDEXING                       ║
║                     (run once via build_db.py)                      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║   data/raw_pdfs/                                                     ║
║   ┌─────────────┐                                                    ║
║   │  10 × PDFs  │                                                    ║
║   └──────┬──────┘                                                    ║
║          │                                                           ║
║          ▼                                                           ║
║   ┌─────────────────────────────────────┐                           ║
║   │  PDFLoader  (LangChain + PyPDF)     │  page-by-page extraction  ║
║   └──────────────────┬──────────────────┘                           ║
║                       │                                              ║
║                       ▼                                              ║
║   ┌─────────────────────────────────────┐                           ║
║   │  Metadata Normalisation             │  source → basename only   ║
║   └──────────────────┬──────────────────┘                           ║
║                       │                                              ║
║                       ▼                                              ║
║   ┌─────────────────────────────────────┐                           ║
║   │  RecursiveCharacterTextSplitter     │  chunk_size=1000          ║
║   │  (LangChain Text Splitters)         │  chunk_overlap=200        ║
║   └──────────────────┬──────────────────┘                           ║
║                       │                                              ║
║                       ▼                                              ║
║   ┌─────────────────────────────────────┐                           ║
║   │  EmbeddingGenerator                 │  all-MiniLM-L6-v2        ║
║   │  (SentenceTransformer — local)      │  384-dim dense vectors    ║
║   └──────────────────┬──────────────────┘                           ║
║                       │                                              ║
║                       ▼                                              ║
║   ┌─────────────────────────────────────┐                           ║
║   │  ChromaDB  (local persistent store) │  saved to chroma_db/     ║
║   └─────────────────────────────────────┘                           ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝


╔══════════════════════════════════════════════════════════════════════╗
║               PHASE 2 — RUNTIME QUERY PROCESSING                    ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║   ┌──────────────────┐        ┌──────────────────┐                  ║
║   │  Claude Desktop  │        │  Streamlit UI     │                  ║
║   │  (MCP Client)    │        │  src/ui/app.py    │                  ║
║   └────────┬─────────┘        └────────┬──────────┘                 ║
║            │  User Question             │  User Question             ║
║            ▼                           ▼                            ║
║   ┌─────────────────────────────────────────────────┐               ║
║   │             FastMCP Server                      │               ║
║   │             src/mcp/server.py                   │               ║
║   │                                                 │               ║
║   │   Tool 1: search_documents(query, k)            │               ║
║   │   Tool 2: summarise_document(doc_id)            │               ║
║   └──────────────────────┬──────────────────────────┘               ║
║                           │                                          ║
║                           ▼                                          ║
║   ┌─────────────────────────────────────┐                           ║
║   │  Retriever  (src/retrieval/)        │                           ║
║   │                                     │                           ║
║   │  Step 1 → MMR Search (k=12)         │  diverse results          ║
║   │  Step 2 → Similarity Search (k=12) │  closest semantic match   ║
║   │  Step 3 → Deduplicate & merge       │  unique top-k chunks      ║
║   └──────────────────┬──────────────────┘                           ║
║          query        │  embed via SentenceTransformer               ║
║          vectors      ▼                                              ║
║   ┌─────────────────────────────────────┐                           ║
║   │  ChromaDB  (chroma_db/ on disk)     │  vector similarity lookup ║
║   └──────────────────┬──────────────────┘                           ║
║                       │  raw document chunks                         ║
║                       ▼                                              ║
║   ┌─────────────────────────────────────┐                           ║
║   │  Context Builder  (RAGPipeline)     │                           ║
║   │                                     │                           ║
║   │  Formats each chunk as:             │                           ║
║   │    Source N | Document | Page       │                           ║
║   │    Content: <chunk text>            │                           ║
║   └──────────────────┬──────────────────┘                           ║
║                       │  structured context string                   ║
║                       ▼                                              ║
║   ┌─────────────────────────────────────┐                           ║
║   │  ClaudeGenerator                    │                           ║
║   │  Model: claude-haiku-4-5-20251001   │                           ║
║   │  via Anthropic Python SDK           │                           ║
║   │                                     │                           ║
║   │  Prompt rules:                      │                           ║
║   │  • Strict grounding (no hallucin.)  │                           ║
║   │  • Cite every claim as [Source N]   │                           ║
║   │  • "Not found in PDFs" if no match  │                           ║
║   └──────────────────┬──────────────────┘                           ║
║                       │  answer + [Source N] citations               ║
║                       ▼                                              ║
║   ┌─────────────────────────────────────┐                           ║
║   │  Response returned to client        │                           ║
║   │  • answer text with inline [Src N]  │                           ║
║   │  • sources[]: filename, page, index │                           ║
║   └─────────────────────────────────────┘                           ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝


╔══════════════════════════════════════════════════════════════════════╗
║                     SECRETS MANAGEMENT                              ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║   Environment Variables                                              ║
║   INFISICAL_CLIENT_ID     ──┐                                        ║
║   INFISICAL_CLIENT_SECRET ──┼──► Infisical Universal Auth SDK        ║
║   INFISICAL_PROJECT_ID    ──┘         │                              ║
║                                       ▼                              ║
║                              Infisical Cloud API                     ║
║                                       │                              ║
║                                       ▼                              ║
║                              ANTHROPIC_API_KEY  (fetched at runtime) ║
║                                       │                              ║
║                                       ▼                              ║
║                              ClaudeGenerator  (injected on init)     ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## Technical Stack

| Layer               | Technology                                             |
|---------------------|--------------------------------------------------------|
| **PDF Loading**     | LangChain `PyPDFLoader` (`langchain-community`)        |
| **Text Splitting**  | `RecursiveCharacterTextSplitter` (chunk=1000, overlap=200) |
| **Embeddings**      | `all-MiniLM-L6-v2` via `sentence-transformers` (local) |
| **Vector Store**    | ChromaDB — local persisted SQLite store (`chroma_db/`) |
| **Retrieval**       | MMR + Similarity Search, deduplicated, top-k=12        |
| **LLM**             | Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) — Anthropic SDK |
| **Secrets**         | Infisical Universal Auth SDK                           |
| **MCP Protocol**    | FastMCP (official Anthropic MCP Python SDK)            |
| **UI**              | Streamlit                                              |

---

## Knowledge Base — Indexed Documents

| # | Filename | Topic |
|---|----------|-------|
| 1 | `Attention Is All You Need.pdf` | Transformer architecture, self-attention, multi-head attention |
| 2 | `BERT.pdf` | Bidirectional encoders, masked language modelling |
| 3 | `Chain-of-Thought.pdf` | Chain-of-thought prompting, reasoning traces |
| 4 | `GPT-3.pdf` | Large language models, few-shot learning |
| 5 | `InstructGPT Paper.pdf` | RLHF, instruction following, alignment |
| 6 | `LORA.pdf` | Low-rank adaptation, parameter-efficient fine-tuning |
| 7 | `Llama 2.pdf` | Open-source LLM, chat fine-tuning |
| 8 | `RAG.pdf` | Retrieval-augmented generation |
| 9 | `ReAct Paper.pdf` | Reasoning + acting, agent frameworks |
| 10 | `Sentence-BERT Paper.pdf` | Sentence embeddings, semantic similarity |

---

## MCP Tools Specification

The MCP server exposes **exactly two tools** as required by the specification:

### `search_documents(query: str, k: int = 5)`
Performs a semantic search over the ChromaDB vector store and returns the top-k most relevant chunks.

**Returns:** List of `{ source, page, content }` objects.

### `summarise_document(doc_id: str)`
Fetches all chunks belonging to a specific document (matched by exact filename), sorts them by page, and sends the combined text to Claude Haiku 4.5 for a comprehensive summary.

**`doc_id` must be an exact filename** from the table above (e.g. `"Attention Is All You Need.pdf"`).

**Returns:** `{ answer: "<summary text>" }`

---

## Setup & Ingest Pipeline

### 1. Prerequisites

Python 3.10+ required. Create and activate a virtual environment:

```bash
# Create virtual environment
python -m venv .venv

# Activate — Windows
.venv\Scripts\activate

# Activate — Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Infisical Secrets

This project uses **Infisical** as a secrets manager. The `ANTHROPIC_API_KEY` is stored in Infisical and fetched at runtime — it is never hardcoded or committed to the repository.

You must create an Infisical project, add `ANTHROPIC_API_KEY` as a secret in the `dev` environment, and obtain a **Universal Auth** client ID and secret.

Set the following environment variables before running:

```powershell
# PowerShell (Windows)
$env:INFISICAL_CLIENT_ID="your_infisical_client_id"
$env:INFISICAL_CLIENT_SECRET="your_infisical_client_secret"
$env:INFISICAL_PROJECT_ID="your_infisical_project_id"
```

```bash
# Bash (Linux / macOS)
export INFISICAL_CLIENT_ID="your_infisical_client_id"
export INFISICAL_CLIENT_SECRET="your_infisical_client_secret"
export INFISICAL_PROJECT_ID="your_infisical_project_id"
```

### 3. Build the Vector Database

Place your PDFs in `data/raw_pdfs/` and run the ingestion script once:

```bash
python build_db.py
```

This will:
1. Load and parse all PDFs page-by-page
2. Normalise metadata (source stored as base filename only)
3. Split text into 1000-character chunks with 200-character overlap
4. Generate 384-dimensional embeddings locally via `all-MiniLM-L6-v2`
5. Persist the ChromaDB vector store to `chroma_db/`

---

## Running the Applications

### Option A — Streamlit Chat UI

Run locally:
```bash
streamlit run src/ui/app.py
```

Opens at `http://localhost:8501`. Provides a chat interface backed by the full RAG pipeline with inline `[Source N]` citations.

Alternatively, you can access the hosted version at the **Live Demo URL:** [https://5hjfvsgxrf5zsk9bvpn3mu.streamlit.app/](https://5hjfvsgxrf5zsk9bvpn3mu.streamlit.app/)

### Option B — MCP Server (standalone)

```bash
python -m src.mcp.server
```

The server pre-warms the embedding model and vector database on startup (~25–30 seconds), then listens on stdio for MCP JSON-RPC messages.

### Option C — MCP Server inside Claude Desktop

Add the following block to your `claude_desktop_config.json`:

- **Standard install:** `%APPDATA%\Roaming\Claude\claude_desktop_config.json`
- **Microsoft Store (UWP):** `%LOCALAPPDATA%\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Roaming\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "rag-pdf-server": {
      "command": "C:\\intern_task\\.venv\\Scripts\\python.exe",
      "args": ["-m", "src.mcp.server"],
      "cwd": "C:\\intern_task",
      "env": {
        "PYTHONPATH": "C:\\intern_task",
        "INFISICAL_CLIENT_ID": "your_infisical_client_id",
        "INFISICAL_CLIENT_SECRET": "your_infisical_client_secret",
        "INFISICAL_PROJECT_ID": "your_infisical_project_id"
      }
    }
  }
}
```

Restart Claude Desktop after saving. The server will appear as `rag-pdf-server` with two tools available.

---

## Compliance Verification

An automated verification script is included to validate the full specification:

```bash
# Basic checks (no API calls)
python test/verify_compliance.py

# Full checks including live Anthropic API call
python test/verify_compliance.py --live
```

The script validates:
1. ChromaDB loads and contains indexed documents
2. Exactly two MCP tools are registered (`search_documents`, `summarise_document`)
3. No hardcoded secrets exist in any source or test file
4. `search_documents_logic` returns correct results with source/page metadata

---

## Project Structure

```
intern_task/
├── data/
│   └── raw_pdfs/               # 10 source PDFs (input to ingestion)
├── chroma_db/                  # Persisted ChromaDB vector store
├── src/
│   ├── config/
│   │   └── settings.py         # Model config, paths, Infisical secret fetch
│   ├── loaders/
│   │   └── pdf_loader.py       # PyPDFLoader wrapper
│   ├── chunking/
│   │   └── text_splitter.py    # RecursiveCharacterTextSplitter wrapper
│   ├── embeddings/
│   │   └── embedding_generator.py  # SentenceTransformer (local, CPU-only)
│   ├── vectordb/
│   │   └── chroma_manager.py   # ChromaDB create / load
│   ├── retrieval/
│   │   └── retriever.py        # MMR + similarity search, deduplication
│   ├── generation/
│   │   └── claude_generator.py # Anthropic API call, strict grounding prompt
│   ├── rag/
│   │   └── rag_pipeline.py     # Orchestrates retrieval → context → generation
│   ├── mcp/
│   │   ├── server.py           # FastMCP server, 2 tools, pre-warm on startup
│   │   └── tools.py            # Tool logic implementations
│   └── ui/
│       └── app.py              # Streamlit chat interface
├── scripts/
│   └── ingest.py               # Alternative ingestion entry point
├── test/
│   ├── test1.py                # Infisical connection smoke test
│   └── verify_compliance.py    # Full spec compliance checker
├── build_db.py                 # Main ingestion entry point
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Known Limitations

- **MCP Server startup time:** The server pre-warms the `SentenceTransformer` model and ChromaDB on startup. This takes approximately 25–30 seconds before the first tool call can be served. Claude Desktop users should wait a few seconds after the app opens before using the tools.
- **First-run model download:** On the very first run, `SentenceTransformer` downloads `all-MiniLM-L6-v2` weights (~120 MB) from Hugging Face Hub and caches them locally. Subsequent runs use the local cache.
- **Model access:** The project uses `claude-haiku-4-5-20251001`. Ensure your Anthropic account has access to this model and has sufficient API credits.
- **Infisical connectivity:** An active internet connection is required at startup to authenticate with Infisical and fetch the `ANTHROPIC_API_KEY`.
- **CPU-only inference:** Embeddings run on CPU (`CUDA_VISIBLE_DEVICES` is forced empty to prevent GPU initialisation hangs in sandboxed environments).
