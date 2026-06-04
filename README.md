# PDF RAG Chatbot & MCP Server (POC 01)

A production-grade, standards-compliant Retrieval-Augmented Generation (RAG) chatbot and Model Context Protocol (MCP) server built over a curated collection of 10 seminal machine learning and LLM research papers.

This project implements the foundational layer (**POC 01**) of the AI Engineering Intern assessment.

---

## Architecture Overview

The system operates in two phases: offline indexing and runtime query processing.

```mermaid
flowchart TD
    subgraph Ingest & Index (Offline)
        A[10 raw PDFs] --> B[PDFLoader PyPDF]
        B --> C[ChunkProcessor RecursiveCharacterTextSplitter]
        C --> D[Normalize Metadata: Source Filename Only]
        D --> E[EmbeddingGenerator SentenceTransformer]
        E --> F[Chroma Vector Store chroma_db]
    end

    subgraph Query & Synthesis (Runtime)
        G[MCP Client / Streamlit App] -->|User Question| H[RAGPipeline]
        H -->|1. MMR Search fetch_k=30| F
        H -->|2. Similarity Search k=12| F
        F -->|Return Raw Chunks| H
        H -->|3. Deduplicate Chunks| I[Context Builder]
        I -->|4. Sorted by Page Number| J[Claude 3 Haiku API]
        J -->|5. Synthesized Answer + Sources| G
    end
```

---

## Features & Improvements Over Initial Build

1.  **Environment-Agnostic Metadata (Bug Fix):** Standardized metadata to store only base filenames (`Attention Is All You Need.pdf`) rather than absolute local directories (`C:\user\path\...`). This makes the index portable and ensures search/summarization works seamlessly in the cloud.
2.  **Native Summarization Querying (Performance Fix):** Rewrote the `summarise_document` tool to use ChromaDB's native filtering (`where={"source": doc_id}`) instead of scanning all 3,000+ chunks in-memory.
3.  **Chronological Summary Generation (Logical Fix):** Sorted chunks by page numbers before passing them to the generator. This prevents jumbled summaries and produces cohesive output.
4.  **Lazy Secrets Initialization (Architecture Fix):** Decoupled Infisical keys so that offline database building and testing can run without throwing Universal Auth environment errors.
5.  **Premium UI (Aesthetics Fix):** Upgraded the Streamlit app with an interactive Indigo/Purple dark-themed dashboard, Suggested Question triggers, dynamic sidebar database metrics, and clickable citation badges.

---

## Technical Stack

*   **RAG Orchestration:** LangChain (`langchain-chroma`, `langchain-community`, `langchain-text-splitters`)
*   **Vector Database:** ChromaDB (Local SQLite file-based)
*   **Embeddings:** Local `all-MiniLM-L6-v2` Sentence Transformer (runs locally, 0 cost)
*   **Synthesis LLM:** Claude 3 Haiku via official Anthropic Python SDK
*   **Secrets Manager:** Infisical Universal Auth SDK
*   **User Interface:** Streamlit (v1.57.0)
*   **MCP Protocol:** FastMCP (Anthropic MCP Python SDK)

---

## Setup & Ingest Pipeline

### 1. Prerequisites
Ensure you have Python 3.10+ installed and a virtual environment activated:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables
You must set your Infisical client credentials in the environment to enable LLM synthesis:
```powershell
# PowerShell (Windows)
$env:INFISICAL_CLIENT_ID="25050f8a-32df-4ccf-888d-da3930cfc033"
$env:INFISICAL_CLIENT_SECRET="f3ad9988d2af4d8f1b81764c11bccb0acaca1819757a7e65fd4d396255b0039a"
$env:INFISICAL_PROJECT_ID="71e13dcb-270b-475d-a565-49885b7ec22b"
```

### 3. Rebuild Database (Optional)
The pre-built vector store is located in `chroma_db/`. To re-index the raw PDFs inside `data/raw_pdfs/` from scratch, run:
```bash
python build_db.py
```

---

## Running the Applications

### 1. Launch Streamlit Chat UI
Run the local web dashboard:
```bash
streamlit run src/ui/app.py
```

### 2. Launch MCP Server
To run the server in developer/standalone mode:
```bash
python src/mcp/server.py
```

To configure it inside **Claude Desktop**, add the server definition to your `claude_desktop_config.json` (located at `%APPDATA%\Roaming\Claude\claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "rag-mcp-server": {
      "command": "python",
      "args": [
        "-u",
        "C:/intern_task/src/mcp/server.py"
      ],
      "env": {
        "INFISICAL_CLIENT_ID": "25050f8a-32df-4ccf-888d-da3930cfc033",
        "INFISICAL_CLIENT_SECRET": "f3ad9988d2af4d8f1b81764c11bccb0acaca1819757a7e65fd4d396255b0039a",
        "INFISICAL_PROJECT_ID": "71e13dcb-270b-475d-a565-49885b7ec22b"
      }
    }
  }
}
```

---

## Known Issues & Validation Notes

*   **API Model Name Deprecation (Resolved):** The original project specified a placeholder model `claude-haiku-4-5-20251001` which causes a `404 Not Found` API exception. We updated it to the stable `claude-3-haiku-20240307` in `src/config/settings.py` for standard access.
*   **Hugging Face Rate Limits:** During first run, the local SentenceTransformer downloads weights from HF Hub. You may see a warning about unauthenticated requests, but the weights will download successfully and cache locally.
*   **Key Billing Check:** If you receive a `404 model: claude-3-haiku-20240307` error, it indicates the active key in your Infisical vault is tied to a workspace with an expired or zero-credit balance. Update the `ANTHROPIC_API_KEY` inside your Infisical dashboard under the `dev` environment to restore access.
