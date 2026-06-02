# PDF RAG MCP Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built over PDF documents using **LangChain**, **ChromaDB**, **Sentence Transformers**, **Claude (Anthropic API)**, **MCP (Model Context Protocol)**, and **Streamlit**.

This project enables users to query PDF documents, retrieve relevant context using vector search, and generate grounded answers using Claude while exposing functionality through an MCP server.

---

## Features

* PDF ingestion pipeline
* Recursive document chunking
* Vector similarity search using ChromaDB
* Semantic retrieval using Sentence Transformers
* Claude-powered grounded answer generation
* Source-aware responses (document + page references)
* MCP server exposing retrieval tools
* Streamlit chatbot UI
* Secret management using Infisical

---

## Tech Stack

### Retrieval Layer

* Python
* LangChain
* ChromaDB
* Sentence Transformers (`all-MiniLM-L6-v2`)

### Generation Layer

* Anthropic Claude API (Claude Haiku / Opus)

### UI Layer

* Streamlit

### Secrets Management

* Infisical

### Protocol Layer

* MCP (Model Context Protocol)

---

## Project Structure

```text
intern_task/
│
├── data/
│   └── raw_pdfs/                 # PDF files
│
├── scripts/
│   ├── ingest.py                 # PDF ingestion script
│   └── inspect_langchain_imports.py
│
├── src/
│   ├── chunking/
│   ├── config/
│   ├── embeddings/
│   ├── generation/
│   ├── loaders/
│   ├── mcp/
│   ├── rag/
│   ├── retrieval/
│   ├── ui/
│   └── vectordb/
│
├── chroma_db/                    # Generated vector DB (local only)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Architecture

```text
User Query
     │
     ▼
Retriever (ChromaDB + SentenceTransformer)
     │
     ▼
Top-K Relevant PDF Chunks
     │
     ▼
Claude Generator (Anthropic API)
     │
     ▼
Grounded Answer + Source References
```

---

## Retrieval Pipeline

1. PDFs are loaded using LangChain loaders.
2. Documents are chunked using `RecursiveCharacterTextSplitter`.
3. Chunks are embedded using:

```python
all-MiniLM-L6-v2
```

4. Embeddings are stored inside ChromaDB.
5. User query retrieves top relevant chunks.
6. Claude synthesizes the final grounded answer.

---

## Setup

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd intern_task
```

---

### 2. Create Virtual Environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Mac/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Secret Setup (Infisical)

Store your Anthropic API key inside Infisical.

Required secret:

```text
ANTHROPIC_API_KEY
```

Run project using:

```bash
infisical run -- streamlit run src/ui/app.py
```

---

## Ingest PDFs

Place PDFs inside:

```text
data/raw_pdfs/
```

Then run:

```bash
python scripts/ingest.py
```

This will:

* Load PDFs
* Chunk documents
* Generate embeddings
* Store vectors in ChromaDB

---

## Run Streamlit UI

```bash
infisical run -- streamlit run src/ui/app.py
```

Open:

```text
http://localhost:8501
```

Example query:

```text
What is fine tuning and how is it different from RAG?
```

---

## MCP Server

This project exposes tools through MCP.

Available tools:

### search_documents(query, k)

Search relevant chunks from indexed PDFs.

Example:

```text
search_documents(
    "fine tuning",
    5
)
```

### summarise_document(doc_id)

Summarise a document using retrieved context.

---

## Run MCP Server

```bash
infisical run -- python -m src.mcp.server
```

---

## Claude Desktop Integration

Create or edit:

```text
claude_desktop_config.json
```

Example configuration:

```json
{
  "mcpServers": {
    "rag-pdf-server": {
      "command": "C:\\intern_task\\.venv\\Scripts\\python.exe",
      "args": [
        "-m",
        "src.mcp.server"
      ],
      "cwd": "C:\\intern_task"
    }
  }
}
```

Restart Claude Desktop.

Example prompt:

```text
Use the PDF search tool and tell me what the documents say about fine tuning.
```

---

## Example Output

Question:

```text
What is fine tuning and how is it different from RAG?
```

Output:

* Explanation of fine tuning
* Fine-tuning techniques (LoRA, SFT, RLHF, adapters, prefix tuning)
* Difference between fine-tuning and RAG
* Source references with page numbers

---

## Known Limitations

* Retrieval quality depends on chunking quality
* Very small contexts may reduce answer completeness
* Claude Desktop setup is OS dependent
* Large PDF collections increase retrieval latency

---

## Future Improvements

* Reranking
* Hybrid retrieval (BM25 + vector search)
* Multi-document summarization
* Citation highlighting
* Deployment support

---

## Author

Mayank Goel
