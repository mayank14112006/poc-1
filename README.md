# PDF RAG Chatbot with MCP, Claude & Streamlit

> Production-style Retrieval-Augmented Generation (RAG) system for querying PDF documents using semantic search, vector databases, Claude, and the Model Context Protocol (MCP).

---

# Overview

This project implements an end-to-end **Retrieval-Augmented Generation (RAG)** pipeline capable of:

* Ingesting PDF documents
* Creating semantic embeddings
* Storing vectors in ChromaDB
* Retrieving relevant document chunks
* Generating grounded responses using Claude
* Providing source citations with page references
* Exposing retrieval capabilities through MCP tools
* Serving an interactive Streamlit interface

The goal is to provide accurate, source-grounded answers while minimizing hallucinations by ensuring the LLM always receives relevant context retrieved from indexed documents.

---

# Architecture

```text
                         ┌────────────────────┐
                         │      PDF Files     │
                         └─────────┬──────────┘
                                   │
                                   ▼
                     ┌─────────────────────────┐
                     │      PDF Loader         │
                     └─────────┬───────────────┘
                               │
                               ▼
                 ┌──────────────────────────────┐
                 │ Recursive Character Splitter │
                 └──────────┬───────────────────┘
                            │
                            ▼
               ┌───────────────────────────────┐
               │ Sentence Transformer Encoder  │
               │   all-MiniLM-L6-v2            │
               └──────────┬────────────────────┘
                          │
                          ▼
                  ┌─────────────────┐
                  │    ChromaDB     │
                  │ Vector Database │
                  └────────┬────────┘
                           │
                User Query │
                           ▼
               ┌─────────────────────┐
               │ Similarity Search   │
               │      Top-K Docs     │
               └─────────┬───────────┘
                         │
                         ▼
              ┌────────────────────────┐
              │ Claude (Anthropic API) │
              └──────────┬─────────────┘
                         │
                         ▼
            Answer + Source Citations
```

---

# Why RAG?

Traditional LLM workflow:

```text
Question
   ↓
LLM
   ↓
Answer
```

Problems:

* Hallucinations
* No access to private PDFs
* Difficult source verification
* Knowledge becomes stale

RAG solves this by:

```text
Question
   ↓
Retriever
   ↓
Relevant Context
   ↓
LLM
   ↓
Grounded Answer
```

Benefits:

* Reduced hallucinations
* Source-backed answers
* Private document retrieval
* Updatable knowledge without retraining

---

# Why ChromaDB?

ChromaDB was selected because:

* Native LangChain integration
* Persistent local storage
* Lightweight deployment
* Metadata support
* Simple developer experience

### Alternatives Considered

| Vector Database | Reason Not Selected                      |
| --------------- | ---------------------------------------- |
| Pinecone        | External dependency                      |
| Weaviate        | Additional infrastructure overhead       |
| Qdrant          | More operational complexity than needed  |
| FAISS           | Limited persistence and metadata support |

---

# Why Sentence Transformers Instead of Voyage AI?

The project initially used Voyage AI embeddings.

After evaluation, the embedding layer was migrated to:

```text
all-MiniLM-L6-v2
```

via Sentence Transformers.

### Benefits

#### Fully Local Embeddings

```text
PDF Chunk
   ↓
Local Model
   ↓
Embedding
```

No embedding API calls required.

#### No Rate Limits

External embedding APIs may enforce:

* Requests per minute (RPM)
* Tokens per minute (TPM)
* Billing restrictions

Local embeddings remove these limitations.

#### Easier Deployment

The application can:

* Build vector databases automatically
* Run without embedding API credentials
* Deploy without external embedding services

#### Strong Retrieval Performance

For PDF-based RAG systems:

```text
all-MiniLM-L6-v2
```

offers an excellent balance of:

* Accuracy
* Speed
* Resource efficiency

---

# Why Claude?

Claude was chosen because of:

* Strong document reasoning
* Excellent long-context performance
* Reliable grounded generation
* High-quality responses in RAG systems

Current model:

```text
Claude Haiku
```

Upgradeable to:

* Claude Sonnet
* Claude Opus

without architecture changes.

---

# MCP Integration

This project supports the **Model Context Protocol (MCP)**.

MCP is an open protocol that allows AI assistants to interact with external tools through a standardized interface.

---

## Traditional Tool Integration

```text
AI Assistant
     │
     ├── Custom API A
     ├── Custom API B
     ├── Custom API C
```

Every integration requires custom implementation.

---

## MCP-Based Integration

```text
AI Assistant
       │
       ▼
      MCP
       │
       ▼
   RAG Server
```

Benefits:

* Standardized tooling
* Easier integrations
* Reusable architecture
* Claude Desktop compatibility

---

# MCP Architecture

```text
Claude Desktop
       │
       ▼
MCP Client
       │
       ▼
MCP Server
       │
       ▼
Retriever
       │
       ▼
ChromaDB
```

The assistant can invoke retrieval tools directly and use retrieved context during conversations.

---

# Project Structure

```text
intern_task/
│
├── data/
│   └── raw_pdfs/
│
├── chroma_db/
│
├── src/
│
│   ├── chunking/
│   │   └── text_splitter.py
│
│   ├── config/
│   │   └── settings.py
│
│   ├── embeddings/
│   │   └── embedding_generator.py
│
│   ├── generation/
│   │   └── claude_generator.py
│
│   ├── loaders/
│   │   └── pdf_loader.py
│
│   ├── mcp/
│   │   └── server.py
│
│   ├── rag/
│   │   └── rag_pipeline.py
│
│   ├── retrieval/
│   │   └── retriever.py
│
│   ├── ui/
│   │   └── app.py
│
│   ├── utils/
│   │   └── bootstrap.py
│
│   └── vectordb/
│       └── chroma_manager.py
│
├── requirements.txt
└── README.md
```

---

# Retrieval Pipeline

## Step 1 — Load PDFs

PDF documents are loaded from:

```text
data/raw_pdfs/
```

---

## Step 2 — Chunk Documents

Uses:

```python
RecursiveCharacterTextSplitter
```

Configuration:

```text
Chunk Size: 1000
Overlap: 200
```

Purpose:

* Preserve semantic meaning
* Maintain contextual continuity
* Improve retrieval accuracy

---

## Step 3 — Generate Embeddings

Model:

```text
all-MiniLM-L6-v2
```

Each chunk is converted into a dense vector representation.

---

## Step 4 — Store in ChromaDB

Stored information includes:

* Chunk text
* Source file
* Page number
* Metadata
* Embedding vectors

---

## Step 5 — Retrieve Top-K Chunks

Example query:

```text
What is fine tuning?
```

↓

Vector similarity search

↓

Top-K relevant chunks

---

## Step 6 — Generate Grounded Answer

Claude receives:

```text
Question
+
Retrieved Context
```

and generates:

```text
Answer
+
Source References
```

---

# Setup

## Clone Repository

```bash
git clone <repository-url>
cd intern_task
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Secrets Configuration

Secrets are managed using Infisical.

Required Secret:

```text
ANTHROPIC_API_KEY
```

Required Environment Variables:

```text
INFISICAL_CLIENT_ID
INFISICAL_CLIENT_SECRET
INFISICAL_PROJECT_ID
```

---

# Running the Application

```bash
streamlit run src/ui/app.py
```

Open:

```text
http://localhost:8501
```

---

# Example Query

```text
What is fine tuning and how does it differ from RAG?
```

Example Output:

* Fine-tuning explanation
* RAG explanation
* Comparative analysis
* Source citations with page references

---

# Engineering Decisions

| Component          | Choice                         |
| ------------------ | ------------------------------ |
| Embeddings         | Sentence Transformers          |
| Embedding Model    | all-MiniLM-L6-v2               |
| Vector Database    | ChromaDB                       |
| LLM                | Claude                         |
| UI                 | Streamlit                      |
| Secrets Management | Infisical                      |
| Protocol           | MCP                            |
| Chunking Strategy  | RecursiveCharacterTextSplitter |
| Retrieval Method   | Dense Vector Search            |

---

# Known Limitations

* Retrieval quality depends on chunk quality
* Very large PDF collections increase indexing time
* No reranking layer yet
* Dense retrieval only (no hybrid search)

---

# Author

**Mayank Goel**

AI Engineering • Retrieval-Augmented Generation • MCP Integrations • LLM Applications
