import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from mcp.server.fastmcp import FastMCP

from src.mcp.tools import (
    search_documents_logic,
    summarise_document_logic,
    ask_documents_logic
)


mcp = FastMCP(
    "rag-mcp-server"
)


@mcp.tool()
def search_documents(
    query: str,
    k: int = 5
):
    return search_documents_logic(
        query,
        k
    )


@mcp.tool()
def summarise_document(
    doc_id: str
):
    return summarise_document_logic(
        doc_id
    )


@mcp.tool()
def ask_documents(
    query: str
):
    return ask_documents_logic(
        query
    )


if __name__ == "__main__":
    mcp.run()
