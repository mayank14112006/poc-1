import sys
import os
import argparse
import asyncio
from pathlib import Path

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))

def test_chroma_load():
    print("Checking if Chroma DB loads successfully...")
    try:
        from src.embeddings.embedding_generator import EmbeddingGenerator
        from src.vectordb.chroma_manager import ChromaManager
        
        embedder = EmbeddingGenerator().get_model()
        manager = ChromaManager(embedder)
        vectorstore = manager.load_vector_store()
        
        # Test document lookup
        docs = vectorstore.get(limit=1)
        if docs and docs.get("ids"):
            print(f"[OK] Chroma DB loaded successfully with {len(docs['ids'])} sample records.")
            return True
        else:
            print("[WARN] Chroma DB is empty or could not retrieve documents.")
            return False
    except Exception as e:
        print(f"[ERROR] Failed to load Chroma DB: {e}")
        return False

def test_mcp_tools_registration():
    print("Checking MCP server tools registration...")
    try:
        # Import mcp server instance directly
        from src.mcp.server import mcp
        
        # Run async list_tools in the event loop
        tools = asyncio.run(mcp.list_tools())
        tool_names = [tool.name for tool in tools]
        print(f"Registered tools: {tool_names}")
        
        expected_tools = {"search_documents", "summarise_document"}
        if set(tool_names) == expected_tools:
            print("[OK] Exactly two tools registered successfully.")
            return True
        else:
            print(f"[ERROR] Registered tools do not match specification. Expected exactly {expected_tools}, got {tool_names}.")
            return False
    except Exception as e:
        print(f"[ERROR] Failed to inspect FastMCP tools: {e}")
        return False

def check_no_hardcoded_secrets():
    print("Scanning codebase for hardcoded secrets...")
    import re
    
    suspicious_patterns = [
        re.compile(r'client_secret\s*=\s*["\'](?!your_)[a-fA-F0-9-]{20,}["\']'),
        re.compile(r'client_id\s*=\s*["\'](?!your_)[a-fA-F0-9-]{20,}["\']'),
        re.compile(r'api_key\s*=\s*["\'](?!your_)[a-fA-F0-9-]{20,}["\']'),
    ]
    
    has_secret = False
    for path in root_dir.glob("src/**/*.py"):
        content = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in suspicious_patterns:
            if pattern.search(content):
                print(f"[ERROR] Suspicious secret pattern found in: {path}")
                has_secret = True
                
    for path in root_dir.glob("test/**/*.py"):
        if path.name == "verify_compliance.py":
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in suspicious_patterns:
            if pattern.search(content):
                print(f"[ERROR] Suspicious secret pattern found in: {path}")
                has_secret = True

    if not has_secret:
        print("[OK] No hardcoded secrets detected in source/test code.")
        return True
    return False

def test_rag_logic(live=False):
    print("Testing document retrieval and summary logic...")
    try:
        from src.mcp.tools import search_documents_logic, summarise_document_logic
        
        # Test document search (does not require LLM API call)
        results = search_documents_logic(query="attention", k=2)
        print(f"Search results count: {len(results)}")
        if results:
            print(f"[OK] Search documents logic retrieved: {results[0].get('source')} (Page {results[0].get('page')})")
        else:
            print("[WARN] Search documents logic returned no results (may need ingestion).")
            
        # Test summarisation (requires LLM only if live is True)
        if live:
            print("Running live summarisation test (this will call the Anthropic API)...")
            from src.embeddings.embedding_generator import EmbeddingGenerator
            from src.vectordb.chroma_manager import ChromaManager
            embedder = EmbeddingGenerator().get_model()
            manager = ChromaManager(embedder)
            vectorstore = manager.load_vector_store()
            docs = vectorstore.get(limit=1)
            
            if docs and docs.get("metadatas"):
                doc_id = docs["metadatas"][0].get("source")
                if doc_id:
                    print(f"Summarising doc_id: {doc_id}")
                    summary = summarise_document_logic(doc_id=doc_id)
                    print(f"[OK] Live Summary result: {str(summary)[:200]}...")
                else:
                    print("[WARN] Could not find a source metadata field to test live summarisation.")
            else:
                print("[WARN] Chroma DB is empty, cannot perform live summarisation test.")
        else:
            print("[INFO] Skipping live LLM summarisation test (run with --live to execute).")
            
        return True
    except Exception as e:
        print(f"[ERROR] Failed during RAG logic testing: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Strict Compliance Verification Script")
    parser.add_argument("--live", action="store_true", help="Execute live API tests using Anthropic keys")
    args = parser.parse_args()
    
    print("==================================================")
    print("Running POC 01 Spec Compliance Validation")
    print("==================================================")
    
    success = True
    success &= test_chroma_load()
    print("--------------------------------------------------")
    success &= test_mcp_tools_registration()
    print("--------------------------------------------------")
    success &= check_no_hardcoded_secrets()
    print("--------------------------------------------------")
    success &= test_rag_logic(live=args.live)
    print("==================================================")
    
    if success:
        print("ALL SPECIFICATION COMPLIANCE CHECKS PASSED!")
        sys.exit(0)
    else:
        print("SOME SPECIFICATION COMPLIANCE CHECKS FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    main()
