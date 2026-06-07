import subprocess
import json
import time
import sys

def main():
    print("Launching MCP server subprocess...")
    proc = subprocess.Popen(
        ["C:\\intern_task\\.venv\\Scripts\\python.exe", "-m", "src.mcp.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1  # line buffered
    )

    # Helper to send a message
    def send(msg):
        line = json.dumps(msg) + "\n"
        proc.stdin.write(line)
        proc.stdin.flush()
        print(f"--> Sent: {line.strip()}")

    # Helper to read a message
    def read():
        line = proc.stdout.readline()
        print(f"<-- Recv: {line.strip()}")
        return json.loads(line) if line else None

    # Thread to print stderr in real-time
    import threading
    def read_stderr():
        for line in proc.stderr:
            print(f"[stderr] {line.strip()}", flush=True)
    threading.Thread(target=read_stderr, daemon=True).start()

    time.sleep(1) # wait for startup

    # 1. Initialize
    send({
        "jsonrpc": "2.0",
        "id": 0,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0.0"}
        }
    })
    
    init_res = read()
    
    # 2. Initialized notification
    send({
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    })

    # 3. Call search_documents tool immediately (will block on lock until pre-warm finishes)
    print("Calling search_documents tool...")
    t0 = time.time()
    send({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "search_documents",
            "arguments": {"query": "attention", "k": 2}
        }
    })

    res = read()
    duration = time.time() - t0
    print(f"Tool call finished in {duration:.2f} seconds.")

    # Terminate process
    proc.terminate()
    proc.wait()

if __name__ == "__main__":
    main()
