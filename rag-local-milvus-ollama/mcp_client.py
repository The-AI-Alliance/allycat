#!/usr/bin/env python3
"""
Interactive MCP client for testing the local RAG server
Takes user input and sends it to an already running MCP server until user types /quit
"""

import json
import sys
import select
import threading
import queue
import time

def read_responses(response_queue):
    """Read JSON-RPC responses from stdin in a separate thread"""
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if line:
                response_queue.put(line)
        except Exception as e:
            print(f"\n❌ Error reading response: {e}", file=sys.stderr)
            break

def main():
    """Main entry point - sends JSON-RPC requests to stdout and reads responses from stdin"""
    
    # Print all header information to stderr to avoid interfering with JSON-RPC communication
    print("🚀 MCP RAG Server Interactive Client", file=sys.stderr)
    print("=" * 50, file=sys.stderr)
    print("ℹ️  Usage: Pipe this script to a running MCP server:", file=sys.stderr)
    print("   python mcp_test.py | python rag-local-milvus-ollama/mcp_server.py", file=sys.stderr)
    print("", file=sys.stderr)
    print("   Or use a bidirectional pipe:", file=sys.stderr)
    print("   mkfifo /tmp/mcp_pipe", file=sys.stderr)
    print("   python rag-local-milvus-ollama/mcp_server.py < /tmp/mcp_pipe > /tmp/mcp_pipe &", file=sys.stderr)
    print("   python mcp_test.py > /tmp/mcp_pipe", file=sys.stderr)
    print("=" * 50, file=sys.stderr)
    
    # Queue for responses
    response_queue = queue.Queue()
    
    # Start response reader thread
    response_thread = threading.Thread(target=read_responses, args=(response_queue,), daemon=True)
    response_thread.start()
    
    try:
        # Initialize MCP connection with proper protocol version
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-06-11",
                "capabilities": {},
                "clientInfo": {"name": "interactive-client", "version": "1.0.0"}
            }
        }
        
        print(json.dumps(init_request), flush=True)
        
        # Small delay to let server initialize
        time.sleep(0.1)
        
        # List available tools
        tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        }
        
        print(json.dumps(tools_request), flush=True)
        
        # Small delay to let server process
        time.sleep(0.1)
        
        print("\n" + "=" * 50, file=sys.stderr)
        print("💬 Interactive RAG Query Interface", file=sys.stderr)
        print("📝 Type your questions and press Enter to send", file=sys.stderr)
        print("👋 Type '/quit' to exit", file=sys.stderr)
        print("=" * 50, file=sys.stderr)
        
        # Interactive loop
        request_id = 3
        while True:
            try:
                # Get user input from stderr to avoid mixing with JSON-RPC traffic
                print("\n❓ Your question: ", file=sys.stderr, end="")
                user_input = input().strip()
                
                # Check for quit command
                if user_input.lower() in ['/quit', '/exit', 'quit', 'exit']:
                    print("👋 Goodbye!", file=sys.stderr)
                    break
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Send query to MCP server via stdout
                query_request = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "tools/call",
                    "params": {
                        "name": "allycat_local_query",
                        "arguments": {"query": user_input}
                    }
                }
                
                print(json.dumps(query_request), flush=True)
                
                # Wait for and display response
                print("\n🔄 Processing query...", file=sys.stderr)
                start_time = time.time()
                response_received = False
                while time.time() - start_time < 30 and not response_received:  # 30 second timeout
                    try:
                        response = response_queue.get(timeout=0.1)
                        try:
                            response_data = json.loads(response)
                            if response_data.get("id") == request_id:
                                print(f"\n📥 Query Response:", file=sys.stderr)
                                if "result" in response_data:
                                    # Extract the text content from the result
                                    result = response_data["result"]
                                    if isinstance(result, list) and len(result) > 0:
                                        content = result[0]
                                        if isinstance(content, dict) and content.get("type") == "text":
                                            print(f"\n📝 Answer:\n{content.get('text', 'No text content')}", file=sys.stderr)
                                        else:
                                            print(f"\n📝 Answer:\n{result}", file=sys.stderr)
                                    else:
                                        print(f"\n📝 Answer:\n{result}", file=sys.stderr)
                                elif "error" in response_data:
                                    print(f"\n❌ Error: {response_data['error']}", file=sys.stderr)
                                else:
                                    print(f"\n📥 Raw Response:\n{response}", file=sys.stderr)
                                response_received = True
                            else:
                                # This is not our response, put it back
                                response_queue.put(response)
                                time.sleep(0.01)
                        except json.JSONDecodeError:
                            print(f"\n📥 Non-JSON Response:\n{response}", file=sys.stderr)
                            response_received = True
                    except queue.Empty:
                        continue
                
                if not response_received:
                    print("\n⏰ Timeout waiting for response", file=sys.stderr)
                
                request_id += 1
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted! Goodbye!", file=sys.stderr)
                break
            except EOFError:
                print("\n\n👋 Goodbye!", file=sys.stderr)
                break
            except Exception as e:
                print(f"\n❌ Error processing input: {e}", file=sys.stderr)
        
        print("✅ Interactive session ended successfully!", file=sys.stderr)
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
