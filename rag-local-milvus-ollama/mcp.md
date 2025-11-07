# RAG Local MCP Server Integration

A Model Context Protocol (MCP) server that provides access to the local RAG (Retrieval-Augmented Generation) system functionality using Milvus and Ollama.

## Overview

This MCP server exposes the local RAG query functionality through the MCP protocol, allowing Claude and other MCP clients to query your local vector database.

## Features

- **allycat_local_query tool**: Query the local RAG system with natural language questions
- Uses the same configuration as the local RAG system (from `my_config.py`)
- Connects to the local Milvus vector database
- Uses local Ollama models for embeddings and LLM
- Provides processing time and metadata in responses

## Prerequisites

1. **Python Environment**: Ensure you have Python 3.8+ installed
2. **Dependencies**: Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. **Milvus Database**: Ensure your local Milvus database is populated with data
4. **Ollama**: Ensure Ollama is installed and running locally
5. **Models**: Ensure the required embedding and LLM models are downloaded

## Configuration

The server uses the same configuration as the main system from `my_config.py`:
- Embedding model and settings
- Local vector database connection
- Ollama LLM model configuration
- Collection name and other parameters

Make sure your `.env` file is properly configured with the correct model names and paths.

## VSCode Integration

### Method 1: Using VSCode Settings (Recommended)

1. Open VSCode Settings (Ctrl/Cmd + ,)
2. Search for "MCP" or navigate to Extensions → MCP
3. Add a new MCP server configuration with the following settings:

```json
{
  "mcp.servers": {
    "allycat-local-rag": {
      "command": "python",
      "args": ["${workspaceFolder}/rag-local-milvus-ollama/mcp_server.py"],
      "cwd": "${workspaceFolder}/rag-local-milvus-ollama"
    }
  }
}
```

### Method 2: Using Workspace Settings

1. Open your workspace settings file (`.vscode/settings.json`)
2. Add the MCP server configuration:

```json
{
  "mcp.servers": {
    "allycat-local-rag": {
      "command": "python",
      "args": ["./rag-local-milvus-ollama/mcp_server.py"],
      "cwd": "./rag-local-milvus-ollama"
    }
  }
}
```

### Method 3: Direct Command Line

You can also run the server directly from the terminal:

```bash
cd rag-local-milvus-ollama
python mcp_server.py
```

## Usage in VSCode

Once configured, the MCP server will be available in VSCode:

1. Open a chat window with an MCP-enabled extension (like Claude Chat)
2. The server should automatically connect
3. You can use the `allycat_local_query` tool to ask questions:

Example queries:
- "What is AI Alliance?"
- "What are the main focus areas of AI Alliance?"
- "What are some AI alliance projects?"
- "What are the upcoming events?"
- "How do I join the AI Alliance?"

## Available Tools

- **allycat_local_query**: Query the local RAG system
  - Input: `query` (string) - The question to ask
  - Output: Response from the RAG system with metadata and processing time

## Troubleshooting

### Common Issues

1. **Server not connecting**: 
   - Ensure all dependencies are installed
   - Check that Milvus database is accessible
   - Verify Ollama is running and models are downloaded

2. **Permission errors**:
   - Ensure the Python script has execute permissions
   - Check that the working directory is correct

3. **Model loading issues**:
   - Verify model names in `my_config.py`
   - Ensure models are downloaded via Ollama

### Logs

The server outputs logs to help debug issues. Look for:
- ✅ Success messages for initialization
- ❌ Error messages for failures
- Processing time and metadata in responses

## Development

To test the server locally:

```bash
cd rag-local-milvus-ollama
python mcp_server.py
```

The server will start and listen for MCP connections on stdin/stdout.

## Testing the MCP Server

### Interactive Testing

Run the interactive test client that takes user input and sends it to a running MCP server:

First, start the MCP server in one terminal:
```bash
cd rag-local-milvus-ollama
python mcp_server.py
```

Then in another terminal, run the interactive client:
```bash
cd rag-local-milvus-ollama
python mcp_test.py
```

Or use pipe redirection for a single-terminal experience:
```bash
cd rag-local-milvus-ollama
python mcp_test.py | python mcp_server.py
```

This will start an interactive session where you can:
- Type questions to query the RAG system
- See properly formatted responses from the MCP server
- Type `/quit` to exit the session

The client handles:
- Initializing the MCP connection with proper JSON-RPC format
- Sending your queries to the `allycat_local_query` tool
- Graceful session management

### Manual JSON-RPC Testing

If you prefer manual testing with JSON requests:

### Manual Testing

1. **Start the server:**
   ```bash
   cd rag-local-milvus-ollama
   python mcp_server.py
   ```

2. **Send initialization request:**
   ```json
   {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"capabilities": {}, "clientInfo": {"name": "test", "version": "1.0.0"}}}
   ```

3. **List available tools:**
   ```json
   {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
   ```

4. **Call the query tool:**
   ```json
   {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "allycat_local_query", "arguments": {"query": "What is this website about?"}}}
   ```

**Note:** For manual testing, you'll need to send each JSON request as a single line followed by pressing Enter. The server responds with JSON-RPC formatted responses.

### Expected Responses

- **Initialization**: Should return server capabilities and confirmation
- **Tool listing**: Should show `allycat_local_query` tool with its schema
- **Tool call**: Should return the RAG query response with metadata and timing info

Press Ctrl+C to exit the server when testing manually.

## Dependencies

The server requires the same dependencies as the main RAG system:
- llama-index with HuggingFace embeddings
- Milvus vector store
- LiteLLM for Ollama integration
- python-dotenv
- mcp-server package

Install with:
```bash
pip install -r requirements.txt
