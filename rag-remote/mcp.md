# MCP Server for RAG Query System

This document provides instructions for testing and using the MCP (Model Context Protocol) server that exposes the RAG query functionality.

## Overview

The `mcp_server.py` file implements an MCP server that allows querying the RAG (Retrieval-Augmented Generation) system through the Model Context Protocol. This enables integration with MCP-compatible clients like Claude Desktop, VS Code, and command-line tools.

## Prerequisites

1. **Install Dependencies**

   First, make sure all dependencies are installed:

   ```bash
   uv sync
   ```

   Or if you need to install the MCP package specifically:

   ```bash
   uv add mcp
   ```

2. **Prepare the RAG System**

   Before running the MCP server, ensure you have:
   - Crawled and processed documents (run `1_crawl_site.py` and `2_process_files.py`)
   - Created the vector database (run `3_save_to_vector_db.py`)
   - Set up your `.env` file with the necessary configuration

3. **Activate Python Environment**

   ```bash
   source .venv/bin/activate
   ```

## Testing the MCP Server

### Method 1: Command Line Testing with mcp_client.py (Recommended)

The `mcp_client.py` client provides an easy way to test the MCP server from the command line. It automatically starts the server and handles all MCP protocol communication.

1. **Interactive Mode** (recommended for testing):

   ```bash
   python mcp_client.py
   ```

   This will:
   - Start the MCP server automatically
   - List available tools
   - Start an interactive prompt where you can enter queries
   - Display formatted responses
   - Allow you to quit with 'q', 'quit', or 'exit'

   Example session:
   ```
   ======================================================================
   MCP Test Client for Allycat RAG System
   ======================================================================

   Connecting to MCP server...
   ✅ Connected to MCP server

   Available tools:

     Tool: allycat_rag_remote
     Description: Query the Allycat RAG (Retrieval-Augmented Generation) system...

   ======================================================================
   Interactive Query Mode
   ======================================================================
   Enter your questions below. Type 'quit', 'exit', or 'q' to exit.

   Your question: What is AI Alliance?
   ⏳ Processing query...

   ----------------------------------------------------------------------
   📋 Response:
   ----------------------------------------------------------------------
   **Answer:**
   [Response from the RAG system]

   **Source Documents:** 3 documents retrieved
   ...
   ----------------------------------------------------------------------

   Your question: q
   👋 Goodbye!
   ```

2. **Single Query Mode** (useful for scripting):

   ```bash
   python mcp_client.py "What is AI Alliance?"
   ```

   This will execute a single query and display the result, then exit. Useful for:
   - Testing from shell scripts
   - Automated testing
   - Quick one-off queries

### Method 2: Testing with MCP Inspector

The MCP Inspector provides a web-based interface for testing MCP servers.

**Prerequisites**: You need Node.js installed to use the MCP Inspector.

1. **Install MCP Inspector** (optional - can also use npx):

   You can either install it globally or use `npx` to run it directly:

   **Option A: Install globally** (recommended if you'll use it frequently):
   ```bash
   npm install -g @modelcontextprotocol/inspector
   ```

   **Option B: Use npx** (no installation needed):
   ```bash
   # npx will download and run it automatically
   npx @modelcontextprotocol/inspector python mcp_server.py
   ```

2. **Run the MCP Inspector**:

   If installed globally:
   ```bash
   mcp-inspector python mcp_server.py
   ```

   Or with npx (one-time use):
   ```bash
   npx @modelcontextprotocol/inspector python mcp_server.py
   ```

   This will:
   - Start the MCP server
   - Launch a web interface (usually at http://localhost:5173)
   - Allow you to test the server interactively

3. **Using the Inspector**:
   - Open the web interface in your browser (it usually opens automatically)
   - Click on "Tools" to see available tools
   - Select the `allycat_rag_remote` tool
   - Enter a query like "What is AI Alliance?"
   - Click "Run" to execute the query

## Integration with Claude Desktop

To use this MCP server with Claude Desktop:

1. **Locate Claude Desktop Configuration**

   The configuration file location depends on your OS:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
   - Linux: `~/.config/Claude/claude_desktop_config.json`

2. **Edit the Configuration File**

   Add your MCP server to the `mcpServers` section. You can use either absolute paths or relative paths:

   **Option A: Using Absolute Paths** (recommended for system-wide setup)

   ```json
   {
     "mcpServers": {
       "rag-query": {
         "command": "uv",
         "args": ["run", "python", "mcp_server.py"],
         "cwd": "/absolute/path/to/rag-remote",
         "env": {
           "WORKSPACE_DIR": "/absolute/path/to/rag-remote/workspace"
         }
       }
     }
   }
   ```

   **Option B: Using Relative Paths** (for workspace-specific setup)

   If you're working within a project workspace, you can use relative paths:

   ```json
   {
     "mcpServers": {
       "rag-query": {
         "command": "uv",
         "args": ["run", "python", "mcp_server.py"],
         "cwd": "~/my-stuff/ai-alliance/allycat/upstream-dev-1/rag-remote",
         "env": {
           "WORKSPACE_DIR": "workspace"
         }
       }
     }
   }
   ```

   **Important**:
   - **Absolute paths**: Replace `/absolute/path/to/rag-remote` with the actual full path to your project
   - **Relative paths**: Use `~` for home directory or relative to the working directory
   - Using `uv run` ensures the correct virtual environment and dependencies are used
   - Make sure `uv` is installed and available in your PATH
   - When using relative `WORKSPACE_DIR`, it's relative to the `cwd` specified

3. **Restart Claude Desktop**

   Completely quit and restart Claude Desktop for the changes to take effect.

4. **Test the Integration**

   In Claude Desktop, you should now be able to:
   - See the `allycat_rag_remote` tool available
   - Ask Claude to query your RAG system
   - Example: "Use the allycat_rag_remote tool to ask: What is AI Alliance?"

## Integration with VS Code

To use this MCP server with VS Code (requires MCP extension):

1. **Install MCP Extension**

   Search for "Model Context Protocol" in VS Code extensions and install it.

2. **Configure the Extension**

   Open VS Code settings (JSON) and add the MCP server configuration. You can use either absolute paths or workspace-relative paths:

   **Option A: Using Absolute Paths** (works from any VS Code window)

   ```json
   {
     "mcp.servers": {
       "allycat-rag-remote": {
         "command": "uv",
         "args": ["run", "python", "mcp_server.py"],
         "cwd": "/absolute/path/to/rag-remote"
       }
     }
   }
   ```

   **Option B: Using Workspace-Relative Paths** (for project-specific setup)

   Ctrl+Shift+P --> MCP List Servers --> Add Server --> Add to workspace

   Give absolute path of `uv`

  This will be saved  : `.vscode/mcp.json`


   ```json
  {
    "servers": {
      "allycat-rag-remote": {
        "type": "stdio",
        "command": "~/.local/bin/uv",
        "args": [
          "run",
          "python",
          "mcp_server.py"
        ],
      "cwd": "${workspaceFolder}/rag-remote"
      }
    },
    "inputs": []
  }
   ```

   Or use the tilde (`~`) for home directory:

   ```json
   {
     "mcp.servers": {
       "allycat-rag-remote": {
         "command": "uv",
         "args": ["run", "python", "mcp_server.py"],
         "cwd": "~/my-stuff/ai-alliance/allycat/upstream-dev-1/rag-remote"
       }
     }
   }
   ```

   **Important**:
   - **Absolute paths**: Replace `/absolute/path/to/rag-remote` with the actual full path
   - **Workspace-relative**: Use `${workspaceFolder}` when the project is your workspace root
   - **Home-relative**: Use `~` to reference your home directory
   - Using `uv run` ensures the correct virtual environment and dependencies are used
   - Make sure `uv` is installed and available in your PATH

3. **Restart VS Code**

4. **Use the MCP Server**

   The MCP tools should now be available in your AI assistant panel.

## Integration with Claude Code CLI

To use this MCP server with Claude Code:

1. **Create MCP Configuration**

   Create or edit `~/.config/claude-code/config.json`. You can use either absolute paths or relative paths:

   **Option A: Using Absolute Paths** (recommended for global configuration)

   ```json
   {
     "mcpServers": {
       "rag-query": {
         "command": "uv",
         "args": ["run", "python", "mcp_server.py"],
         "cwd": "/absolute/path/to/rag-remote",
         "env": {
           "WORKSPACE_DIR": "/absolute/path/to/rag-remote/workspace"
         }
       }
     }
   }
   ```

   **Option B: Using Relative Paths** (for home directory or workspace setup)

   Using home directory (`~`):

   ```json
   {
     "mcpServers": {
       "rag-query": {
         "command": "uv",
         "args": ["run", "python", "mcp_server.py"],
         "cwd": "~/my-stuff/ai-alliance/allycat/upstream-dev-1/rag-remote",
         "env": {
           "WORKSPACE_DIR": "workspace"
         }
       }
     }
   }
   ```

   **Important**:
   - **Absolute paths**: Replace `/absolute/path/to/rag-remote` with the actual full path
   - **Home-relative**: Use `~` to reference your home directory (expands to full path)
   - Using `uv run` ensures the correct virtual environment and dependencies are used
   - Make sure `uv` is installed and available in your PATH
   - When using relative `WORKSPACE_DIR`, it's relative to the `cwd` specified

2. **Restart Claude Code**

   ```bash
   claude-code reset
   ```

3. **Test the Integration**

   Start Claude Code and ask it to use the allycat_rag_remote tool:

   ```
   $ claude-code
   > Use the allycat_rag_remote tool to ask: What are the main focus areas?
   ```

## Troubleshooting

### Server Won't Start

- **Check Python version**: Ensure you're using Python 3.12+
- **Check dependencies**: Run `uv sync` to install all dependencies
- **Check environment variables**: Ensure `.env` file is properly configured
- **Check database**: Ensure the vector database exists in the workspace directory

### Query Fails

- **Check database path**: Verify `MY_CONFIG.DB_URI` points to the correct database
- **Check models**: Ensure embedding and LLM models are accessible
- **Check API keys**: If using cloud models, verify API keys in `.env`

### MCP Client Connection Issues

- **Check paths**: Use absolute paths in all configuration files (for Claude Desktop, VS Code, Claude Code)
- **Check Python environment**: Ensure the Python interpreter has access to all dependencies
- **Check logs**: Look for error messages in the server logs

### Common Errors

1. **"RAG system not initialized"**
   - The vector database may not exist
   - Run `3_save_to_vector_db.py` first

2. **"Collection not found"**
   - The specified collection doesn't exist in the database
   - Check `MY_CONFIG.COLLECTION_NAME` in `my_config.py`

3. **"Model not found"**
   - The specified LLM or embedding model is not accessible
   - Check your API keys and model names in `.env`

## Example Queries

Here are some example queries to test with the MCP server:

- "What is AI Alliance?"
- "What are the main focus areas?"
- "What are upcoming events?"
- "How do I join the AI Alliance?"
- "What are some AI Alliance projects?"

## Server Configuration

The server uses configuration from `my_config.py`, which reads from `.env`:

- `EMBEDDING_MODEL`: Embedding model for vector search
- `EMBEDDING_LENGTH`: Dimension of embeddings
- `LLM_MODEL`: Language model for generating responses
- `DB_URI`: Path to the Milvus vector database
- `COLLECTION_NAME`: Name of the collection in the database

## Advanced Usage

### Custom Query Parameters

You can extend the MCP server to accept additional parameters:

1. Edit `mcp_server.py`
2. Modify the `inputSchema` in `handle_list_tools()`
3. Update `handle_call_tool()` to process new parameters
4. Update the query engine configuration accordingly

### Multiple Tools

You can add more tools to the MCP server:

1. Define new tool handlers in `handle_list_tools()`
2. Implement the logic in `handle_call_tool()`
3. Examples: document indexing, collection management, etc.

## Security Considerations

- **API Keys**: Never commit `.env` files with API keys
- **Access Control**: The MCP server has full access to your RAG system
- **Input Validation**: Queries are not sanitized; use with trusted clients only
- **Rate Limiting**: Consider adding rate limiting for production use

## References

- [Model Context Protocol Documentation](https://modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Claude Desktop MCP Guide](https://docs.anthropic.com/claude/docs/model-context-protocol)
