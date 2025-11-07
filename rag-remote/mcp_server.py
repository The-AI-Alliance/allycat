#!/usr/bin/env python3
"""
MCP Server for RAG Query System

This MCP server exposes the RAG query functionality from 4_query.py as an MCP tool.
It allows querying the vector database through the Model Context Protocol.
"""

import os
import sys
import asyncio
import logging
from typing import Any
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import MCP SDK
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp import types

# Import project dependencies
from my_config import MY_CONFIG
from dotenv import load_dotenv

# If connection to https://huggingface.co/ failed, uncomment the following path
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from llama_index.core import Settings
from llama_index.embeddings.litellm import LiteLLMEmbedding
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.litellm import LiteLLM
import query_utils

# Load environment variables
load_dotenv()

# Global query engine (will be initialized on server start)
query_engine = None


def initialize_rag_system():
    """
    Initialize the RAG system components:
    - Embedding model
    - Vector store connection
    - LLM model
    - Query engine
    """
    global query_engine

    try:
        # Setup embeddings
        Settings.embed_model = LiteLLMEmbedding(
            model_name=MY_CONFIG.EMBEDDING_MODEL,
        )
        logger.info(f"✅ Using embedding model: {MY_CONFIG.EMBEDDING_MODEL}")

        # Connect to vector db
        vector_store = MilvusVectorStore(
            uri=MY_CONFIG.DB_URI,
            dim=MY_CONFIG.EMBEDDING_LENGTH,
            collection_name=MY_CONFIG.COLLECTION_NAME,
            overwrite=False  # Load the index from db
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        logger.info(f"✅ Connected to Milvus instance: {MY_CONFIG.DB_URI}")

        # Load Document Index from DB
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, storage_context=storage_context
        )
        logger.info(f"✅ Loaded index from vector db: {MY_CONFIG.DB_URI}")

        # Setup LLM
        logger.info(f"✅ Using LLM model: {MY_CONFIG.LLM_MODEL}")
        Settings.llm = LiteLLM(
            model=MY_CONFIG.LLM_MODEL,
        )

        # Create query engine
        query_engine = index.as_query_engine()
        logger.info("✅ RAG system initialized successfully")

    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {e}")
        raise


async def run_query(query: str) -> dict[str, Any]:
    """
    Execute a query against the RAG system

    Args:
        query: The question to ask

    Returns:
        Dictionary containing the response and metadata
    """
    global query_engine

    if query_engine is None:
        raise RuntimeError("RAG system not initialized")

    try:
        # Tweak query if needed (e.g., for Qwen3 models)
        tweaked_query = query_utils.tweak_query(query, MY_CONFIG.LLM_MODEL)
        logger.info(f"Processing query: {tweaked_query}")

        # Execute query
        response = query_engine.query(tweaked_query)

        # Prepare result
        result = {
            "answer": str(response),
            "metadata": response.metadata if hasattr(response, 'metadata') else {},
            "source_nodes": [
                {
                    "node_id": node.node_id,
                    "score": node.score if hasattr(node, 'score') else None,
                    "text": node.text[:200] + "..." if len(node.text) > 200 else node.text
                }
                for node in response.source_nodes
            ] if hasattr(response, 'source_nodes') else []
        }

        logger.info(f"Query completed successfully")
        return result

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise


# Create MCP server instance
app = Server("rag-query-server")


@app.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    List available tools.
    """
    return [
        types.Tool(
            name="allycat_rag_remote",
            description=f"""This tool can answer questions about The AI Alliance website (aialliance.org).

Query the Allycat RAG system.

Current configuration:
- LLM Model: {MY_CONFIG.LLM_MODEL}
- Embedding Model: {MY_CONFIG.EMBEDDING_MODEL}
- Vector DB: {MY_CONFIG.DB_URI}
- Collection: {MY_CONFIG.COLLECTION_NAME}

Example queries:
- "What is AI Alliance?"
- "What are the main focus areas?"
- "What are upcoming events?"
""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question to ask the RAG system",
                    },
                },
                "required": ["query"],
            },
        ),
    ]


@app.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Handle tool execution requests.
    """
    if name != "allycat_rag_remote":
        raise ValueError(f"Unknown tool: {name}")

    if not arguments or "query" not in arguments:
        raise ValueError("Missing required argument: query")

    query = arguments["query"]

    try:
        result = await run_query(query)

        # Format the response
        response_text = f"""**Answer:**
{result['answer']}

**Source Documents:** {len(result['source_nodes'])} documents retrieved

**Metadata:**
```json
{json.dumps(result['metadata'], indent=2)}
```
"""

        return [types.TextContent(type="text", text=response_text)]

    except Exception as e:
        error_msg = f"Error executing query: {str(e)}"
        logger.error(error_msg)
        return [types.TextContent(type="text", text=error_msg)]


async def main():
    """
    Main entry point for the MCP server.
    """
    logger.info("Starting RAG MCP Server...")

    # Initialize the RAG system
    initialize_rag_system()

    # Run the server
    async with stdio_server() as (read_stream, write_stream):
        logger.info("MCP Server running on stdio")
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="rag-query-server",
                server_version="0.1.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
