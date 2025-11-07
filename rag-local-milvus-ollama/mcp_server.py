#!/usr/bin/env python3

import asyncio
import logging
import sys
import os
import json
import time

# Add common to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'common'))

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, Tool, TextContent, ImageContent, EmbeddedResource
from typing import Any, Sequence

from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.llms.litellm import LiteLLM
import query_utils as query_utils
from my_config import MY_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-local-mcp-server")

# Global query engine
query_engine = None

class RAGLocalMCPServer:
    def __init__(self):
        self.server = Server("rag-local-milvus-ollama")
        self.setup_handlers()
        self.initialize_rag()
    
    def setup_handlers(self):
        @self.server.list_resources()
        async def handle_list_resources() -> list[Resource]:
            return []
        
        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            return [
                Tool(
                    name="allycat_local_query",
                    description="Query the local RAG system for information",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The question to ask the RAG system"
                            }
                        },
                        "required": ["query"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict | None) -> list[TextContent | ImageContent | EmbeddedResource]:
            if name != "allycat_local_query":
                raise ValueError(f"Unknown tool: {name}")
            
            if not arguments or "query" not in arguments:
                raise ValueError("Missing required argument: query")
            
            query = arguments["query"]
            result = await self.process_query(query)
            
            return [TextContent(type="text", text=result)]
    
    def initialize_rag(self):
        """Initialize the RAG system components"""
        global query_engine
        
        try:
            # Load environment variables
            load_dotenv()
            
            # Setup embeddings
            Settings.embed_model = HuggingFaceEmbedding(
                model_name=MY_CONFIG.EMBEDDING_MODEL
            )
            logger.info(f"✅ Using embedding model: {MY_CONFIG.EMBEDDING_MODEL}")
            
            # Connect to vector database
            vector_store = MilvusVectorStore(
                uri=MY_CONFIG.DB_URI,
                dim=MY_CONFIG.EMBEDDING_LENGTH,
                collection_name=MY_CONFIG.COLLECTION_NAME,
                overwrite=False
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            logger.info(f"✅ Connected to Milvus instance: {MY_CONFIG.DB_URI}")
            
            # Load Document Index from DB
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store, storage_context=storage_context
            )
            logger.info(f"✅ Loaded index from vector db")
            
            # Setup LLM
            Settings.llm = LiteLLM(model=MY_CONFIG.LLM_MODEL)
            logger.info(f"✅ Using LLM model: {MY_CONFIG.LLM_MODEL}")
            
            # Create query engine
            query_engine = index.as_query_engine()
            logger.info("✅ RAG system initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG system: {e}")
            raise
    
    async def process_query(self, query: str) -> str:
        """Process a query through the RAG system"""
        global query_engine
        
        if query_engine is None:
            return "❌ RAG system not properly initialized"
        
        try:
            # Tweak query if needed
            tweaked_query = query_utils.tweak_query(query, MY_CONFIG.LLM_MODEL)
            logger.info(f"Processing query: {tweaked_query}")
            
            # Run the query
            start_time = time.time()
            response = query_engine.query(tweaked_query)
            end_time = time.time()
            
            # Format response with timing and metadata
            result = str(response)
            result += f"\n\n⏱️ **Processing time:** {(end_time - start_time):.1f} seconds"
            
            if hasattr(response, 'metadata') and response.metadata:
                result += f"\n\n📊 **Metadata:**\n```json\n{json.dumps(response.metadata, indent=2)}\n```"
            
            logger.info(f"Query processed successfully")
            return result
            
        except Exception as e:
            error_msg = f"❌ Error processing query: {str(e)}"
            logger.error(error_msg)
            return error_msg

async def main():
    # Initialize the server
    rag_server = RAGLocalMCPServer()
    
    # Run the server
    async with stdio_server() as (read_stream, write_stream):
        await rag_server.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="rag-local-milvus-ollama",
                server_version="1.0.0",
                capabilities=rag_server.server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())
