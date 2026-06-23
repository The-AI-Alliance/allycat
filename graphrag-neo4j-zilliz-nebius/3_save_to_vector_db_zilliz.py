"""
Cloud Vector Database Setup

Creates vector database collections on cloud infrastructure.
Supports both vector search and graph-based retrieval systems.
"""

from my_config import MY_CONFIG
import os
import sys
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from pymilvus import MilvusClient
from llama_index.core import StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import VectorStoreIndex
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Validate cloud database configuration
if not MY_CONFIG.ZILLIZ_CLUSTER_ENDPOINT:
    raise ValueError("Cloud endpoint configuration missing")
if not MY_CONFIG.ZILLIZ_TOKEN:
    raise ValueError("Cloud authentication token missing")

def main():
    logger.info("Initializing cloud database connection")

    # Load source documents
    logger.info("Loading documents")
    reader = SimpleDirectoryReader(input_dir=MY_CONFIG.PROCESSED_DATA_DIR, recursive=False, required_exts=[".md"])
    documents = reader.load_data()
    logger.info(f"Loaded {len(documents)} documents")

    # Process document chunks
    logger.info("Processing document chunks")
    parser = SentenceSplitter(chunk_size=MY_CONFIG.CHUNK_SIZE, chunk_overlap=MY_CONFIG.CHUNK_OVERLAP)
    nodes = parser.get_nodes_from_documents(documents)
    logger.info(f"Created {len(nodes)} chunks")

    # Initialize embedding model
    logger.info("Configuring embedding model")
    os.environ['HF_ENDPOINT'] = MY_CONFIG.HF_ENDPOINT

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=MY_CONFIG.EMBEDDING_MODEL
    )

    # Create cloud database collection
    logger.info("Creating database collection")
    collection_name = MY_CONFIG.COLLECTION_NAME

    milvus_client = None
    try:
        # Connect to cloud database
        milvus_client = MilvusClient(
            uri=MY_CONFIG.ZILLIZ_CLUSTER_ENDPOINT,
            token=MY_CONFIG.ZILLIZ_TOKEN
        )

        # Remove existing collection if present
        if milvus_client.has_collection(collection_name=collection_name):
            milvus_client.drop_collection(collection_name=collection_name)

        # Initialize vector store
        vector_store = MilvusVectorStore(
            uri=MY_CONFIG.ZILLIZ_CLUSTER_ENDPOINT,
            token=MY_CONFIG.ZILLIZ_TOKEN,
            collection_name=collection_name,
            dim=MY_CONFIG.EMBEDDING_LENGTH,
            overwrite=True
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Store document vectors
        logger.info(f"Processing {len(nodes)} document chunks")
        VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
        )

        logger.info(f"Database collection '{collection_name}' created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create collection: {str(e)}")
        raise
    finally:
        if milvus_client:
            milvus_client.close()

    logger.info("Cloud database setup completed successfully")

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)