from my_config import MY_CONFIG
import os
import glob
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
# logger.setLevel(logging.INFO)

# Step-1: Read Markdown files
pattern = os.path.join(MY_CONFIG.PROCESSED_DATA_DIR, '*.md')
md_file_count = len(glob.glob(pattern, recursive=True))

reader = SimpleDirectoryReader(input_dir=MY_CONFIG.PROCESSED_DATA_DIR, recursive=False, required_exts=[".md"])
documents = reader.load_data()
logger.info (f"Loaded {len(documents)} documents from {md_file_count} files")

# Step-2: Create Chunks
parser = SentenceSplitter(chunk_size=MY_CONFIG.CHUNK_SIZE, chunk_overlap=MY_CONFIG.CHUNK_OVERLAP)
nodes = parser.get_nodes_from_documents(documents)
logger.info (f"Created {len(nodes)} chunks from {len(documents)} documents")

# Step-3: Setup Embedding Model
os.environ['HF_ENDPOINT'] = MY_CONFIG.HF_ENDPOINT

Settings.embed_model = HuggingFaceEmbedding(
    model_name = MY_CONFIG.EMBEDDING_MODEL
)

# Step-4: Create 2 Vector Databases (Vector RAG and Hybrid GraphRAG databases)

databases_to_create = [
    {
        "name": "Vector RAG Only",
        "uri": MY_CONFIG.MILVUS_URI_VECTOR,
        "description": "For Vector RAG systems"
    },
    {
        "name": "Hybrid GraphRAG", 
        "uri": MY_CONFIG.MILVUS_URI_HYBRID_GRAPH,
        "description": "For Hybrid GraphRAG systems"
    }
]

for db_config in databases_to_create:
    logger.info(f"📦 Creating {db_config['name']} database...")
    
    # Connect to Milvus for this database
    milvus_client = MilvusClient(db_config['uri'])
    logger.info(f"✅ Connected to: {db_config['uri']}")

    if milvus_client.has_collection(collection_name = MY_CONFIG.COLLECTION_NAME):
        milvus_client.drop_collection(collection_name = MY_CONFIG.COLLECTION_NAME)
        logger.info(f"✅ Cleared collection: {MY_CONFIG.COLLECTION_NAME}")

    # Connect llama-index to vector db
    vector_store = MilvusVectorStore(
        uri = db_config['uri'],
        dim = MY_CONFIG.EMBEDDING_LENGTH,
        collection_name = MY_CONFIG.COLLECTION_NAME,
        overwrite=True
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Save chunks into vector db
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
    )

    logger.info(f"✅ Stored {len(nodes)} chunks in {db_config['name']}")
    milvus_client.close()

logger.info("🎉 Both databases created!")
logger.info(f"   • Vector RAG: {MY_CONFIG.MILVUS_URI_VECTOR}")
logger.info(f"   • Hybrid GraphRAG: {MY_CONFIG.MILVUS_URI_HYBRID_GRAPH}")