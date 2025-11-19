import os
import sys
# add ../common to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'common'))

from my_config import MY_CONFIG
import glob
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.embeddings.litellm import LiteLLMEmbedding
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
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


Settings.embed_model = LiteLLMEmbedding(
        model_name=MY_CONFIG.EMBEDDING_MODEL,
        embed_batch_size=50,  # Batch size for embedding (default is 10)
    )
logger.info (f"✅ Using embedding model: {MY_CONFIG.EMBEDDING_MODEL}")


# Step-4: Connect to Milvus
milvus_client = MilvusClient(MY_CONFIG.DB_URI)
logger.info (f"✅ Connected to Milvus instance: {MY_CONFIG.DB_URI}")

if milvus_client.has_collection(collection_name = MY_CONFIG.COLLECTION_NAME):
    milvus_client.drop_collection(collection_name = MY_CONFIG.COLLECTION_NAME)
    logger.info (f"✅ Cleared collection : {MY_CONFIG.COLLECTION_NAME}")

# Connect llama-index to vector db
vector_store = MilvusVectorStore(
    uri = MY_CONFIG.DB_URI,
    dim = MY_CONFIG.EMBEDDING_LENGTH,
    collection_name = MY_CONFIG.COLLECTION_NAME,
    overwrite=True
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
logger.info (f"✅ Connected Llama-index to Milvus instance: {MY_CONFIG.DB_URI}")

# Step-5: Save to DB
# Save chunks into vector db
index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
)

logger.info (f"✅ Successfully stored {len(nodes)} chunks in Milvus collection '{MY_CONFIG.COLLECTION_NAME}'")

milvus_client.close()