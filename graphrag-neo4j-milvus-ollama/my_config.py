import os 
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

## Configuration

class MyConfig:
    pass

MY_CONFIG = MyConfig ()

## All of these settings can be overridden by .env file
## And it will be loaded automatically by load_dotenv()
## And they will take precedence over the default values below
## See sample .env file 'env.sample.txt' for reference

## HuggingFace config
MY_CONFIG.HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://thealliance.ai/")

## Crawl settings
MY_CONFIG.WEBSITE_URL = os.getenv("WEBSITE_URL", "")
MY_CONFIG.CRAWL_MAX_DOWNLOADS = int(os.getenv("CRAWL_MAX_DOWNLOADS", 100))
MY_CONFIG.CRAWL_MAX_DEPTH = int(os.getenv("CRAWL_MAX_DEPTH", 3))
MY_CONFIG.WAITTIME_BETWEEN_REQUESTS = float(os.getenv("WAITTIME_BETWEEN_REQUESTS", 0.1)) # in seconds
MY_CONFIG.CRAWL_MIME_TYPE = 'text/html'


## Directories
MY_CONFIG.WORKSPACE_DIR = os.path.join(os.getenv('WORKSPACE_DIR', 'workspace'))
MY_CONFIG.CRAWL_DIR = os.path.join( MY_CONFIG.WORKSPACE_DIR, "crawled")
MY_CONFIG.PROCESSED_DATA_DIR = os.path.join( MY_CONFIG.WORKSPACE_DIR, "processed")

## llama index will download the models to this directory
os.environ["LLAMA_INDEX_CACHE_DIR"] = os.path.join(MY_CONFIG.WORKSPACE_DIR, "llama_index_cache")
### -------------------------------

# Find embedding models: https://huggingface.co/spaces/mteb/leaderboard

MY_CONFIG.EMBEDDING_MODEL =  os.getenv("EMBEDDING_MODEL", 'ibm-granite/granite-embedding-30m-english')
MY_CONFIG.EMBEDDING_LENGTH = int(os.getenv("EMBEDDING_LENGTH", 384))

## Chunking
MY_CONFIG.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
MY_CONFIG.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 20))


### Milvus config  
MY_CONFIG.COLLECTION_NAME = 'pages'

# Separate Milvus databases for different RAG approaches
# This allows running Vector RAG and Hybrid GraphRAG simultaneously without conflicts
MY_CONFIG.MILVUS_URI_VECTOR = os.path.join( MY_CONFIG.WORKSPACE_DIR, 'vector_only_milvus.db')  # Vector RAG only
MY_CONFIG.MILVUS_URI_HYBRID_GRAPH = os.path.join( MY_CONFIG.WORKSPACE_DIR, 'hybrid_graph_milvus.db')  # Hybrid GraphRAG

# Vector Database Configuration
MY_CONFIG.VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "local")  # Options: "local" or "cloud_zilliz"

# Zilliz Cloud Configuration (for cloud deployment)
MY_CONFIG.ZILLIZ_CLUSTER_ENDPOINT = os.getenv("ZILLIZ_CLUSTER_ENDPOINT")
MY_CONFIG.ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")



## ---- LLM settings ----
## Choose one: We can do local or cloud LLMs
# LLM_RUN_ENV controls which LLM backend to use: 'local_ollama' for local Ollama, 'cloud' for cloud LLMs
# Set LLM_RUN_ENV in your .env file. Default is 'cloud' for production deployment.
## Local LLMs are run on your machine using Ollama
## Cloud LLMs are run on any LiteLLM supported service like Replicate / Nebius / Cerebras / etc
## For running Ollama locally, please check the instructions in the docs/llm-local.md file


MY_CONFIG.LLM_RUN_ENV = os.getenv("LLM_RUN_ENV", "local_ollama")


MY_CONFIG.LLM_MODEL = os.getenv("LLM_MODEL", 'ollama/gemma3:1b')

# Replicate API token (if using Replicate)
MY_CONFIG.REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", None)
# Nebius API key (if using Nebius)
MY_CONFIG.NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY", None)

# --- GraphRAG LLM API keys ---
MY_CONFIG.CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", None)

# --- GraphRAG LLM Provider Configuration ---
# Provider for graph extraction: 'cerebras' (65K context, FREE) or 'nebius' (131K context, paid)
# Default is 'cerebras' (FREE) for local setup
MY_CONFIG.GRAPHRAG_LLM_PROVIDER = os.getenv("GRAPHRAG_LLM_PROVIDER", "cerebras")

# Provider-specific configurations (all controllable via environment variables)
MY_CONFIG.GRAPHRAG_PROVIDERS = {
    "cerebras": {
        "base_url": os.getenv("GRAPHRAG_CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1"),
        "model": os.getenv("GRAPHRAG_CEREBRAS_MODEL", "llama-3.3-70b"),
        "api_key_env": "CEREBRAS_API_KEY",
        "max_tokens": int(os.getenv("GRAPHRAG_CEREBRAS_MAX_TOKENS", "65536")),
        "max_output": int(os.getenv("GRAPHRAG_CEREBRAS_MAX_OUTPUT", "8000")),
        "rate_delay": float(os.getenv("GRAPHRAG_CEREBRAS_RATE_DELAY", "2.5")),
        "batch_reduction": int(os.getenv("GRAPHRAG_CEREBRAS_BATCH_REDUCTION", "15000")),
        "chars_per_token": int(os.getenv("GRAPHRAG_CEREBRAS_CHARS_PER_TOKEN", "4"))
    },
    "nebius": {
        "base_url": os.getenv("GRAPHRAG_NEBIUS_BASE_URL", "https://api.studio.nebius.ai/v1/"),
        "model": os.getenv("GRAPHRAG_NEBIUS_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
        "api_key_env": "NEBIUS_API_KEY",
        "max_tokens": int(os.getenv("GRAPHRAG_NEBIUS_MAX_TOKENS", "131072")),
        "max_output": int(os.getenv("GRAPHRAG_NEBIUS_MAX_OUTPUT", "8000")),
        "rate_delay": float(os.getenv("GRAPHRAG_NEBIUS_RATE_DELAY", "1.5")),
        "batch_reduction": int(os.getenv("GRAPHRAG_NEBIUS_BATCH_REDUCTION", "35000")),
        "chars_per_token": int(os.getenv("GRAPHRAG_NEBIUS_CHARS_PER_TOKEN", "4"))
    }
}

# --- Graph entity/relationship extraction config ---
MY_CONFIG.GRAPH_MIN_ENTITIES = int(os.getenv("GRAPH_MIN_ENTITIES", 5))
MY_CONFIG.GRAPH_MAX_ENTITIES = int(os.getenv("GRAPH_MAX_ENTITIES", 15))
MY_CONFIG.GRAPH_MIN_RELATIONSHIPS = int(os.getenv("GRAPH_MIN_RELATIONSHIPS", 5))
MY_CONFIG.GRAPH_MAX_RELATIONSHIPS = int(os.getenv("GRAPH_MAX_RELATIONSHIPS", 30))  
MY_CONFIG.GRAPH_MIN_CONFIDENCE = float(os.getenv("GRAPH_MIN_CONFIDENCE", 0.60))  


## --- GraphRAG ---
# --- Neo4j config ---
MY_CONFIG.NEO4J_URI = os.getenv("NEO4J_URI")
MY_CONFIG.NEO4J_USER = os.getenv("NEO4J_USERNAME")
MY_CONFIG.NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
MY_CONFIG.NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")
MY_CONFIG.GRAPH_DATA_DIR = os.path.join(MY_CONFIG.WORKSPACE_DIR, "graph_data")





## --- UI settings ---
MY_CONFIG.STARTER_PROMPTS_STR = os.getenv("UI_STARTER_PROMPTS", 'What is this website?  |  What are upcoming events?  | Who are some of the partners?')


MY_CONFIG.STARTER_PROMPTS = MY_CONFIG.STARTER_PROMPTS_STR.split("|") if MY_CONFIG.STARTER_PROMPTS_STR else []


## --- Port Configuration ---
# Flask apps (auto-configured via MY_CONFIG)
MY_CONFIG.FLASK_VECTOR_PORT = int(os.getenv("FLASK_VECTOR_PORT", 8081))  # app_flask.py (vector RAG)
MY_CONFIG.FLASK_GRAPH_PORT = int(os.getenv("FLASK_GRAPH_PORT", 8080))   # app_flask_graph.py (GraphRAG)

# Chainlit apps (default port: 8000, custom ports for Docker deployments)
MY_CONFIG.CHAINLIT_VECTOR_PORT = int(os.getenv("CHAINLIT_VECTOR_PORT", 8082))  # app_chainlit.py (Docker: 8082, Native: 8000)
MY_CONFIG.CHAINLIT_GRAPH_PORT = int(os.getenv("CHAINLIT_GRAPH_PORT", 8083))    # app_chainlit_graph.py (Docker: 8083, Native: 8000)

# Docker and external services
MY_CONFIG.DOCKER_PORT = int(os.getenv("DOCKER_PORT", 8080))  # External host port (maps to DOCKER_APP_PORT)
MY_CONFIG.DOCKER_APP_PORT = int(os.getenv("DOCKER_APP_PORT", 8080))  # Internal container port (all apps use this in Docker)
MY_CONFIG.OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", 11434))  # Ollama server port