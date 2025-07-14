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

## Crawl settings
MY_CONFIG.CRAWL_MAX_DOWNLOADS = 100
MY_CONFIG.CRAWL_MAX_DEPTH = 3
MY_CONFIG.WAITTIME_BETWEEN_REQUESTS = int(os.getenv("WAITTIME_BETWEEN_REQUESTS", 0.1)) # in seconds
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
MY_CONFIG.DB_URI = os.path.join( MY_CONFIG.WORKSPACE_DIR, 'rag_website_milvus.db')  # For embedded instance
MY_CONFIG.COLLECTION_NAME = 'pages'

## ---- LLM settings ----
## Choose one: We can do local or cloud LLMs
## Local LLMs are run on your machine using Ollama
## Cloud LLMs are run on any LiteLLM supported service like Replicate / Nebius / etc
## For running Ollama locally, please check the instructions in the docs/ollama.md file

MY_CONFIG.LLM_MODEL = os.getenv("LLM_MODEL", 'ollama/gemma3:1b')

