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

MY_CONFIG.EMBEDDING_MODEL =  os.getenv("EMBEDDING_MODEL", 'nebius/Qwen/Qwen3-Embedding-8B')
MY_CONFIG.EMBEDDING_LENGTH = int(os.getenv("EMBEDDING_LENGTH", 4096))

## Chunking
MY_CONFIG.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1024))
MY_CONFIG.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))


### Milvus config
MY_CONFIG.DB_URI = os.path.join( MY_CONFIG.WORKSPACE_DIR, 'rag_website_milvus.db')  # For embedded instance
MY_CONFIG.COLLECTION_NAME = 'pages'

## ---- LLM settings ----
## We use LiteLLM  which supports multiple LLM backends, including local and cloud LLMs.
## Local LLMs are run on your machine using Ollama
## Cloud LLMs are run on any LiteLLM supported service like Replicate / Nebius / etc
## For running Ollama locally, please check the instructions in the docs/llm-local.md file



MY_CONFIG.LLM_MODEL = os.getenv("LLM_MODEL", 'nebius/Qwen/Qwen3-30B-A3B-Instruct-2507')


## --- UI settings ---
MY_CONFIG.STARTER_PROMPTS_STR = os.getenv("UI_STARTER_PROMPTS", 'What is this website?  |  What are upcoming events?  | Who are some of the partners?')
MY_CONFIG.STARTER_PROMPTS = MY_CONFIG.STARTER_PROMPTS_STR.split("|") if MY_CONFIG.STARTER_PROMPTS_STR else []