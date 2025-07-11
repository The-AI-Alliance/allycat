import os 

## Configuration
class MyConfig:
    pass 

MY_CONFIG = MyConfig ()

## Crawl settings
# MY_CONFIG.CRAWL_URL_BASE = 'https://thealliance.ai/'
MY_CONFIG.CRAWL_MAX_DOWNLOADS = 500
MY_CONFIG.CRAWL_MAX_DEPTH = 5
MY_CONFIG.CRAWL_MIME_TYPE = 'text/html'

## Directories
MY_CONFIG.WORKSPACE_DIR = os.path.join('workspace')
MY_CONFIG.CRAWL_DIR = os.path.join( MY_CONFIG.WORKSPACE_DIR, "crawled")
MY_CONFIG.PROCESSED_DATA_DIR = os.path.join( MY_CONFIG.WORKSPACE_DIR, "processed")

## llama index will download the models to this directory
os.environ["LLAMA_INDEX_CACHE_DIR"] = os.path.join(MY_CONFIG.WORKSPACE_DIR, "llama_index_cache")
### -------------------------------

# Find embedding models: https://huggingface.co/spaces/mteb/leaderboard

# MY_CONFIG.EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
# MY_CONFIG.EMBEDDING_LENGTH = 384

# MY_CONFIG.EMBEDDING_MODEL = 'BAAI/bge-small-en-v1.5'
# MY_CONFIG.EMBEDDING_LENGTH = 384

MY_CONFIG.EMBEDDING_MODEL = 'ibm-granite/granite-embedding-30m-english'
MY_CONFIG.EMBEDDING_LENGTH = 384

## Chunking
MY_CONFIG.CHUNK_SIZE = 512
MY_CONFIG.CHUNK_OVERLAP = 20


### Milvus config
MY_CONFIG.DB_URI = os.path.join( MY_CONFIG.WORKSPACE_DIR, 'rag_website_milvus.db')  # For embedded instance
MY_CONFIG.COLLECTION_NAME = 'pages'

## ---- LLM settings ----
## Pluggable provider configuration
MY_CONFIG.LLM_PROVIDERS = {
    'ollama': {
        'model': 'gemma3:1b',  # 815MB - available models: https://ollama.com/models
        'request_timeout': 30.0,  # Optional: set to None to use default
        'temperature': 0.1        # Optional: set to None to use default
        # Other available models:
        # 'qwen3:0.6b'      # 522MB
        # 'tinyllama'       # 638MB  
        # 'llama3.2:1b'     # 1.2GB
        # 'qwen3:1.7b'      # 1.4 GB
        # 'gemma3:2b'       # 1.5GB
        # 'gemma3:4b'       # 3.3GB
        # 'llama3.2:8b'     # 8.1GB
        # 'gemma3:8b'       # 8.1GB
    },
    'replicate': {
        'model': 'meta/meta-llama-3-8b-instruct',  # available models: https://replicate.com/explore
        'temperature': 0.1  # Optional: set to None to use default
        # Other available models:
        # 'meta/meta-llama-3-70b-instruct'
        # 'ibm-granite/granite-3.1-2b-instruct'
        # 'ibm-granite/granite-3.2-8b-instruct'
    },
    'vllm': {
        'model': 'RedHatAI/gemma-3-4b-it-quantized.w4a16',  # Model name for vLLM
        'api_key': 'fake',  # vLLM doesn't require real API key
        'api_base': 'http://localhost:8000/v1',  # vLLM server URL
        'temperature': 0.1,  # Optional: set to None to use default
        'context_window': 8192,  # Should match your `vllm serve --max-model-len` setting
                                 # This is for LlamaIndex's internal optimization (chunking, prompt planning)
                                 # It doesn't control the vLLM server - just tells LlamaIndex what to expect
        'is_chat_model': True  # Important for chat-based models
    }
}

## Active provider - change this to switch providers
MY_CONFIG.ACTIVE_PROVIDER = 'vllm'  # 'ollama', 'replicate', or 'vllm'

## Legacy support - for backwards compatibility with existing code
MY_CONFIG.LLM_RUN_ENV = 'local_ollama' if MY_CONFIG.ACTIVE_PROVIDER == 'ollama' else MY_CONFIG.ACTIVE_PROVIDER
MY_CONFIG.LLM_MODEL = MY_CONFIG.LLM_PROVIDERS[MY_CONFIG.ACTIVE_PROVIDER]['model']
