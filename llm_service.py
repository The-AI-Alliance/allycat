import importlib
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from my_config import MY_CONFIG

class ProviderFactory:
    """Factory class for creating LLM providers dynamically"""
    
    PROVIDERS = {
        'ollama': ('llama_index.llms.ollama', 'Ollama'),
        'replicate': ('llama_index.llms.replicate', 'Replicate'),
        'vllm': ('llama_index.llms.openai_like', 'OpenAILike')
    }
    
    @classmethod
    def create_llm(cls, provider_name, config):
        """Create LLM instance dynamically based on provider name and config"""
        if provider_name not in cls.PROVIDERS:
            raise ValueError(f"❌ Unsupported provider: {provider_name}")
        
        module_name, class_name = cls.PROVIDERS[provider_name]
        
        try:
            # Dynamic import
            module = importlib.import_module(module_name)
            llm_class = getattr(module, class_name)
            
            # Filter out None values to make parameters optional
            filtered_config = {k: v for k, v in config.items() if v is not None}
            
            # Create instance with filtered config
            return llm_class(**filtered_config)
        except ImportError as e:
            raise ValueError(f"❌ Failed to import {module_name}: {e}")
        except Exception as e:
            raise ValueError(f"❌ Failed to create {provider_name} LLM: {e}")

def initialize_llm_service():
    """Initialize LLM service using pluggable provider architecture"""
    
    # Configure embedding (same for all providers)
    Settings.embed_model = HuggingFaceEmbedding(model_name=MY_CONFIG.EMBEDDING_MODEL)
    print("✅ Using embedding model:", MY_CONFIG.EMBEDDING_MODEL)
    
    # Get active provider configuration
    provider_name = MY_CONFIG.ACTIVE_PROVIDER
    provider_config = MY_CONFIG.LLM_PROVIDERS[provider_name]
    
    # Create LLM instance dynamically
    llm = ProviderFactory.create_llm(provider_name, provider_config)
    
    Settings.llm = llm
    print("✅ LLM provider:", provider_name)    
    print("✅ Using LLM model:", provider_config.get('model'))
    
    return llm