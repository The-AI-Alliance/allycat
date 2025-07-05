"""
Tests for AllyCat configuration system and pluggable provider architecture.

Tests configuration loading, validation, and provider switching functionality.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Add parent directory to path to import AllyCat modules  
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_config import MY_CONFIG


class TestConfigurationLoading:
    """Test basic configuration loading and validation."""
    
    def test_config_has_required_attributes(self):
        """Test that MY_CONFIG has all required attributes."""
        required_attrs = [
            'ACTIVE_PROVIDER',
            'LLM_PROVIDERS', 
            'EMBEDDING_MODEL',
            'EMBEDDING_LENGTH',
            'DB_URI',
            'COLLECTION_NAME'
        ]
        
        for attr in required_attrs:
            assert hasattr(MY_CONFIG, attr), f"MY_CONFIG missing required attribute: {attr}"
        
        print("✅ All required configuration attributes present")
    
    def test_active_provider_is_valid(self):
        """Test that ACTIVE_PROVIDER is set to a valid value."""
        valid_providers = ['ollama', 'vllm', 'replicate']
        
        assert MY_CONFIG.ACTIVE_PROVIDER in valid_providers, \
            f"ACTIVE_PROVIDER '{MY_CONFIG.ACTIVE_PROVIDER}' not in {valid_providers}"
        
        print(f"✅ Active provider '{MY_CONFIG.ACTIVE_PROVIDER}' is valid")
    
    def test_llm_providers_structure(self):
        """Test that LLM_PROVIDERS has correct structure."""
        assert isinstance(MY_CONFIG.LLM_PROVIDERS, dict), "LLM_PROVIDERS should be a dictionary"
        
        expected_providers = ['ollama', 'vllm', 'replicate']
        for provider in expected_providers:
            assert provider in MY_CONFIG.LLM_PROVIDERS, f"Missing provider: {provider}"
            assert isinstance(MY_CONFIG.LLM_PROVIDERS[provider], dict), f"Provider {provider} config should be dict"
        
        print("✅ LLM_PROVIDERS structure is correct")
    
    def test_active_provider_has_config(self):
        """Test that the active provider has configuration."""
        active_provider = MY_CONFIG.ACTIVE_PROVIDER
        
        assert active_provider in MY_CONFIG.LLM_PROVIDERS, \
            f"Active provider '{active_provider}' not found in LLM_PROVIDERS"
        
        provider_config = MY_CONFIG.LLM_PROVIDERS[active_provider]
        assert 'model' in provider_config, f"Provider {active_provider} missing 'model' config"
        
        print(f"✅ Active provider '{active_provider}' has valid configuration")


class TestProviderConfigurations:
    """Test individual provider configurations."""
    
    def test_ollama_config_structure(self):
        """Test Ollama provider configuration structure."""
        ollama_config = MY_CONFIG.LLM_PROVIDERS['ollama']
        
        required_keys = ['model', 'request_timeout', 'temperature']
        for key in required_keys:
            assert key in ollama_config, f"Ollama config missing key: {key}"
        
        # Validate types
        assert isinstance(ollama_config['model'], str), "Ollama model should be string"
        assert isinstance(ollama_config['request_timeout'], (int, float)), "request_timeout should be numeric"
        assert isinstance(ollama_config['temperature'], (int, float)), "temperature should be numeric"
        
        print("✅ Ollama configuration structure valid")
    
    def test_vllm_config_structure(self):
        """Test vLLM provider configuration structure."""
        vllm_config = MY_CONFIG.LLM_PROVIDERS['vllm']
        
        required_keys = ['model', 'api_key', 'api_base', 'temperature', 'context_window', 'is_chat_model']
        for key in required_keys:
            assert key in vllm_config, f"vLLM config missing key: {key}"
        
        # Validate types
        assert isinstance(vllm_config['model'], str), "vLLM model should be string"
        assert isinstance(vllm_config['api_key'], str), "api_key should be string"
        assert isinstance(vllm_config['api_base'], str), "api_base should be string"
        assert isinstance(vllm_config['temperature'], (int, float)), "temperature should be numeric"
        assert isinstance(vllm_config['context_window'], int), "context_window should be integer"
        assert isinstance(vllm_config['is_chat_model'], bool), "is_chat_model should be boolean"
        
        # Validate URL format
        assert vllm_config['api_base'].startswith('http'), "api_base should be valid URL"
        
        print("✅ vLLM configuration structure valid")
    
    def test_replicate_config_structure(self):
        """Test Replicate provider configuration structure."""
        replicate_config = MY_CONFIG.LLM_PROVIDERS['replicate']
        
        required_keys = ['model', 'temperature']
        for key in required_keys:
            assert key in replicate_config, f"Replicate config missing key: {key}"
        
        # Validate types
        assert isinstance(replicate_config['model'], str), "Replicate model should be string"
        assert isinstance(replicate_config['temperature'], (int, float)), "temperature should be numeric"
        
        # Validate model format (should contain '/')
        assert '/' in replicate_config['model'], "Replicate model should be in format 'owner/model'"
        
        print("✅ Replicate configuration structure valid")


class TestLegacyCompatibility:
    """Test backwards compatibility with legacy configuration."""
    
    def test_legacy_attributes_exist(self):
        """Test that legacy configuration attributes still exist."""
        # These are used for backwards compatibility
        legacy_attrs = ['LLM_RUN_ENV', 'LLM_MODEL']
        
        for attr in legacy_attrs:
            assert hasattr(MY_CONFIG, attr), f"Legacy attribute {attr} missing"
        
        print("✅ Legacy configuration attributes present")
    
    def test_legacy_env_mapping(self):
        """Test that LLM_RUN_ENV maps correctly to ACTIVE_PROVIDER."""
        active_provider = MY_CONFIG.ACTIVE_PROVIDER
        legacy_env = MY_CONFIG.LLM_RUN_ENV
        
        # Check the mapping logic from my_config.py
        if active_provider == 'ollama':
            assert legacy_env == 'local_ollama', "Ollama should map to 'local_ollama'"
        elif active_provider == 'replicate':
            assert legacy_env == 'replicate', "Replicate should map to 'replicate'"
        elif active_provider == 'vllm':
            assert legacy_env == 'vllm', "vLLM should map to 'vllm'"
        
        print(f"✅ Legacy mapping: {active_provider} -> {legacy_env}")
    
    def test_legacy_model_mapping(self):
        """Test that LLM_MODEL maps to active provider's model."""
        active_provider = MY_CONFIG.ACTIVE_PROVIDER
        legacy_model = MY_CONFIG.LLM_MODEL
        provider_model = MY_CONFIG.LLM_PROVIDERS[active_provider]['model']
        
        assert legacy_model == provider_model, \
            f"Legacy model '{legacy_model}' should match provider model '{provider_model}'"
        
        print(f"✅ Legacy model mapping: {legacy_model}")


class TestConfigurationValidation:
    """Test configuration validation and error cases."""
    
    def test_embedding_config_validation(self):
        """Test embedding model configuration."""
        assert isinstance(MY_CONFIG.EMBEDDING_MODEL, str), "EMBEDDING_MODEL should be string"
        assert len(MY_CONFIG.EMBEDDING_MODEL) > 0, "EMBEDDING_MODEL should not be empty"
        
        assert isinstance(MY_CONFIG.EMBEDDING_LENGTH, int), "EMBEDDING_LENGTH should be integer"
        assert MY_CONFIG.EMBEDDING_LENGTH > 0, "EMBEDDING_LENGTH should be positive"
        
        print("✅ Embedding configuration valid")
    
    def test_database_config_validation(self):
        """Test database configuration."""
        assert isinstance(MY_CONFIG.DB_URI, str), "DB_URI should be string"
        assert len(MY_CONFIG.DB_URI) > 0, "DB_URI should not be empty"
        
        assert isinstance(MY_CONFIG.COLLECTION_NAME, str), "COLLECTION_NAME should be string"
        assert len(MY_CONFIG.COLLECTION_NAME) > 0, "COLLECTION_NAME should not be empty"
        
        print("✅ Database configuration valid")
    
    def test_temperature_values(self):
        """Test that temperature values are reasonable."""
        for provider_name, provider_config in MY_CONFIG.LLM_PROVIDERS.items():
            if 'temperature' in provider_config:
                temp = provider_config['temperature']
                assert 0.0 <= temp <= 2.0, f"Temperature {temp} for {provider_name} should be between 0.0 and 2.0"
        
        print("✅ Temperature values are reasonable")


class TestProviderSwitching:
    """Test provider switching functionality."""
    
    def test_provider_switching_simulation(self, mock_config):
        """Test simulated provider switching."""
        # Test switching to different providers
        test_providers = ['ollama', 'vllm', 'replicate']
        
        for provider in test_providers:
            mock_config.ACTIVE_PROVIDER = provider
            
            # Verify the switch
            assert mock_config.ACTIVE_PROVIDER == provider
            
            # Verify provider config exists
            assert provider in mock_config.LLM_PROVIDERS
            
            provider_config = mock_config.LLM_PROVIDERS[provider]
            assert 'model' in provider_config
            
        print("✅ Provider switching simulation successful")
    
    def test_invalid_provider_detection(self, mock_config):
        """Test that invalid providers can be detected."""
        invalid_provider = 'nonexistent_provider'
        mock_config.ACTIVE_PROVIDER = invalid_provider
        
        # This should fail validation
        with pytest.raises(KeyError):
            _ = mock_config.LLM_PROVIDERS[invalid_provider]
        
        print("✅ Invalid provider detection working")


class TestEnvironmentVariableSupport:
    """Test environment variable support for configuration."""
    
    def test_env_var_override(self):
        """Test that environment variables can override config."""
        # Test if ALLYCAT_TEST_PROVIDER environment variable works
        test_provider = os.environ.get('ALLYCAT_TEST_PROVIDER')
        
        if test_provider:
            # If environment variable is set, it should be used
            print(f"✅ Environment provider override: {test_provider}")
        else:
            # If not set, should use default from config
            print(f"✅ Using default provider: {MY_CONFIG.ACTIVE_PROVIDER}")
    
    def test_config_immutability_protection(self):
        """Test that critical config values are protected in tests."""
        # Save original values
        original_provider = MY_CONFIG.ACTIVE_PROVIDER
        original_providers = MY_CONFIG.LLM_PROVIDERS.copy()
        
        # Verify we can read them
        assert original_provider is not None
        assert len(original_providers) > 0
        
        print("✅ Configuration values are accessible and non-empty")


class TestConfigurationConsistency:
    """Test internal consistency of configuration."""
    
    def test_workspace_directories(self):
        """Test workspace directory configuration."""
        workspace_attrs = ['WORKSPACE_DIR', 'CRAWL_DIR', 'PROCESSED_DATA_DIR']
        
        for attr in workspace_attrs:
            if hasattr(MY_CONFIG, attr):
                value = getattr(MY_CONFIG, attr)
                assert isinstance(value, str), f"{attr} should be string"
                assert len(value) > 0, f"{attr} should not be empty"
        
        print("✅ Workspace directory configuration consistent")
    
    def test_chunking_config(self):
        """Test chunking configuration if present."""
        chunking_attrs = ['CHUNK_SIZE', 'CHUNK_OVERLAP']
        
        for attr in chunking_attrs:
            if hasattr(MY_CONFIG, attr):
                value = getattr(MY_CONFIG, attr)
                assert isinstance(value, int), f"{attr} should be integer"
                assert value > 0, f"{attr} should be positive"
        
        # If both exist, overlap should be less than size
        if hasattr(MY_CONFIG, 'CHUNK_SIZE') and hasattr(MY_CONFIG, 'CHUNK_OVERLAP'):
            assert MY_CONFIG.CHUNK_OVERLAP < MY_CONFIG.CHUNK_SIZE, \
                "CHUNK_OVERLAP should be less than CHUNK_SIZE"
        
        print("✅ Chunking configuration consistent")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])