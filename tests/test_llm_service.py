"""
Tests for AllyCat's pluggable LLM service architecture.

Tests the provider factory, initialization, and provider switching functionality.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Add parent directory to path to import AllyCat modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_service import ProviderFactory, initialize_llm_service
from my_config import MY_CONFIG
from llama_index.core import Settings


class TestProviderFactory:
    """Test the ProviderFactory class and its methods."""
    
    def test_provider_factory_providers_list(self):
        """Test that ProviderFactory knows about all supported providers."""
        expected_providers = ['ollama', 'vllm', 'replicate']
        
        for provider in expected_providers:
            assert provider in ProviderFactory.PROVIDERS, f"Provider {provider} not in factory"
        
        # Check provider mappings exist
        for provider, (module_name, class_name) in ProviderFactory.PROVIDERS.items():
            assert isinstance(module_name, str), f"Module name for {provider} should be string"
            assert isinstance(class_name, str), f"Class name for {provider} should be string"
            assert '.' in module_name, f"Module name {module_name} should be fully qualified"
        
        print("✅ ProviderFactory has all expected providers")
    
    def test_provider_factory_create_llm_validation(self):
        """Test provider factory input validation."""
        # Test invalid provider
        with pytest.raises(ValueError) as exc_info:
            ProviderFactory.create_llm('nonexistent_provider', {})
        
        assert "Unsupported provider" in str(exc_info.value)
        print("✅ ProviderFactory validates provider names")
    
    def test_provider_factory_config_filtering(self):
        """Test that factory filters None values from config."""
        # This is a unit test that doesn't require actual LLM providers
        test_config = {
            'model': 'test-model',
            'temperature': 0.1,
            'timeout': None,  # Should be filtered out
            'other_param': 'value'
        }
        
        # Mock the import and class creation to test filtering logic
        with patch('importlib.import_module') as mock_import:
            mock_module = MagicMock()
            mock_class = MagicMock()
            mock_module.TestClass = mock_class
            mock_import.return_value = mock_module
            
            # Override the provider mapping for testing
            original_providers = ProviderFactory.PROVIDERS.copy()
            ProviderFactory.PROVIDERS['test_provider'] = ('test.module', 'TestClass')
            
            try:
                ProviderFactory.create_llm('test_provider', test_config)
                
                # Verify the class was called without None values
                mock_class.assert_called_once_with(
                    model='test-model',
                    temperature=0.1,
                    other_param='value'
                )
                print("✅ ProviderFactory filters None values from config")
                
            finally:
                # Restore original providers
                ProviderFactory.PROVIDERS = original_providers
    
    def test_provider_factory_import_error_handling(self):
        """Test factory error handling for missing modules."""
        # Test with non-existent module
        original_providers = ProviderFactory.PROVIDERS.copy()
        ProviderFactory.PROVIDERS['test_provider'] = ('nonexistent.module', 'TestClass')
        
        try:
            with pytest.raises(ValueError) as exc_info:
                ProviderFactory.create_llm('test_provider', {'model': 'test'})
            
            assert "Failed to import" in str(exc_info.value)
            print("✅ ProviderFactory handles import errors")
            
        finally:
            # Restore original providers
            ProviderFactory.PROVIDERS = original_providers


class TestLLMServiceInitialization:
    """Test the initialize_llm_service function."""
    
    def test_initialize_llm_service_sets_embedding_model(self):
        """Test that initialization sets up embedding model."""
        # Save original settings
        original_embed_model = Settings.embed_model
        original_llm = Settings.llm
        
        try:
            # Initialize service
            initialize_llm_service()
            
            # Check embedding model is set
            assert Settings.embed_model is not None
            assert hasattr(Settings.embed_model, 'model_name')
            assert MY_CONFIG.EMBEDDING_MODEL in Settings.embed_model.model_name
            print("✅ LLM service sets embedding model correctly")
            
        finally:
            # Restore original settings
            Settings.embed_model = original_embed_model
            Settings.llm = original_llm
    
    def test_initialize_llm_service_sets_llm(self):
        """Test that initialization sets up LLM."""
        # Save original settings
        original_embed_model = Settings.embed_model  
        original_llm = Settings.llm
        
        try:
            # Initialize service
            initialize_llm_service()
            
            # Check LLM is set
            assert Settings.llm is not None
            print(f"✅ LLM service sets LLM correctly for {MY_CONFIG.ACTIVE_PROVIDER}")
            
        finally:
            # Restore original settings
            Settings.embed_model = original_embed_model
            Settings.llm = original_llm
    
    def test_initialize_llm_service_uses_active_provider(self):
        """Test that initialization uses the configured active provider."""
        # Save original settings
        original_embed_model = Settings.embed_model
        original_llm = Settings.llm
        
        try:
            # Initialize service
            llm_instance = initialize_llm_service()
            
            # Verify an LLM was created
            assert llm_instance is not None
            assert Settings.llm is not None
            
            # The specific type depends on the provider, but it should be set
            print(f"✅ LLM service uses active provider {MY_CONFIG.ACTIVE_PROVIDER}")
            
        finally:
            # Restore original settings
            Settings.embed_model = original_embed_model
            Settings.llm = original_llm


class TestProviderSwitching:
    """Test provider switching functionality."""
    
    def test_provider_config_access(self):
        """Test accessing different provider configurations."""
        # Test that all providers have valid configurations
        providers = ['ollama', 'vllm', 'replicate']
        
        for provider in providers:
            assert provider in MY_CONFIG.LLM_PROVIDERS, f"Provider {provider} not in config"
            config = MY_CONFIG.LLM_PROVIDERS[provider]
            
            assert isinstance(config, dict), f"Config for {provider} should be dict"
            assert 'model' in config, f"Provider {provider} missing model config"
            assert isinstance(config['model'], str), f"Model for {provider} should be string"
        
        print("✅ All provider configurations are accessible")
    
    def test_active_provider_mapping(self):
        """Test that active provider maps to valid configuration."""
        active_provider = MY_CONFIG.ACTIVE_PROVIDER
        
        # Check active provider exists in LLM_PROVIDERS
        assert active_provider in MY_CONFIG.LLM_PROVIDERS
        
        # Check configuration is valid
        config = MY_CONFIG.LLM_PROVIDERS[active_provider]
        assert isinstance(config, dict)
        assert 'model' in config
        
        print(f"✅ Active provider {active_provider} has valid configuration")
    
    def test_legacy_compatibility(self):
        """Test that legacy configuration attributes still work."""
        # Test legacy attributes exist
        assert hasattr(MY_CONFIG, 'LLM_RUN_ENV')
        assert hasattr(MY_CONFIG, 'LLM_MODEL')
        
        # Test they map correctly to new system
        active_provider = MY_CONFIG.ACTIVE_PROVIDER
        legacy_model = MY_CONFIG.LLM_MODEL
        provider_model = MY_CONFIG.LLM_PROVIDERS[active_provider]['model']
        
        assert legacy_model == provider_model, "Legacy model should match provider model"
        
        print("✅ Legacy configuration compatibility maintained")


class TestProviderSpecificLogic:
    """Test provider-specific logic and configurations."""
    
    def test_ollama_provider_config(self):
        """Test Ollama provider configuration structure."""
        if 'ollama' not in MY_CONFIG.LLM_PROVIDERS:
            pytest.skip("Ollama provider not configured")
        
        config = MY_CONFIG.LLM_PROVIDERS['ollama']
        
        # Check required keys
        required_keys = ['model', 'request_timeout', 'temperature']
        for key in required_keys:
            assert key in config, f"Ollama config missing {key}"
        
        # Check types
        assert isinstance(config['model'], str)
        assert isinstance(config['request_timeout'], (int, float))
        assert isinstance(config['temperature'], (int, float))
        
        print("✅ Ollama provider configuration structure valid")
    
    def test_vllm_provider_config(self):
        """Test vLLM provider configuration structure."""
        if 'vllm' not in MY_CONFIG.LLM_PROVIDERS:
            pytest.skip("vLLM provider not configured")
        
        config = MY_CONFIG.LLM_PROVIDERS['vllm']
        
        # Check required keys
        required_keys = ['model', 'api_key', 'api_base', 'context_window', 'is_chat_model']
        for key in required_keys:
            assert key in config, f"vLLM config missing {key}"
        
        # Check types
        assert isinstance(config['model'], str)
        assert isinstance(config['api_key'], str)
        assert isinstance(config['api_base'], str)
        assert isinstance(config['context_window'], int)
        assert isinstance(config['is_chat_model'], bool)
        
        # Check URL format
        assert config['api_base'].startswith('http'), "api_base should be valid URL"
        
        print("✅ vLLM provider configuration structure valid")
    
    def test_replicate_provider_config(self):
        """Test Replicate provider configuration structure."""
        if 'replicate' not in MY_CONFIG.LLM_PROVIDERS:
            pytest.skip("Replicate provider not configured")
        
        config = MY_CONFIG.LLM_PROVIDERS['replicate']
        
        # Check required keys
        required_keys = ['model', 'temperature']
        for key in required_keys:
            assert key in config, f"Replicate config missing {key}"
        
        # Check types
        assert isinstance(config['model'], str)
        assert isinstance(config['temperature'], (int, float))
        
        # Check model format (should contain '/')
        assert '/' in config['model'], "Replicate model should be in 'owner/model' format"
        
        print("✅ Replicate provider configuration structure valid")


class TestErrorHandling:
    """Test error handling in the LLM service."""
    
    def test_invalid_provider_error(self):
        """Test error handling for invalid provider."""
        # This tests the factory validation
        with pytest.raises(ValueError) as exc_info:
            ProviderFactory.create_llm('invalid_provider', {'model': 'test'})
        
        assert "Unsupported provider" in str(exc_info.value)
        print("✅ Invalid provider error handling working")
    
    def test_missing_config_key(self):
        """Test error handling for missing configuration keys."""
        # Test with empty config - this should raise an exception
        # Use a known provider that requires specific config keys
        with pytest.raises(Exception):  # Could be KeyError, TypeError, etc.
            ProviderFactory.create_llm('ollama', {})  # Ollama requires 'model' key
        
        print("✅ Missing config key error handling working")


class TestServiceIntegration:
    """Test integration between different parts of the LLM service."""
    
    def test_service_initialization_complete_flow(self):
        """Test complete initialization flow."""
        # Save original settings
        original_embed_model = Settings.embed_model
        original_llm = Settings.llm
        
        try:
            # Test complete initialization
            result = initialize_llm_service()
            
            # Verify all components are set up
            assert result is not None
            assert Settings.embed_model is not None  
            assert Settings.llm is not None
            
            print(f"✅ Complete LLM service initialization successful with {MY_CONFIG.ACTIVE_PROVIDER}")
            
        finally:
            # Restore original settings
            Settings.embed_model = original_embed_model
            Settings.llm = original_llm
    
    def test_service_provider_consistency(self):
        """Test that service uses consistent provider throughout."""
        active_provider = MY_CONFIG.ACTIVE_PROVIDER
        
        # Check that the active provider is valid
        assert active_provider in ProviderFactory.PROVIDERS
        assert active_provider in MY_CONFIG.LLM_PROVIDERS
        
        # Check configuration consistency
        provider_config = MY_CONFIG.LLM_PROVIDERS[active_provider]
        assert 'model' in provider_config
        
        print(f"✅ LLM service provider consistency verified for {active_provider}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])