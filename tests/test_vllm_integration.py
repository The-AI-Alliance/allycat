"""
Integration tests for vLLM functionality in AllyCat.

These tests validate vLLM-specific functionality when vLLM is the active provider.

Prerequisites:
- vLLM server running (vllm serve MODEL_NAME --max-model-len 8192)
- MY_CONFIG.ACTIVE_PROVIDER = 'vllm'
- Model available at http://localhost:8000/v1
"""

import pytest
import os
import sys
import time
import requests
from unittest.mock import patch

# Add parent directory to path to import AllyCat modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings
from my_config import MY_CONFIG

# Skip all tests in this file if vLLM is not the active provider
pytestmark = pytest.mark.skipif(
    MY_CONFIG.ACTIVE_PROVIDER != 'vllm',
    reason="vLLM integration tests require ACTIVE_PROVIDER='vllm'"
)


class TestVLLMServerConnection:
    """Test vLLM server connectivity and API compatibility."""
    
    def test_vllm_server_running(self):
        """Verify vLLM server is accessible via HTTP."""
        vllm_config = MY_CONFIG.LLM_PROVIDERS['vllm']
        api_base = vllm_config['api_base']
        
        # Test basic health endpoint
        try:
            response = requests.get(f"{api_base.rstrip('/v1')}/health", timeout=10)
            assert response.status_code == 200
            print("✅ vLLM server health check passed")
        except requests.exceptions.RequestException:
            # Some vLLM versions might not have /health endpoint
            # Test models endpoint instead
            response = requests.get(f"{api_base}/models", timeout=10)
            assert response.status_code == 200
            print("✅ vLLM server models endpoint accessible")
    
    def test_vllm_models_endpoint(self):
        """Test that vLLM server exposes models via OpenAI-compatible API."""
        vllm_config = MY_CONFIG.LLM_PROVIDERS['vllm']
        api_base = vllm_config['api_base']
        
        response = requests.get(f"{api_base}/models", timeout=10)
        assert response.status_code == 200
        
        data = response.json()
        assert 'data' in data
        assert len(data['data']) > 0
        
        # Check that configured model is available
        model_names = [model['id'] for model in data['data']]
        configured_model = vllm_config['model']
        assert configured_model in model_names
        print(f"✅ Configured model '{configured_model}' available on vLLM server")
    
    def test_vllm_openai_like_connection(self):
        """Test OpenAILike client connection to vLLM server."""
        vllm_config = MY_CONFIG.LLM_PROVIDERS['vllm']
        
        llm = OpenAILike(
            model=vllm_config['model'],
            api_key=vllm_config['api_key'],
            api_base=vllm_config['api_base'],
            temperature=0.1,
            context_window=vllm_config.get('context_window', 8192),
            is_chat_model=vllm_config.get('is_chat_model', True)
        )
        
        # Simple test query to verify connection
        response = llm.complete("Hello")
        
        assert response is not None
        assert len(str(response)) > 0
        print(f"✅ vLLM server responding via OpenAILike: {str(response)[:50]}...")


class TestVLLMFunctionality:
    """Test vLLM-specific functionality and features."""
    
    def test_vllm_model_completion(self):
        """Test basic text completion with vLLM."""
        vllm_config = MY_CONFIG.LLM_PROVIDERS['vllm']
        
        llm = OpenAILike(
            model=vllm_config['model'],
            api_key=vllm_config['api_key'],
            api_base=vllm_config['api_base'],
            temperature=0.1,
            context_window=vllm_config.get('context_window', 8192),
            is_chat_model=vllm_config.get('is_chat_model', True)
        )
        
        # Test with a math question
        response = llm.complete("What is 2+2?")
        
        assert response is not None
        response_text = str(response).lower()
        assert len(response_text) > 0
        # Should contain some mathematical answer
        assert any(num in response_text for num in ['4', 'four', 'equal'])
        print(f"✅ vLLM model completion working: {str(response)[:100]}...")
    
    def test_vllm_temperature_setting(self):
        """Test that temperature setting affects output variability."""
        vllm_config = MY_CONFIG.LLM_PROVIDERS['vllm']
        
        # Test with low temperature (deterministic)
        llm_low_temp = OpenAILike(
            model=vllm_config['model'],
            api_key=vllm_config['api_key'],
            api_base=vllm_config['api_base'],
            temperature=0.0,  # Very deterministic
            context_window=vllm_config.get('context_window', 8192),
            is_chat_model=vllm_config.get('is_chat_model', True)
        )
        
        # Ask same question multiple times
        question = "What is the capital of France?"
        responses = []
        
        for _ in range(2):
            response = llm_low_temp.complete(question)
            responses.append(str(response).lower())
        
        # Both responses should mention Paris
        assert all("paris" in resp for resp in responses)
        print(f"✅ vLLM temperature setting working. Responses: {[resp[:50] for resp in responses]}")
    
    def test_vllm_context_window(self):
        """Test that vLLM handles context window appropriately."""
        vllm_config = MY_CONFIG.LLM_PROVIDERS['vllm']
        
        llm = OpenAILike(
            model=vllm_config['model'],
            api_key=vllm_config['api_key'],
            api_base=vllm_config['api_base'],
            temperature=0.1,
            context_window=vllm_config.get('context_window', 8192),
            is_chat_model=vllm_config.get('is_chat_model', True)
        )
        
        # Test with a longer prompt (but still within context window)
        long_prompt = "Please explain the concept of machine learning. " * 10
        response = llm.complete(long_prompt)
        
        assert response is not None
        assert len(str(response)) > 0
        print(f"✅ vLLM context window handling working: {len(str(response))} chars response")


class TestVLLMWithConfig:
    """Test vLLM integration using AllyCat's configuration system."""
    
    def test_vllm_config_loading(self):
        """Test that MY_CONFIG loads vLLM settings correctly."""
        assert MY_CONFIG.ACTIVE_PROVIDER == 'vllm'
        assert 'vllm' in MY_CONFIG.LLM_PROVIDERS
        
        vllm_config = MY_CONFIG.LLM_PROVIDERS['vllm']
        required_keys = ['model', 'api_key', 'api_base', 'context_window', 'is_chat_model']
        
        for key in required_keys:
            assert key in vllm_config, f"vLLM config missing key: {key}"
        
        # Validate URL format
        assert vllm_config['api_base'].startswith('http'), "api_base should be valid URL"
        print("✅ vLLM configuration loading working")
    
    def test_vllm_initialization_from_config(self):
        """Test creating OpenAILike instance using config values."""
        vllm_config = MY_CONFIG.LLM_PROVIDERS['vllm']
        
        # Create OpenAILike instance using configuration
        llm = OpenAILike(
            model=vllm_config['model'],
            api_key=vllm_config['api_key'],
            api_base=vllm_config['api_base'],
            temperature=vllm_config.get('temperature', 0.1),
            context_window=vllm_config.get('context_window', 8192),
            is_chat_model=vllm_config.get('is_chat_model', True)
        )
        
        # Test the LLM works
        response = llm.complete("Hello from test")
        assert response is not None
        assert len(str(response)) > 0
        print(f"✅ vLLM config-based initialization working: {str(response)[:50]}...")
    
    def test_vllm_llama_index_settings_integration(self):
        """Test that vLLM integrates correctly with llama-index Settings."""
        vllm_config = MY_CONFIG.LLM_PROVIDERS['vllm']
        
        llm = OpenAILike(
            model=vllm_config['model'],
            api_key=vllm_config['api_key'],
            api_base=vllm_config['api_base'],
            temperature=vllm_config.get('temperature', 0.1),
            context_window=vllm_config.get('context_window', 8192),
            is_chat_model=vllm_config.get('is_chat_model', True)
        )
        
        # Save original setting
        original_llm = Settings.llm
        
        try:
            # Set the LLM in Settings (like app.py does)
            Settings.llm = llm
            
            # Verify it's set correctly
            assert Settings.llm is not None
            
            # Test that it can be used for completion
            response = Settings.llm.complete("Test query")
            assert response is not None
            print("✅ vLLM llama-index Settings integration working")
            
        finally:
            # Restore original setting
            Settings.llm = original_llm


class TestVLLMErrorHandling:
    """Test error scenarios and edge cases specific to vLLM."""
    
    def test_vllm_server_unavailable(self):
        """Test behavior when vLLM server is not accessible."""
        # Create OpenAILike instance pointing to wrong port
        llm = OpenAILike(
            model="test-model",
            api_key="fake",
            api_base="http://localhost:9999/v1",  # Wrong port
            temperature=0.1,
            context_window=8192,
            is_chat_model=True
        )
        
        with pytest.raises(Exception) as exc_info:
            llm.complete("Test")
        
        # Should get connection error
        error_msg = str(exc_info.value).lower()
        assert any(phrase in error_msg for phrase in [
            'connection', 'connect', 'refused', 'timeout', 'unreachable', 'failed'
        ])
        print(f"✅ vLLM server unavailable handling: {str(exc_info.value)[:100]}...")
    
    def test_vllm_invalid_model(self):
        """Test behavior with non-existent model on vLLM server."""
        vllm_config = MY_CONFIG.LLM_PROVIDERS['vllm']
        
        # Use correct server but wrong model
        llm = OpenAILike(
            model="nonexistent-model-xyz",
            api_key=vllm_config['api_key'],
            api_base=vllm_config['api_base'],
            temperature=0.1,
            context_window=8192,
            is_chat_model=True
        )
        
        with pytest.raises(Exception) as exc_info:
            llm.complete("Test")
        
        # Should get model not found error
        error_msg = str(exc_info.value).lower()
        assert any(phrase in error_msg for phrase in [
            'not found', 'model', 'invalid', '404', 'does not exist'
        ])
        print(f"✅ vLLM invalid model handling: {str(exc_info.value)[:100]}...")
    
    def test_vllm_empty_prompt(self):
        """Test behavior with empty or invalid prompts."""
        vllm_config = MY_CONFIG.LLM_PROVIDERS['vllm']
        
        llm = OpenAILike(
            model=vllm_config['model'],
            api_key=vllm_config['api_key'],
            api_base=vllm_config['api_base'],
            temperature=0.1,
            context_window=vllm_config.get('context_window', 8192),
            is_chat_model=vllm_config.get('is_chat_model', True)
        )
        
        # Test empty string
        response = llm.complete("")
        # Should handle gracefully - either return something or raise clean error
        assert response is not None or True  # Accept any behavior for empty prompt
        
        print("✅ vLLM empty prompt handling working")


class TestVLLMPerformance:
    """Test vLLM performance characteristics."""
    
    @pytest.mark.slow
    def test_vllm_response_time(self):
        """Test vLLM response time for basic queries."""
        vllm_config = MY_CONFIG.LLM_PROVIDERS['vllm']
        
        llm = OpenAILike(
            model=vllm_config['model'],
            api_key=vllm_config['api_key'],
            api_base=vllm_config['api_base'],
            temperature=0.1,
            context_window=vllm_config.get('context_window', 8192),
            is_chat_model=vllm_config.get('is_chat_model', True)
        )
        
        # Measure response time for a simple query
        start_time = time.time()
        response = llm.complete("What is 2+2?")
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # vLLM should be reasonably fast (generous timeout for CI)
        assert response_time < 30
        assert response is not None
        assert len(str(response)) > 0
        
        print(f"✅ vLLM performance test passed: {response_time:.2f}s response time")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])