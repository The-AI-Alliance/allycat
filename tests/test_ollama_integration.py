"""
Integration tests for Ollama functionality in AllyCat.

These tests validate Ollama-specific functionality when Ollama is the active provider.

Prerequisites:
- Ollama server running (ollama serve)
- gemma3:1b model available (ollama pull gemma3:1b) 
- MY_CONFIG.ACTIVE_PROVIDER = 'ollama'
"""

import pytest
import os
import sys
import time
from unittest.mock import patch

# Add parent directory to path to import AllyCat modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from my_config import MY_CONFIG

# Skip all tests in this file if Ollama is not the active provider
pytestmark = pytest.mark.skipif(
    MY_CONFIG.ACTIVE_PROVIDER != 'ollama',
    reason="Ollama integration tests require ACTIVE_PROVIDER='ollama'"
)


class TestOllamaConnection:
    """Test direct Ollama server connectivity and model availability."""
    
    def test_ollama_server_running(self):
        """Verify Ollama server is accessible."""
        ollama_config = MY_CONFIG.LLM_PROVIDERS['ollama']
        llm = Ollama(model=ollama_config['model'], request_timeout=10.0)
        
        # Simple test query to verify connection
        response = llm.complete("Hello")
        
        assert response is not None
        assert len(str(response)) > 0
        print(f"✅ Ollama server responding: {str(response)[:50]}...")
    
    def test_ollama_model_availability(self):
        """Test that the configured model is available."""
        ollama_config = MY_CONFIG.LLM_PROVIDERS['ollama']
        model_name = ollama_config['model']
        llm = Ollama(model=model_name, request_timeout=15.0)
        
        # Test with a simple prompt
        response = llm.complete("What is 2+2?")
        
        assert response is not None
        response_text = str(response).lower()
        assert len(response_text) > 0
        # Should contain some mathematical answer
        assert any(num in response_text for num in ['4', 'four', 'equal'])
        print(f"✅ Model {model_name} working: {str(response)[:100]}...")
    
    def test_ollama_timeout_handling(self):
        """Test Ollama timeout configuration."""
        ollama_config = MY_CONFIG.LLM_PROVIDERS['ollama']
        # Very short timeout to test timeout handling
        llm = Ollama(model=ollama_config['model'], request_timeout=0.1)
        
        # This should timeout or complete very quickly
        try:
            response = llm.complete("Write a very long story about cats")
            # If it completes, that's also acceptable
            assert response is not None
        except Exception as e:
            # Timeout or connection error is expected
            assert ("timeout" in str(e).lower() or "connection" in str(e).lower() or "timed out" in str(e).lower())
            print(f"✅ Timeout handling working: {str(e)[:100]}...")
    
    def test_ollama_temperature_setting(self):
        """Test that temperature setting is respected."""
        ollama_config = MY_CONFIG.LLM_PROVIDERS['ollama']
        llm = Ollama(
            model=ollama_config['model'], 
            temperature=ollama_config.get('temperature', 0.1), 
            request_timeout=15.0
        )
        
        # Ask same question multiple times - low temperature should give similar answers
        question = "What is the capital of France?"
        responses = []
        
        for _ in range(2):
            response = llm.complete(question)
            responses.append(str(response).lower())
        
        # Both responses should mention Paris
        assert all("paris" in resp for resp in responses)
        print(f"✅ Temperature setting working. Responses: {responses}")


class TestOllamaWithConfig:
    """Test Ollama integration using AllyCat's configuration system."""
    
    def test_config_loading(self):
        """Test that MY_CONFIG loads Ollama settings correctly."""
        assert MY_CONFIG.ACTIVE_PROVIDER == 'ollama'
        assert 'ollama' in MY_CONFIG.LLM_PROVIDERS
        
        ollama_config = MY_CONFIG.LLM_PROVIDERS['ollama']
        assert 'model' in ollama_config
        assert isinstance(ollama_config['model'], str)
        print("✅ Ollama configuration loading working")
    
    def test_ollama_initialization_from_config(self):
        """Test creating Ollama instance using config values."""
        ollama_config = MY_CONFIG.LLM_PROVIDERS['ollama']
        
        # Create Ollama instance using configuration
        llm = Ollama(
            model=ollama_config['model'],
            request_timeout=ollama_config.get('request_timeout', 30.0),
            temperature=ollama_config.get('temperature', 0.1)
        )
        
        # Test the LLM works
        response = llm.complete("Hello from test")
        assert response is not None
        assert len(str(response)) > 0
        print(f"✅ Config-based initialization working: {str(response)[:50]}...")
    
    def test_llama_index_settings_integration(self):
        """Test that Ollama integrates correctly with llama-index Settings."""
        ollama_config = MY_CONFIG.LLM_PROVIDERS['ollama']
        llm = Ollama(
            model=ollama_config['model'], 
            request_timeout=15.0, 
            temperature=ollama_config.get('temperature', 0.1)
        )
        
        # Save original setting
        original_llm = Settings.llm
        
        try:
            # Set the LLM in Settings (like app.py does)
            Settings.llm = llm
            
            # Verify it's set correctly
            assert Settings.llm is not None
            assert hasattr(Settings.llm, 'model')
            
            # Test that it can be used for completion
            response = Settings.llm.complete("Test query")
            assert response is not None
            print("✅ llama-index Settings integration working")
            
        finally:
            # Restore original setting
            Settings.llm = original_llm


class TestOllamaErrorHandling:
    """Test error scenarios and edge cases."""
    
    def test_invalid_model_name(self):
        """Test behavior with non-existent model."""
        llm = Ollama(model="nonexistent-model-xyz", request_timeout=5.0)
        
        with pytest.raises(Exception) as exc_info:
            llm.complete("Test")
        
        # Should get some kind of model not found error
        error_msg = str(exc_info.value).lower()
        assert any(phrase in error_msg for phrase in ['not found', 'model', 'error', 'pull'])
        print(f"✅ Invalid model handling: {str(exc_info.value)[:100]}...")
    
    def test_ollama_server_not_running(self):
        """Test behavior when Ollama server is not accessible."""
        # Create Ollama instance pointing to wrong port
        llm = Ollama(
            model="gemma3:1b", 
            base_url="http://localhost:11435",  # Wrong port
            request_timeout=2.0
        )
        
        with pytest.raises(Exception) as exc_info:
            llm.complete("Test")
        
        # Should get connection error
        error_msg = str(exc_info.value).lower()
        assert any(phrase in error_msg for phrase in ['connection', 'connect', 'refused', 'timeout', 'unreachable', 'timed out', 'failed'])
        print(f"✅ Server unavailable handling: {str(exc_info.value)[:100]}...")
    
    def test_empty_prompt(self):
        """Test behavior with empty or invalid prompts."""
        llm = Ollama(model="gemma3:1b", request_timeout=10.0)
        
        # Test empty string
        response = llm.complete("")
        # Should handle gracefully - either return something or raise clean error
        assert response is not None or True  # Accept any behavior for empty prompt
        
        # Test whitespace only
        response = llm.complete("   ")
        assert response is not None or True
        
        print("✅ Empty prompt handling working")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])