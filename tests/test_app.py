"""
Integration tests for Flask app.py functionality.

Tests the web application's pluggable provider integration, initialization,
and chat endpoint behavior with provider-agnostic approach.

Prerequisites:
- Active LLM provider properly configured (see MY_CONFIG.ACTIVE_PROVIDER)
- Vector database populated with test data
- Provider-specific services running (e.g., Ollama, vLLM server, Replicate API key)
"""

import pytest
import json
import sys
import os
import time
from unittest.mock import patch, MagicMock

# Add parent directory to path to import AllyCat modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_config import MY_CONFIG


@pytest.fixture
def app():
    """Create test Flask app instance using current provider configuration."""
    # Import app after ensuring provider is configured
    from app import app as flask_app
    flask_app.config['TESTING'] = True
    
    # Log which provider we're testing with
    print(f"Testing with provider: {MY_CONFIG.ACTIVE_PROVIDER}")
    
    yield flask_app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


class TestAppInitialization:
    """Test Flask app initialization with pluggable provider system."""
    
    def test_app_initialization_success(self, app):
        """Test that app initializes successfully with current provider."""
        # Import the app module
        import app as app_module
        
        # Call initialize (should succeed with properly configured provider)
        try:
            app_module.initialize()
            assert app_module.initialization_complete == True
            print(f"✅ App initialization successful with {MY_CONFIG.ACTIVE_PROVIDER}")
        except Exception as e:
            pytest.fail(f"App initialization failed with {MY_CONFIG.ACTIVE_PROVIDER}: {e}")
    
    def test_app_vector_index_creation(self, app):
        """Test that vector index is created during initialization."""
        import app as app_module
        
        # Initialize the app
        app_module.initialize()
        
        # Check that vector index exists
        assert app_module.vector_index is not None
        assert hasattr(app_module.vector_index, 'as_query_engine')
        print("✅ Vector index created successfully")
    
    def test_app_llm_settings_applied(self, app):
        """Test that LLM settings are correctly applied via llm_service."""
        import app as app_module
        from llama_index.core import Settings
        
        # Initialize the app
        app_module.initialize()
        
        # Check that Settings.llm is configured
        assert Settings.llm is not None
        print(f"✅ LLM settings applied to llama-index via {MY_CONFIG.ACTIVE_PROVIDER}")


class TestAppRoutes:
    """Test Flask app routes and responses."""
    
    def test_index_route(self, client):
        """Test the main index route."""
        response = client.get('/')
        
        assert response.status_code == 200
        assert b'html' in response.data.lower()
        print("✅ Index route working")
    
    def test_index_route_with_init_error(self, app, client):
        """Test index route behavior when initialization fails."""
        # Set an init error
        app.config['INIT_ERROR'] = 'Test initialization error'
        
        response = client.get('/')
        
        assert response.status_code == 200
        # Should contain error message in response
        assert b'error' in response.data.lower() or b'init' in response.data.lower()
        print("✅ Index route handles init errors")


class TestChatEndpoint:
    """Test the /chat endpoint with current provider integration."""
    
    def test_chat_endpoint_basic(self, client, provider_helper):
        """Test basic chat functionality."""
        test_message = provider_helper.get_test_query()
        
        response = client.post('/chat', 
                             json={'message': test_message},
                             content_type='application/json')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'response' in data
        assert provider_helper.validate_response(data['response'])
        assert 'time taken' in data['response']  # Should include timing
        print(f"✅ Basic chat working with {MY_CONFIG.ACTIVE_PROVIDER}: {data['response'][:100]}...")
    
    def test_chat_endpoint_with_rag_query(self, client, provider_helper):
        """Test chat with a query that should use RAG."""
        test_message = provider_helper.get_rag_test_query()
        
        response = client.post('/chat',
                             json={'message': test_message},
                             content_type='application/json')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'response' in data
        assert len(data['response']) > 10  # Should be substantial response
        
        # Response should contain relevant information
        assert provider_helper.validate_rag_response(data['response'])
        print(f"✅ RAG query working with {MY_CONFIG.ACTIVE_PROVIDER}: {data['response'][:150]}...")
    
    def test_chat_endpoint_empty_message(self, client):
        """Test chat endpoint with empty message."""
        response = client.post('/chat',
                             json={'message': ''},
                             content_type='application/json')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'response' in data
        # Should handle empty message gracefully
        assert len(data['response']) > 0
        print("✅ Empty message handling working")
    
    def test_chat_endpoint_missing_message(self, client):
        """Test chat endpoint with missing message field."""
        response = client.post('/chat',
                             json={},
                             content_type='application/json')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'response' in data
        # Should handle missing message gracefully
        print("✅ Missing message handling working")
    
    def test_chat_endpoint_invalid_json(self, client):
        """Test chat endpoint with invalid JSON."""
        response = client.post('/chat',
                             data='invalid json',
                             content_type='application/json')
        
        # Should return 400 for invalid JSON
        assert response.status_code == 400
        print("✅ Invalid JSON handling working")
    
    @pytest.mark.slow
    def test_chat_endpoint_performance(self, client, provider_helper, medium_timeout):
        """Test that chat endpoint responds within reasonable time."""
        test_message = provider_helper.get_math_query()
        
        start_time = time.time()
        response = client.post('/chat',
                             json={'message': test_message},
                             content_type='application/json')
        end_time = time.time()
        
        assert response.status_code == 200
        
        # Should respond within configured timeout
        response_time = end_time - start_time
        assert response_time < medium_timeout
        
        data = json.loads(response.data)
        # Response should include timing information
        assert 'time taken' in data['response']
        print(f"✅ Performance test passed with {MY_CONFIG.ACTIVE_PROVIDER}: {response_time:.1f}s response time")


class TestAppErrorHandling:
    """Test error handling in the Flask app."""
    
    def test_app_handles_llm_errors(self, client):
        """Test app behavior when LLM encounters errors."""
        # Mock the get_llm_response function to return an error message
        # (simulating the internal error handling within get_llm_response)
        with patch('app.get_llm_response') as mock_response:
            mock_response.return_value = "Sorry, I encountered an error while processing your request: Test LLM error"
            
            response = client.post('/chat',
                                 json={'message': 'test'},
                                 content_type='application/json')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'response' in data
            assert 'error' in data['response'].lower()
            print("✅ LLM error handling working")
    
    def test_app_handles_vector_store_errors(self, app):
        """Test app behavior when vector store is unavailable."""
        # This would require mocking the vector store initialization
        # For now, just verify the app can handle initialization errors
        import app as app_module
        
        # If initialization failed, app should still serve basic routes
        if not app_module.initialization_complete:
            with app.test_client() as client:
                response = client.get('/')
                assert response.status_code == 200
                print("✅ Vector store error handling working")
        else:
            print("✅ Vector store available, skipping error test")


class TestAppConfiguration:
    """Test app configuration and settings."""
    
    def test_app_uses_config_settings(self, app):
        """Test that app uses MY_CONFIG settings correctly."""
        import app as app_module
        from llama_index.core import Settings
        
        app_module.initialize()
        
        # Verify settings were applied
        assert Settings.llm is not None
        print(f"✅ App configuration integration working with {MY_CONFIG.ACTIVE_PROVIDER}")
    
    def test_app_embedding_model_setup(self, app):
        """Test that embedding model is configured correctly."""
        import app as app_module
        from llama_index.core import Settings
        
        app_module.initialize()
        
        # Check that embedding model is set
        assert Settings.embed_model is not None
        assert hasattr(Settings.embed_model, 'model_name')
        print("✅ Embedding model configuration working")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])