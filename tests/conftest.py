"""
Pytest configuration and shared fixtures for AllyCat test suite.

Provides provider-agnostic testing infrastructure that works with
the pluggable LLM provider architecture.
"""

import pytest
import os
import sys
import time
from unittest.mock import patch, MagicMock

# Add parent directory to path to import AllyCat modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_config import MY_CONFIG


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "ollama: mark test as requiring Ollama provider"
    )
    config.addinivalue_line(
        "markers", "vllm: mark test as requiring vLLM provider"
    )
    config.addinivalue_line(
        "markers", "replicate: mark test as requiring Replicate provider"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )


@pytest.fixture(scope="session")
def active_provider():
    """Get the currently active LLM provider from configuration."""
    return MY_CONFIG.ACTIVE_PROVIDER


@pytest.fixture(scope="session")
def provider_config(active_provider):
    """Get the configuration for the active provider."""
    return MY_CONFIG.LLM_PROVIDERS[active_provider]


@pytest.fixture(scope="session")
def is_ollama_active(active_provider):
    """Check if Ollama is the active provider."""
    return active_provider == 'ollama'


@pytest.fixture(scope="session")
def is_vllm_active(active_provider):
    """Check if vLLM is the active provider.""" 
    return active_provider == 'vllm'


@pytest.fixture(scope="session")
def is_replicate_active(active_provider):
    """Check if Replicate is the active provider."""
    return active_provider == 'replicate'


@pytest.fixture
def mock_config():
    """Provide a mock configuration for testing."""
    class MockConfig:
        def __init__(self):
            self.ACTIVE_PROVIDER = 'ollama'
            self.LLM_PROVIDERS = {
                'ollama': {
                    'model': 'gemma3:1b',
                    'request_timeout': 30.0,
                    'temperature': 0.1
                },
                'vllm': {
                    'model': 'test-model',
                    'api_key': 'fake',
                    'api_base': 'http://localhost:8000/v1',
                    'temperature': 0.1,
                    'context_window': 8192,
                    'is_chat_model': True
                },
                'replicate': {
                    'model': 'meta/test-model',
                    'temperature': 0.1
                }
            }
            self.EMBEDDING_MODEL = 'ibm-granite/granite-embedding-30m-english'
            self.EMBEDDING_LENGTH = 384
            self.DB_URI = 'test.db'
            self.COLLECTION_NAME = 'test_collection'
    
    return MockConfig()


@pytest.fixture
def llm_service():
    """Initialize LLM service for testing."""
    try:
        from llm_service import initialize_llm_service
        initialize_llm_service()
        print(f"✅ LLM service initialized with provider: {MY_CONFIG.ACTIVE_PROVIDER}")
        return True
    except Exception as e:
        pytest.skip(f"LLM service initialization failed: {e}")


@pytest.fixture
def mock_vector_store():
    """Provide a mock vector store for testing."""
    mock_store = MagicMock()
    mock_store.query.return_value = MagicMock()
    mock_store.query.return_value.response = "Mock response from vector store"
    return mock_store


@pytest.fixture
def flask_app():
    """Create a Flask app instance for testing."""
    # Avoid importing app module at test collection time
    # Import only when needed in the test
    return None


@pytest.fixture
def flask_client(flask_app):
    """Create a Flask test client."""
    if flask_app is None:
        pytest.skip("Flask app not available")
    return flask_app.test_client()


# Skip decorators for provider-specific tests
skip_if_not_ollama = pytest.mark.skipif(
    MY_CONFIG.ACTIVE_PROVIDER != 'ollama',
    reason="Test requires Ollama provider"
)

skip_if_not_vllm = pytest.mark.skipif(
    MY_CONFIG.ACTIVE_PROVIDER != 'vllm', 
    reason="Test requires vLLM provider"
)

skip_if_not_replicate = pytest.mark.skipif(
    MY_CONFIG.ACTIVE_PROVIDER != 'replicate',
    reason="Test requires Replicate provider"
)


def pytest_runtest_setup(item):
    """Setup hook that runs before each test."""
    # Skip tests based on provider markers
    if item.get_closest_marker("ollama") and MY_CONFIG.ACTIVE_PROVIDER != 'ollama':
        pytest.skip("Test requires Ollama provider")
    
    if item.get_closest_marker("vllm") and MY_CONFIG.ACTIVE_PROVIDER != 'vllm':
        pytest.skip("Test requires vLLM provider")
    
    if item.get_closest_marker("replicate") and MY_CONFIG.ACTIVE_PROVIDER != 'replicate':
        pytest.skip("Test requires Replicate provider")


class ProviderTestHelper:
    """Helper class for provider-agnostic testing."""
    
    @staticmethod
    def get_test_query():
        """Get a simple test query that should work with any provider."""
        return "Hello, how are you?"
    
    @staticmethod
    def get_rag_test_query():
        """Get a query that should use RAG context."""
        return "What is AI Alliance?"
    
    @staticmethod
    def get_math_query():
        """Get a math query for testing basic reasoning."""
        return "What is 2+2?"
    
    @staticmethod
    def validate_response(response_text):
        """Validate that a response looks reasonable."""
        if not response_text:
            return False
        
        # Basic validation - should be non-empty string
        if not isinstance(response_text, str):
            return False
            
        # Should have some content
        if len(response_text.strip()) < 3:
            return False
            
        return True
    
    @staticmethod
    def validate_rag_response(response_text):
        """Validate that a response contains RAG-relevant content."""
        if not ProviderTestHelper.validate_response(response_text):
            return False
            
        # Should contain relevant keywords for AI Alliance
        response_lower = response_text.lower()
        relevant_keywords = ['alliance', 'ai', 'artificial', 'intelligence', 'collaboration']
        
        return any(keyword in response_lower for keyword in relevant_keywords)
    
    @staticmethod
    def extract_timing_from_response(response_text):
        """Extract timing information from response if present."""
        import re
        
        # Look for pattern like "time taken: X.X secs"
        timing_pattern = r'time taken:\s*(\d+\.?\d*)\s*secs?'
        match = re.search(timing_pattern, response_text, re.IGNORECASE)
        
        if match:
            return float(match.group(1))
        return None


@pytest.fixture
def provider_helper():
    """Provide the ProviderTestHelper for tests."""
    return ProviderTestHelper


# Timeout fixtures for different test types
@pytest.fixture
def short_timeout():
    """Short timeout for quick tests."""
    return 10.0


@pytest.fixture  
def medium_timeout():
    """Medium timeout for normal tests."""
    return 30.0


@pytest.fixture
def long_timeout():
    """Long timeout for slow/integration tests."""
    return 120.0


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add slow marker to performance tests
        if "performance" in item.name.lower():
            item.add_marker(pytest.mark.slow)
        
        # Add integration marker to integration tests    
        if "integration" in item.name.lower():
            item.add_marker(pytest.mark.integration)
        
        # Add provider markers based on test file names
        if "ollama" in str(item.fspath):
            item.add_marker(pytest.mark.ollama)
        elif "vllm" in str(item.fspath):
            item.add_marker(pytest.mark.vllm)
        elif "replicate" in str(item.fspath):
            item.add_marker(pytest.mark.replicate)


# Environment variable support for test configuration
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment variables and configuration."""
    # Allow overriding provider for tests via environment variable
    test_provider = os.environ.get('ALLYCAT_TEST_PROVIDER')
    if test_provider:
        original_provider = MY_CONFIG.ACTIVE_PROVIDER
        MY_CONFIG.ACTIVE_PROVIDER = test_provider
        print(f"Test environment: Using provider {test_provider}")
        
        yield
        
        # Restore original provider
        MY_CONFIG.ACTIVE_PROVIDER = original_provider
    else:
        yield