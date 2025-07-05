# AllyCat Test Suite

This directory contains the comprehensive test suite for AllyCat's pluggable LLM provider architecture.

## Test Structure

### Provider-Agnostic Tests
- `test_app.py` - Flask web application integration tests
- `test_query.py` - CLI query functionality tests  
- `test_config.py` - Configuration validation tests
- `test_llm_service.py` - Pluggable provider factory tests

### Provider-Specific Tests
- `test_ollama_integration.py` - Ollama-specific functionality tests
- `test_vllm_integration.py` - vLLM-specific functionality tests

### Test Infrastructure
- `conftest.py` - Shared fixtures and provider-agnostic test helpers

## Prerequisites for Running Tests

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Stop Running Applications
**IMPORTANT**: Stop any running instances of AllyCat before running tests:
- Stop `app.py` (Flask web server)
- Stop `4_query.py` (CLI interface)
- Any other processes using the configured LLM provider

This prevents resource conflicts and GPU memory issues.

### 3. Provider-Specific Requirements

#### For Ollama Tests (`ACTIVE_PROVIDER='ollama'`)
- Ollama server running: `ollama serve`
- Required model available: `ollama pull gemma3:1b` (or your configured model)
- Sufficient system memory for model loading

#### For vLLM Tests (`ACTIVE_PROVIDER='vllm'`)
- vLLM server running: `vllm serve MODEL_NAME --max-model-len 8192`
- Server accessible at configured `api_base` (default: `http://localhost:8000/v1`)
- Sufficient GPU memory available (tests load actual models)
- No other processes using GPU memory

#### For Replicate Tests (`ACTIVE_PROVIDER='replicate'`)
- Valid `REPLICATE_API_TOKEN` environment variable set
- Active internet connection
- API quota available

### 4. Vector Database
- Milvus Lite database with populated test data
- Database accessible at configured location
- Embedding model available for loading

## Running Tests

### Run All Tests
```bash
python -m pytest tests/ -v
```

### Run Specific Test Categories
```bash
# Provider-agnostic tests only
python -m pytest tests/test_app.py tests/test_query.py tests/test_config.py tests/test_llm_service.py -v

# Current provider integration tests only
python -m pytest tests/ -k "not (ollama_integration or vllm_integration)" -v

# Specific provider tests
python -m pytest tests/test_ollama_integration.py -v  # Only if ACTIVE_PROVIDER='ollama'
python -m pytest tests/test_vllm_integration.py -v   # Only if ACTIVE_PROVIDER='vllm'
```

### Run Tests with Timing
```bash
# Exclude slow tests
python -m pytest tests/ -v -m "not slow"

# Include all tests with timing
python -m pytest tests/ -v --durations=10
```

## Test Behavior

### Conditional Execution
- Provider-specific tests automatically skip when not using that provider
- Tests use `MY_CONFIG.ACTIVE_PROVIDER` to determine which provider to test
- No mocking - tests validate actual service integration

### Real Service Testing
These tests intentionally **do not use mocking** for LLM services. They:
- Load actual models into memory
- Connect to real provider services  
- Test true end-to-end functionality
- Validate production-ready integrations

This approach provides confidence that the system works with real services but requires:
- Actual provider services running
- Sufficient system resources
- Proper configuration

### GPU Memory Considerations
- vLLM tests require significant GPU memory
- Stop other GPU processes before running tests
- Consider running tests on systems with adequate GPU resources
- Tests will fail with CUDA out-of-memory if insufficient GPU memory

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
- Stop running AllyCat applications
- Stop other GPU processes: `nvidia-smi` to check
- Reduce model size or context window in configuration
- Run tests on system with more GPU memory

**Provider Connection Errors**
- Verify provider service is running and accessible
- Check configuration in `my_config.py`
- Validate API keys and endpoints
- Check network connectivity for cloud providers

**Test Failures with "DID NOT RAISE"**
- Ensure provider services are stopped before running factory tests
- Check that test configuration matches actual provider requirements

**Import or Module Errors**
- Install all dependencies: `pip install -r requirements.txt`
- Ensure Python path includes AllyCat modules
- Check for conflicting package versions

### Debug Mode
Run with verbose output and no capture:
```bash
python -m pytest tests/ -v -s --tb=long
```

## Test Configuration

Tests use the same configuration as the main application:
- `MY_CONFIG.ACTIVE_PROVIDER` determines which provider to test
- `MY_CONFIG.LLM_PROVIDERS` provides provider-specific settings
- `MY_CONFIG.EMBEDDING_MODEL` for embedding model configuration

Modify `my_config.py` to test different provider configurations.