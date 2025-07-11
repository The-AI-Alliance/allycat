# Customizing Allycat

AllyCat uses a **pluggable provider architecture** that makes it easy to switch between different LLM providers by changing a single configuration setting. All provider configurations are centralized in `my_config.py`, so you never need to modify code files.

## Quick Provider Switching

To switch providers, simply change the `ACTIVE_PROVIDER` setting in [my_config.py](../my_config.py):

```python
MY_CONFIG.ACTIVE_PROVIDER = 'ollama'    # or 'vllm' or 'replicate'
```

## Available Providers

- [1 - Ollama (Local, CPU/GPU)](#1---to-try-a-different-llm-with-ollama)
- [2 - Replicate (Cloud Service)](#2---trying-a-different-model-with-replicate)
- [3 - vLLM (High-Performance Local)](#3---trying-vllm-for-high-performance-inference)
- [4 - Embedding Models](#4---trying-various-embedding-models)


## 1 - To try a different LLM with Ollama

**Step 1: Set Ollama as Active Provider**

Edit file [my_config.py](../my_config.py) and set:

```python
MY_CONFIG.ACTIVE_PROVIDER = 'ollama'
```

**Step 2: Customize Ollama Model**

The Ollama configuration is in the `LLM_PROVIDERS` section:

```python
'ollama': {
    'model': 'gemma3:1b',  # Change this to your desired model
    'request_timeout': 30.0,
    'temperature': 0.1
}
```

To try a different model, change the model name:

```python
'ollama': {
    'model': 'qwen3:1.7b',  # Updated model
    'request_timeout': 30.0,
    'temperature': 0.1
}
```

**Step 3: Download the Ollama Model**

```bash
ollama pull qwen3:1.7b
```

Verify the model is ready:

```bash
ollama list
```

Make sure the model is listed. Now the model is ready to be used.

**Step 4: Run Your Query**

```bash
python 4_query.py
```

## 2 - Trying a different model with Replicate

**Step 1: Set Replicate as Active Provider**

Edit file [my_config.py](../my_config.py) and set:

```python
MY_CONFIG.ACTIVE_PROVIDER = 'replicate'
```

**Step 2: Customize Replicate Model**

The Replicate configuration is in the `LLM_PROVIDERS` section:

```python
'replicate': {
    'model': 'meta/meta-llama-3-8b-instruct',  # Change this to your desired model
    'temperature': 0.1
}
```

To try a different model, change the model name:

```python
'replicate': {
    'model': 'ibm-granite/granite-3.2-8b-instruct',  # Updated model
    'temperature': 0.1
}
```

**Step 3: Ensure API Token is Set**

Make sure you have your Replicate API token in your `.env` file:

```bash
REPLICATE_API_TOKEN=your_token_here
```

**Step 4: Run Your Query**

```bash
python 4_query.py
```

## 3 - Trying a different model with vLLM 


**Step 1: Install vLLM**

See the [vLLM Installation Guide](https://docs.vllm.ai/en/latest/getting_started/installation.html) for detailed instructions.

Quick install:
```bash
pip install vllm
```

**Step 2: Set vLLM as Active Provider**

Edit file [my_config.py](../my_config.py) and set:

```python
MY_CONFIG.ACTIVE_PROVIDER = 'vllm'
```

**Step 3: Customize vLLM Configuration**

The vLLM configuration is in the `LLM_PROVIDERS` section:

```python
'vllm': {
    'model': 'RedHatAI/gemma-3-4b-it-quantized.w4a16',  # Change to your desired model
    'api_key': 'fake',  # vLLM doesn't require real API key
    'api_base': 'http://localhost:8000/v1',  # vLLM server URL
    'temperature': 0.1,
    'context_window': 8192,  # Should match your --max-model-len setting or default model context length
    'is_chat_model': True
}
```

To try a different model, update the configuration:

```python
'vllm': {
    'model': 'Qwen/Qwen2.5-1.5B-Instruct',  # Different model
    'api_key': 'fake',
    'api_base': 'http://localhost:8000/v1',
    'temperature': 0.1,
    'context_window': 120000,  # Should match your --max-model-len setting or default model context length
    'is_chat_model': True
}
```
*Note* The model name you provide in this section must match the model name you pass to `vllm serve`


**Step 4: Start vLLM Server**

Before running AllyCat, start the vLLM server:

```bash
# Start vLLM server with your model
vllm serve RedHatAI/gemma-3-4b-it-quantized.w4a16 --max-model-len 8192

# Server will be available at http://localhost:8000
```

or

```bash
# Start vLLM server with your model
vllm serve Qwen/Qwen2.5-1.5B-Instruct 

# Server will be available at http://localhost:8000
```

**Step 5: Run Your Query**

```bash
python 4_query.py
```

**Key Points:**
- vLLM requires GPU for best performance
- The `context_window` setting should match your `--max-model-len` parameter
- vLLM server must be running before starting AllyCat
- AllyCat connects via OpenAI-compatible API

## 4 - Trying various embedding models

**Edit file [my_config.py](../my_config.py)** and change these lines:

```python
MY_CONFIG.EMBEDDING_MODEL = 'ibm-granite/granite-embedding-30m-english'
MY_CONFIG.EMBEDDING_LENGTH = 384
```

You can find [embedding models here](https://huggingface.co/spaces/mteb/leaderboard)

Once embedding model is changed:

1) Rerun chunking and embedding again

```bash
python  3_save_to_vector_db.py
```

2) Run query

```bash
python   4_query.py
```


