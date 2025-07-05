<img src="assets/allycat.png" alt="Alley Cat" width="200"/>

[![License](https://img.shields.io/github/license/The-AI-Alliance/allycat)](https://github.com/The-AI-Alliance/allycat/blob/main/LICENSE)
[![Issues](https://img.shields.io/github/issues/The-AI-Alliance/allycat)](https://github.com/The-AI-Alliance/allycat/issues)
![GitHub stars](https://img.shields.io/github/stars/The-AI-Alliance/allycat?style=social)

# AllyCat

**AllyCat** is a full-stack, open source chatbot with **pluggable LLM providers** that uses GenAI to answer questions about your website. It is simple by design and scales from laptop to production server. 

## Why?

AllyCat is purposefully simple so it can be used by developers to learn how RAG-based GenAI works. Yet it is powerful enough to use with your website, You may also extend it for your own purposes. 

## How does it work? 
AllyCat uses your choice of LLM and vector database to implement a chatbot written in Python using [RAG](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) architecture.
AllyCat also includes web scraping tools that extract data from your website (or any website). 

## 🌟🌟 Features 🌟🌟 

1. Chatbot with interface to answer questions with text scraped from a website.
2. Includes web crawling & scraping, text extraction, data/HTML processing, conversion to markdown.
   - **Currently uses:** [Data Prep Kit Connector](https://github.com/data-prep-kit/data-prep-kit/blob/dev/data-connector-lib/doc/overview.md) and [Docling](https://github.com/docling-project/docling)
3. Processing Chunking, vector embedding creation, saving to vector database.
   - **Currently uses:** [Llama Index](https://docs.llamaindex.ai/en/stable/) and [Granite Embedding Model](https://huggingface.co/ibm-granite/granite-embedding-30m-english)
4. Supports multiple LLMs.
   - **Currently:** [Llama](https://www.llama.com) or [Granite](https://huggingface.co/collections/ibm-granite/granite-33-language-models-67f65d0cca24bcbd1d3a08e3)
5. Supports multiple vector databases.
   - **Currently:** [Milvus](https://milvus.io/) or [Weaviate](https://weaviate.io)
6. End User and New Contributor Friendly.
   - **Currently:** Run locally with [Ollama](https://ollama.com/) or [vLLM](https://docs.vllm.ai/en/latest/index.html), or as-a-service on [Replicate](https://replicate.com)
7. Easily switch between inference providers - **Pluggable LLM Provider Architecture** - 
   - **Local:** [Ollama](https://ollama.com/) (lightweight, CPU/GPU)
   - **High-Performance:** [vLLM](https://docs.vllm.ai/) (optimized GPU inference)  
   - **Cloud:** [Replicate](https://replicate.com) (managed service)
   - **Extensible:** Add any LLM provider with a [LlamaIndex implementation](https://docs.llamaindex.ai/en/stable/module_guides/models/llms/)
8. Supports multiple vector databases
   - **Currently:** [Milvus](https://milvus.io/) or [Weaviate](https://weaviate.io)
9. Simple Configuration-Driven Setup
   - **One-line provider switching:** Change `ACTIVE_PROVIDER` in config file
   - **No code changes required:** All provider settings centralized in `my_config.py`
   - **Beginner-friendly:** Works with local Ollama or cloud services

## 🔌 Pluggable Architecture

AllyCat uses a **pluggable provider system** that makes it easy to:

- **Switch LLM providers** by changing one config setting
- **Add new providers** that have LlamaIndex implementations  
- **Configure per-provider settings** without touching code
- **Scale from laptop to production** using the same codebase

**Supported Providers:**
| Provider | Best For | Setup |
|----------|----------|-------|
| [Ollama](https://ollama.com/) | Local development, learning | `pip install` + model download |
| [vLLM](https://docs.vllm.ai/) | High-performance inference | `pip install vllm` + GPU |
| [Replicate](https://replicate.com) | Cloud inference, scaling | API key setup |

**Adding New Providers:** Any LLM with a [LlamaIndex integration](https://docs.llamaindex.ai/en/stable/module_guides/models/llms/) can be added to the factory pattern in `llm_service.py`.

## ⚡️⚡️Quickstart ⚡️⚡️

There are two ways to run Allycat.

### Option 1: Use the Docker image

A great option for a quick evaluation.  
See [running AllyCat using docker](docs/running-in-docker.md)

### Option 2: Run natively (for tweaking, developing)

Choose this option if you want to tweak AllyCat to fit your needs. For example, experimenting with embedding models or LLMs.  
See [running AllyCat natively](docs/running-natively.md)

## AllyCat Workflow

![](assets/rag-website-1.png)

See [running allycat](docs/running-allycat.md)

## Customizing AllyCat

See [customizing allycat](docs/customizing-allycat.md)

## Deploying AllyCat

See [deployment guide](docs/deploy.md)

## Developing AllyCat

See [developing allycat](docs/developing-allycat.md)

## Why the name **AllyCat**?

Originally AllianceChat, we shortened it to AllyCat when we learned chat means cat in French. Who doesn't love cats?!


