<img src="assets/allycat.png" alt="Alley Cat" width="200"/>

![License](https://img.shields.io/github/license/The-AI-Alliance/allycat)
![Issues](https://img.shields.io/github/issues/The-AI-Alliance/allycat)
![GitHub stars](https://img.shields.io/github/stars/The-AI-Alliance/allycat?style=social)

# AllyCat

**AllyCat** is full stack, open source chatbot that uses GenAI LLMs to answer questions about your website. It is simple by design and will run on your laptop or server. 

## Why? ##

AllyCat is purposefully simple so it can be used by developers to learn how RAG-based GenAI works. Yet it is powerful enough to use with your website, You may also extend it for your own purposes. 

## How does it work? 
AllyCat uses your choice of LLM and vector database to implement a chatbot written in Python using [RAG](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) architecture.
AllyCat also includes web scraping tools that extract data from your website (or any website). 

## 🌟🌟 Features 🌟🌟 

1. Chatbot with interface to answer questions with text scraped from a website
   - **Default website:** [thealliance.ai](https://thealliance.ai)
2. Includes web crawling & scraping, text extraction, data/HTML processing, conversion to markdown
   - **Current:** [Data Prep Kit Connector](https://github.com/data-prep-kit/data-prep-kit/blob/dev/data-connector-lib/doc/overview.md), [Docling](https://github.com/docling-project/docling)
3. Processing Chunking, vector embedding creation, saving to vector database
   - **Current:** [Llama Index](https://docs.llamaindex.ai/en/stable/), [Granite Embedding](https://huggingface.co/ibm-granite/granite-embedding-30m-english)
4. Supports multiple LLMs
   - **Current:** [Llama](https://www.llama.com), [Granite](https://huggingface.co/collections/ibm-granite/granite-33-language-models-67f65d0cca24bcbd1d3a08e3)
5. Supports multiple vector databases
   - **Current:** [Milvus](https://milvus.io/), [Weaviate](https://weaviate.io)
6. End User and New Contributor Friendly
   - **Current:** Run locally with [Ollama](https://ollama.com/), or as-a-service on [Replicate](https://replicate.com)

## ⚡️⚡️Quickstart ⚡️⚡️

There are two ways to run Allycat.

### Option 1: Quick start using the AllyCat docker image

This is great option if you want a quick evaluation.  
See [running AllyCat using docker](docs/running-in-docker.md)

### Option 2: Run natively (for tweaking, developing)

Choose this  option if you like to tweak AllyCat to fit your needs. For example, experimenting with embedding models or LLMs.  
See [running AllyCat natively](docs/running-natively.md)


## Why the name **AllyCat**?

Originally AllianceChat, we shortened it to AllyCat when we learned chat means cat in French. Who doesn't love cats?!

## AllyCat Workflow

![](assets/rag-website-1.png)

See [running allycat](docs/running-allycat.md)

## Customizing AllyCat

See [customizing allycat](docs/customizing-allycat.md)

## Deploying AllyCat

See [deployment guide](docs/deploy.md)

## Developing AllyCat

See [developing allycat](docs/developing-allycat.md)
