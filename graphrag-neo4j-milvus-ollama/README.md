# AllyCAT GraphRAG - Local Setup (Ollama + Milvus)

This is the **local heavy version** of AllyCAT with full GraphRAG implementation. Everything runs on your machine - no cloud dependencies.

## What's Inside

**Full GraphRAG Pipeline:**
- Entity and relationship extraction
- Community detection (Louvain algorithm)
- Community summaries generation
- Neo4j graph database storage
- Local Milvus vector database

**Local Infrastructure:**
- Ollama for running LLMs locally
- Milvus Lite for vector storage
- Neo4j for graph database
- No API costs, full privacy

**Query System:**
- 7-phase GraphRAG query with query-aware ranking
- Vector search augmentation
- Flask and Chainlit web interfaces

## Quick Start

### 1. Setup Environment

Copy and configure your environment:
```bash
cp env.sample.txt .env
# Edit .env with your settings
```

Required settings:
- `WEBSITE_URL` - website to crawl
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` - Neo4j connection
- `LLM_RUN_ENV=local_ollama` - use local Ollama
- `VECTOR_DB_TYPE=local` - use local Milvus

### 2. Run with Docker

```bash
# Build and start
docker-compose up --build

# Or run in background
docker-compose up -d
```

Access the app at `http://localhost:8080`

Ollama runs on port `11434`

### 3. Auto-Run Pipeline (Optional)

To automatically run the full pipeline on startup:
```bash
# In .env file:
AUTO_RUN_PIPELINE=true
```

This will:
1. Crawl your website
2. Process files to markdown
3. Save to Milvus vector DB
4. Process GraphRAG (entities, communities, summaries)
5. Save graph to Neo4j
6. Start the web app

### 4. Run Manually (Without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline step by step
python 1_crawl_site.py
python 2_process_files.py
python 3_save_to_vector_db.py
python 2b_process_graph_data.py
python 3_save_to_neo4j.py

# Start app
python app_flask_graph.py
# or
chainlit run app_chainlit_graph.py
```

## Configuration

### Ollama Models

Default model: `gemma3:1b` (small, fast)

Change in `.env`:
```
OLLAMA_MODEL=gemma2:2b
# or
OLLAMA_MODEL=llama3.1:8b
```

First run downloads the model automatically.

### GraphRAG Settings

In `my_config.py`:
- `GRAPHRAG_LLM_PROVIDER` - which LLM for GraphRAG processing (default: cerebras - free tier)
- Entity extraction prompts
- Community detection parameters
- Summary generation settings

### Memory Optimization

Docker automatically cleans up pipeline dependencies after processing to save RAM (350-500MB saved).

Disable cleanup:
```bash
# In .env:
CLEANUP_PIPELINE_DEPS=false
```

## File Structure

```
├── 1_crawl_site.py              # Website crawler
├── 2_process_files.py           # Convert to markdown
├── 2b_process_graph_data.py     # GraphRAG processing (all phases)
├── 3_save_to_vector_db.py       # Save to Milvus
├── 3_save_to_neo4j.py           # Save graph to Neo4j
├── 4_query_graph.py             # CLI query interface
├── app_flask_graph.py           # Flask web app
├── app_chainlit_graph.py        # Chainlit chat app
├── process_graph_data/          # GraphRAG phase implementations
├── query_functions/             # 7-phase query system
├── Dockerfile                   # Docker image (with Ollama)
└── docker-compose.yml           # Docker setup
```

## System Requirements

**Minimum:**
- 8GB RAM (16GB recommended)
- 10GB disk space
- CPU: 4 cores+

**For larger models:**
- 16GB+ RAM
- GPU (optional, speeds up inference)

## Troubleshooting

**Ollama not starting:**
- Check logs: `docker logs allycat-local`
- Verify port 11434 is free
- Try smaller model (gemma3:1b)

**Out of memory:**
- Use smaller Ollama model
- Reduce batch sizes in `my_config.py`
- Enable cleanup: `CLEANUP_PIPELINE_DEPS=true`

**Neo4j connection failed:**
- Check Neo4j is running
- Verify credentials in `.env`
- Check firewall/network settings

## Why Local?

✅ **Privacy** - your data never leaves your machine  
✅ **No API costs** - run unlimited queries  
✅ **Full control** - customize everything  
✅ **Offline capable** - no internet needed after setup  

## Need Cloud Instead?

Check out `../graphrag-neo4j-zilliz-nebius/` for cloud-based lightweight version with Nebius LLM and Zilliz Cloud.

## Support

Issues? Questions? Open an issue on GitHub or ask in discussions.
