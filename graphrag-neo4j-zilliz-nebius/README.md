# AllyCAT GraphRAG - Cloud Setup (Nebius + Zilliz)

This is the **cloud lightweight version** of AllyCAT with full GraphRAG implementation. Uses cloud services - minimal local resources needed.

## What's Inside

**Full GraphRAG Pipeline:**
- Entity and relationship extraction
- Community detection (Louvain algorithm)  
- Community summaries generation
- Neo4j Aura graph database (cloud)
- Zilliz Cloud vector database

**Cloud Infrastructure:**
- Nebius/OpenAI for LLM processing
- Zilliz Cloud for vector storage
- Neo4j Aura for graph database
- Lightweight Docker image (no Ollama)

**Query System:**
- 7-phase GraphRAG query with query-aware ranking
- Vector search augmentation
- Flask and Chainlit web interfaces

## Quick Start

### 1. Setup Environment

Copy and configure your environment:
```bash
cp env.sample.txt .env
# Edit .env with your cloud credentials
```

Required settings:
- `WEBSITE_URL` - website to crawl
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` - Neo4j Aura connection
- `ZILLIZ_URI`, `ZILLIZ_TOKEN` - Zilliz Cloud credentials
- `LLM_RUN_ENV=cloud` - use cloud LLM
- `VECTOR_DB_TYPE=cloud_zilliz` - use Zilliz Cloud
- `NEBIUS_API_KEY` or `OPENAI_API_KEY` - LLM API key

### 2. Run with Docker

```bash
# Build and start
docker-compose up --build

# Or run in background
docker-compose up -d
```

Access the app at `http://localhost:8080`

### 3. Auto-Run Pipeline (Optional)

To automatically run the full pipeline on startup:
```bash
# In .env file:
AUTO_RUN_PIPELINE=true
```

This will:
1. Crawl your website
2. Process files to markdown
3. Save to Zilliz Cloud vector DB
4. Process GraphRAG (entities, communities, summaries)
5. Save graph to Neo4j Aura
6. Start the web app

### 4. Run Manually (Without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline step by step
python 1_crawl_site.py
python 2_process_files.py
python 3_save_to_vector_db_zilliz.py
python 2b_process_graph_data.py
python 3_save_to_neo4j.py

# Start app
python app_flask_graph.py
# or
chainlit run app_chainlit_graph.py
```

## Configuration

### Cloud LLM Models

Default: Nebius `meta-llama/Meta-Llama-3.1-8B-Instruct`

Change in `.env`:
```
LLM_MODEL=nebius/meta-llama/Meta-Llama-3.1-70B-Instruct
# or use OpenAI
LLM_MODEL=openai/gpt-4
```

### GraphRAG Settings

In `my_config.py`:
- `GRAPHRAG_LLM_PROVIDER` - which LLM for GraphRAG processing (default: nebius)
- Entity extraction prompts
- Community detection parameters
- Summary generation settings

### Cloud Services

**Zilliz Cloud:**
- Free tier: 2 CU, good for testing
- Paid plans start at $0.12/hour

**Neo4j Aura:**
- Free tier available
- Pay-as-you-go pricing

**Nebius:**
- Free tier with limits
- Pay per token after

## File Structure

```
├── 1_crawl_site.py              # Website crawler
├── 2_process_files.py           # Convert to markdown
├── 2b_process_graph_data.py     # GraphRAG processing (all phases)
├── 3_save_to_vector_db_zilliz.py # Save to Zilliz Cloud
├── 3_save_to_neo4j.py           # Save graph to Neo4j Aura
├── 4_query_graph.py             # CLI query interface
├── app_flask_graph.py           # Flask web app
├── app_chainlit_graph.py        # Chainlit chat app
├── process_graph_data/          # GraphRAG phase implementations
├── query_functions/             # 7-phase query system
├── Dockerfile                   # Docker image (lightweight, no Ollama)
└── docker-compose.yml           # Docker setup
```

## System Requirements

**Minimum:**
- 2GB RAM (very light!)
- 2GB disk space
- CPU: 2 cores

Much lighter than local version since no Ollama or local Milvus.

## Cost Estimation

**For small website (100 pages):**
- Nebius LLM: ~$0.50-2.00
- Zilliz Cloud: ~$0.01-0.10/day
- Neo4j Aura: Free tier OK

**For medium website (1000 pages):**
- Nebius LLM: ~$5-15
- Zilliz Cloud: ~$0.50-2.00/day
- Neo4j Aura: ~$10-20/month

Use free tiers for testing!

## Troubleshooting

**Zilliz connection failed:**
- Check URI and token in `.env`
- Verify cluster is running
- Check network/firewall

**Neo4j Aura connection failed:**
- Verify credentials in `.env`
- Check cluster status
- Whitelist your IP if needed

**LLM API errors:**
- Check API key is valid
- Verify you have credits/quota
- Try different model if rate limited

## Why Cloud?

✅ **Lightweight** - minimal local resources needed  
✅ **Scalable** - handles large datasets easily  
✅ **Fast** - cloud LLMs respond quickly  
✅ **No maintenance** - managed services  
✅ **Deploy anywhere** - works on small VPS  

## Need Local Instead?

Check out `../graphrag-neo4j-milvus-ollama/` for local version with Ollama (no API costs, full privacy).

## Support

Issues? Questions? Open an issue on GitHub or ask in discussions.
