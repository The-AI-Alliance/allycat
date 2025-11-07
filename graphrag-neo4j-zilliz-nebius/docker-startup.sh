#!/bin/bash

echo "=== AllyCAT GraphRAG Docker Startup ==="

# Check deployment mode from environment
LLM_MODE=${LLM_RUN_ENV:-cloud}
VECTOR_MODE=${VECTOR_DB_TYPE:-cloud_zilliz}

echo "LLM Mode: $LLM_MODE (Cloud)"
echo "Vector DB Mode: $VECTOR_MODE (Zilliz Cloud)"

# Using cloud LLM (Nebius/OpenAI) - no Ollama needed
echo "✅ Using cloud LLM mode"

# Using Zilliz Cloud - no local setup needed
echo "✅ Using Zilliz Cloud for vector database"
mkdir -p /allycat/workspace

# Run GraphRAG pipeline if AUTO_RUN_PIPELINE is enabled and in deploy mode
if [ "$1" == "deploy" ] && [ "${AUTO_RUN_PIPELINE:-false}" = "true" ]; then
    echo ""
    echo "=== Running GraphRAG Pipeline Automatically ==="
    echo ""
    
    # Step 1: Crawl website
    if [ -n "$WEBSITE_URL" ]; then
        echo "Step 1/5: Crawling website: $WEBSITE_URL"
        python3 1_crawl_site.py || echo "⚠️  Warning: Crawl failed, continuing..."
        echo "✅ Step 1 complete"
        echo ""
    else
        echo "⚠️  Skipping crawl - WEBSITE_URL not set"
    fi
    
    # Step 2: Process files to markdown
    echo "Step 2/5: Processing files to markdown..."
    python3 2_process_files.py || echo "⚠️  Warning: Processing failed, continuing..."
    echo "✅ Step 2 complete"
    echo ""
    
    # Step 3: Save to Zilliz Cloud vector database
    echo "Step 3/5: Saving to Zilliz Cloud vector database..."
    python3 3_save_to_vector_db_zilliz.py || echo "⚠️  Warning: Vector DB save failed, continuing..."
    echo "✅ Step 3 complete"
    echo ""
    
    # Step 4: Process graph data (orchestrator script)
    echo "Step 4/5: Processing graph data (GraphRAG phases)..."
    python3 2_process_graph_data.py || echo "⚠️  Warning: Graph processing failed, continuing..."
    echo "✅ Step 4 complete"
    echo ""
    
    # Step 5: Save to graph database (Neo4j)
    echo "Step 5/5: Saving to graph database (Neo4j)..."
    python3 3_save_to_neo4j.py || echo "⚠️  Warning: Graph DB save failed, continuing..."
    echo "✅ Step 5 complete"
    echo ""
    
    echo "=== ✅ Pipeline Complete - Starting Application ==="
    echo ""
    
    # OPTIMIZATION: Clean up pipeline dependencies to save RAM
    if [ "${CLEANUP_PIPELINE_DEPS:-false}" = "true" ]; then
        echo ""
        echo "=== 🧹 Cleaning Up Pipeline Dependencies ==="
        echo "This will save ~350-500 MB of RAM"
        echo ""
        chmod +x ./cleanup_pipeline_deps.sh
        ./cleanup_pipeline_deps.sh
        echo ""
        echo "=== ✅ Cleanup Complete ==="
        echo ""
    else
        echo ""
        echo "💡 TIP: Set CLEANUP_PIPELINE_DEPS=true in .env to save ~350-500 MB RAM"
        echo "        after pipeline completes (reduces OOM errors on 1GB containers)"
        echo ""
    fi
fi

# Start the appropriate web application
APP_TYPE=${APP_TYPE:-flask_graph}
DOCKER_APP_PORT=${DOCKER_APP_PORT:-8080}
FLASK_GRAPH_PORT=${FLASK_GRAPH_PORT:-8080}
FLASK_VECTOR_PORT=${FLASK_VECTOR_PORT:-8081}
CHAINLIT_GRAPH_PORT=${CHAINLIT_GRAPH_PORT:-8083}
CHAINLIT_VECTOR_PORT=${CHAINLIT_VECTOR_PORT:-8082}

# Log port configuration
echo ""
echo "=== Port Configuration ==="
echo "DOCKER_APP_PORT (internal container): $DOCKER_APP_PORT"
echo "FLASK_GRAPH_PORT: $FLASK_GRAPH_PORT"
echo "FLASK_VECTOR_PORT: $FLASK_VECTOR_PORT"
echo "CHAINLIT_GRAPH_PORT: $CHAINLIT_GRAPH_PORT"
echo "CHAINLIT_VECTOR_PORT: $CHAINLIT_VECTOR_PORT"
echo ""

# Determine which port will be used based on APP_TYPE
case $APP_TYPE in
    "flask_graph")
        APP_PORT=$FLASK_GRAPH_PORT
        ;;
    "chainlit_graph")
        APP_PORT=$CHAINLIT_GRAPH_PORT
        ;;
    "flask")
        APP_PORT=$FLASK_VECTOR_PORT
        ;;
    "chainlit")
        APP_PORT=$CHAINLIT_VECTOR_PORT
        ;;
    *)
        APP_PORT=$FLASK_GRAPH_PORT
        ;;
esac

echo "Selected APP_TYPE: $APP_TYPE will run on port: $APP_PORT"
echo "Container will expose application on port: $DOCKER_APP_PORT (mapped to host DOCKER_PORT)"
echo ""

if [ "$1" == "deploy" ]; then
    echo "In deploy mode..."
    
    case $APP_TYPE in
        "flask_graph")
            echo "Starting Flask GraphRAG app on port $FLASK_GRAPH_PORT..."
            python3 app_flask_graph.py
            ;;
        "chainlit_graph")
            echo "Starting Chainlit GraphRAG app on port $CHAINLIT_GRAPH_PORT..."
            chainlit run app_chainlit_graph.py --host 0.0.0.0 --port $CHAINLIT_GRAPH_PORT
            ;;
        "flask")
            echo "Starting Flask Vector RAG app on port $FLASK_VECTOR_PORT..."
            python3 app_flask.py
            ;;
        "chainlit")
            echo "Starting Chainlit Vector RAG app on port $CHAINLIT_VECTOR_PORT..."
            chainlit run app_chainlit.py --host 0.0.0.0 --port $CHAINLIT_VECTOR_PORT
            ;;
        *)
            echo "Starting default Flask GraphRAG app on port $FLASK_GRAPH_PORT..."
            python3 app_flask_graph.py
            ;;
    esac
else
    echo "Not in deploy mode, entering interactive shell."
    echo ""
    echo "Available commands:"
    echo "  python3 app_flask_graph.py       - Start Flask GraphRAG app"
    echo "  python3 app_flask.py             - Start Flask VectorRAG app"
    echo "  chainlit run app_chainlit_graph.py - Start Chainlit GraphRAG app"
    echo "  chainlit run app_chainlit.py     - Start Chainlit VectorRAG app"
    
    if [ "$LLM_MODE" = "local_ollama" ]; then
        echo "  ollama pull $ollama_model        - Download Ollama model"
    fi
    echo ""
    /bin/bash
fi
