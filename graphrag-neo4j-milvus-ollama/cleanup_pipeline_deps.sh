#!/bin/bash
# ============================================
# Cleanup Pipeline Dependencies Script
# ============================================
# This script removes heavy packages that are only needed for the pipeline
# to save ~350-500 MB of RAM in production Docker containers.
#
# Run this AFTER the pipeline completes successfully.
# ============================================

echo "============================================"
echo "Starting Pipeline Dependency Cleanup"
echo "============================================"
echo "This will remove packages only needed for:"
echo "  - Document processing (docling, html2text)"
echo "  - Graph community detection (igraph, leidenalg, etc.)"
echo "  - Development tools (ipykernel, tqdm, etc.)"
echo ""
echo "Estimated RAM savings: 350-500 MB"
echo "============================================"

# Document processing packages
echo "Removing document processing packages..."
pip uninstall -y docling html2text 2>/dev/null || echo "  (already removed or not installed)"

# Graph community detection packages
echo "Removing graph community detection packages..."
pip uninstall -y python-louvain igraph leidenalg graspologic 2>/dev/null || echo "  (already removed or not installed)"

# Development tools
echo "Removing development tools..."
pip uninstall -y tqdm ipykernel fastmcp 2>/dev/null || echo "  (already removed or not installed)"

# Milvus Lite (if using cloud Zilliz)
if [ "$VECTOR_DB_TYPE" = "cloud_zilliz" ]; then
    echo "Removing Milvus Lite (using cloud Zilliz)..."
    pip uninstall -y milvus-lite 2>/dev/null || echo "  (already removed or not installed)"
fi

# Chainlit (if using Flask only)
if [ "$APP_TYPE" = "flask_graph" ] || [ "$APP_TYPE" = "flask" ]; then
    echo "Removing Chainlit (using Flask app)..."
    pip uninstall -y chainlit 2>/dev/null || echo "  (already removed or not installed)"
fi

echo ""
echo "============================================"
echo "Cleanup Complete!"
echo "============================================"
echo "Before cleanup: ~800 MB"
echo "After cleanup:  ~300-450 MB (depending on config)"
echo ""
echo "Note: If you redeploy and AUTO_RUN_PIPELINE=true,"
echo "      all packages will be reinstalled automatically."
echo "============================================"
