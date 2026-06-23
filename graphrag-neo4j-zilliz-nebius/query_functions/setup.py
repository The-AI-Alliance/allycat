"""Graph setup module for database and model initialization. Phase A (Steps 1-2)"""

import os
import logging
import asyncio
from typing import Dict, Optional, Any
import sys
sys.path.append('..')  # Add parent directory to path for imports

from my_config import MY_CONFIG
from neo4j import GraphDatabase
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.litellm import LiteLLM

try:
    from litellm_patch import cleanup_all_async_tasks
except ImportError:
    cleanup_all_async_tasks = None

# Set up environment
os.environ['HF_ENDPOINT'] = MY_CONFIG.HF_ENDPOINT

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Neo4jConnection:
    """
    Neo4j database connection manager.
    """
    
    def __init__(self):
        self.uri = MY_CONFIG.NEO4J_URI
        self.username = MY_CONFIG.NEO4J_USER
        self.password = MY_CONFIG.NEO4J_PASSWORD
        self.database = getattr(MY_CONFIG, "NEO4J_DATABASE", None)
        
        # Validate required configuration
        if not self.uri:
            raise ValueError("NEO4J_URI config is required")
        if not self.username:
            raise ValueError("NEO4J_USERNAME config is required")
        if not self.password:
            raise ValueError("NEO4J_PASSWORD config is required")
        if not self.database:
            raise ValueError("NEO4J_DATABASE config is required")
            
        self.driver: Optional[GraphDatabase.driver] = None

    def connect(self):
        """STEP 1.2: Initialize Neo4j driver with verification"""
        if self.driver is None:
            try:
                self.driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.username, self.password)
                )
                self.driver.verify_connectivity()
                logger.info(f"Connected to Neo4j at {self.uri}")
            except Exception as e:
                logger.error(f"❌ STEP 1.2 FAILED: Neo4j connection error: {e}")
                self.driver = None

    def disconnect(self):
        """Clean up Neo4j connection"""
        if self.driver:
            self.driver.close()
            self.driver = None
            logger.info("Neo4j connection closed")

    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None):
        """Execute Cypher query with error handling"""
        if not self.driver:
            raise ConnectionError("Not connected to Neo4j database")
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            records = [record.data() for record in result]
        return records


class GraphRAGSetup:
    """
    Main setup class for graph-based retrieval system.
    
    Handles core initialization and configuration:
    - Database connections (Neo4j and vector database)
    - Model initialization and configuration
    - Graph statistics and validation
    - Search configuration loading
    """
    
    def __init__(self):
        logger.info("Starting graph system initialization")
        
        # Initialize core components
        self.config = MY_CONFIG  # Add config attribute for GraphQueryEngine
        self.neo4j_conn = None
        self.query_engine = None
        self.graph_stats = {}
        self.drift_config = {}
        self.llm = None
        self.embedding_model = None
        
        # Execute Step 1 initialization sequence
        self._execute_step1_sequence()
        
        logger.info("Graph system initialization complete")

    def _execute_step1_sequence(self):
        """Execute complete Step 1 initialization sequence"""
        # STEP 1.1-1.6: Initialize all components
        self._setup_neo4j()          # STEP 1.2
        self._setup_vector_search()  # STEP 1.3-1.6
        self._load_graph_statistics() # STEP 2.1-2.4
        self._load_drift_configuration() # STEP 2.5

    def _setup_neo4j(self):
        """STEP 1.2: Initialize Neo4j driver with verification"""
        try:
            logger.info("Initializing Neo4j connection...")
            self.neo4j_conn = Neo4jConnection()
            self.neo4j_conn.connect()
            
            # Verify connection with test query
            if self.neo4j_conn.driver:
                test_result = self.neo4j_conn.execute_query("MATCH (n) RETURN count(n) as total_nodes LIMIT 1")
                node_count = test_result[0]['total_nodes'] if test_result else 0
                logger.info(f"Neo4j connected - {node_count} nodes found")
            
        except Exception as e:
            logger.error(f"Neo4j connection error: {e}")
            self.neo4j_conn = None
    
    def _setup_vector_search(self):
        """STEP 1.3-1.5: Initialize vector database and LLM components"""
        try:
            logger.info("Setting up vector search and LLM...")
            
            # STEP 1.5: Load embedding model
            self.embedding_model = HuggingFaceEmbedding(
                model_name=MY_CONFIG.EMBEDDING_MODEL
            )
            Settings.embed_model = self.embedding_model
            logger.info(f"Embedding model loaded: {MY_CONFIG.EMBEDDING_MODEL}")

            # STEP 1.6: Connect to vector database based on configuration
            if MY_CONFIG.VECTOR_DB_TYPE == "cloud_zilliz":
                if not MY_CONFIG.ZILLIZ_CLUSTER_ENDPOINT or not MY_CONFIG.ZILLIZ_TOKEN:
                    raise ValueError("Cloud database configuration missing. Set ZILLIZ_CLUSTER_ENDPOINT and ZILLIZ_TOKEN in .env")
                
                vector_store = MilvusVectorStore(
                    uri=MY_CONFIG.ZILLIZ_CLUSTER_ENDPOINT,
                    token=MY_CONFIG.ZILLIZ_TOKEN,
                    dim=MY_CONFIG.EMBEDDING_LENGTH,
                    collection_name=MY_CONFIG.COLLECTION_NAME,
                    overwrite=False
                )
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                logger.info("Connected to cloud vector database")
            else:
                vector_store = MilvusVectorStore(
                    uri=MY_CONFIG.MILVUS_URI_HYBRID_GRAPH,
                    dim=MY_CONFIG.EMBEDDING_LENGTH,
                    collection_name=MY_CONFIG.COLLECTION_NAME, 
                    overwrite=False
                )
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                logger.info("Connected to local vector database")

            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store, storage_context=storage_context)
            logger.info("Vector index loaded successfully")

            # STEP 1.4: Initialize LLM provider
            llm_model = MY_CONFIG.LLM_MODEL
            self.llm = LiteLLM(model=llm_model)
            Settings.llm = self.llm
            logger.info(f"LLM initialized: {llm_model}")
             
            self.query_engine = index.as_query_engine()
            
        except Exception as e:
            logger.error(f"Vector setup error: {e}")
            self.query_engine = None

    def _load_graph_statistics(self):
        """STEP 2.1-2.4: Load and validate graph data structure"""
        try:
            logger.info("Loading graph statistics and validation...")
            
            if not self.neo4j_conn or not self.neo4j_conn.driver:
                logger.warning("No Neo4j connection for statistics")
                return
            
            # STEP 2.1: Get node and relationship counts
            stats_query = """
            MATCH (n) 
            OPTIONAL MATCH ()-[r]-()
            RETURN count(DISTINCT n) as node_count, 
                   count(DISTINCT r) as relationship_count,
                   count(DISTINCT n.community_id) as community_count
            """
            
            result = self.neo4j_conn.execute_query(stats_query)
            if result:
                stats = result[0]
                self.graph_stats = {
                    'node_count': stats.get('node_count', 0),
                    'relationship_count': stats.get('relationship_count', 0),
                    'community_count': stats.get('community_count', 0)
                }
                
                logger.info(f"Graph validated - {self.graph_stats['node_count']} nodes, "
                           f"{self.graph_stats['relationship_count']} relationships, "
                           f"{self.graph_stats['community_count']} communities")
            
        except Exception as e:
            logger.error(f"Graph statistics error: {e}")
            self.graph_stats = {}

    def _load_drift_configuration(self):
        """STEP 2.5: Load DRIFT search metadata and configuration"""
        logger.info("Loading search configuration...")
        
        if not self.neo4j_conn or not self.neo4j_conn.driver:
            logger.warning("No Neo4j connection for search configuration")
            self.drift_config = {}
            return
        
        # Query for metadata nodes
        drift_metadata_query = """
        OPTIONAL MATCH (dm:DriftMetadata)
        OPTIONAL MATCH (gm:GlobalMetadata)
        OPTIONAL MATCH (cm:CommunitiesMetadata)
        RETURN dm, gm, cm
        """
        
        result = self.neo4j_conn.execute_query(drift_metadata_query)
        if result and result[0]:
            record = result[0]
            drift_config = {}
            
            # Extract DriftMetadata properties 
            if record.get('dm'):
                dm_props = dict(record['dm'])
                drift_config.update(dm_props)
                logger.info("DriftMetadata node found")
            
            # Extract GlobalMetadata properties
            if record.get('gm'):
                gm_props = dict(record['gm'])
                drift_config['global_metadata'] = gm_props
                logger.info("GlobalMetadata node found")
            
            # Extract CommunitiesMetadata properties
            if record.get('cm'):
                cm_props = dict(record['cm'])
                drift_config['communities_metadata'] = cm_props
                logger.info("CommunitiesMetadata node found")
            
            self.drift_config = drift_config
            logger.info("Search configuration loaded from Neo4j nodes")
            
        else:
            logger.warning("No metadata nodes found in Neo4j")
            self.drift_config = {}

    def validate_system_readiness(self):
        """Validate all required components are initialized"""
        ready = True
        
        if not self.neo4j_conn or not self.neo4j_conn.driver:
            logger.error("Neo4j connection not available")
            ready = False
        
        if not self.query_engine:
            logger.error("Vector query engine not available")
            ready = False
        
        if not self.graph_stats:
            logger.warning("Graph statistics not loaded")
        
        if ready:
            logger.info("System readiness validated")
            
        return ready

    def get_system_status(self):
        """Get detailed system status information"""
        return {
            "neo4j_connected": bool(self.neo4j_conn and self.neo4j_conn.driver),
            "vector_engine_ready": bool(self.query_engine),
            "graph_stats_loaded": bool(self.graph_stats),
            "drift_config_loaded": bool(self.drift_config),
            "llm_ready": bool(self.llm),
            "graph_stats": self.graph_stats,
            "drift_config": self.drift_config
        }

    async def cleanup_async_tasks(self, timeout: float = 2.0) -> None:
        """
        Clean up async tasks and pending operations.
        
        Handles proper cleanup of LiteLLM and other async tasks to prevent
        'Task was destroyed but it is pending!' warnings.
        """
        try:
            # Import cleanup function if available
            if cleanup_all_async_tasks is not None:
                await cleanup_all_async_tasks(timeout=timeout)
                logger.info(f"Cleaned up async tasks with timeout {timeout}s")
            else:
                # Fallback: Cancel pending tasks manually
                pending_tasks = [task for task in asyncio.all_tasks() if not task.done()]
                if pending_tasks:
                    logger.info(f"Cancelling {len(pending_tasks)} pending tasks")
                    for task in pending_tasks:
                        task.cancel()
                    
                    # Wait for cancellation with timeout
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*pending_tasks, return_exceptions=True),
                            timeout=timeout
                        )
                    except asyncio.TimeoutError:
                        logger.warning("Some tasks did not complete within timeout")
                        
        except Exception as e:
            logger.error(f"Error during async cleanup: {e}")

    def close(self):
        """Clean up all connections"""
        if self.neo4j_conn:
            self.neo4j_conn.disconnect()
        logger.info("Setup cleanup complete")


def create_graphrag_setup():
    """Factory function to create GraphRAG setup instance"""
    return GraphRAGSetup()


# Exports
__all__ = ['GraphRAGSetup', 'create_graphrag_setup']