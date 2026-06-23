import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, Optional
from my_config import MY_CONFIG
from neo4j import GraphDatabase, Driver
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GRAPH_DATA_DIR = MY_CONFIG.GRAPH_DATA_DIR
GRAPH_DATA_FILE = os.path.join(GRAPH_DATA_DIR, "graph-data-final.json")

class Neo4jConnection:
    def __init__(self):
        self.uri = MY_CONFIG.NEO4J_URI
        self.username = MY_CONFIG.NEO4J_USER
        self.password = MY_CONFIG.NEO4J_PASSWORD
        self.database = getattr(MY_CONFIG, "NEO4J_DATABASE", None)
        if not self.uri or not self.username or not self.password or not self.database:
            raise ValueError("Neo4j configuration incomplete")
        self.driver: Optional[Driver] = None
    
    async def connect(self):
        if self.driver is None:
            try:
                self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
                await asyncio.get_event_loop().run_in_executor(None, self.driver.verify_connectivity)
                # Get safe URI without credentials
                safe_uri = self.uri.split('@')[-1] if '@' in self.uri else self.uri.split('//')[1] if '//' in self.uri else self.uri
                logger.info(f"✓ Connected to Neo4j: {safe_uri}")
            except Exception as e:
                logger.error(f"✗ Connection failed")
                self.driver = None
    
    async def disconnect(self):
        if self.driver:
            await asyncio.get_event_loop().run_in_executor(None, self.driver.close)
            self.driver = None
    
    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None):
        if not self.driver:
            raise ConnectionError("Not connected to Neo4j")
        
        def run_query():
            with self.driver.session(database=self.database) as session:
                result = session.run(query, parameters or {})
                records = [record.data() for record in result]
                summary = result.consume()
                return records, summary
        
        return await asyncio.get_event_loop().run_in_executor(None, run_query)

neo4j_connection = Neo4jConnection()

async def clear_database():
    try:
        if not neo4j_connection.driver:
            await neo4j_connection.connect()
        
        node_records, _ = await neo4j_connection.execute_query("MATCH (n) RETURN count(n) as count")
        rel_records, _ = await neo4j_connection.execute_query("MATCH ()-[r]->() RETURN count(r) as count")
        
        nodes_before = node_records[0]["count"] if node_records else 0
        rels_before = rel_records[0]["count"] if rel_records else 0
        
        await neo4j_connection.execute_query("MATCH ()-[r]->() DELETE r")
        await neo4j_connection.execute_query("MATCH (n) DELETE n")
        
        print(f" Cleared: {nodes_before} nodes, {rels_before} relationships")
        return True
    except Exception as e:
        logger.error("✗ Clear failed")
        return False

async def upload_graph_data():
    try:
        if not neo4j_connection.driver:
            await neo4j_connection.connect()
        
        if not os.path.exists(GRAPH_DATA_FILE):
            logger.error("✗ Graph data file not found")
            return False
        
        # Clear database
        if not await clear_database():
            return False
        
        # Load data
        print("\n Loading graph data...")
        with open(GRAPH_DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        nodes = data.get('nodes', [])
        relationships = data.get('relationships', [])
        community_reports = data.get('community_reports', [])
        communities_data = data.get('communities', {})
        drift_metadata = data.get('drift_search_metadata', {})
        global_metadata = data.get('metadata', {})
        
        print(f" Data loaded:")
        print(f"   • {len(nodes)} nodes")
        print(f"   • {len(relationships)} relationships")
        print(f"   • {len(community_reports)} community reports")
        print(f"   • Communities metadata: {communities_data.get('total_communities', 0)} communities")
        print(f"   • DRIFT metadata: {'✓' if drift_metadata else '✗'}")
        print(f"   • Global metadata: {'✓' if global_metadata else '✗'}")
        
        stats = {"nodes": 0, "relationships": 0, "reports": 0, "metadata": 0, "errors": []}
        
        # 1. Upload Nodes
        print(f"\n Uploading {len(nodes)} nodes...")
        with tqdm(total=len(nodes), desc="Nodes", ncols=80) as pbar:
            for node in nodes:
                try:
                    labels_str = ':'.join([f"`{label}`" for label in node['labels']])
                    query = f"MERGE (n:{labels_str} {{id: $id}}) SET n += $props"
                    await neo4j_connection.execute_query(query, {
                        "id": node['id'],
                        "props": node.get('properties', {})
                    })
                    stats["nodes"] += 1
                except Exception as e:
                    stats["errors"].append(f"Node {node.get('id')}: {str(e)}")
                pbar.update(1)
        
        # 2. Upload Relationships
        print(f"\n Uploading {len(relationships)} relationships...")
        with tqdm(total=len(relationships), desc="Relationships", ncols=80) as pbar:
            for rel in relationships:
                try:
                    props = {
                        "id": rel.get('id'),
                        "description": rel.get('description', ''),
                        "evidence": rel.get('evidence', ''),
                        "confidence": rel.get('confidence'),
                        "weight": rel.get('weight'),
                        "source_file": rel.get('source_file', '')
                    }
                    query = f"""
                    MATCH (a {{id: $start}})
                    MATCH (b {{id: $end}})
                    MERGE (a)-[r:`{rel['type']}`]->(b)
                    SET r += $props
                    """
                    await neo4j_connection.execute_query(query, {
                        "start": rel['startNode'],
                        "end": rel['endNode'],
                        "props": props
                    })
                    stats["relationships"] += 1
                except Exception as e:
                    stats["errors"].append(f"Relationship {rel.get('type')}: {str(e)}")
                pbar.update(1)
        
        # 3. Upload Community Reports
        print(f"\n Uploading {len(community_reports)} community reports...")
        with tqdm(total=len(community_reports), desc="Reports", ncols=80) as pbar:
            for report in community_reports:
                try:
                    props = {
                        "community_id": report.get('community'),
                        "parent": report.get('parent'),
                        "children": report.get('children', []),
                        "level": report.get('level'),
                        "title": report.get('title', ''),
                        "summary": report.get('summary', ''),
                        "full_content": report.get('full_content', ''),
                        "rank": report.get('rank', 0),
                        "rank_explanation": report.get('rank_explanation', ''),
                        "rating_explanation": report.get('rating_explanation', ''),
                        "findings": json.dumps(report.get('findings', [])),
                        "full_content_json": json.dumps(report.get('full_content_json', {})),
                        "period": report.get('period', ''),
                        "size": report.get('size', 0),
                        "generated_at": report.get('generated_at', ''),
                        "report_embedding": report.get('report_embedding', []),
                        "key_entities_ranked": json.dumps(report.get('key_entities_ranked', [])),
                        "key_relationships_ranked": json.dumps(report.get('key_relationships_ranked', [])),
                        "llm_model": report.get('llm_model', ''),
                        "generation_confidence": report.get('generation_confidence', 0.0),
                        "token_count": report.get('token_count', 0),
                        "created_date": report.get('created_date', '')
                    }
                    
                    query = "MERGE (cr:CommunityReport {community_id: $id}) SET cr += $props"
                    await neo4j_connection.execute_query(query, {
                        "id": report.get('community'),
                        "props": props
                    })
                    stats["reports"] += 1
                except Exception as e:
                    stats["errors"].append(f"Report {report.get('community')}: {str(e)}")
                pbar.update(1)
        
        # 4. Upload Metadata Nodes
        print("\nUploading metadata...")
        
        # Communities Metadata
        if communities_data:
            try:
                props = {
                    "algorithm": communities_data.get('algorithm', ''),
                    "total_communities": communities_data.get('total_communities', 0),
                    "modularity_score": communities_data.get('modularity_score', 0.0),
                    "hierarchical": communities_data.get('hierarchical', False),
                    "levels": communities_data.get('levels', 0),
                    "summaries": json.dumps(communities_data.get('summaries', {})),
                    "statistics": json.dumps(communities_data.get('statistics', {})),
                    "hierarchy": json.dumps(communities_data.get('hierarchy', []))
                }
                query = "MERGE (cm:CommunitiesMetadata {id: 'global'}) SET cm += $props"
                await neo4j_connection.execute_query(query, {"props": props})
                stats["metadata"] += 1
                print("   ✓ Communities metadata")
            except Exception as e:
                stats["errors"].append(f"Communities metadata: {str(e)}")
        
        # DRIFT Metadata
        if drift_metadata:
            try:
                props = {
                    "configuration": json.dumps(drift_metadata.get('configuration', {})),
                    "query_routing_config": json.dumps(drift_metadata.get('query_routing_config', {})),
                    "community_search_index": json.dumps(drift_metadata.get('community_search_index', {})),
                    "entity_search_index": json.dumps(drift_metadata.get('entity_search_index', {})),
                    "relationship_search_index": json.dumps(drift_metadata.get('relationship_search_index', {})),
                    "vector_index_config": json.dumps(drift_metadata.get('vector_index_config', {})),
                    "fulltext_index_config": json.dumps(drift_metadata.get('fulltext_index_config', {})),
                    "composite_index_config": json.dumps(drift_metadata.get('composite_index_config', {})),
                    "range_index_config": json.dumps(drift_metadata.get('range_index_config', {})),
                    "centrality_index_config": json.dumps(drift_metadata.get('centrality_index_config', {})),
                    "semantic_embedding_index": json.dumps(drift_metadata.get('semantic_embedding_index', {})),
                    "graph_topology_index": json.dumps(drift_metadata.get('graph_topology_index', {})),
                    "search_optimization": json.dumps(drift_metadata.get('search_optimization', {}))
                }
                query = "MERGE (dm:DriftMetadata {id: 'global'}) SET dm += $props"
                await neo4j_connection.execute_query(query, {"props": props})
                stats["metadata"] += 1
                print("   ✓ DRIFT metadata")
            except Exception as e:
                stats["errors"].append(f"DRIFT metadata: {str(e)}")
        
        # Global Metadata
        if global_metadata:
            try:
                props = {
                    "phase": global_metadata.get('phase', ''),
                    "total_reports": global_metadata.get('total_reports', 0),
                    "llm_model": global_metadata.get('llm_model', ''),
                    "llm_provider": global_metadata.get('llm_provider', ''),
                    "created_date": global_metadata.get('created_date', ''),
                    "embedding_model": global_metadata.get('embedding_model', ''),
                    "embedding_dimensions": global_metadata.get('embedding_dimensions', 0),
                    "generated_at": global_metadata.get('generated_at', ''),
                    "validation": json.dumps(global_metadata.get('validation', {}))
                }
                query = "MERGE (gm:GlobalMetadata {id: 'global'}) SET gm += $props"
                await neo4j_connection.execute_query(query, {"props": props})
                stats["metadata"] += 1
                print("   ✓ Global metadata")
            except Exception as e:
                stats["errors"].append(f"Global metadata: {str(e)}")
        
        # 5. Create Community Relationships
        print("\nCreating community relationships...")
        try:
            query = """
            MATCH (n) WHERE n.community_id IS NOT NULL
            MATCH (cr:CommunityReport {community_id: n.community_id})
            MERGE (n)-[:BELONGS_TO_COMMUNITY]->(cr)
            """
            await neo4j_connection.execute_query(query)
            print("   ✓ Community relationships created")
        except Exception as e:
            stats["errors"].append(f"Community relationships: {str(e)}")
        
        # 6. Create Hierarchy Relationships (PARENT_OF/CHILD_OF)
        print("\nCreating hierarchy relationships...")
        try:
            hierarchy_count = 0
            for report in community_reports:
                comm_id = report.get('community')
                parent_id = report.get('parent')
                children = report.get('children', [])
                
                # Create PARENT_OF relationship from this community to parent
                if parent_id is not None:
                    query = """
                    MATCH (child:CommunityReport {community_id: $child_id})
                    MATCH (parent:CommunityReport {community_id: $parent_id})
                    MERGE (parent)-[:PARENT_OF]->(child)
                    MERGE (child)-[:CHILD_OF]->(parent)
                    """
                    await neo4j_connection.execute_query(query, {
                        "child_id": comm_id,
                        "parent_id": parent_id
                    })
                    hierarchy_count += 1
            
            print(f"   ✓ Hierarchy relationships created ({hierarchy_count} parent-child links)")
        except Exception as e:
            stats["errors"].append(f"Hierarchy relationships: {str(e)}")
        
        # Summary
        total = stats["nodes"] + stats["relationships"] + stats["reports"] + stats["metadata"]
        expected = len(nodes) + len(relationships) + len(community_reports) + 3
        success_rate = (total / expected * 100) if expected else 0
        
        print(f"\n✅ Upload Complete!")
        print(f"   Total: {total}/{expected} items ({success_rate:.1f}%)")
        print(f"   • Nodes: {stats['nodes']}/{len(nodes)}")
        print(f"   • Relationships: {stats['relationships']}/{len(relationships)}")
        print(f"   • Community Reports: {stats['reports']}/{len(community_reports)}")
        print(f"   • Metadata: {stats['metadata']}/3")
        
        if stats['errors']:
            print(f"\n⚠️  {len(stats['errors'])} errors")
            for err in stats['errors'][:5]:
                print(f"   {err}")
        
        return len(stats['errors']) == 0
        
    except Exception as e:
        logger.error("✗ Upload failed")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(upload_graph_data())
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error("✗ Fatal error occurred")
        sys.exit(1)
