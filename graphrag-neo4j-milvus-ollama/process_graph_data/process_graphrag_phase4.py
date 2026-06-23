"""
Phase 4: Entity & Report Embeddings + DRIFT Metadata
Input: workspace/graph_data/phase3_output.json
Output: workspace/graph_data/phase4_output.json
"""

import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer
from my_config import MY_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextEmbedder:
    
    def __init__(self):
        self.embedding_model = MY_CONFIG.EMBEDDING_MODEL
        self.embedding_dim = MY_CONFIG.EMBEDDING_LENGTH
        self.model = SentenceTransformer(self.embedding_model)
        logger.info(f"Initialized {self.embedding_model} ({self.embedding_dim}D)")
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()
    
    def embed_entities(self, nodes: List[Dict]) -> List[Dict]:
        texts = [
            f"{node['properties']['title']}: {node['properties'].get('content', '')[:500]}"
            for node in nodes
        ]
        
        embeddings = self.generate_embeddings_batch(texts)
        
        for node, embedding in zip(nodes, embeddings):
            node["properties"]["embedding_vector"] = embedding
            node["properties"]["embedding_model_version"] = self.embedding_model
            
            quality_factors = [
                node["properties"].get("confidence", 0.0),
                node["properties"].get("degree_centrality", 0.0),
                node["properties"].get("betweenness_centrality", 0.0),
                node["properties"].get("closeness_centrality", 0.0)
            ]
            
            node["properties"]["data_quality_score"] = round(sum(quality_factors) / len(quality_factors), 4)
        
        return nodes
    
    def embed_community_reports(self, reports: List[Dict]) -> List[Dict]:
        texts = [f"{report['title']}: {report['summary']}" for report in reports]
        embeddings = self.generate_embeddings_batch(texts)
        
        for report, embedding in zip(reports, embeddings):
            report["report_embedding"] = embedding
        
        return reports
    
    def build_drift_search_metadata(
        self,
        nodes: List[Dict],
        relationships: List[Dict],
        communities: Dict,
        community_reports: List[Dict]
    ) -> Dict:
        
        drift_metadata = {
            "configuration": {
                "tiers": ["lightweight_drift", "standard_drift", "comprehensive_drift"],
                "default_tier": "standard_drift",
                "fallback_enabled": True,
                "max_retries": 3,
                "timeout_seconds": 30
            },
            
            "query_routing_config": self._build_query_routing_config(),
            "community_search_index": self._build_community_search_index(community_reports),
            "entity_search_index": self._build_entity_search_index(nodes),
            "relationship_search_index": self._build_relationship_search_index(relationships),
            "vector_index_config": self._build_vector_index_config(),
            "fulltext_index_config": self._build_fulltext_index_config(),
            "composite_index_config": self._build_composite_index_config(),
            "range_index_config": self._build_range_index_config(),
            "centrality_index_config": self._build_centrality_index_config(),
            "semantic_embedding_index": self._build_semantic_index(nodes, community_reports),
            "graph_topology_index": self._build_topology_index(nodes, relationships),
            "search_optimization": self._calculate_search_stats(nodes, relationships, communities)
        }
        
        return drift_metadata
    
    def _build_query_routing_config(self) -> Dict:
        return {
            "lightweight_drift": {
                "triggers": ["simple factual question", "single entity lookup", "definition request"],
                "config": {
                    "primer_communities": 3,
                    "follow_up_iterations": 1,
                    "confidence_threshold": 0.6,
                    "max_tokens": 2000,
                    "community_level": 0
                },
                "cache_config": {
                    "cache_enabled": True,
                    "cache_ttl_seconds": 3600,
                    "cache_strategy": "lru",
                    "max_cache_size_mb": 100
                },
                "performance_thresholds": {
                    "max_latency_ms": 2000,
                    "target_accuracy": 0.7,
                    "min_confidence": 0.5,
                    "timeout_ms": 10000
                },
                "fallback_tier": "standard_drift"
            },
            "standard_drift": {
                "triggers": ["multi-entity question", "relationship query", "moderate complexity"],
                "config": {
                    "primer_communities": 10,
                    "follow_up_iterations": 2,
                    "confidence_threshold": 0.75,
                    "max_tokens": 8000,
                    "community_level": 1
                },
                "cache_config": {
                    "cache_enabled": True,
                    "cache_ttl_seconds": 1800,
                    "cache_strategy": "lru",
                    "max_cache_size_mb": 200
                },
                "performance_thresholds": {
                    "max_latency_ms": 5000,
                    "target_accuracy": 0.85,
                    "min_confidence": 0.7,
                    "timeout_ms": 20000
                },
                "fallback_tier": "comprehensive_drift"
            },
            "comprehensive_drift": {
                "triggers": ["complex multi-hop reasoning", "cross-community analysis", "high-stakes query"],
                "config": {
                    "primer_communities": 25,
                    "follow_up_iterations": 3,
                    "confidence_threshold": 0.85,
                    "max_tokens": 32000,
                    "community_level": 2
                },
                "cache_config": {
                    "cache_enabled": True,
                    "cache_ttl_seconds": 900,
                    "cache_strategy": "lru",
                    "max_cache_size_mb": 500
                },
                "performance_thresholds": {
                    "max_latency_ms": 15000,
                    "target_accuracy": 0.95,
                    "min_confidence": 0.8,
                    "timeout_ms": 60000
                },
                "fallback_tier": "none"
            }
        }
    
    def _build_community_search_index(self, community_reports: List[Dict]) -> Dict:
        index = {}
        for report in community_reports:
            index[report["community"]] = {
                "title": report["title"],
                "summary": report["summary"],
                "rank": report.get("rank", 5.0),
                "key_entities": [e["entity_id"] for e in report.get("key_entities_ranked", [])[:10]],
                "level": report["level"]
            }
        return index
    
    def _build_entity_search_index(self, nodes: List[Dict]) -> Dict:
        by_title = {}
        by_type = defaultdict(list)
        by_community = defaultdict(list)
        
        for node in nodes:
            node_id = node["id"]
            props = node["properties"]
            
            title = props["title"].lower()
            by_title[title] = node_id
            
            entity_types = node.get("labels", [])
            for entity_type in entity_types:
                by_type[entity_type].append(node_id)
            
            comm_id = props.get("community_id", "unknown")
            by_community[comm_id].append(node_id)
        
        return {
            "by_title": by_title,
            "by_type": dict(by_type),
            "by_community": dict(by_community)
        }
    
    def _build_relationship_search_index(self, relationships: List[Dict]) -> Dict:
        by_type = defaultdict(list)
        by_entity_pair = {}
        
        for rel in relationships:
            rel_type = rel["type"]
            by_type[rel_type].append(rel["id"])
            
            pair_key = f"{rel['startNode']}|{rel['endNode']}"
            by_entity_pair[pair_key] = rel["id"]
        
        return {
            "by_type": dict(by_type),
            "by_entity_pair": by_entity_pair
        }
    
    def _build_semantic_index(self, nodes: List[Dict], reports: List[Dict]) -> Dict:
        return {
            "entity_embeddings": {
                node["id"]: node["properties"].get("embedding_vector", [])
                for node in nodes
            },
            "community_report_embeddings": {
                report["community"]: report.get("report_embedding", [])
                for report in reports
            },
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dim,
            "similarity_metric": "cosine"
        }
    
    def _build_topology_index(self, nodes: List[Dict], relationships: List[Dict]) -> Dict:
        return {
            "node_count": len(nodes),
            "relationship_count": len(relationships),
            "avg_degree": len(relationships) * 2 / len(nodes) if nodes else 0,
            "graph_density": (2 * len(relationships)) / (len(nodes) * (len(nodes) - 1)) if len(nodes) > 1 else 0
        }
    
    def _build_vector_index_config(self) -> Dict:
        return {
            "entity_embedding_index": {
                "indexed_field": "embedding_vector",
                "dimensions": self.embedding_dim,
                "similarity_function": "cosine",
                "index_type": "hnsw"
            },
            "community_report_embedding_index": {
                "indexed_field": "report_embedding",
                "dimensions": self.embedding_dim,
                "similarity_function": "cosine",
                "index_type": "hnsw"
            }
        }
    
    def _build_fulltext_index_config(self) -> Dict:
        return {
            "entity_fulltext_index": {
                "indexed_fields": ["title", "content"],
                "analyzer": "standard",
                "language": "en"
            },
            "community_report_fulltext_index": {
                "indexed_fields": ["title", "summary", "full_content"],
                "analyzer": "standard",
                "language": "en"
            }
        }
    
    def _build_composite_index_config(self) -> Dict:
        return {
            "entity_community_quality": {
                "fields": ["community_id", "data_quality_score"],
                "order": "DESC"
            },
            "entity_type_centrality": {
                "fields": ["labels", "degree_centrality"],
                "order": "DESC"
            },
            "relationship_pair_type": {
                "fields": ["startNode", "type", "endNode"]
            }
        }
    
    def _build_range_index_config(self) -> Dict:
        return {
            "entity_id_index": {
                "field": "id",
                "index_type": "range"
            },
            "entity_title_index": {
                "field": "title",
                "index_type": "range"
            },
            "relationship_type_index": {
                "field": "type",
                "index_type": "range"
            },
            "community_id_index": {
                "field": "community_id",
                "index_type": "range"
            }
        }
    
    def _build_centrality_index_config(self) -> Dict:
        return {
            "degree_centrality_index": {
                "field": "degree_centrality",
                "order": "DESC"
            },
            "betweenness_centrality_index": {
                "field": "betweenness_centrality",
                "order": "DESC"
            },
            "data_quality_score_index": {
                "field": "data_quality_score",
                "order": "DESC"
            },
            "relationship_weight_index": {
                "field": "weight",
                "order": "DESC"
            }
        }
    
    def _calculate_search_stats(self, nodes: List[Dict], relationships: List[Dict], communities: Dict) -> Dict:
        return {
            "total_entities": len(nodes),
            "total_relationships": len(relationships),
            "total_communities": communities.get("total_communities", 0),
            "avg_community_size": len(nodes) / communities.get("total_communities", 1) if communities.get("total_communities", 0) > 0 else 0,
            "graph_modularity": communities.get("modularity_score", 0.0)
        }
    
    def run(self, input_file: str, output_file: str):
        logger.info(f"Loading {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        nodes = data['nodes']
        relationships = data['relationships']
        communities = data['communities']
        community_reports = data['community_reports']
        metadata = data['metadata']
        
        logger.info(f"Loaded {len(nodes)} nodes, {len(relationships)} relationships, {len(community_reports)} reports")
        
        nodes = self.embed_entities(nodes)
        community_reports = self.embed_community_reports(community_reports)
        
        logger.info("Building DRIFT search metadata with 12 indexes")
        drift_metadata = self.build_drift_search_metadata(
            nodes, relationships, communities, community_reports
        )
        
        metadata['phase'] = 'phase7_embeddings_complete'
        metadata['embedding_model'] = self.embedding_model
        metadata['embedding_dimensions'] = self.embedding_dim
        metadata['generated_at'] = datetime.now().isoformat()
        
        output = {
            "nodes": nodes,
            "relationships": relationships,
            "communities": communities,
            "community_reports": community_reports,
            "metadata": metadata,
            "drift_search_metadata": drift_metadata
        }
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved to {output_file}")


def main():
    embedder = TextEmbedder()
    embedder.run(
        input_file="workspace/graph_data/phase3_output.json",
        output_file="workspace/graph_data/phase4_output.json"
    )


if __name__ == "__main__":
    main()
