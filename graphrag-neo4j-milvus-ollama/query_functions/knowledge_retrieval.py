"""
Knowledge Retrieval Module - Phase C (Steps 6-8)

Performs community search and data extraction using graph database structures.
Handles community retrieval, data extraction, and initial answer generation.
"""

import logging
import numpy as np
import json
import re
import time
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

from .setup import GraphRAGSetup
from .query_preprocessing import DriftRoutingResult, SearchStrategy


@dataclass
class CommunityResult:
    """Community result with hierarchy support."""
    community_id: str
    similarity_score: float
    summary: str
    key_entities: List[str]
    member_ids: List[str]  # Direct member access
    modularity_score: float  # Community quality
    level: int
    internal_edges: int
    member_count: int
    centrality_stats: Dict[str, float]  # Aggregated centrality measures
    confidence_score: float
    search_index: str  # Optimized search key
    termination_criteria: Dict[str, Any]
    parent_id: int = None  # Hierarchy support
    children_ids: List[int] = None  # Child communities
    
    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []
    
    @property
    def is_parent(self) -> bool:
        """Has children."""
        return len(self.children_ids) > 0
    
    @property
    def is_leaf(self) -> bool:
        """No children."""
        return not self.is_parent


@dataclass
class EntityResult:
    """Entity result with attributes from graph database."""
    entity_id: str
    name: str
    content: str
    confidence: float
    degree_centrality: float
    betweenness_centrality: float
    closeness_centrality: float
    community_id: str
    node_type: str
    semantic_tags: list = None
    bridge_score: float = 0.0
    entity_importance_rank: float = 0.0
    
    def __post_init__(self):
        if self.semantic_tags is None:
            self.semantic_tags = []


@dataclass
class RelationshipResult:
    """Relationship result with graph database attributes."""
    start_node: str
    end_node: str
    relationship_type: str
    confidence: float
    start_node_title: str = ""
    end_node_title: str = ""
    description: str = ""
    evidence: str = ""
    weight: float = 0.0


class CommunitySearchEngine:
    """Knowledge retrieval engine for community search and entity extraction."""
    
    def __init__(self, setup: GraphRAGSetup):
        self.setup = setup
        self.neo4j_conn = setup.neo4j_conn
        self.config = setup.config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize search optimization
        self.community_search_index = {}
        self.centrality_cache = {}
        self.recent_communities = []  # Track recently used communities for diversity
        
    async def execute_primer_phase(self,
                                 query_embedding: List[float],
                                 routing_result: DriftRoutingResult) -> Dict[str, Any]:
        """Execute community search and knowledge retrieval."""
        start_time = datetime.now()
        
        try:
            # Community retrieval
            self.logger.info("Starting community retrieval")
            communities = await self._retrieve_communities_enhanced(
                query_embedding, routing_result
            )
            
            # Data extraction
            self.logger.info("Starting data extraction")
            extracted_data = await self._extract_community_data_enhanced(communities)
            
            # Answer generation
            self.logger.info("Starting answer generation")
            initial_answer = await self._generate_initial_answer_enhanced(
                extracted_data, routing_result
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'communities': communities,
                'extracted_data': extracted_data,
                'initial_answer': initial_answer,
                'execution_time': execution_time,
                'metadata': {
                    'communities_retrieved': len(communities),
                    'entities_extracted': len(extracted_data.get('entities', [])),
                    'relationships_extracted': len(extracted_data.get('relationships', [])),
                    'phase': 'primer',
                    'step_range': '6-8'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Primer phase execution failed: {e}")
            raise
    
    async def _retrieve_communities_enhanced(self,
                                           query_embedding: List[float],
                                           routing_result: DriftRoutingResult) -> List[CommunityResult]:
        """
        Step 6: Enhanced community retrieval using comprehensive properties.
        
        Retrieves relevant communities based on query embedding similarity.
        """
        try:
            # Retrieve HyDE embeddings
            hyde_embeddings = await self._retrieve_hyde_embeddings_enhanced()
            
            if not hyde_embeddings:
                self.logger.warning("No HyDE embeddings found")
                return []
            
            # Compute similarities
            similarities = self._compute_hyde_similarities_enhanced(
                query_embedding, hyde_embeddings
            )
            
            # Rank communities
            ranked_communities = self._rank_communities_enhanced(
                similarities, routing_result
            )
            
            # Apply criteria
            filtered_communities = self._apply_termination_criteria(
                ranked_communities, routing_result
            )
            
            # Fetch community details
            community_results = await self._fetch_community_details_enhanced(
                filtered_communities
            )
            
            # Track used communities for diversity in future queries
            for community in community_results:
                if community.community_id not in self.recent_communities[-3:]:  # Don't duplicate immediate repeats
                    self.recent_communities.append(community.community_id)
            
            # Keep only last 10 communities to limit memory
            if len(self.recent_communities) > 10:
                self.recent_communities = self.recent_communities[-10:]
            
            self.logger.info(f"Retrieved {len(community_results)} enhanced communities")
            return community_results
            
        except Exception as e:
            self.logger.error(f"Enhanced community retrieval failed: {e}")
            return []
    
    async def _load_community_search_index(self):
        """Load optimized community search index from Neo4j."""
        try:
            query = """
            MATCH (meta:DriftMetadata)
            WHERE meta.community_search_index IS NOT NULL
            RETURN meta.community_search_index as search_index,
                   meta.total_communities as total_communities
            """
            
            results = self.neo4j_conn.execute_query(query)
            
            for record in results:
                # The search index is a nested JSON structure with community IDs as keys
                search_index_data = record['search_index']
                if isinstance(search_index_data, dict):
                    # Each community in the search index 
                    for community_id, community_data in search_index_data.items():
                        self.community_search_index[community_id] = community_data
                else:
                    self.logger.warning(f"Unexpected search index format: {type(search_index_data)}")
            
            self.logger.info(f"Loaded search index for {len(self.community_search_index)} communities")
            
        except Exception as e:
            self.logger.error(f"Failed to load community search index: {e}")
    
    async def _retrieve_hyde_embeddings_enhanced(self) -> Dict[str, Dict[str, Any]]:
        """Retrieve HyDE embeddings and metadata."""
        try:
            # Retrieve community embeddings
            query = """
            MATCH (cr:CommunityReport)
            WHERE cr.report_embedding IS NOT NULL
              AND cr.rank >= 5.0
              AND cr.generation_confidence >= 0.7
            OPTIONAL MATCH (meta:CommunitiesMetadata)
            RETURN cr.community_id as community_id,
                   cr.report_embedding as hyde_embeddings,
                   cr.summary as summary,
                   cr.key_entities_ranked as key_entities,
                   cr.size as member_count,
                   size(cr.report_embedding) as embedding_size,
                   meta.modularity_score as global_modularity_score
            """
            
            results = self.neo4j_conn.execute_query(query)
            hyde_embeddings = {}
            
            for record in results:
                community_id = record['community_id']
                embeddings_data = record.get('hyde_embeddings')
                
                if embeddings_data and community_id:
                    hyde_embeddings[community_id] = {
                        'embeddings': embeddings_data,
                        'summary': record.get('summary', ''),
                        'key_entities': record.get('key_entities', []),
                        'member_count': record.get('member_count', 0),
                        'embedding_size': record.get('embedding_size', 0),
                        'global_modularity_score': record.get('global_modularity_score', 0.0),
                        'embedding_type': 'report_embedding'
                    }
            
            self.logger.info(f"Retrieved enhanced HyDE embeddings for {len(hyde_embeddings)} communities")
            return hyde_embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve enhanced HyDE embeddings: {e}")
            # Retry logic for embeddings
            self.logger.info("Attempting retry for HyDE embeddings...")
            try:
                time.sleep(2)  # Brief delay before retry
                results = self.neo4j_conn.execute_query(query)
                hyde_embeddings = {}
                
                for record in results:
                    community_id = record['community_id']
                    embeddings_data = record.get('hyde_embeddings')
                    
                    if embeddings_data and community_id:
                        hyde_embeddings[community_id] = {
                            'embeddings': embeddings_data,
                            'summary': record.get('summary', ''),
                            'key_entities': record.get('key_entities', []),
                            'member_count': record.get('member_count', 0),
                            'embedding_size': record.get('embedding_size', 0),
                            'global_modularity': record.get('global_modularity_score', 0.0)
                        }
                
                self.logger.info(f"Retry successful: Retrieved enhanced HyDE embeddings for {len(hyde_embeddings)} communities")
                return hyde_embeddings
                
            except Exception as retry_error:
                self.logger.error(f"Retry also failed: {retry_error}")
                return {}
    
    def _compute_hyde_similarities_enhanced(self,
                                          query_embedding: List[float],
                                          hyde_embeddings: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Similarity computation with global modularity weighting.
        
        Calculates similarity scores between query embedding and community embeddings.
        """
        similarities = {}
        query_vec = np.array(query_embedding)
        query_norm = np.linalg.norm(query_vec)
        
        if query_norm == 0:
            self.logger.warning("Query embedding has zero norm")
            return similarities
        
        for community_id, embedding_data in hyde_embeddings.items():
            embeddings_list = embedding_data['embeddings']
            global_modularity = embedding_data.get('global_modularity_score', 0.0)
            
            max_similarity = 0.0
            
            # Compute similarity
            try:
                # Parse embedding string
                if isinstance(embeddings_list, str):
                    embeddings_list = json.loads(embeddings_list)
                
                # Process embeddings
                if isinstance(embeddings_list, list) and len(embeddings_list) > 0:
                    # Use first embedding
                    hyde_vec = np.array(embeddings_list[0] if isinstance(embeddings_list[0], list) else embeddings_list)
                else:
                    hyde_vec = np.array(embeddings_list)
                
                hyde_norm = np.linalg.norm(hyde_vec)
                
                if hyde_norm > 0:
                    # Calculate similarity
                    base_similarity = np.dot(query_vec, hyde_vec) / (query_norm * hyde_norm)
                    
                    # Apply weighting
                    weighted_similarity = base_similarity * (1 + 0.2 * global_modularity)
                    max_similarity = weighted_similarity
                        
            except Exception as e:
                self.logger.warning(f"Error computing similarity for community {community_id}: {e}")
                continue
            
            similarities[community_id] = {
                'similarity': max_similarity,
                'global_modularity_score': global_modularity,
                'embedding_size': embedding_data.get('embedding_size', 0)
            }
        
        self.logger.info(f"Computed enhanced similarities for {len(similarities)} communities")
        return similarities
    
    def _rank_communities_enhanced(self,
                                 similarities: Dict[str, Dict[str, float]],
                                 routing_result: DriftRoutingResult) -> List[Tuple[str, Dict[str, float]]]:
        """
        Ranking using global modularity and similarity with diversity penalty.
        
        Ranks communities based on similarity and modularity, penalizing recently used communities.
        """
        
        def ranking_score(item):
            cid, scores = item
            similarity = scores['similarity']
            global_modularity = scores['global_modularity_score']
            
            # Apply diversity penalty for recently used communities
            diversity_penalty = 0.0
            if cid in self.recent_communities[-5:]:  # Last 5 queries
                position = self.recent_communities[::-1].index(cid)
                diversity_penalty = 0.1 * (1.0 - position / 5.0)  # Stronger penalty for more recent
            
            # Weighted combination with diversity consideration
            base_score = 0.8 * similarity + 0.2 * global_modularity
            return base_score - diversity_penalty
        
        # Sort by combined ranking score
        ranked = sorted(similarities.items(), key=ranking_score, reverse=True)
        
        # Apply similarity threshold
        similarity_threshold = routing_result.parameters.get('similarity_threshold', 0.7)
        filtered_ranked = [
            (cid, scores) for cid, scores in ranked 
            if scores['similarity'] >= similarity_threshold
        ]
        
        self.logger.info(f"Enhanced ranking: {len(filtered_ranked)} communities above threshold {similarity_threshold}")
        return filtered_ranked
    
    def _apply_termination_criteria(self,
                                  ranked_communities: List[Tuple[str, Dict[str, float]]],
                                  routing_result: DriftRoutingResult) -> List[Tuple[str, Dict[str, float]]]:
        """
        Apply termination criteria for community selection.
        Adaptive max_communities based on strategy.
        """
        strategy = routing_result.search_strategy
        if strategy == SearchStrategy.GLOBAL_SEARCH:
            max_communities = routing_result.parameters.get('max_communities', 10)
        elif strategy == SearchStrategy.HYBRID_SEARCH:
            max_communities = routing_result.parameters.get('max_communities', 5)
        else:  # LOCAL_SEARCH
            max_communities = routing_result.parameters.get('max_communities', 3)
        
        min_global_modularity = routing_result.parameters.get('min_global_modularity', 0.3)
        
        self.logger.info(f"Termination: max={max_communities} (strategy={strategy.value})")
        
        # Apply criteria
        filtered = []
        for community_id, scores in ranked_communities:
            if len(filtered) >= max_communities:
                break
                
            # Check global modularity threshold
            if scores['global_modularity_score'] >= min_global_modularity:
                filtered.append((community_id, scores))
        
        self.logger.info(f"Applied termination criteria: {len(filtered)} communities selected")
        return filtered
    
    async def _fetch_community_details_enhanced(self,
                                              ranked_communities: List[Tuple[str, Dict[str, float]]]) -> List[CommunityResult]:
        """
        Fetch comprehensive community details with all properties.
        
        Retrieves detailed information about selected communities including summaries,
        key entities, and member IDs.
        """
        community_results = []
        
        for community_id, scores in ranked_communities:
            try:
                # Query with hierarchy support - MUST use toInteger() for community_id comparison!
                detail_query = """
                MATCH (cr:CommunityReport)
                WHERE cr.community_id = toInteger($community_id) AND cr.report_embedding IS NOT NULL
                OPTIONAL MATCH (meta:CommunitiesMetadata {id: 'global'})
                OPTIONAL MATCH (cr)-[:CHILD_OF]->(parent:CommunityReport)
                OPTIONAL MATCH (cr)-[:PARENT_OF]->(child:CommunityReport)
                OPTIONAL MATCH (entity)-[:BELONGS_TO_COMMUNITY]->(cr)
                WHERE NOT entity:CommunityReport
                WITH cr, meta, parent, 
                     collect(DISTINCT child.community_id) as children_ids,
                     collect(DISTINCT entity.id)[0..50] as member_ids
                RETURN cr.summary as summary,
                       cr.full_content as full_content,
                       cr.findings as findings,
                       cr.rank as importance_rank,
                       cr.key_entities_ranked as key_entities,
                       cr.size as member_count,
                       cr.level as level,
                       cr.generation_confidence as generation_confidence,
                       meta.modularity_score as modularity_score,
                       cr.community_id as id,
                       parent.community_id as parent_id,
                       children_ids,
                       member_ids
                LIMIT 1
                """
                
                results = self.neo4j_conn.execute_query(
                    detail_query, 
                    {'community_id': community_id}
                )
                
                if results:
                    record = results[0]
                    
                    # Include hierarchy in result
                    community_result = CommunityResult(
                        community_id=community_id,
                        similarity_score=scores['similarity'],
                        summary=record.get('summary', ''),
                        key_entities=record.get('key_entities', []),
                        member_ids=record.get('member_ids', []),  # Now populated
                        modularity_score=record.get('modularity_score', 0.0),
                        level=record.get('level', 1),
                        internal_edges=0,
                        member_count=record.get('member_count', 0),
                        confidence_score=scores.get('confidence_score', 0.5),
                        search_index='',
                        termination_criteria={},
                        centrality_stats={},
                        parent_id=record.get('parent_id'),  # Hierarchy
                        children_ids=record.get('children_ids', [])  # Hierarchy
                    )
                    
                    community_results.append(community_result)
                    
            except Exception as e:
                self.logger.error(f"Failed to fetch details for community {community_id}: {e}")
                continue
        
        self.logger.info(f"Fetched enhanced details for {len(community_results)} communities")
        return community_results
    
    async def _extract_community_data_enhanced(self,
                                             communities: List[CommunityResult]) -> Dict[str, Any]:
        """
        Step 7: Data extraction with centrality measures.
        
        Extracts:
        - Entities with degree/betweenness/closeness centrality
        - Relationships with confidence scores
        - Community statistics and properties
        """
        try:
            all_entities = []
            all_relationships = []
            community_stats = []
            
            for community in communities:
                # Extract entities with centrality measures
                entities = await self._extract_entities_with_centrality(community)
                all_entities.extend(entities)
                
                # Extract relationships with properties
                relationships = await self._extract_relationships_enhanced(community)
                all_relationships.extend(relationships)
                
                # Collect community statistics
                community_stats.append({
                    'community_id': community.community_id,
                    'member_count': community.member_count,
                    'modularity_score': community.modularity_score,
                    'confidence_score': community.confidence_score,
                    'centrality_stats': community.centrality_stats
                })
            
            extracted_data = {
                'entities': all_entities,
                'relationships': all_relationships,
                'community_stats': community_stats,
                'extraction_metadata': {
                    'communities_processed': len(communities),
                    'entities_extracted': len(all_entities),
                    'relationships_extracted': len(all_relationships),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            self.logger.info(f"Enhanced extraction completed: {len(all_entities)} entities, {len(all_relationships)} relationships")
            return extracted_data
            
        except Exception as e:
            self.logger.error(f"Enhanced data extraction failed: {e}")
            return {'entities': [], 'relationships': [], 'community_stats': []}
    
    async def _extract_entities_with_centrality(self,
                                              community: CommunityResult) -> List[EntityResult]:
        """
        Extract entities with comprehensive centrality measures.
        
        Retrieves entities from the community with their associated centrality metrics.
        """
        try:
            # Use member_ids for direct access if available
            member_ids = community.member_ids if community.member_ids else []
            
            if member_ids:
                # Direct member access query based on actual schema
                entity_query = """
                MATCH (n)
                WHERE n.id IN $member_ids
                  AND n.title IS NOT NULL 
                  AND n.content IS NOT NULL
                RETURN n.id as entity_id,
                       n.title as name,
                       n.content as content,
                       n.confidence as confidence,
                       n.degree_centrality as degree_centrality,
                       n.betweenness_centrality as betweenness_centrality,
                       n.closeness_centrality as closeness_centrality,
                       n.entity_importance_rank as entity_importance_rank,
                       n.bridge_score as bridge_score,
                       n.semantic_tags as semantic_tags,
                       labels(n) as node_types
                ORDER BY n.entity_importance_rank DESC, n.bridge_score DESC, n.degree_centrality DESC
                """
                
                results = self.neo4j_conn.execute_query(
                    entity_query,
                    {'member_ids': member_ids}
                )
            else:
                # Method 1: Direct BELONGS_TO_COMMUNITY relationship
                entity_query_direct = """
                MATCH (n)-[:BELONGS_TO_COMMUNITY]->(report:CommunityReport {community_id: toInteger($community_id)})
                WHERE n.title IS NOT NULL 
                  AND n.content IS NOT NULL
                  AND NOT n:CommunityReport
                RETURN n.id as entity_id,
                       n.title as name,
                       n.content as content,
                       n.confidence as confidence,
                       n.degree_centrality as degree_centrality,
                       n.betweenness_centrality as betweenness_centrality,
                       n.closeness_centrality as closeness_centrality,
                       n.entity_importance_rank as entity_importance_rank,
                       n.bridge_score as bridge_score,
                       n.semantic_tags as semantic_tags,
                       labels(n) as node_types
                ORDER BY n.entity_importance_rank DESC, n.bridge_score DESC, n.degree_centrality DESC
                LIMIT 50
                """
                
                results_method1 = self.neo4j_conn.execute_query(
                    entity_query_direct,
                    {'community_id': int(community.community_id)}
                )
                
                # Method 2: community_id property
                entity_query_property = """
                MATCH (n)
                WHERE n.community_id = toInteger($community_id)
                  AND n.title IS NOT NULL 
                  AND n.content IS NOT NULL
                  AND NOT n:CommunityReport
                RETURN n.id as entity_id,
                       n.title as name,
                       n.content as content,
                       n.confidence as confidence,
                       n.degree_centrality as degree_centrality,
                       n.betweenness_centrality as betweenness_centrality,
                       n.closeness_centrality as closeness_centrality,
                       n.entity_importance_rank as entity_importance_rank,
                       n.bridge_score as bridge_score,
                       n.semantic_tags as semantic_tags,
                       labels(n) as node_types
                ORDER BY n.entity_importance_rank DESC, n.bridge_score DESC, n.degree_centrality DESC
                LIMIT 50
                """
                
                results_method2 = self.neo4j_conn.execute_query(
                    entity_query_property,
                    {'community_id': int(community.community_id)}
                )
                
                # Method 3: Aggregate from children (for parent communities)
                results_method3 = []
                if community.is_parent:
                    entity_query_children = """
                    MATCH (cr:CommunityReport {community_id: toInteger($community_id)})-[:PARENT_OF*1..2]->(child:CommunityReport)
                    MATCH (n)-[:BELONGS_TO_COMMUNITY]->(child)
                    WHERE n.title IS NOT NULL 
                      AND n.content IS NOT NULL
                      AND NOT n:CommunityReport
                    RETURN DISTINCT n.id as entity_id,
                           n.title as name,
                           n.content as content,
                           n.confidence as confidence,
                           n.degree_centrality as degree_centrality,
                           n.betweenness_centrality as betweenness_centrality,
                           n.closeness_centrality as closeness_centrality,
                           n.entity_importance_rank as entity_importance_rank,
                           n.bridge_score as bridge_score,
                           n.semantic_tags as semantic_tags,
                           labels(n) as node_types
                    ORDER BY n.entity_importance_rank DESC, n.bridge_score DESC, n.degree_centrality DESC
                    LIMIT 100
                    """
                    
                    results_method3 = self.neo4j_conn.execute_query(
                        entity_query_children,
                        {'community_id': int(community.community_id)}
                    )
                
                # Merge all results with deduplication
                self.logger.info(f"  Method 1 (direct): {len(results_method1)} entities")
                self.logger.info(f"  Method 2 (property): {len(results_method2)} entities")
                self.logger.info(f"  Method 3 (children): {len(results_method3)} entities")
                
                seen_entities = set()
                results = []
                for record in results_method1 + results_method2 + results_method3:
                    entity_id = record['entity_id']
                    if entity_id not in seen_entities:
                        results.append(record)
                        seen_entities.add(entity_id)
                
                self.logger.info(f"  Total unique entities: {len(results)}")
            
            entities = []
            for record in results:
                # Handle multiple labels properly
                node_types = record.get('node_types', ['Unknown'])
                primary_type = self._get_primary_entity_type(node_types) if node_types else 'Unknown'
                
                entity = EntityResult(
                    entity_id=record['entity_id'],
                    name=record.get('name', ''),
                    content=record.get('content', ''),
                    confidence=record.get('confidence', 0.0),
                    degree_centrality=record.get('degree_centrality', 0.0),
                    betweenness_centrality=record.get('betweenness_centrality', 0.0),
                    closeness_centrality=record.get('closeness_centrality', 0.0),
                    community_id=community.community_id,
                    node_type=primary_type,
                    semantic_tags=record.get('semantic_tags', []),
                    bridge_score=record.get('bridge_score', 0.0),
                    entity_importance_rank=record.get('entity_importance_rank', 0.0)
                )
                entities.append(entity)
            
            return entities
            
        except Exception as e:
            self.logger.error(f"Failed to extract entities for community {community.community_id}: {e}")
            return []
    
    async def _extract_relationships_enhanced(self,
                                            community: CommunityResult) -> List[RelationshipResult]:
        """
        Extract relationships using multiple methods, combine unique results.
        
        Method 1: community_id property (for leaf communities)
        Method 2: BELONGS_TO_COMMUNITY relationship (for hierarchical communities)
        Method 3: Entity 1-hop traversal (for parent communities)
        """
        try:
            all_relationships = []
            seen_pairs = set()
            
            # METHOD 1: Via community_id property
            query_method1 = """
            MATCH (a)-[r]->(b)
            WHERE (a.community_id = toInteger($community_id) OR b.community_id = toInteger($community_id))
              AND NOT type(r) = 'BELONGS_TO_COMMUNITY'
              AND NOT a:CommunityReport
              AND NOT b:CommunityReport
              AND (r.confidence IS NULL OR r.confidence >= 0.2)
            RETURN a.id as start_node,
                   a.title as start_node_title,
                   b.id as end_node,
                   b.title as end_node_title,
                   type(r) as relationship_type,
                   r.description as description,
                   r.evidence as evidence,
                   COALESCE(r.confidence, 0.5) as confidence,
                   r.weight as weight
            ORDER BY confidence DESC, weight DESC
            LIMIT 50
            """
            
            # METHOD 1: Direct relationship query
            results1 = self.neo4j_conn.execute_query(query_method1, {'community_id': int(community.community_id)})
            
            # METHOD 2: Via BELONGS_TO_COMMUNITY relationship
            query_method2 = """
            MATCH (entity)-[:BELONGS_TO_COMMUNITY]->(report:CommunityReport {community_id: toInteger($community_id)})
            WITH collect(entity) as community_entities
            UNWIND community_entities as a
            MATCH (a)-[r]->(b)
            WHERE NOT type(r) = 'BELONGS_TO_COMMUNITY'
              AND NOT b:CommunityReport
              AND (r.confidence IS NULL OR r.confidence >= 0.2)
            RETURN a.id as start_node,
                   a.title as start_node_title,
                   b.id as end_node,
                   b.title as end_node_title,
                   type(r) as relationship_type,
                   r.description as description,
                   r.evidence as evidence,
                   COALESCE(r.confidence, 0.5) as confidence,
                   r.weight as weight
            ORDER BY confidence DESC, weight DESC
            LIMIT 100
            """
            
            results2 = self.neo4j_conn.execute_query(query_method2, {'community_id': int(community.community_id)})
            
            # METHOD 3: Aggregate from children (for parent communities)
            results3 = []
            if community.is_parent:
                query_method3 = """
                MATCH (cr:CommunityReport {community_id: toInteger($community_id)})-[:PARENT_OF*1..2]->(child:CommunityReport)
                MATCH (a)-[:BELONGS_TO_COMMUNITY]->(child)
                MATCH (a)-[r]->(b)
                WHERE NOT type(r) = 'BELONGS_TO_COMMUNITY'
                  AND NOT a:CommunityReport
                  AND NOT b:CommunityReport
                  AND (r.confidence IS NULL OR r.confidence >= 0.2)
                RETURN DISTINCT a.id as start_node,
                       a.title as start_node_title,
                       b.id as end_node,
                       b.title as end_node_title,
                       type(r) as relationship_type,
                       r.description as description,
                       r.evidence as evidence,
                       COALESCE(r.confidence, 0.5) as confidence,
                       r.weight as weight
                ORDER BY confidence DESC, weight DESC
                LIMIT 200
                """
                
                results3 = self.neo4j_conn.execute_query(query_method3, {'community_id': int(community.community_id)})
            
            # Merge all results with deduplication
            self.logger.info(f"  Method 1 (direct): {len(results1)} relationships")
            self.logger.info(f"  Method 2 (BELONGS_TO): {len(results2)} relationships")
            self.logger.info(f"  Method 3 (children): {len(results3)} relationships")
            
            for record in results1 + results2 + results3:
                pair_key = (record['start_node'], record['end_node'], record['relationship_type'])
                if pair_key not in seen_pairs:
                    all_relationships.append(RelationshipResult(
                        start_node=record['start_node'],
                        end_node=record['end_node'],
                        relationship_type=record['relationship_type'],
                        confidence=record['confidence'],
                        start_node_title=record.get('start_node_title', ''),
                        end_node_title=record.get('end_node_title', ''),
                        description=record.get('description', ''),
                        evidence=record.get('evidence', ''),
                        weight=record.get('weight', 0.0)
                    ))
                    seen_pairs.add(pair_key)
            
            self.logger.info(f"  Total unique relationships: {len(all_relationships)}")
            
            return all_relationships
            
        except Exception as e:
            self.logger.error(f"Failed to extract relationships for community {community.community_id}: {e}")
            return []
    
    async def _generate_initial_answer_enhanced(self, 
                                              extracted_data: Dict[str, Any],
                                              routing_result: DriftRoutingResult) -> Dict[str, Any]:
        """
        Step 8: Context-aware initial answer generation.
        
        Query-aware entity selection instead of pure centrality ranking.
        This prevents hallucination by showing LLM entities relevant to the query.
        """
        try:
            entities = extracted_data['entities']
            relationships = extracted_data['relationships']
            community_stats = extracted_data['community_stats']
            
            # Detect target entity types from query
            query_lower = routing_result.original_query.lower()
            target_types = self._detect_query_entity_types(query_lower)
            self.logger.info(f"Query-aware filtering: detected types = {target_types}")
            
            # Select entities with query-aware prioritization
            important_entities = self._select_query_relevant_entities(
                entities, target_types, max_entities=15
            )
            
            # Log entity type distribution for debugging
            type_dist = {}
            for e in important_entities:
                type_dist[e.node_type] = type_dist.get(e.node_type, 0) + 1
            self.logger.info(f"Entities shown to LLM by type: {type_dist}")
            
            # Select high-confidence relationships
            strong_relationships = [
                r for r in relationships 
                if r.confidence >= 0.7
            ]
            
            # Prepare context for LLM
            llm_context = self._prepare_llm_context_enhanced(
                important_entities, strong_relationships, community_stats, routing_result
            )
            
            # Generate initial answer using configured LLM
            llm_response = await self._generate_llm_answer(llm_context, routing_result)
            
            initial_answer = {
                'content': llm_response['answer'],
                'llm_context': llm_context,
                'context_used': {
                    'important_entities': len(important_entities),
                    'strong_relationships': len(strong_relationships),
                    'communities_analyzed': len(community_stats)
                },
                'confidence_metrics': {
                    'avg_entity_centrality': np.mean([e.degree_centrality for e in important_entities]) if important_entities else 0,
                    'avg_relationship_confidence': np.mean([r.confidence for r in strong_relationships]) if strong_relationships else 0,
                    'avg_community_modularity': np.mean([c['modularity_score'] for c in community_stats]) if community_stats else 0,
                    'llm_confidence': llm_response['confidence']
                },
                'follow_up_questions': llm_response['follow_up_questions'],
                'reasoning': llm_response['reasoning']
            }
            
            self.logger.info("Enhanced initial answer generated with comprehensive context")
            return initial_answer
            
        except Exception as e:
            self.logger.error(f"Enhanced answer generation failed: {e}")
            return {'content': 'Error generating initial answer', 'error': str(e)}
    
    def _prepare_llm_context_enhanced(self, 
                                    entities: List[EntityResult],
                                    relationships: List[RelationshipResult],
                                    community_stats: List[Dict[str, Any]],
                                    routing_result: DriftRoutingResult) -> str:
        """Prepare enhanced context for LLM with comprehensive information."""
        
        context_parts = [
            f"Query: {routing_result.original_query}",
            f"Search Strategy: {routing_result.search_strategy.value}",
            "",
            "=== IMPORTANT ENTITIES (Use these specific names in your answer) ===",
        ]
        
        # Log entity types being shown to LLM for debugging
        entity_type_counts = {}
        for e in entities[:15]:
            entity_type_counts[e.node_type] = entity_type_counts.get(e.node_type, 0) + 1
        self.logger.info(f"Entities shown to LLM (top 15): {entity_type_counts}")
        
        for i, entity in enumerate(entities[:10], 1):  # Show more entities
            tags_str = f"[{', '.join(entity.semantic_tags[:3])}]" if entity.semantic_tags else "[no tags]"
            context_parts.append(
                f"{i}. **{entity.name}** ({entity.node_type}) {tags_str}\n"
                f"   Content: {entity.content[:150]}...\n"
                f"   Metrics: Centrality={entity.degree_centrality:.3f}, Confidence={entity.confidence:.3f}, Importance={entity.entity_importance_rank:.3f}"
            )
        
        context_parts.extend([
            "",
            "=== KEY RELATIONSHIPS (Use these connections in your answer) ===",
        ])
        
        # Create ID to name mapping
        entity_id_to_name = {entity.entity_id: entity.name for entity in entities}
        
        for i, rel in enumerate(relationships[:8], 1):  # Show more relationships
            # Use titles from relationship if available, otherwise map from entities
            start_name = rel.start_node_title or entity_id_to_name.get(rel.start_node, rel.start_node)
            end_name = rel.end_node_title or entity_id_to_name.get(rel.end_node, rel.end_node)
            
            # Show description and evidence if available
            desc_str = f"\n   Description: {rel.description[:100]}..." if rel.description else ""
            evidence_str = f"\n   Evidence: {rel.evidence[:80]}..." if rel.evidence else ""
            
            context_parts.append(
                f"{i}. **{start_name}** --[{rel.relationship_type}]--> **{end_name}**\n"
                f"   Confidence: {rel.confidence:.3f}, Weight: {rel.weight:.2f}"
                f"{desc_str}{evidence_str}"
            )
        
        # Add entity type distribution statistics
        type_counts = {}
        for entity in entities:
            type_counts[entity.node_type] = type_counts.get(entity.node_type, 0) + 1
        
        type_summary = ", ".join([f"{count} {etype}(s)" for etype, count in sorted(type_counts.items())])
        
        # Add quick reference list of all entity names
        entity_names = [entity.name for entity in entities[:15]]
        context_parts.extend([
            "",
            "=== ENTITY TYPE DISTRIBUTION ===",
            f"Entity types in this context: {type_summary}",
            f"Total entities provided: {len(entities)}",
            "",
            "=== ENTITY NAMES FOR REFERENCE ===",
            f"Available entities: {', '.join(entity_names)}",
            "",
            "=== COMMUNITY STATISTICS ===",
        ])
        
        for stat in community_stats:
            context_parts.append(
                f"Community {stat['community_id']}: {stat['member_count']} members, "
                f"modularity: {stat['modularity_score']:.3f}"
            )
        
        context_parts.extend([
            "",
            "REMEMBER: Use the specific entity names listed above in your answer!"
        ])
        
        return "\n".join(context_parts)
    
    async def _generate_llm_answer(self, 
                                 context: str, 
                                 routing_result: DriftRoutingResult) -> Dict[str, Any]:
        """
        Generate actual LLM response using the configured LLM.
        
        Uses the LLM from GraphRAGSetup to generate answers with follow-up questions.
        """
        try:
            # Construct comprehensive prompt for LLM
            prompt = f"""
You are an expert knowledge analyst. Answer the user's query by extracting and using the specific entity names from the graph data below.

GRAPH DATA CONTEXT:
{context}

USER QUERY: {routing_result.original_query}

INSTRUCTIONS:
1. Look at the "IMPORTANT ENTITIES" and "ENTITY NAMES FOR REFERENCE" sections above
2. Answer the query by listing or describing the ACTUAL ENTITY NAMES you see there
3. Do not write generic descriptions - use the real names from the data
4. If asked for a list, provide the entity names with their types
5. If asked for explanation, use the real names and relationships shown

RESPONSE FORMAT:
Answer: [Direct answer using actual entity names from the data above]
Confidence: [0.0-1.0]
Reasoning: [Why these specific entities answer the query]
Follow-up Questions:
1. [Specific question about entities found]
2. [Question about relationships discovered]
3. [Question about community connections]
4. [Question for deeper exploration]
5. [Question about related entities]
"""

            # Call the configured LLM
            llm_response = await self.setup.llm.acomplete(prompt)
            response_text = llm_response.text
            
            # Parse LLM response
            parsed_response = self._parse_llm_response(response_text)
            
            self.logger.info(f"LLM generated answer with confidence: {parsed_response['confidence']}")
            return parsed_response
            
        except Exception as e:
            self.logger.error(f"LLM answer generation failed: {e}")
            # Fallback response
            return {
                'answer': f"Based on the graph analysis, I found relevant information but encountered an issue generating the full response: {str(e)}",
                'confidence': 0.3,
                'reasoning': "LLM generation encountered an error, providing basic analysis from graph data.",
                'follow_up_questions': [
                    "What specific aspects would you like me to explore further?",
                    "Are there particular entities or relationships of interest?",
                    "Should I focus on a specific community or time period?"
                ]
            }
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse structured LLM response into components."""
        try:
            lines = response_text.strip().split('\n')
            
            answer = ""
            confidence = 0.5
            reasoning = ""
            follow_up_questions = []
            
            current_section = None
            
            for line in lines:
                line = line.strip()
                
                if line.startswith("Answer:"):
                    current_section = "answer"
                    answer = line.replace("Answer:", "").strip()
                elif line.startswith("Confidence:"):
                    confidence_text = line.replace("Confidence:", "").strip()
                    try:
                        confidence = float(confidence_text)
                    except (ValueError, TypeError):
                        confidence = 0.5
                elif line.startswith("Reasoning:"):
                    current_section = "reasoning"
                    reasoning = line.replace("Reasoning:", "").strip()
                elif line.startswith("Follow-up Questions:"):
                    current_section = "questions"
                elif current_section == "answer" and line:
                    answer += " " + line
                elif current_section == "reasoning" and line:
                    reasoning += " " + line
                elif current_section == "questions" and line.startswith(("1.", "2.", "3.", "4.", "5.")):
                    question = line[2:].strip()
                    # Filter out questions containing UUIDs
                    uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
                    if not re.search(uuid_pattern, question, re.IGNORECASE):
                        follow_up_questions.append(question)
            
            return {
                'answer': answer.strip() if answer else "Unable to generate answer from available context.",
                'confidence': max(0.0, min(1.0, confidence)),
                'reasoning': reasoning.strip() if reasoning else "Analysis based on graph structure and entity relationships.",
                'follow_up_questions': follow_up_questions if follow_up_questions else [
                    "What additional information would be helpful?",
                    "Are there specific aspects to explore further?",
                    "Should I analyze different communities or relationships?"
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            return {
                'answer': response_text[:500] if response_text else "No response generated.",
                'confidence': 0.4,
                'reasoning': "Direct LLM output due to parsing issues.",
                'follow_up_questions': ["What would you like to know more about?"]
            }
    
    def _get_primary_entity_type(self, node_types: List[str]) -> str:
        """
        Determine primary entity type when node has multiple labels.
        
        Priority order (most specific to least specific):
        1. Project, Organization, Person (domain entities)
        2. Technology, Library, Model, Tool (technical entities)
        3. Publication, Document (content entities)
        4. Event, Location (contextual entities)
        5. Concept, Topic (abstract entities)
        6. Unknown (fallback)
        """
        if not node_types:
            return 'Unknown'
        
        # Priority ranking
        priority_order = [
            'Project', 'Organization', 'Person',
            'Technology', 'Library', 'Model', 'Tool',
            'Publication', 'Document', 'Dataset',
            'Event', 'Location',
            'Concept', 'Topic'
        ]
        
        # Return first match in priority order
        for priority_type in priority_order:
            if priority_type in node_types:
                return priority_type
        
        # Fall back to first label if no priority match
        return node_types[0]
    
    def _detect_query_entity_types(self, query_lower: str) -> List[str]:
        """
        Detect what entity types the user is asking about.
        This prevents showing Organizations when they ask for Projects.
        """
        target_types = []
        
        # Project-related keywords
        if any(word in query_lower for word in ['project', 'projects', 'initiative', 'initiatives']):
            target_types.append('Project')
        
        # Organization-related keywords
        if any(word in query_lower for word in ['organization', 'organizations', 'company', 'companies', 'member', 'members']):
            target_types.append('Organization')
        
        # Person-related keywords
        if any(word in query_lower for word in ['person', 'people', 'who', 'researcher', 'researchers', 'author', 'authors']):
            target_types.append('Person')
        
        # Technology/Tool keywords
        if any(word in query_lower for word in ['tool', 'tools', 'technology', 'technologies', 'library', 'libraries', 'model', 'models']):
            target_types.extend(['Technology', 'Library', 'Model'])
        
        # Document/Publication keywords
        if any(word in query_lower for word in ['document', 'documents', 'publication', 'publications', 'paper', 'papers']):
            target_types.extend(['Publication', 'Document'])
        
        # Event keywords
        if any(word in query_lower for word in ['event', 'events', 'conference', 'conferences', 'meeting', 'meetings']):
            target_types.append('Event')
        
        # Concept keywords
        if any(word in query_lower for word in ['concept', 'concepts', 'topic', 'topics', 'area', 'areas']):
            target_types.append('Concept')
        
        return target_types
    
    def _select_query_relevant_entities(self, entities: List[EntityResult], 
                                       target_types: List[str], 
                                       max_entities: int = 15) -> List[EntityResult]:
        """
        Select entities based on query relevance, not just centrality.
        
        Strategy:
        1. If target types detected, prioritize those entities
        2. Use hybrid scoring: type_match_bonus + centrality
        3. Ensure diversity in results
        4. Fall back to centrality if no type matches
        """
        if not target_types:
            # No specific type detected, use pure centrality
            self.logger.info("No target types detected, using centrality ranking")
            return sorted(
                entities,
                key=lambda e: (e.degree_centrality + e.betweenness_centrality) / 2,
                reverse=True
            )[:max_entities]
        
        # Score entities with type relevance bonus
        scored_entities = []
        for entity in entities:
            # Base centrality score (0-1 range)
            centrality_score = (entity.degree_centrality + entity.betweenness_centrality) / 2
            
            # Type match bonus (2.0 if matches, 0.5 if not)
            # This strongly prioritizes relevant types
            type_bonus = 2.0 if entity.node_type in target_types else 0.5
            
            # Hybrid score: type relevance is more important than centrality
            final_score = (type_bonus * 0.7) + (centrality_score * 0.3)
            
            scored_entities.append((entity, final_score, entity.node_type))
        
        # Sort by hybrid score
        scored_entities.sort(key=lambda x: x[1], reverse=True)
        
        # Take top entities
        selected = [e[0] for e in scored_entities[:max_entities]]
        
        # Log selection reasoning
        type_counts = {}
        for _, _, node_type in scored_entities[:max_entities]:
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        
        self.logger.info(f"Selected {len(selected)} entities with target types {target_types}")
        self.logger.info(f"Type distribution: {type_counts}")
        
        return selected


# Exports
__all__ = ['CommunitySearchEngine', 'CommunityResult', 'EntityResult', 'RelationshipResult']