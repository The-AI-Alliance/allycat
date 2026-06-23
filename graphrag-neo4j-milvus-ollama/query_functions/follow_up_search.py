"""Follow-up search module for local graph traversal. - Phase D (Steps 9-12)"""

import logging
from typing import Dict, List, Any
from dataclasses import dataclass
import re
from datetime import datetime

# Project imports
from .setup import GraphRAGSetup
from .query_preprocessing import DriftRoutingResult
from .knowledge_retrieval import CommunityResult, EntityResult, RelationshipResult


@dataclass
class FollowUpQuestion:
    """Represents a follow-up question from Phase C."""
    question: str
    question_id: int
    extracted_entities: List[str]
    query_type: str
    confidence: float


@dataclass  
class LocalSearchResult:
    """Results from local graph traversal."""
    seed_entities: List[EntityResult]
    traversed_entities: List[EntityResult] 
    traversed_relationships: List[RelationshipResult]
    search_depth: int
    total_nodes_visited: int


@dataclass
class IntermediateAnswer:
    """Intermediate answer for a follow-up question."""
    question_id: int
    question: str
    answer: str
    confidence: float
    reasoning: str
    supporting_entities: List[str]
    supporting_evidence: List[str]


class FollowUpSearch:
    """Follow-up search module for local graph traversal."""
    def __init__(self, setup: GraphRAGSetup):
        self.setup = setup
        self.neo4j_conn = setup.neo4j_conn
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.max_traversal_depth = 2
        self.max_entities_per_hop = 20
        self.min_entity_confidence = 0.7
        self.min_relationship_confidence = 0.6
        
    async def execute_follow_up_phase(self, 
                                    phase_c_results: Dict[str, Any],
                                    routing_result: DriftRoutingResult) -> Dict[str, Any]:
        """
        Execute follow-up search pipeline based on initial results.
        
        Args:
            phase_c_results: Results from community search with follow-up questions
            routing_result: Routing configuration parameters
            
        Returns:
            Dictionary with intermediate answers and entity information
        """
        try:
            self.logger.info("Starting Follow-up Search (Steps 9-12)")
            
            # Process follow-up questions
            self.logger.info("Starting Step 9: Follow-up Question Processing")
            follow_up_questions = await self._process_follow_up_questions(
                phase_c_results.get('initial_answer', {}).get('follow_up_questions', []),
                routing_result
            )
            self.logger.info(f"Step 9 completed: {len(follow_up_questions)} questions processed")
            
            # Local graph traversal  
            self.logger.info("Starting Step 10: Local Graph Traversal")
            local_search_results = await self._execute_local_traversal(
                follow_up_questions,
                phase_c_results.get('communities', []),
                routing_result
            )
            self.logger.info(f"Step 10 completed: {len(local_search_results)} searches performed")
            
            # Entity extraction
            self.logger.info("Starting Step 11: Detailed Entity Extraction") 
            detailed_entities = await self._extract_detailed_entities(
                local_search_results,
                routing_result
            )
            self.logger.info(f"Step 11 completed: {len(detailed_entities)} detailed entities extracted")
            
            # Generate intermediate answers
            self.logger.info("Starting Step 12: Intermediate Answer Generation")
            intermediate_answers = await self._generate_intermediate_answers(
                follow_up_questions,
                local_search_results,
                detailed_entities,
                routing_result
            )
            self.logger.info(f"Step 12 completed: {len(intermediate_answers)} intermediate answers generated")
            
            # Compile results
            phase_d_results = {
                'follow_up_questions': follow_up_questions,
                'local_search_results': local_search_results,
                'detailed_entities': detailed_entities,
                'intermediate_answers': intermediate_answers,
                'execution_stats': {
                    'questions_processed': len(follow_up_questions),
                    'local_searches_executed': len(local_search_results), 
                    'entities_extracted': len(detailed_entities),
                    'answers_generated': len(intermediate_answers),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            self.logger.info(f"Phase D completed: {len(intermediate_answers)} detailed answers generated")
            return phase_d_results
            
        except Exception as e:
            self.logger.error(f"Phase D execution failed: {e}")
            return {'error': str(e), 'intermediate_answers': []}
    
    async def _process_follow_up_questions(self, 
                                         questions: List[str],
                                         routing_result: DriftRoutingResult) -> List[FollowUpQuestion]:
        """Extract keywords from follow-up questions for entity search."""
        processed_questions = []
        
        # Define stopwords to exclude
        stopwords = {
            'what', 'which', 'who', 'how', 'are', 'the', 'does', 'is', 'can',
            'that', 'this', 'there', 'would', 'could', 'should', 'where', 'when',
            'why', 'will', 'with', 'from', 'have', 'has', 'been', 'being'
        }
        
        for i, question in enumerate(questions):
            # IMPROVED: Extract both capitalized AND important lowercase words
            # 1. Capitalized words (proper nouns, acronyms)
            capitalized = re.findall(r'\b[A-Z][a-z]+\b|\b[A-Z]{2,}\b', question)
            capitalized = [k for k in capitalized if k.lower() not in stopwords]
            
            # 2. Important lowercase words (4+ chars, not stopwords)
            lowercase = re.findall(r'\b[a-z]{4,}\b', question)
            lowercase = [w for w in lowercase if w not in stopwords]
            
            # Combine: prioritize capitalized, then add lowercase
            keywords = capitalized + lowercase
            
            # Remove duplicates while preserving order
            seen = set()
            keywords = [k for k in keywords if not (k.lower() in seen or seen.add(k.lower()))]
            
            # Take top 5 keywords (increased from 3)
            keywords = keywords[:5]
            
            # Fallback: if still no keywords, extract ALL words 4+ chars
            if not keywords:
                all_words = re.findall(r'\b\w{4,}\b', question)
                keywords = [w for w in all_words if w.lower() not in stopwords][:5]
            
            follow_up = FollowUpQuestion(
                question=question,
                question_id=i + 1,
                extracted_entities=keywords,
                query_type='search',
                confidence=0.8
            )
            
            processed_questions.append(follow_up)
            self.logger.info(f"Question {i+1}: {question} -> Keywords: {keywords}")
                
        return processed_questions
    

    
    async def _execute_local_traversal(self, 
                                     questions: List[FollowUpQuestion],
                                     communities: List[CommunityResult], 
                                     routing_result: DriftRoutingResult) -> List[LocalSearchResult]:
        """
        Step 10: Execute local graph traversal for each follow-up question.
        
        Performs multi-hop traversal from seed entities to find detailed information.
        """
        local_results = []
        
        for question in questions:
            try:
                # Skip if no keywords extracted
                if not question.extracted_entities:
                    self.logger.warning(f"No keywords for Q{question.question_id}, skipping traversal")
                    continue
                
                # Find seed entities
                seed_entities = await self._find_seed_entities(
                    question.extracted_entities, 
                    communities
                )
                
                if not seed_entities:
                    self.logger.warning(f"No seed entities found for Q{question.question_id}")
                    continue
                
                # Multi-hop traversal
                traversal_result = await self._multi_hop_traversal(
                    seed_entities,
                    question,
                    routing_result
                )
                
                local_results.append(traversal_result)
                self.logger.info(f"   Traversal for Q{question.question_id}: {traversal_result.total_nodes_visited} nodes visited")
                
            except Exception as e:
                self.logger.error(f"Local traversal failed for Q{question.question_id}: {e}")
                
        return local_results
    
    async def _find_seed_entities(self, 
                                entity_names: List[str],
                                communities: List[CommunityResult]) -> List[EntityResult]:
        """Just search the graph for entities matching the keywords."""
        if not entity_names:
            return []
        
        # Search query using exact schema
        conditions = " OR ".join([f"toLower(n.title) CONTAINS toLower('{name}')" for name in entity_names])
        query = f"""
        MATCH (n)
        WHERE n.title IS NOT NULL AND ({conditions})
          AND NOT n:CommunityReport
          AND n.data_quality_score >= 0.2
        RETURN n.id as entity_id, 
               n.title as name, 
               n.content as content,
               n.confidence as confidence,
               n.degree_centrality as degree_centrality,
               n.betweenness_centrality as betweenness_centrality,
               n.closeness_centrality as closeness_centrality,
               n.entity_importance_rank as entity_importance_rank,
               n.bridge_score as bridge_score,
               labels(n) as node_types
        ORDER BY n.entity_importance_rank ASC, n.degree_centrality DESC
        LIMIT 20
        """
        
        try:
            results = self.neo4j_conn.execute_query(query, {})
            entities = []
            
            for record in results:
                entity = EntityResult(
                    entity_id=record['entity_id'],
                    name=record['name'],
                    content=record['content'],
                    confidence=record['confidence'],
                    degree_centrality=record['degree_centrality'],
                    betweenness_centrality=record['betweenness_centrality'],
                    closeness_centrality=record['closeness_centrality'],
                    # Set community info
                    community_id='found',
                    node_type=', '.join(record['node_types']) if record['node_types'] else 'Entity'
                )
                entities.append(entity)
            
            return entities
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    async def _multi_hop_traversal(self, 
                                 seed_entities: List[EntityResult],
                                 question: FollowUpQuestion,
                                 routing_result: DriftRoutingResult) -> LocalSearchResult:
        """Execute multi-hop graph traversal from seed entities."""
        
        all_entities = list(seed_entities)
        all_relationships = []
        visited_node_ids = {entity.entity_id for entity in seed_entities}
        
        current_entities = seed_entities
        
        for hop in range(self.max_traversal_depth):
            if not current_entities:
                break
                
            # Get entity IDs for this hop
            current_ids = [entity.entity_id for entity in current_entities]
            
            # Multi-hop traversal query
            traversal_query = """
            MATCH (seed)-[r]-(neighbor)
            WHERE seed.id IN $current_ids
              AND NOT (neighbor.id IN $visited_ids)
              AND r.confidence >= $min_rel_confidence
              AND neighbor.confidence >= $min_entity_confidence
              AND neighbor.data_quality_score >= 0.2
              AND neighbor.title IS NOT NULL
              AND neighbor.content IS NOT NULL
              AND NOT neighbor:CommunityReport
            RETURN DISTINCT
                   seed.id as seed_id,
                   neighbor.id as neighbor_id,
                   neighbor.title as neighbor_name,
                   neighbor.content as neighbor_content,
                   neighbor.confidence as neighbor_confidence,
                   neighbor.degree_centrality as degree_centrality,
                   neighbor.betweenness_centrality as betweenness_centrality, 
                   neighbor.closeness_centrality as closeness_centrality,
                   neighbor.entity_importance_rank as entity_importance_rank,
                   neighbor.bridge_score as bridge_score,
                   labels(neighbor) as neighbor_types,
                   type(r) as relationship_type,
                   r.confidence as relationship_confidence,
                   r.weight as relationship_weight
            ORDER BY neighbor.entity_importance_rank DESC, r.confidence DESC
            LIMIT $max_results
            """
            
            try:
                results = self.neo4j_conn.execute_query(
                    traversal_query,
                    {
                        'current_ids': current_ids,
                        'visited_ids': list(visited_node_ids),
                        'min_rel_confidence': self.min_relationship_confidence,
                        'min_entity_confidence': self.min_entity_confidence,
                        'max_results': self.max_entities_per_hop
                    }
                )
                
                next_hop_entities = []
                
                for record in results:
                    neighbor_id = record['neighbor_id']
                    
                    if neighbor_id not in visited_node_ids:
                        # Create entity result
                        entity = EntityResult(
                            entity_id=neighbor_id,
                            name=record['neighbor_name'],
                            content=record['neighbor_content'],
                            confidence=record['neighbor_confidence'],
                            degree_centrality=record['degree_centrality'] or 0.0,
                            betweenness_centrality=record['betweenness_centrality'] or 0.0,
                            closeness_centrality=record['closeness_centrality'] or 0.0,
                            # Set community info
                            community_id='unknown',
                            node_type=', '.join(record['neighbor_types']) if record['neighbor_types'] else 'Entity'
                        )
                        
                        all_entities.append(entity)
                        next_hop_entities.append(entity)
                        visited_node_ids.add(neighbor_id)
                    
                    # Create relationship result using REAL schema attributes
                    relationship = RelationshipResult(
                        start_node=record['seed_id'],
                        end_node=neighbor_id,
                        relationship_type=record['relationship_type'],
                        confidence=record['relationship_confidence']
                        # Using REAL schema: startNode, endNode
                    )
                    
                    all_relationships.append(relationship)
                
                current_entities = next_hop_entities
                self.logger.info(f"     Hop {hop + 1}: Found {len(next_hop_entities)} new entities")
                
            except Exception as e:
                self.logger.error(f"Multi-hop traversal failed at hop {hop + 1}: {e}")
                break
        
        return LocalSearchResult(
            seed_entities=seed_entities,
            traversed_entities=all_entities,
            traversed_relationships=all_relationships,
            search_depth=min(hop + 1, self.max_traversal_depth),
            total_nodes_visited=len(visited_node_ids)
        )
    
    async def _extract_detailed_entities(self, 
                                       local_results: List[LocalSearchResult],
                                       routing_result: DriftRoutingResult) -> List[EntityResult]:
        """
        Step 11: Extract detailed entity information from local search results.
        
        Combines and ranks entities from all local searches.
        """
        all_entities = []
        entity_scores = {}
        
        # Collect all entities and calculate importance scores
        for search_result in local_results:
            for entity in search_result.traversed_entities:
                if entity.entity_id not in entity_scores:
                    # Calculate entity importance score
                    importance_score = (
                        0.4 * entity.confidence +
                        0.3 * entity.degree_centrality +
                        0.2 * entity.betweenness_centrality +
                        0.1 * entity.closeness_centrality
                    )
                    
                    entity_scores[entity.entity_id] = {
                        'entity': entity,
                        'importance_score': importance_score,
                        'appearance_count': 1
                    }
                    all_entities.append(entity)
                else:
                    # Increment appearance count for entities found in multiple searches
                    entity_scores[entity.entity_id]['appearance_count'] += 1
        
        # Sort entities by importance score and appearance frequency
        sorted_entities = sorted(
            entity_scores.values(),
            key=lambda x: (x['appearance_count'], x['importance_score']),
            reverse=True
        )
        
        # Return top entities
        max_entities = routing_result.parameters.get('max_detailed_entities', 50)
        detailed_entities = [item['entity'] for item in sorted_entities[:max_entities]]
        
        self.logger.info(f"Extracted {len(detailed_entities)} detailed entities from {len(all_entities)} total")
        return detailed_entities
    
    async def _generate_intermediate_answers(self, 
                                           questions: List[FollowUpQuestion],
                                           local_results: List[LocalSearchResult],
                                           detailed_entities: List[EntityResult],
                                           routing_result: DriftRoutingResult) -> List[IntermediateAnswer]:
        """Simple: just list the entity names we found."""
        answers = []
        
        for i, question in enumerate(questions):
            # Get entities from search result
            entities = local_results[i].traversed_entities if i < len(local_results) else []
            entity_names = [e.name for e in entities[:10]]
            
            # Simple answer with entity names
            answer_text = f"Found entities: {', '.join(entity_names)}" if entity_names else "No specific entities found."
            
            answer = IntermediateAnswer(
                question_id=question.question_id,
                question=question.question,
                answer=answer_text,
                confidence=0.8,
                reasoning=f"Found {len(entity_names)} entities matching the search criteria.",
                supporting_entities=entity_names,
                supporting_evidence=[]
            )
            answers.append(answer)
                
        return answers


# Exports
__all__ = ['FollowUpSearch', 'FollowUpQuestion', 'LocalSearchResult', 'IntermediateAnswer']
    
