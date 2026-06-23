"""
    Vector augmentation engine implementing Phase E (Steps 13-14).
    
    Handles vector search operations and result fusion:
    - Vector similarity search for additional context (Step 13)
    - Result fusion strategy for enhanced answers (Step 14)
"""

import logging
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

from .setup import GraphRAGSetup
from .query_preprocessing import DriftRoutingResult, SearchStrategy


@dataclass
class VectorSearchResult:
    """Vector search result with similarity and content."""
    document_id: str
    content: str
    similarity_score: float
    metadata: Dict[str, Any]
    source_type: str
    relevance_score: float
    file_path: str = None


@dataclass  
class AugmentationResult:
    """Phase E augmentation result with enhanced context."""
    vector_results: List[VectorSearchResult]
    enhanced_context: str
    fusion_strategy: str
    augmentation_confidence: float
    execution_time: float
    metadata: Dict[str, Any]


class VectorAugmentationEngine:
    def __init__(self, setup: GraphRAGSetup):
        self.setup = setup
        self.vector_engine = setup.query_engine  # Milvus vector engine
        self.embedding_model = setup.embedding_model
        self.config = setup.config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Vector search parameters (adaptive defaults)
        self.base_similarity_threshold = 0.65  # Lower base for better coverage
        self.max_vector_results = 20  # Increased from 10 for better context
        
    async def execute_vector_augmentation_phase(self,
                                              query_embedding: List[float],
                                              graph_results: Dict[str, Any],
                                              routing_result: DriftRoutingResult) -> AugmentationResult:
        """
        Execute vector augmentation phase with similarity search.
        
        Args:
            query_embedding: Query vector for similarity matching
            graph_results: Results from graph-based search
            routing_result: Routing decision parameters
            
        Returns:
            Augmentation results with vector context
        """
        start_time = datetime.now()
        
        try:
            # Step 13: Vector Similarity Search
            self.logger.info("Starting Step 13: Vector Similarity Search")
            vector_results = await self._perform_vector_search(
                query_embedding, routing_result
            )
            
            # Step 14: Result Fusion and Enhancement
            self.logger.info("Starting Step 14: Result Fusion and Enhancement")
            enhanced_context = await self._fuse_results(
                vector_results, graph_results, routing_result
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            augmentation_result = AugmentationResult(
                vector_results=vector_results,
                enhanced_context=enhanced_context,
                fusion_strategy='graph_vector_hybrid',
                augmentation_confidence=self._calculate_augmentation_confidence(vector_results),
                execution_time=execution_time,
                metadata={
                    'vector_results_count': len(vector_results),
                    'avg_similarity': np.mean([r.similarity_score for r in vector_results]) if vector_results else 0,
                    'phase': 'vector_augmentation',
                    'step_range': '13-14'
                }
            )
            
            self.logger.info(f"Phase E completed: {len(vector_results)} vector results, augmentation confidence: {augmentation_result.augmentation_confidence:.3f}")
            return augmentation_result
            
        except Exception as e:
            self.logger.error(f"Vector augmentation phase failed: {e}")
            # Return empty augmentation on failure
            return AugmentationResult(
                vector_results=[],
                enhanced_context="",
                fusion_strategy='graph_only',
                augmentation_confidence=0.0,
                execution_time=(datetime.now() - start_time).total_seconds(),
                metadata={'error': str(e), 'fallback': True}
            )
    
    async def _perform_vector_search(self, 
                                   query_embedding: List[float],
                                   routing_result: DriftRoutingResult) -> List[VectorSearchResult]:
        """
        Step 13: Perform comprehensive vector similarity search.
        
        Uses the Milvus vector database to find semantically similar content.
        Adaptive threshold based on query complexity.
        """
        try:
            strategy = routing_result.search_strategy
            if strategy == SearchStrategy.GLOBAL_SEARCH:
                similarity_threshold = 0.60
            elif strategy == SearchStrategy.HYBRID_SEARCH:
                similarity_threshold = 0.65
            else:  # LOCAL_SEARCH
                similarity_threshold = 0.70
            
            similarity_threshold = routing_result.parameters.get('similarity_threshold', similarity_threshold)
            max_results = routing_result.parameters.get('max_vector_results', self.max_vector_results)
            
            self.logger.info(f"Vector search: strategy={strategy.value}, threshold={similarity_threshold:.2f}, max={max_results}")
            
            vector_results = []
            
            # Use the existing vector query engine for similarity search
            if self.vector_engine:
                # Query the vector database with the embedding
                search_results = self.vector_engine.query(routing_result.original_query)
                
                # Extract vector search results from the response
                if hasattr(search_results, 'source_nodes') and search_results.source_nodes:
                    self.logger.info(f"Vector DB returned {len(search_results.source_nodes)} raw results (max={max_results})")
                    
                    for i, node in enumerate(search_results.source_nodes[:max_results]):
                        # Calculate similarity score (handle different node types)
                        similarity_score = 0.8  # Default similarity
                        if hasattr(node, 'score'):
                            similarity_score = node.score
                        elif hasattr(node, 'similarity'):
                            similarity_score = node.similarity
                        elif hasattr(node, 'metadata') and 'score' in node.metadata:
                            similarity_score = node.metadata['score']
                        
                        # Log first 5 scores for debugging
                        if i < 5:
                            self.logger.info(f"  Result {i+1}: score={similarity_score:.4f}, threshold={similarity_threshold:.4f}")
                        
                        # Extract content (handle different node types)
                        content = ""
                        if hasattr(node, 'text'):
                            content = node.text
                        elif hasattr(node, 'content'):
                            content = node.content
                        elif hasattr(node, 'get_content'):
                            content = node.get_content()
                        else:
                            content = str(node)
                        
                        # Extract metadata and file path
                        node_metadata = {}
                        if hasattr(node, 'metadata') and node.metadata:
                            node_metadata = node.metadata
                        elif hasattr(node, 'extra_info') and node.extra_info:
                            node_metadata = node.extra_info
                        
                        # Extract file path from multiple possible locations
                        file_path = node_metadata.get('file_path') or node_metadata.get('source') or node_metadata.get('filename')
                        if not file_path and hasattr(node, 'node_id'):
                            file_path = node.node_id
                        
                        vector_result = VectorSearchResult(
                            document_id=node_metadata.get('doc_id', f"doc_{i}"),
                            content=content,
                            similarity_score=similarity_score,
                            metadata=node_metadata,
                            source_type='vector_db',
                            relevance_score=similarity_score * 0.9,
                            file_path=file_path
                        )
                        
                        # Only include results above adaptive similarity threshold
                        if similarity_score >= similarity_threshold:
                            vector_results.append(vector_result)
                
                self.logger.info(f"Vector search completed: {len(vector_results)} results above threshold {similarity_threshold}")
            else:
                self.logger.warning("Vector engine not available, skipping vector search")
            
            return vector_results
            
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return []
    
    async def _fuse_results(self,
                          vector_results: List[VectorSearchResult],
                          graph_results: Dict[str, Any],
                          routing_result: DriftRoutingResult) -> str:
        """
        Step 14: Fuse vector and graph results for enhanced context.
        
        Combines graph-based entity relationships with vector similarity content.
        Includes cross-verification between graph and vector sources.
        """
        try:
            fusion_parts = []
            
            # CROSS-VERIFICATION: Extract sources from graph results
            graph_sources = set()
            if 'entities' in graph_results:
                for entity in graph_results.get('entities', []):
                    if hasattr(entity, 'source'):
                        graph_sources.add(entity.source)
            if 'relationships' in graph_results:
                for rel in graph_results.get('relationships', []):
                    if hasattr(rel, 'source_file'):
                        graph_sources.add(rel.source_file)
            
            # Extract sources from vector results
            vector_sources = set()
            for result in vector_results:
                if result.file_path:
                    vector_sources.add(result.file_path)
            
            # Log cross-verification
            common_sources = graph_sources & vector_sources
            if common_sources:
                self.logger.info(f"CROSS-VERIFICATION: {len(common_sources)} common sources between graph and vector")
            else:
                self.logger.warning(f"CROSS-VERIFICATION: NO common sources! Graph={len(graph_sources)}, Vector={len(vector_sources)}")
            
            # Start with graph-based context (Phase C & D results)
            if 'initial_answer' in graph_results:
                initial_answer = graph_results['initial_answer']
                if isinstance(initial_answer, dict) and 'content' in initial_answer:
                    fusion_parts.extend([
                        "=== GRAPH-BASED KNOWLEDGE ===",
                        initial_answer['content'],
                        ""
                    ])
            
            # Add vector-based augmentation
            if vector_results:
                fusion_parts.extend([
                    "=== SEMANTIC AUGMENTATION ===",
                    "Additional relevant information from vector similarity search:",
                    ""
                ])
                
                for i, result in enumerate(vector_results[:5], 1):  # Top 5 vector results
                    fusion_parts.extend([
                        f"**{i}. Vector Result (Similarity: {result.similarity_score:.3f})**",
                        result.content,  # Show full content without truncation
                        ""
                    ])
            
            # Add fusion methodology explanation
            fusion_parts.extend([
                "=== FUSION METHODOLOGY ===",
                "This enhanced answer combines graph-based entity relationships with vector semantic similarity search.",
                "Graph results provide structured knowledge connections, while vector search adds contextual depth.",
                ""
            ])
            
            enhanced_context = "\n".join(fusion_parts)
            
            self.logger.info(f"Result fusion completed: {len(fusion_parts)} context sections")
            return enhanced_context
            
        except Exception as e:
            self.logger.error(f"Result fusion failed: {e}")
            return "Graph-based results only (vector fusion failed)"
    
    def _calculate_augmentation_confidence(self, vector_results: List[VectorSearchResult]) -> float:
        """Calculate confidence score for the augmentation results."""
        if not vector_results:
            return 0.0
        
        # Base confidence on average similarity and result count
        avg_similarity = np.mean([r.similarity_score for r in vector_results])
        count_factor = min(len(vector_results) / 10, 1.0)  # Normalize to max 10 results
        
        # Combined confidence
        confidence = (avg_similarity * 0.7) + (count_factor * 0.3)
        
        return min(confidence, 1.0)
    
    def get_augmentation_stats(self) -> Dict[str, Any]:
        """Get statistics about vector augmentation performance."""
        return {
            'similarity_threshold': self.similarity_threshold,
            'max_vector_results': self.max_vector_results,
            'vector_engine_ready': bool(self.vector_engine),
            'embedding_model': str(self.embedding_model) if self.embedding_model else None
        }


# Export main class
__all__ = ['VectorAugmentationEngine', 'VectorSearchResult', 'AugmentationResult']