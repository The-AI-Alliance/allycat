"""Query preprocessing for analysis, routing, and vectorization - Phase B (Steps 3-5)."""

import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import re

# System imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from my_config import MY_CONFIG
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class QueryType(Enum):
    """Query type classifications for DRIFT routing."""
    SPECIFIC_ENTITY = "specific_entity"
    RELATIONSHIP_QUERY = "relationship_query"
    BROAD_THEMATIC = "broad_thematic"
    COMPARATIVE = "comparative"
    COMPLEX_REASONING = "complex_reasoning"
    FACTUAL_LOOKUP = "factual_lookup"


class SearchStrategy(Enum):
    """Search strategy determined by DRIFT routing."""
    LOCAL_SEARCH = "local_search"
    GLOBAL_SEARCH = "global_search"
    HYBRID_SEARCH = "hybrid_search"


@dataclass
class QueryAnalysis:
    """Results of query analysis step."""
    query_type: QueryType
    complexity_score: float  # 0.0 to 1.0
    entities_mentioned: List[str]
    key_concepts: List[str]
    intent_description: str
    context_requirements: Dict[str, Any]
    estimated_scope: str  # "narrow", "moderate", "broad"


@dataclass
@dataclass
class DriftRoutingResult:
    """Results of DRIFT routing decision."""
    search_strategy: SearchStrategy
    reasoning: str
    confidence: float  # 0.0 to 1.0
    parameters: Dict[str, Any]
    original_query: str  # Added to fix answer generation
    fallback_strategy: Optional[SearchStrategy] = None


@dataclass
class VectorizedQuery:
    """Results of query vectorization."""
    embedding: List[float]
    embedding_model: str
    normalized_query: str
    semantic_keywords: List[str]
    similarity_threshold: float


class QueryAnalyzer:
    """Handles Step 3: Query Analysis with intent detection and complexity assessment."""
    
    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger('graphrag_query')
        
        # Entity extraction patterns
        self.entity_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns
            r'\b(?:company|organization|person|place|event)\s+(?:named|called)?\s*["\']?([^"\']+)["\']?',
            r'\bwho\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\bwhat\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        ]
        
        # Complexity indicators
        self.complexity_indicators = {
            'high': ['compare', 'analyze', 'evaluate', 'relationship', 'impact', 'why', 'how'],
            'medium': ['describe', 'explain', 'summarize', 'list', 'identify'],
            'low': ['who', 'what', 'when', 'where', 'is', 'are']
        }
        
        self.logger.info("QueryAnalyzer initialized for Step 3 processing")
    
    async def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query for intent, complexity, and entities."""
        self.logger.info(f"Starting Step 3: Query Analysis for: {query[:100]}...")
        
        try:
            # Extract entities and concepts
            entities = self._extract_entities(query)
            concepts = self._extract_key_concepts(query)
            query_type = self._classify_query_type(query, entities, concepts)
            complexity = self._calculate_complexity(query, query_type)
            intent = self._determine_intent(query, query_type)
            scope = self._estimate_scope(query, entities, concepts, complexity)
            
            # Build context
            context_reqs = self._analyze_context_requirements(query, query_type, entities)
            
            analysis = QueryAnalysis(
                query_type=query_type,
                complexity_score=complexity,
                entities_mentioned=entities,
                key_concepts=concepts,
                intent_description=intent,
                context_requirements=context_reqs,
                estimated_scope=scope
            )
            
            self.logger.info(f"Step 3 completed: Query type={query_type.value}, "
                           f"complexity={complexity:.2f}, entities={len(entities)}, scope={scope}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Step 3 Query Analysis failed: {e}")
            raise
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from query text."""
        entities = set()
        
        for pattern in self.entity_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities.update(matches)
        
        # Filter entities
        filtered_entities = [
            entity.strip() for entity in entities 
            if len(entity.strip()) > 2 and entity.lower() not in 
            {'the', 'and', 'are', 'is', 'was', 'were', 'this', 'that', 'what', 'who', 'how'}
        ]
        
        return list(set(filtered_entities))
    
    def _extract_key_concepts(self, query: str) -> List[str]:
        """Extract key conceptual terms from query."""
        # Extract concepts
        concepts = []
        
        # Find domain terms
        domain_terms = [
            'revenue', 'profit', 'growth', 'market', 'strategy', 'technology',
            'product', 'service', 'customer', 'partnership', 'acquisition',
            'investment', 'research', 'development', 'innovation', 'competition'
        ]
        
        query_lower = query.lower()
        for term in domain_terms:
            if term in query_lower:
                concepts.append(term)
        
        return concepts
    
    def _classify_query_type(self, query: str, entities: List[str], concepts: List[str]) -> QueryType:
        """Classify the type of query for routing decisions."""
        query_lower = query.lower()
        
        # Check patterns
        if any(word in query_lower for word in ['compare', 'versus', 'vs', 'difference']):
            return QueryType.COMPARATIVE
        
        if any(word in query_lower for word in ['relationship', 'connect', 'related', 'between']):
            return QueryType.RELATIONSHIP_QUERY
        
        if len(entities) > 0 and any(word in query_lower for word in ['who is', 'what is', 'about']):
            return QueryType.SPECIFIC_ENTITY
        
        if any(word in query_lower for word in ['analyze', 'evaluate', 'why', 'how', 'impact']):
            return QueryType.COMPLEX_REASONING
        
        if len(concepts) > 2 or any(word in query_lower for word in ['overall', 'general', 'trend']):
            return QueryType.BROAD_THEMATIC
        
        return QueryType.FACTUAL_LOOKUP
    
    def _calculate_complexity(self, query: str, query_type: QueryType) -> float:
        """Calculate query complexity score (0.0 to 1.0)."""
        base_score = 0.3
        query_lower = query.lower()
        
        # Base complexity
        type_scores = {
            QueryType.FACTUAL_LOOKUP: 0.2,
            QueryType.SPECIFIC_ENTITY: 0.3,
            QueryType.RELATIONSHIP_QUERY: 0.6,
            QueryType.BROAD_THEMATIC: 0.7,
            QueryType.COMPARATIVE: 0.8,
            QueryType.COMPLEX_REASONING: 0.9
        }
        
        base_score = type_scores.get(query_type, 0.5)
        
        # Adjust complexity
        for level, indicators in self.complexity_indicators.items():
            count = sum(1 for indicator in indicators if indicator in query_lower)
            if level == 'high':
                base_score += count * 0.2
            elif level == 'medium':
                base_score += count * 0.1
            else:
                base_score -= count * 0.05
        
        # Query length and structure
        if len(query.split()) > 15:
            base_score += 0.1
        if '?' in query and len(query.split('?')) > 2:
            base_score += 0.15
        
        return min(1.0, max(0.0, base_score))
    
    def _determine_intent(self, query: str, query_type: QueryType) -> str:
        """Determine the user's intent based on query analysis."""
        intent_map = {
            QueryType.FACTUAL_LOOKUP: "Seeking specific factual information",
            QueryType.SPECIFIC_ENTITY: "Requesting details about a particular entity",
            QueryType.RELATIONSHIP_QUERY: "Exploring connections and relationships",
            QueryType.BROAD_THEMATIC: "Understanding broad themes or patterns",
            QueryType.COMPARATIVE: "Comparing entities or concepts",
            QueryType.COMPLEX_REASONING: "Requiring analytical reasoning and insights"
        }
        
        return intent_map.get(query_type, "General information seeking")
    
    def _estimate_scope(self, query: str, entities: List[str], concepts: List[str], complexity: float) -> str:
        """Estimate the scope of information needed."""
        if len(entities) == 1 and complexity < 0.4:
            return "narrow"
        elif len(entities) > 3 or len(concepts) > 3 or complexity > 0.7:
            return "broad"
        else:
            return "moderate"
    
    def _analyze_context_requirements(self, query: str, query_type: QueryType, entities: List[str]) -> Dict[str, Any]:
        """Analyze what context information is needed."""
        return {
            "requires_entity_details": len(entities) > 0,
            "requires_relationships": query_type in [QueryType.RELATIONSHIP_QUERY, QueryType.COMPARATIVE],
            "requires_historical_context": any(word in query.lower() for word in ['history', 'past', 'previous', 'before']),
            "requires_quantitative_data": any(word in query.lower() for word in ['number', 'amount', 'count', 'revenue', 'profit']),
            "primary_entities": entities[:3]  # Focus on top 3 entities
        }


class DriftRouter:
    """Handles Step 4: DRIFT Routing for optimal search strategy selection."""
    
    def __init__(self, config: Any, graph_stats: Dict[str, Any]):
        self.config = config
        self.graph_stats = graph_stats
        self.logger = logging.getLogger('graphrag_query')
        
        # Routing thresholds
        self.local_search_threshold = 0.4
        self.global_search_threshold = 0.7
        self.entity_count_threshold = 10  # Based on graph size
        
        self.logger.info("DriftRouter initialized for Step 4 processing")
    
    async def determine_search_strategy(self, query_analysis: QueryAnalysis, original_query: str) -> DriftRoutingResult:
        """
        Determine optimal search strategy using DRIFT methodology (Step 4).
        
        Args:
            query_analysis: Results from Step 3 query analysis
            original_query: The original user query
            
        Returns:
            DriftRoutingResult with search strategy and parameters
        """
        self.logger.info(f"Starting Step 4: DRIFT Routing for {query_analysis.query_type.value}")
        
        try:
            # Apply routing logic
            strategy, reasoning, confidence, params = self._apply_drift_logic(query_analysis)
            
            # Fallback strategy
            fallback = self._determine_fallback_strategy(strategy)
            
            result = DriftRoutingResult(
                search_strategy=strategy,
                reasoning=reasoning,
                confidence=confidence,
                parameters=params,
                original_query=original_query,
                fallback_strategy=fallback
            )
            
            self.logger.info(f"Step 4 completed: Strategy={strategy.value}, "
                           f"confidence={confidence:.2f}, reasoning={reasoning[:50]}...")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Step 4 DRIFT Routing failed: {e}")
            raise
    
    def _apply_drift_logic(self, analysis: QueryAnalysis) -> Tuple[SearchStrategy, str, float, Dict[str, Any]]:
        """Apply DRIFT (Distributed Retrieval and Information Filtering Technique) logic."""
        
        # Decision factors
        complexity = analysis.complexity_score
        entity_count = len(analysis.entities_mentioned)
        scope = analysis.estimated_scope
        query_type = analysis.query_type
        
        # Local search conditions
        if (query_type == QueryType.SPECIFIC_ENTITY and 
            entity_count <= 2 and 
            complexity < self.local_search_threshold):
            
            return (
                SearchStrategy.LOCAL_SEARCH,
                f"Specific entity query with low complexity ({complexity:.2f})",
                0.9,
                {
                    "max_depth": 2,
                    "entity_focus": analysis.entities_mentioned,
                    "include_neighbors": True,
                    "max_results": 20
                }
            )
        
        # Global search conditions
        if (complexity > self.global_search_threshold or
            scope == "broad" or
            query_type in [QueryType.BROAD_THEMATIC, QueryType.COMPLEX_REASONING]):
            
            return (
                SearchStrategy.GLOBAL_SEARCH,
                f"High complexity ({complexity:.2f}) or broad scope requiring global context",
                0.85,
                {
                    "community_level": "high",
                    "max_communities": 10,
                    "include_summary": True,
                    "max_results": 50
                }
            )
        
        # Hybrid search for intermediate cases
        if (query_type == QueryType.RELATIONSHIP_QUERY or
            query_type == QueryType.COMPARATIVE or
            entity_count > 2):
            
            return (
                SearchStrategy.HYBRID_SEARCH,
                f"Relationship/comparative query or multiple entities ({entity_count})",
                0.75,
                {
                    "max_communities": 5,
                    "local_depth": 2,
                    "global_communities": 5,
                    "balance_weight": 0.6,  # Favor local over global
                    "max_results": 35
                }
            )
        
        # Default to local search with moderate confidence
        return (
            SearchStrategy.LOCAL_SEARCH,
            "Default local search for moderate complexity query",
            0.6,
            {
                "max_communities": 3,
                "max_depth": 3,
                "entity_focus": analysis.entities_mentioned,
                "include_neighbors": True,
                "max_results": 25
            }
        )
    
    def _determine_fallback_strategy(self, primary_strategy: SearchStrategy) -> Optional[SearchStrategy]:
        """Determine fallback strategy if primary fails."""
        fallback_map = {
            SearchStrategy.LOCAL_SEARCH: SearchStrategy.GLOBAL_SEARCH,
            SearchStrategy.GLOBAL_SEARCH: SearchStrategy.LOCAL_SEARCH,
            SearchStrategy.HYBRID_SEARCH: SearchStrategy.LOCAL_SEARCH
        }
        
        return fallback_map.get(primary_strategy)


class QueryVectorizer:
    """Handles Step 5: Query Vectorization for semantic similarity matching."""
    
    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger('graphrag_query')
        
        # Initialize embedding model using same pattern as other files
        self.embedding_model = HuggingFaceEmbedding(
            model_name=MY_CONFIG.EMBEDDING_MODEL
        )
        
        self.model_name = MY_CONFIG.EMBEDDING_MODEL
        self.embedding_dimension = MY_CONFIG.EMBEDDING_LENGTH
        
        self.logger.info(f"QueryVectorizer initialized with {self.model_name}")
    
    async def vectorize_query(self, query: str, query_analysis: QueryAnalysis) -> VectorizedQuery:
        """
        Generate query embeddings for similarity matching (Step 5).
        
        Args:
            query: Original query text
            query_analysis: Results from Step 3
            
        Returns:
            VectorizedQuery with embeddings and metadata
        """
        self.logger.info(f"Starting Step 5: Query Vectorization for: {query[:100]}...")
        
        try:
            # Normalize query
            normalized_query = self._normalize_query(query, query_analysis)
            
            # Generate embedding
            embedding = await self._generate_embedding(normalized_query)
            
            # Extract keywords
            semantic_keywords = self._extract_semantic_keywords(query, query_analysis)
            
            # Set similarity threshold
            similarity_threshold = self._calculate_similarity_threshold(query_analysis)
            
            result = VectorizedQuery(
                embedding=embedding,
                embedding_model=self.model_name,
                normalized_query=normalized_query,
                semantic_keywords=semantic_keywords,
                similarity_threshold=similarity_threshold
            )
            
            self.logger.info(f"Step 5 completed: Embedding dimension={len(embedding)}, "
                           f"threshold={similarity_threshold:.3f}, keywords={len(semantic_keywords)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Step 5 Query Vectorization failed: {e}")
            raise
    
    def _normalize_query(self, query: str, analysis: QueryAnalysis) -> str:
        """Normalize query text for better embedding quality."""
        # Start with original query
        normalized = query.strip()
        
        # Add important entities and concepts for context
        if analysis.entities_mentioned:
            entity_context = " ".join(analysis.entities_mentioned[:3])
            normalized = f"{normalized} [Entities: {entity_context}]"
        
        if analysis.key_concepts:
            concept_context = " ".join(analysis.key_concepts[:3])
            normalized = f"{normalized} [Concepts: {concept_context}]"
        
        return normalized
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using configured model."""
        try:
            embedding = await self.embedding_model.aget_text_embedding(text)
            return embedding
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            # Fallback to synchronous call if async fails
            return self.embedding_model.get_text_embedding(text)
    
    def _extract_semantic_keywords(self, query: str, analysis: QueryAnalysis) -> List[str]:
        """Extract semantic keywords for additional matching."""
        keywords = set()
        
        # Add entities and concepts
        keywords.update(analysis.entities_mentioned)
        keywords.update(analysis.key_concepts)
        
        # Add query-specific terms based on type
        if analysis.query_type == QueryType.RELATIONSHIP_QUERY:
            keywords.update(['relationship', 'connection', 'related', 'linked'])
        elif analysis.query_type == QueryType.COMPARATIVE:
            keywords.update(['comparison', 'versus', 'difference', 'similar'])
        elif analysis.query_type == QueryType.BROAD_THEMATIC:
            keywords.update(['theme', 'pattern', 'trend', 'overview'])
        
        # Filter and return as list
        return [kw for kw in keywords if len(kw) > 2]
    
    def _calculate_similarity_threshold(self, analysis: QueryAnalysis) -> float:
        """Calculate appropriate similarity threshold based on query characteristics."""
        base_threshold = 0.7
        
        # Adjust based on query complexity
        if analysis.complexity_score > 0.7:
            base_threshold -= 0.1  # Lower threshold for complex queries
        elif analysis.complexity_score < 0.3:
            base_threshold += 0.1  # Higher threshold for simple queries
        
        # Adjust based on scope
        if analysis.estimated_scope == "narrow":
            base_threshold += 0.05
        elif analysis.estimated_scope == "broad":
            base_threshold -= 0.05
        
        # Ensure reasonable bounds
        return max(0.5, min(0.9, base_threshold))


class QueryPreprocessor:
    """Main class coordinating all query preprocessing steps (Steps 3-5)."""
    
    def __init__(self, config: Any, graph_stats: Dict[str, Any]):
        self.config = config
        self.graph_stats = graph_stats
        self.logger = logging.getLogger('graphrag_query')
        
        # Initialize component processors
        self.analyzer = QueryAnalyzer(config)
        self.router = DriftRouter(config, graph_stats)
        self.vectorizer = QueryVectorizer(config)
        
        self.logger.info("QueryPreprocessor initialized for Steps 3-5")
    
    async def preprocess_query(self, query: str) -> Tuple[QueryAnalysis, DriftRoutingResult, VectorizedQuery]:
        """
        Execute complete query preprocessing pipeline (Steps 3-5).
        
        Args:
            query: User's natural language query
            
        Returns:
            Tuple of (analysis, routing, vectorization) results
        """
        self.logger.info(f"Starting Phase B: Query Preprocessing Pipeline for: {query[:100]}...")
        
        try:
            # Query analysis
            analysis = await self.analyzer.analyze_query(query)
            
            # Query routing
            routing = await self.router.determine_search_strategy(analysis, query)
            
            # Query vectorization
            vectorization = await self.vectorizer.vectorize_query(query, analysis)
            
            self.logger.info(f"Phase B completed successfully: "
                           f"Type={analysis.query_type.value}, "
                           f"Strategy={routing.search_strategy.value}, "
                           f"Embedding_dim={len(vectorization.embedding)}")
            
            return analysis, routing, vectorization
            
        except Exception as e:
            self.logger.error(f"Query preprocessing pipeline failed: {e}")
            raise


# Exports
async def create_query_preprocessor(config: Any, graph_stats: Dict[str, Any]) -> QueryPreprocessor:
    """Create and initialize QueryPreprocessor."""
    return QueryPreprocessor(config, graph_stats)


async def preprocess_query_pipeline(query: str, config: Any, graph_stats: Dict[str, Any]) -> Tuple[QueryAnalysis, DriftRoutingResult, VectorizedQuery]:
    """
    Convenience function for complete query preprocessing.
    
    Args:
        query: User's natural language query
        config: Application configuration
        graph_stats: Graph database statistics
        
    Returns:
        Complete preprocessing results
    """
    preprocessor = await create_query_preprocessor(config, graph_stats)
    return await preprocessor.preprocess_query(query)


__all__ = [
    'QueryAnalyzer', 'DriftRouter', 'QueryVectorizer', 'QueryPreprocessor',
    'create_query_preprocessor', 'preprocess_query_pipeline',
    'QueryAnalysis', 'DriftRoutingResult', 'VectorizedQuery',
    'QueryType', 'SearchStrategy'
]