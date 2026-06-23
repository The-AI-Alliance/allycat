"""
Query Graph Functions Package

Core modules for graph-based retrieval augmentation implementation.

Package contents:
- setup.py: Initialization and connection functionality (Phase A: Steps 1-2)
- query_preprocessing.py: Query analysis, routing, and vectorization (Phase B: Steps 3-5)
- knowledge_retrieval.py: Community search and data extraction (Phase C: Steps 6-8)
- follow_up_search.py: Follow-up search and entity extraction (Phase D: Steps 9-12)
- vector_augmentation.py: Vector search enhancement (Phase E: Steps 13-14)
- answer_synthesis.py: Final answer generation (Phase F: Steps 15-16)
- response_management.py: Metadata generation and file persistence (Phase G: Steps 17-20)
"""

# Phase A: Initialization (Steps 1-2)
from .setup import GraphRAGSetup, create_graphrag_setup

# Phase B: Query Preprocessing (Steps 3-5)
from .query_preprocessing import (
    QueryAnalyzer,
    DriftRouter, 
    QueryVectorizer,
    QueryPreprocessor,
    create_query_preprocessor,
    preprocess_query_pipeline,
    QueryAnalysis,
    DriftRoutingResult, 
    VectorizedQuery,
    QueryType,
    SearchStrategy
)

# Phase C: Knowledge Retrieval (Steps 6-8)
from .knowledge_retrieval import (
    CommunitySearchEngine,
    CommunityResult,
    EntityResult,
    RelationshipResult
)

# Phase D: Follow-up Search (Steps 9-12)
from .follow_up_search import (
    FollowUpSearch,
    FollowUpQuestion,
    LocalSearchResult,
    IntermediateAnswer
)

# Phase E: Vector Search Augmentation (Steps 13-14)
from .vector_augmentation import (
    VectorAugmentationEngine,
    VectorSearchResult,
    AugmentationResult
)

# Phase F: Answer Synthesis (Steps 15-16)
from .answer_synthesis import (
    AnswerSynthesisEngine,
    SynthesisResult,
    SourceEvidence
)

# Phase G: Response Management (Steps 17-20)
from .response_management import (
    ResponseManager,
    ResponseMetadata
)

__version__ = "1.3.0"
__author__ = "AllyCat GraphRAG Team"
__description__ = "Graph-based retrieval augmentation implementation for AllyCat"

# Export main classes and functions
__all__ = [
    # Phase A: Initialization
    "GraphRAGSetup",
    "create_graphrag_setup",
    # Phase B: Query Preprocessing  
    "QueryAnalyzer",
    "DriftRouter", 
    "QueryVectorizer",
    "QueryPreprocessor",
    "create_query_preprocessor",
    "preprocess_query_pipeline",
    "QueryAnalysis",
    "DriftRoutingResult", 
    "VectorizedQuery", 
    "QueryType",
    "SearchStrategy",
    # Phase C: Knowledge Retrieval
    "CommunitySearchEngine",
    "CommunityResult",
    "EntityResult",
    "RelationshipResult",
    # Phase D: Follow-up Search
    "FollowUpSearch",
    "FollowUpQuestion",
    "LocalSearchResult",
    "IntermediateAnswer",
    # Phase E: Vector Augmentation
    "VectorAugmentationEngine",
    "VectorSearchResult",
    "AugmentationResult",
    # Phase F: Answer Synthesis
    "AnswerSynthesisEngine",
    "SynthesisResult",
    "SourceEvidence",
    # Phase G: Response Management
    "ResponseManager",
    "ResponseMetadata"
]