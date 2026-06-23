"""Response management module for metadata generation and file I/O operations - Phase G (Steps 17-20)."""

import time
import json
import logging
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

from my_config import MY_CONFIG
from .setup import GraphRAGSetup
from .query_preprocessing import QueryAnalysis, DriftRoutingResult, VectorizedQuery
from .answer_synthesis import SynthesisResult


@dataclass
class ResponseMetadata:
    """Complete response metadata structure."""
    query_type: str
    search_strategy: str
    complexity_score: float
    total_time_seconds: float
    phases_completed: List[str]
    status: str
    phase_details: Dict[str, Any]
    database_stats: Dict[str, Any]


class ResponseManager:
    def __init__(self, setup: GraphRAGSetup):
        self.setup = setup
        self.config = setup.config
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def generate_comprehensive_metadata(self,
                                      analysis: QueryAnalysis,
                                      routing: DriftRoutingResult,
                                      vectorization: VectorizedQuery,
                                      community_results: Dict[str, Any],
                                      follow_up_results: Dict[str, Any],
                                      augmentation_results: Any,
                                      synthesis_results: SynthesisResult,
                                      total_time: float) -> Dict[str, Any]:
        """
        Generate comprehensive metadata for query response.
        
        Consolidates all phase results into structured metadata format.
        """
        try:
            communities = community_results.get('communities', [])
            
            metadata = {
                # Execution Summary
                "query_type": analysis.query_type.value,
                "search_strategy": routing.search_strategy.value,
                "complexity_score": analysis.complexity_score,
                "total_time_seconds": round(total_time, 2),
                "phases_completed": ["A-Init", "B-Preprocess", "C-Communities", "D-Followup", "E-Vector", "F-Synthesis"],
                "status": "success",
                
                # Phase A: Initialization
                "phase_a": self._generate_phase_a_metadata(),
                
                # Phase B: Query Preprocessing  
                "phase_b": self._generate_phase_b_metadata(analysis, vectorization, routing),
                
                # Phase C: Community Search
                "phase_c": self._generate_phase_c_metadata(communities, community_results),
                
                # Phase D: Follow-up Search
                "phase_d": self._generate_phase_d_metadata(follow_up_results),
                
                # Phase E: Vector Augmentation
                "phase_e": self._generate_phase_e_metadata(augmentation_results),
                
                # Phase F: Answer Synthesis
                "phase_f": self._generate_phase_f_metadata(synthesis_results),
                
                # Database Statistics
                "database_stats": self._generate_database_stats(follow_up_results, communities, augmentation_results)
            }
            
            self.logger.info("Generated comprehensive metadata with all phase details")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to generate metadata: {e}")
            return self._generate_fallback_metadata(str(e))
    
    def _generate_phase_a_metadata(self) -> Dict[str, Any]:
        """Generate Phase A initialization metadata."""
        return {
            "neo4j_connected": bool(self.setup.neo4j_conn),
            "vector_db_ready": bool(self.setup.query_engine),
            "llm_model": getattr(MY_CONFIG, 'LLM_MODEL', 'unknown'),
            "embedding_model": getattr(MY_CONFIG, 'EMBEDDING_MODEL', 'unknown'),
            "drift_config_loaded": bool(self.setup.drift_config)
        }
    
    def _generate_phase_b_metadata(self, analysis: QueryAnalysis, vectorization: VectorizedQuery, routing: DriftRoutingResult) -> Dict[str, Any]:
        """Generate Phase B query preprocessing metadata."""
        return {
            "entities_extracted": len(analysis.entities_mentioned),
            "semantic_keywords": len(vectorization.semantic_keywords),
            "embedding_dimensions": len(vectorization.embedding),
            "similarity_threshold": vectorization.similarity_threshold,
            "routing_confidence": round(routing.confidence, 3)
        }
    
    def _generate_phase_c_metadata(self, communities: List[Any], community_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Phase C community search metadata."""
        return {
            "communities_found": len(communities),
            "community_ids": [c.community_id for c in communities[:5]],
            "similarities": [round(c.similarity_score, 3) for c in communities[:5]],
            "entities_extracted": len(community_results.get('extracted_data', {}).get('entities', [])),
            "relationships_extracted": len(community_results.get('extracted_data', {}).get('relationships', []))
        }
    
    def _generate_phase_d_metadata(self, follow_up_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Phase D follow-up search metadata."""
        intermediate_answers = follow_up_results.get('intermediate_answers', [])
        avg_confidence = 0.0
        if intermediate_answers:
            avg_confidence = sum(a.confidence for a in intermediate_answers) / len(intermediate_answers)
        
        return {
            "questions_generated": len(follow_up_results.get('follow_up_questions', [])),
            "graph_traversals": len(follow_up_results.get('local_search_results', [])),
            "entities_found": len(follow_up_results.get('detailed_entities', [])),
            "intermediate_answers": len(intermediate_answers),
            "avg_confidence": round(avg_confidence, 3)
        }
    
    def _generate_phase_e_metadata(self, augmentation_results: Any) -> Dict[str, Any]:
        """Generate Phase E vector augmentation metadata."""
        if not augmentation_results:
            return {"vector_results_count": 0, "augmentation_confidence": 0.0}
        
        vector_files = []
        if hasattr(augmentation_results, 'vector_results'):
            for i, result in enumerate(augmentation_results.vector_results):
                file_info = {
                    "file_id": i + 1,
                    "file_path": getattr(result, 'file_path', 'unknown'),
                    "similarity": round(result.similarity_score, 3),
                    "content_length": len(result.content),
                    "relevance": round(getattr(result, 'relevance_score', 0.0), 3)
                }
                vector_files.append(file_info)
        
        return {
            "vector_results_count": len(augmentation_results.vector_results) if hasattr(augmentation_results, 'vector_results') else 0,
            "augmentation_confidence": round(augmentation_results.augmentation_confidence, 3) if hasattr(augmentation_results, 'augmentation_confidence') else 0.0,
            "execution_time": round(augmentation_results.execution_time, 2) if hasattr(augmentation_results, 'execution_time') else 0.0,
            "similarity_threshold": 0.75,
            "vector_files": vector_files
        }
    
    def _generate_phase_f_metadata(self, synthesis_results: SynthesisResult) -> Dict[str, Any]:
        """Generate Phase F answer synthesis metadata."""
        return {
            "synthesis_confidence": round(synthesis_results.confidence_score, 3),
            "sources_integrated": len(synthesis_results.source_evidence),
            "final_answer_length": len(synthesis_results.final_answer),
            "synthesis_method": getattr(synthesis_results, 'synthesis_method', 'comprehensive_fusion')
        }
    
    def _generate_database_stats(self, follow_up_results: Dict[str, Any], communities: List[Any], augmentation_results: Any) -> Dict[str, Any]:
        """Generate database statistics metadata."""
        vector_docs_used = 0
        if augmentation_results and hasattr(augmentation_results, 'vector_results'):
            vector_docs_used = len(augmentation_results.vector_results)
        
        return {
            "total_nodes": self.setup.graph_stats.get('node_count', 0),
            "total_relationships": self.setup.graph_stats.get('relationship_count', 0),
            "total_communities": self.setup.graph_stats.get('community_count', 0),
            "nodes_accessed": len(follow_up_results.get('detailed_entities', [])),
            "communities_searched": len(communities),
            "vector_docs_used": vector_docs_used
        }
    
    def _generate_fallback_metadata(self, error: str) -> Dict[str, Any]:
        """Generate minimal metadata when full generation fails."""
        return {
            "status": "metadata_generation_error",
            "error": error,
            "phases_completed": "incomplete",
            "total_time_seconds": 0.0
        }
    
    def save_response_to_files(self, user_query: str, result: Dict[str, Any]) -> None:
        """
        Save query response and metadata to separate files.
        
        Handles file I/O operations for response persistence.
        """
        try:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Save response to response file
            self._save_response_file(user_query, result, timestamp)
            
            # Save metadata to metadata file
            self._save_metadata_file(user_query, result, timestamp)
            
            self.logger.info(f"Saved response and metadata for query: {user_query[:50]}...")
            
        except Exception as e:
            self.logger.error(f"Failed to save response files: {e}")
    
    def _save_response_file(self, user_query: str, result: Dict[str, Any], timestamp: str) -> None:
        """Save response content to response file."""
        try:
            with open('logs/graphrag_query/graphrag_responses.txt', 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"QUERY [{timestamp}]: {user_query}\n")
                f.write(f"{'='*80}\n")
                f.write(f"RESPONSE: {result['answer']}\n")
                f.write(f"{'='*80}\n\n")
        except Exception as e:
            self.logger.error(f"Failed to save response file: {e}")
    
    def _save_metadata_file(self, user_query: str, result: Dict[str, Any], timestamp: str) -> None:
        """Save metadata to metadata file."""
        try:
            with open('logs/graphrag_query/graphrag_metadata.txt', 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"METADATA [{timestamp}]: {user_query}\n")
                f.write(f"{'='*80}\n")
                f.write(json.dumps(result['metadata'], indent=2, default=str))
                f.write(f"\n{'='*80}\n\n")
        except Exception as e:
            self.logger.error(f"Failed to save metadata file: {e}")
    
    def format_error_response(self, error_message: str) -> Dict[str, Any]:
        """
        Generate standardized error response with metadata.
        
        Creates consistent error format for failed queries.
        """
        return {
            "answer": f"Sorry, I encountered an error: {error_message}",
            "metadata": {
                "status": "error",
                "error_message": error_message,
                "phases_completed": "incomplete",
                "neo4j_connected": bool(self.setup.neo4j_conn) if self.setup.neo4j_conn else False,
                "vector_engine_ready": bool(self.setup.query_engine) if self.setup.query_engine else False,
                "timestamp": datetime.now().isoformat()
            }
        }


# Exports
__all__ = ['ResponseManager', 'ResponseMetadata']