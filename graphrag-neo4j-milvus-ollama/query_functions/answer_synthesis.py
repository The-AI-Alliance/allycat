"""Answer synthesis module for final response generation. - Phase F (Steps 15-16)"""

import logging
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from .setup import GraphRAGSetup
from .query_preprocessing import DriftRoutingResult, QueryAnalysis
from .vector_augmentation import AugmentationResult


@dataclass
class SourceEvidence:
    """Evidence source with attribution and confidence."""
    source_type: str  # 'community', 'entity', 'relationship', 'vector_doc'
    source_id: str
    content: str
    confidence: float
    phase: str  # 'C', 'D', 'E'


@dataclass
class SynthesisResult:
    """Phase F synthesis result with comprehensive answer."""
    final_answer: str
    confidence_score: float
    source_evidence: List[SourceEvidence]
    synthesis_strategy: str
    coverage_assessment: Dict[str, float]
    execution_time: float
    metadata: Dict[str, Any]


class AnswerSynthesisEngine:
    """
    Answer synthesis engine implementing Phase F (Steps 15-16).
    
    Handles final answer generation process:
    - Context assembly and evidence ranking (Step 15)
    - Final answer generation with confidence scoring (Step 16)
    """
    
    def __init__(self, setup: GraphRAGSetup):
        self.setup = setup
        self.llm = setup.llm
        self.config = setup.config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Synthesis parameters
        self.min_confidence_threshold = 0.7
        self.max_synthesis_length = 2000
        
    async def execute_answer_synthesis_phase(self,
                                           analysis: QueryAnalysis,
                                           routing: DriftRoutingResult,
                                           community_results: Dict[str, Any],
                                           follow_up_results: Dict[str, Any],
                                           augmentation_results: AugmentationResult) -> SynthesisResult:
        """
        Execute answer synthesis phase with comprehensive integration.
        
        Args:
            analysis: Query analysis results
            routing: Routing decision parameters  
            community_results: Community search results
            follow_up_results: Follow-up search results
            augmentation_results: Vector augmentation results
            
        Returns:
            Synthesis result with final answer
        """
        start_time = datetime.now()
        
        try:
            # Context assembly
            self.logger.info("Starting Step 15: Context Assembly and Ranking")
            assembled_context = await self._assemble_and_rank_context(
                analysis, community_results, follow_up_results, augmentation_results
            )
            
            # Final answer generation
            self.logger.info("Starting Step 16: Final Answer Generation")
            final_answer, confidence = await self._generate_final_answer(
                analysis, routing, assembled_context
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            synthesis_result = SynthesisResult(
                final_answer=final_answer,
                confidence_score=confidence,
                source_evidence=assembled_context['evidence'],
                synthesis_strategy='comprehensive_drift',
                coverage_assessment=assembled_context['coverage'],
                execution_time=execution_time,
                metadata={
                    'sources_integrated': len(assembled_context['evidence']),
                    'phase_coverage': assembled_context['phase_coverage'],
                    'synthesis_method': 'llm_guided',
                    'phase': 'answer_synthesis',
                    'step_range': '15-16'
                }
            )
            
            self.logger.info(f"Phase F completed: confidence {confidence:.3f}, {len(assembled_context['evidence'])} sources integrated")
            return synthesis_result
            
        except Exception as e:
            self.logger.error(f"Answer synthesis phase failed: {e}")
            # Return fallback synthesis on failure
            return self._create_fallback_synthesis(
                community_results, follow_up_results, 
                (datetime.now() - start_time).total_seconds(), str(e)
            )
    
    async def _assemble_and_rank_context(self,
                                       analysis: QueryAnalysis,
                                       community_results: Dict[str, Any],
                                       follow_up_results: Dict[str, Any],
                                       augmentation_results: AugmentationResult) -> Dict[str, Any]:
        """
        Step 15: Assemble and rank all context from Phases C, D, and E.
        
        Prioritizes information by relevance, confidence, and source diversity.
        """
        evidence_sources = []
        
        # Extract community evidence
        if 'communities' in community_results:
            for community in community_results['communities']:
                evidence_sources.append(SourceEvidence(
                    source_type='community',
                    source_id=community.community_id,
                    content=community.summary,
                    confidence=community.similarity_score,
                    phase='C'
                ))
        
        # Extract follow-up evidence  
        if 'intermediate_answers' in follow_up_results:
            for answer in follow_up_results['intermediate_answers']:
                evidence_sources.append(SourceEvidence(
                    source_type='entity_search',
                    source_id=f"followup_{len(evidence_sources)}",
                    content=f"Q: {answer.question}\nA: {answer.answer}",
                    confidence=answer.confidence,
                    phase='D'
                ))
        
        # Extract vector evidence
        if augmentation_results and augmentation_results.vector_results:
            for i, vector_result in enumerate(augmentation_results.vector_results):
                evidence_sources.append(SourceEvidence(
                    source_type='vector_doc',
                    source_id=f"vector_{i}",
                    content=vector_result.content,
                    confidence=vector_result.similarity_score,
                    phase='E'
                ))
        
        # Rank evidence
        ranked_evidence = sorted(evidence_sources, key=lambda x: x.confidence, reverse=True)
        
        # Calculate coverage
        coverage = {
            'community_coverage': len([e for e in ranked_evidence if e.phase == 'C']) / max(1, len(community_results.get('communities', []))),
            'entity_coverage': len([e for e in ranked_evidence if e.phase == 'D']) / max(1, len(follow_up_results.get('intermediate_answers', []))),
            'vector_coverage': len([e for e in ranked_evidence if e.phase == 'E']) / max(1, len(augmentation_results.vector_results) if augmentation_results else 1),
            'overall_confidence': sum(e.confidence for e in ranked_evidence) / max(1, len(ranked_evidence))
        }
        
        phase_coverage = {
            'phase_c': len([e for e in ranked_evidence if e.phase == 'C']),
            'phase_d': len([e for e in ranked_evidence if e.phase == 'D']),
            'phase_e': len([e for e in ranked_evidence if e.phase == 'E'])
        }
        
        return {
            'evidence': ranked_evidence[:15],  # Top 15 pieces of evidence
            'coverage': coverage,
            'phase_coverage': phase_coverage
        }
    
    async def _generate_final_answer(self,
                                   analysis: QueryAnalysis,
                                   routing: DriftRoutingResult,
                                   assembled_context: Dict[str, Any]) -> tuple[str, float]:
        """
        Step 16: Generate comprehensive final answer using LLM synthesis.
        
        Creates structured, comprehensive response with proper source attribution.
        """
        try:
            # Prepare prompt
            synthesis_prompt = self._create_synthesis_prompt(
                routing.original_query,
                assembled_context['evidence']
            )
            
            # Generate answer
            response = self.llm.complete(synthesis_prompt)
            final_answer = str(response).strip()
            
            # Calculate confidence
            synthesis_confidence = self._calculate_synthesis_confidence(
                assembled_context['evidence'], assembled_context['coverage']
            )
            
            # Format final answer
            formatted_answer = self._format_final_answer(
                final_answer, assembled_context['evidence'], synthesis_confidence
            )
            
            return formatted_answer, synthesis_confidence
            
        except Exception as e:
            self.logger.error(f"Final answer generation failed: {e}")
            return self._create_fallback_answer(assembled_context['evidence']), 0.5
    
    def _create_synthesis_prompt(self, original_query: str, evidence: List[SourceEvidence]) -> str:
        """Create comprehensive synthesis prompt for LLM."""
        prompt_parts = [
            f"# Query: {original_query}",
            "",
            "You are an expert synthesizing information from multiple sources.",
            "Create a comprehensive, accurate answer using the following evidence:",
            "",
            "## Evidence Sources:",
            ""
        ]
        
        for i, source in enumerate(evidence[:10], 1):  # Top 10 sources
            prompt_parts.extend([
                f"### Source {i} ({source.phase} - {source.source_type}, confidence: {source.confidence:.3f})",
                source.content[:1000] + ("..." if len(source.content) > 500 else ""),
                ""
            ])
        
        prompt_parts.extend([
            "## Instructions:",
            "1. Synthesize a comprehensive answer addressing the original query",
            "2. Prioritize high-confidence sources (>0.8)",
            "3. Include specific details and examples from the evidence",
            "4. Structure the response clearly with sections if appropriate",
            "5. Do not mention source IDs or technical details",
            "6. Focus on factual accuracy and completeness",
            "",
            "## Comprehensive Answer:"
        ])
        
        return "\n".join(prompt_parts)
    
    def _calculate_synthesis_confidence(self, evidence: List[SourceEvidence], coverage: Dict[str, float]) -> float:
        """Calculate overall synthesis confidence based on evidence quality and coverage."""
        if not evidence:
            return 0.0
        
        # Weight evidence
        evidence_confidence = sum(e.confidence for e in evidence) / len(evidence)
        coverage_score = sum(coverage.values()) / len(coverage)
        
        # Coverage bonus
        phase_diversity = len(set(e.phase for e in evidence)) / 3.0  # 3 phases max
        
        # Combined score
        synthesis_confidence = (evidence_confidence * 0.5) + (coverage_score * 0.3) + (phase_diversity * 0.2)
        
        return min(synthesis_confidence, 1.0)
    
    def _format_final_answer(self, answer: str, evidence: List[SourceEvidence], confidence: float) -> str:
        """Format the final answer with proper structure and attribution."""
        formatted_parts = [
            "# Comprehensive Answer",
            "",
            answer,
            "",
            "---",
            "",
            f"**Answer Confidence**: {confidence:.1%}",
            f"**Sources Integrated**: {len(evidence)} evidence sources",
            f"**Multi-Phase Coverage**: {len(set(e.phase for e in evidence))} phases (C: Community, D: Entity, E: Vector)",
            ""
        ]
        
        return "\n".join(formatted_parts)
    
    def _create_fallback_answer(self, evidence: List[SourceEvidence]) -> str:
        """Create fallback answer when LLM synthesis fails."""
        if not evidence:
            return "Unable to generate answer due to insufficient evidence."
        
        # Simple concatenation of top evidence
        fallback_parts = [
            "# Answer Summary",
            "",
            "Based on available evidence:",
            ""
        ]
        
        for i, source in enumerate(evidence[:3], 1):
            fallback_parts.extend([
                f"## Source {i} (Confidence: {source.confidence:.2f})",
                source.content[:300] + ("..." if len(source.content) > 300 else ""),
                ""
            ])
        
        return "\n".join(fallback_parts)
    
    def _create_fallback_synthesis(self, community_results: Dict, follow_up_results: Dict, 
                                 execution_time: float, error: str) -> SynthesisResult:
        """Create fallback synthesis result when phase fails."""
        return SynthesisResult(
            final_answer=" Response failed due to technical error. Please try again.",
            confidence_score=0.0,
            source_evidence=[],
            synthesis_strategy='fallback',
            coverage_assessment={'overall_confidence': 0.0},
            execution_time=execution_time,
            metadata={'error': error, 'fallback': True}
        )
    
    def combine_phase_results(self, 
                            phase_c_answer: str, 
                            follow_up_results: Dict[str, Any], 
                            augmentation_results=None) -> str:
        """
        Combine Phase C, D, and E results into enhanced answer.
        
        Creates comprehensive response by integrating results from multiple phases.
        """
        try:
            intermediate_answers = follow_up_results.get('intermediate_answers', [])
            
            if not intermediate_answers:
                return phase_c_answer
            
            # Start with Phase C answer
            enhanced_parts = [
                "## Global Context (Phase C)",
                phase_c_answer.strip(),
                "",
                "## Detailed Information (Phase D)"
            ]
            
            # Add intermediate answers from Phase D
            for i, answer in enumerate(intermediate_answers, 1):
                enhanced_parts.extend([
                    f"**{i}. {answer.question}**",
                    answer.answer,
                    f"*Confidence: {answer.confidence:.2f}*",
                    ""
                ])
            
            # Add Phase E vector augmentation if available
            if augmentation_results and hasattr(augmentation_results, 'vector_results') and augmentation_results.vector_results:
                enhanced_parts.extend([
                    "## Vector Augmentation (Phase E)",
                    f"**Semantic Enhancement** (Confidence: {augmentation_results.augmentation_confidence:.2f})",
                    ""
                ])
                
                # Add top vector results
                for i, vector_result in enumerate(augmentation_results.vector_results[:3], 1):
                    enhanced_parts.extend([
                        f"**Vector Result {i}** (Similarity: {vector_result.similarity_score:.3f})",
                        vector_result.content,  # Show full content without truncation
                        ""
                    ])
            
            # Add supporting evidence if available
            if intermediate_answers:
                enhanced_parts.extend([
                    "## Supporting Evidence",
                    "**Key Entities Found:** " + ", ".join(
                        set(entity for answer in intermediate_answers 
                            for entity in answer.supporting_entities[:3])
                    ),
                    ""
                ])
            
            return "\n".join(enhanced_parts)
            
        except Exception as e:
            self.logger.error(f"Failed to combine phase results: {e}")
            return phase_c_answer
    
    def generate_error_response(self, error_message: str) -> Dict[str, Any]:
        """
        Generate standardized error response.
        
        Creates consistent error format for failed synthesis operations.
        """
        return {
            "answer": f"Sorry, I encountered an error during answer synthesis: {error_message}",
            "metadata": {
                "status": "synthesis_error",
                "error_message": error_message,
                "synthesis_stage": "failed",
                "confidence_score": 0.0,
                "timestamp": datetime.now().isoformat()
            }
        }


# Exports
__all__ = ['AnswerSynthesisEngine', 'SynthesisResult', 'SourceEvidence']