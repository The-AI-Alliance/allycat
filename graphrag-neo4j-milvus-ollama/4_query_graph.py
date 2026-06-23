"""
GraphRAG Query 
"""

import time
import logging
import json
import importlib
import sys
import os
import asyncio
from typing import Dict, Any

# Apply nest_asyncio to allow nested event loops
import nest_asyncio 
nest_asyncio.apply()

# Import Step 1 functionality from setup module
from query_graph_functions.setup import create_graphrag_setup
# Import Steps 3-5 functionality from query preprocessing module  
from query_graph_functions.query_preprocessing import (
    create_query_preprocessor,
    preprocess_query_pipeline
)
# Import Steps 6-8 functionality from knowledge retrieval module
from query_graph_functions.knowledge_retrieval import CommunitySearchEngine
# Import Steps 9-12 functionality from follow-up search module
from query_graph_functions.follow_up_search import FollowUpSearch
# Import Steps 13-14 functionality from vector augmentation module
from query_graph_functions.vector_augmentation import VectorAugmentationEngine
# Import Steps 15-16 functionality from answer synthesis module
from query_graph_functions.answer_synthesis import AnswerSynthesisEngine
# Import Steps 17-20 functionality from response management module
from query_graph_functions.response_management import ResponseManager


from my_config import MY_CONFIG
import query_utils

# Create logs directory if it doesn't exist
os.makedirs('logs/graphrag_query', exist_ok=True)

# Configure logging - Save to file and console
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    handlers=[
        logging.FileHandler('logs/graphrag_query/graphrag_query_log.txt', mode='a'),  # Save to file
        logging.StreamHandler()  # Also show in console
    ],
    force=True
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Log session start
logger.info("=" * 80)
logger.info(f"GraphRAG Session Started - {time.strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("=" * 80)


class GraphQueryEngine:
    def __init__(self):
        logger.info("GraphRAG Query Engine Initializing")
        
        # Initialize using setup module (Step 1)
        self.setup = create_graphrag_setup()
        
        # Extract components from setup
        self.neo4j_conn = self.setup.neo4j_conn
        self.query_engine = self.setup.query_engine
        self.graph_stats = self.setup.graph_stats
        self.drift_config = self.setup.drift_config
        self.llm = self.setup.llm
        self.config = self.setup.config
        
        # Initialize query preprocessor (Steps 3-5) - will be created async
        self.query_preprocessor = None
        
        # Initialize response manager (Steps 17-20)
        self.response_manager = ResponseManager(self.setup)
        
        logger.info("GraphRAG Query Engine Ready")

    async def run_query_async(self, user_query: str) -> Dict[str, Any]:
        """
        GraphRAG Query Pipeline - Main Entry Point (Async)
        
        Implements Phase B (Steps 3-5) of the 25-step DRIFT search methodology
        """
        logger.info("=" * 60)
        logger.info("GraphRAG Query Pipeline Starting")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Apply query optimization
        optimized_query = query_utils.tweak_query(user_query, MY_CONFIG.LLM_MODEL)
        logger.info(f"Original Query: {user_query}")
        if optimized_query != user_query:
            logger.info(f"Optimized Query: {optimized_query}")
        
        try:
            # Validate system readiness using setup module
            if not self.setup.validate_system_readiness():
                return self._generate_error_response("System not properly initialized")
            
            # PHASE B: QUERY PREPROCESSING (Steps 3-5)
            logger.info("Phase B: Starting Query Preprocessing (Steps 3-5)")
            
            # Initialize query preprocessor if needed
            if not self.query_preprocessor:
                self.query_preprocessor = await create_query_preprocessor(
                    self.config, self.graph_stats
                )
            
            # Execute complete preprocessing pipeline
            analysis, routing, vectorization = await preprocess_query_pipeline(
                optimized_query, self.config, self.graph_stats
            )
            
            logger.info(f"Phase B Completed: "
                       f"Type={analysis.query_type.value}, "
                       f"Strategy={routing.search_strategy.value}")
            
            # PHASE C: COMMUNITY RETRIEVAL (Steps 6-7)
            logger.info("Phase C: Starting Community Retrieval (Steps 6-7)")
            
            # Create community search engine
            community_engine = CommunitySearchEngine(self.setup)
            
            # Execute the primer phase (Steps 6-8)
            community_results = await community_engine.execute_primer_phase(
                vectorization.embedding, routing
            )
            
            # Extract communities for Phase D
            communities = community_results['communities']
            
            logger.info(f"Phase C Completed: Retrieved {len(communities)} communities")
            
            # PHASE D: FOLLOW-UP SEARCH (Steps 9-12)
            logger.info("Phase D: Starting Follow-up Search (Steps 9-12)")
            
            # Create follow-up search engine
            follow_up_engine = FollowUpSearch(self.setup)
            
            # Execute follow-up search phase
            follow_up_results = await follow_up_engine.execute_follow_up_phase(
                community_results, routing
            )
            
            logger.info(f"Phase D Completed: Generated {len(follow_up_results.get('intermediate_answers', []))} detailed answers")
            
            # PHASE E: VECTOR SEARCH AUGMENTATION (Steps 13-14)
            logger.info("Phase E: Starting Vector Search Augmentation (Steps 13-14)")
            
            # Create vector augmentation engine
            vector_engine = VectorAugmentationEngine(self.setup)
            
            # Execute vector augmentation phase
            augmentation_results = await vector_engine.execute_vector_augmentation_phase(
                vectorization.embedding, 
                {'communities': communities, 'initial_answer': community_results['initial_answer'], 'follow_up_results': follow_up_results},
                routing
            )
            
            logger.info(f"Phase E Completed: Vector augmentation confidence: {augmentation_results.augmentation_confidence:.3f}")
            
            # PHASE F: ANSWER SYNTHESIS (Steps 15-16)
            logger.info("Phase F: Starting Answer Synthesis (Steps 15-16)")
            
            # Create answer synthesis engine
            synthesis_engine = AnswerSynthesisEngine(self.setup)
            
            # Execute comprehensive answer synthesis
            synthesis_results = await synthesis_engine.execute_answer_synthesis_phase(
                analysis, routing, community_results, follow_up_results, augmentation_results
            )
            
            logger.info(f"Phase F Completed: Final synthesis confidence: {synthesis_results.confidence_score:.3f}")
            
            # PHASE G: RESPONSE MANAGEMENT (Steps 17-20)
            logger.info("Phase G: Starting Response Management (Steps 17-20)")
            
            # Enhanced implementation using preprocessing results
            if self.query_engine:
                # Use the vectorized query for better results  
                _ = self.query_engine.query(vectorization.normalized_query)
                total_time = time.time() - start_time
                
                logger.info(f"Enhanced Query Completed in {total_time:.2f}s")
                logger.info("=" * 60)
                
                # Use Phase F synthesis result as the final answer
                enhanced_answer = synthesis_results.final_answer
                
                # Generate comprehensive metadata using ResponseManager
                metadata = self.response_manager.generate_comprehensive_metadata(
                    analysis=analysis,
                    routing=routing, 
                    vectorization=vectorization,
                    community_results=community_results,
                    follow_up_results=follow_up_results,
                    augmentation_results=augmentation_results,
                    synthesis_results=synthesis_results,
                    total_time=total_time
                )
                
                result = {
                    "answer": enhanced_answer,
                    "metadata": metadata
                }
                
                # Save response and metadata to files using ResponseManager
                self.response_manager.save_response_to_files(user_query, result)
                
                logger.info("Phase G Completed: Response management finished")
                
                return result
            else:
                return synthesis_engine.generate_error_response("Query engine not available")
                
        except Exception as e:
            logger.error(f"Query Pipeline Failed: {e}")
            synthesis_engine = AnswerSynthesisEngine(self.setup)
            return synthesis_engine.generate_error_response(f"Query processing error: {e}")
    
    def run_query(self, user_query: str) -> Dict[str, Any]:
        """
        Synchronous wrapper for async query processing.
        
        This maintains backward compatibility while using the new async pipeline.
        Uses nest_asyncio and our LiteLLM patch to properly handle async tasks.
        """
        try:
            # Use the current event loop since nest_asyncio.apply() has been called
            loop = asyncio.get_event_loop()
            
            # Create a future to gather all tasks and wait for completion
            async def run_with_cleanup():
                try:
                    # Run the main query
                    result = await self.run_query_async(user_query)
                    
                    # Use setup module's cleanup function
                    await self.setup.cleanup_async_tasks(timeout=2.0)
                    
                    return result
                except Exception as e:
                    logger.error(f"Async Query Execution Failed: {e}")
                    raise e
                
            # Run the async function with cleanup
            return loop.run_until_complete(run_with_cleanup())
            
        except Exception as e:
            logger.error(f"Sync Query Wrapper Failed: {e}")
            # Use synthesis engine for error handling
            synthesis_engine = AnswerSynthesisEngine(self.setup)
            return synthesis_engine.generate_error_response(f"Query processing error: {e}")

    def close(self):
        """Clean up connections using setup module"""
        if self.setup:
            self.setup.close()
        logger.info("GraphQueryEngine cleanup complete")


if __name__ == "__main__":
    print("GraphRAG Implementation - Reload Enabled")
    print("=" * 50)
    print("Step 1: Initialization and Connection")
    print("Hot Reload: Type 'r' to reload modules")
    print("=" * 50)
    
    engine = GraphQueryEngine()
    
    try:
        # Create an event loop for the main thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while True:
            user_query = input("\nEnter your question ('q' to exit, 'r' to reload): ")
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if user_query.lower() == 'r':
                print("Reloading...")
                engine.close()
                
                # Run cleanup tasks before reloading using setup module
                loop.run_until_complete(
                    engine.setup.cleanup_async_tasks(timeout=3.0) if engine.setup else None
                )
                
                engine = GraphQueryEngine()
                print("Reloaded!")
                continue
            
            if user_query.strip() == "":
                continue
            
            # Direct method call - clean forward-only implementation
            result = engine.run_query(user_query)
            
            # Print results
            print("\n" + "=" * 60)
            print("GraphRAG Query Results")
            print("=" * 60)
            print(f"Answer: {result['answer']}")
            print(f"\nMetadata: {json.dumps(result['metadata'], indent=2)}")
            print("=" * 60)
            
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        print(f"Error processing query: {e}")
    finally:
        # Run final cleanup before exiting using setup module
        if 'loop' in locals() and 'engine' in locals():
            loop.run_until_complete(
                engine.setup.cleanup_async_tasks(timeout=5.0) if engine.setup else None
            )
            loop.close()
        if 'engine' in locals():
            engine.close()