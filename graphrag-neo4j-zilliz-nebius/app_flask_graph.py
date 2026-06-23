"""
GraphRAG Flask Web Application
"""

from flask import Flask, render_template, request, jsonify
import os
import logging
import time
import asyncio
import nest_asyncio
from query_graph_functions.setup import create_graphrag_setup
from query_graph_functions.query_preprocessing import create_query_preprocessor, preprocess_query_pipeline
from query_graph_functions.knowledge_retrieval import CommunitySearchEngine
from query_graph_functions.follow_up_search import FollowUpSearch
from query_graph_functions.vector_augmentation import VectorAugmentationEngine
from query_graph_functions.answer_synthesis import AnswerSynthesisEngine
from query_graph_functions.response_management import ResponseManager
from my_config import MY_CONFIG
import query_utils


nest_asyncio.apply()
os.environ['HF_ENDPOINT'] = MY_CONFIG.HF_ENDPOINT

app = Flask(__name__)

# Global GraphRAG engine
graph_engine = None
initialization_complete = False

def initialize():
    """
    Initialize GraphRAG system
    """
    global graph_engine, initialization_complete
    
    if initialization_complete:
        return
    
    logging.info("Initializing GraphRAG system...")
    
    try:
        # Initialize setup module (Step 1)
        setup = create_graphrag_setup()
        
        # Create GraphRAG engine wrapper
        class GraphQueryEngine:
            def __init__(self, setup):
                self.setup = setup
                self.neo4j_conn = setup.neo4j_conn
                self.query_engine = setup.query_engine
                self.graph_stats = setup.graph_stats
                self.drift_config = setup.drift_config
                self.llm = setup.llm
                self.config = setup.config
                self.query_preprocessor = None
                self.response_manager = ResponseManager(setup)
            
            async def run_query_async(self, user_query):
                """Execute GraphRAG query pipeline (Steps 3-20)"""
                start_time = time.time()
                
                optimized_query = query_utils.tweak_query(user_query, MY_CONFIG.LLM_MODEL)
                
                try:
                    if not self.setup.validate_system_readiness():
                        return {"answer": "System not ready", "metadata": {}}
                    
                    # Initialize query preprocessor if needed
                    if not self.query_preprocessor:
                        self.query_preprocessor = await create_query_preprocessor(
                            self.config, self.graph_stats
                        )
                    
                    # Phase B: Query Preprocessing (Steps 3-5)
                    analysis, routing, vectorization = await preprocess_query_pipeline(
                        optimized_query, self.config, self.graph_stats
                    )
                    
                    # Phase C: Community Retrieval (Steps 6-7)
                    community_engine = CommunitySearchEngine(self.setup)
                    community_results = await community_engine.execute_primer_phase(
                        vectorization.embedding, routing
                    )
                    
                    # Phase D: Follow-up Search (Steps 9-12)
                    follow_up_engine = FollowUpSearch(self.setup)
                    follow_up_results = await follow_up_engine.execute_follow_up_phase(
                        community_results, routing
                    )
                    
                    # Phase E: Vector Search Augmentation (Steps 13-14)
                    vector_engine = VectorAugmentationEngine(self.setup)
                    augmentation_results = await vector_engine.execute_vector_augmentation_phase(
                        vectorization.embedding,
                        {'communities': community_results['communities'], 
                         'initial_answer': community_results['initial_answer'], 
                         'follow_up_results': follow_up_results},
                        routing
                    )
                    
                    # Phase F: Answer Synthesis (Steps 15-16)
                    synthesis_engine = AnswerSynthesisEngine(self.setup)
                    synthesis_results = await synthesis_engine.execute_answer_synthesis_phase(
                        analysis, routing, community_results, follow_up_results, augmentation_results
                    )
                    
                    total_time = time.time() - start_time
                    
                    # Generate metadata
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
                    
                    # Cleanup async tasks
                    await self.setup.cleanup_async_tasks(timeout=2.0)
                    
                    return {
                        "answer": synthesis_results.final_answer,
                        "metadata": metadata
                    }
                    
                except Exception as e:
                    logging.error(f"Query pipeline error: {e}")
                    synthesis_engine = AnswerSynthesisEngine(self.setup)
                    return synthesis_engine.generate_error_response(f"Query error: {e}")
            
            def run_query(self, user_query):
                """Synchronous wrapper for async query"""
                try:
                    loop = asyncio.get_event_loop()
                    return loop.run_until_complete(self.run_query_async(user_query))
                except Exception as e:
                    logging.error(f"Query execution error: {e}")
                    return {"answer": f"Error: {e}", "metadata": {}}
        
        graph_engine = GraphQueryEngine(setup)
        
        print("✅ GraphRAG system initialized")
        print(f"✅ Using LLM: {MY_CONFIG.LLM_MODEL}")
        print(f"✅ Using embedding: {MY_CONFIG.EMBEDDING_MODEL}")
        
        logging.info("GraphRAG system ready")
        initialization_complete = True
        
    except Exception as e:
        initialization_complete = False
        logging.error(f"GraphRAG initialization error: {str(e)}")
        raise

## ----
@app.route('/')
def index():
    init_error = app.config.get('INIT_ERROR', '')
    # init_error = g.get('init_error', None)
    return render_template('index.html', init_error=init_error)
## end --- def index():


## ----
@app.route('/health')
def health():
    """Health check endpoint for deployment platforms"""
    if initialization_complete:
        return jsonify({"status": "healthy", "graphrag": "initialized"}), 200
    else:
        return jsonify({"status": "initializing"}), 503
## end --- def health():


## -----
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    
    
    # Get response from LLM
    response = get_llm_response(user_message)
    # print (response)
    
    return jsonify({'response': response})
## end : def chat():


def get_llm_response(message):
    """
    Process user message using complete GraphRAG pipeline.
    Implements the full 25-step DRIFT search methodology.
    """
    global graph_engine, initialization_complete
    
    if not initialization_complete or graph_engine is None:
        return "System not initialized. Please try again later."
    
    start_time = time.time()
    
    try:
        # Execute GraphRAG query pipeline
        result = graph_engine.run_query(message)
        
        # Extract answer and timing
        full_response = result.get('answer', 'No response generated')
        
        # Filter out metadata section more robustly
        lines = full_response.split('\n')
        filtered_lines = []
        
        for line in lines:
            stripped = line.strip()
            # Skip these metadata lines completely
            if (stripped.startswith('## Comprehensive Answer') or 
                stripped.startswith('# Comprehensive Answer') or
                stripped.startswith('---') or
                stripped.startswith('**Answer Confidence**:') or
                stripped.startswith('**Sources Integrated**:') or
                stripped.startswith('**Multi-Phase Coverage**:')):
                continue
            
            filtered_lines.append(line)
        
        response_text = '\n'.join(filtered_lines).strip()
        end_time = time.time()
        
        # Add timing information
        response_text += f"\n\n⏱️ *Total time: {(end_time - start_time):.1f} seconds*"
        
        return response_text
        
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        return f"Sorry, I encountered an error:\n{str(e)}"


    

## -------
if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.info("App starting up...")
    
    # Initialize LLM and vector database
    try:
        initialize()
    except Exception as e:
        logging.warning("Starting without LLM and vector database. Responses will be limited.")
        app.config['INIT_ERROR'] = str(e)
        # g.init_error = str(e)
        
    
    # GraphRAG Flask App - Configurable port via environment
    PORT = MY_CONFIG.FLASK_GRAPH_PORT
    print(f"🚀 GraphRAG Flask app starting on port {PORT}")
    app.run(host="0.0.0.0", debug=False, port=PORT)