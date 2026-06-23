"""
GraphRAG Chainlit Application
"""

import chainlit as cl
import os
import logging
from dotenv import load_dotenv
import time
import asyncio
import re
from typing import Dict, Any, Tuple

# Apply nest_asyncio to allow nested event loops
import nest_asyncio
nest_asyncio.apply()

# Import Step 1 functionality from setup module
from query_graph_functions.setup import create_graphrag_setup
# Import Steps 3-5 functionality from query preprocessing module
from query_graph_functions.query_preprocessing import create_query_preprocessor, preprocess_query_pipeline
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

# Configure environment
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Load environment variables
load_dotenv()

# Create logs directory if it doesn't exist
os.makedirs('logs/chainlit', exist_ok=True)

# Configure logging - Save to file and console
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('logs/chainlit/chainlit_graph.log', mode='a'),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Log session start
logger.info("=" * 80)
logger.info(f"Chainlit GraphRAG Session Started - {time.strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("=" * 80)

# Global GraphRAG engine instance
graph_engine = None
initialization_complete = False

def initialize():
    global graph_engine, initialization_complete
    
    if initialization_complete:
        return
    
    logger.info("Initializing GraphRAG system...")
    
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
                        "metadata": metadata,
                        "analysis": analysis,
                        "routing": routing,
                        "community_results": community_results,
                        "follow_up_results": follow_up_results,
                        "augmentation_results": augmentation_results
                    }
                    
                except Exception as e:
                    logger.error(f"Query pipeline error: {e}")
                    synthesis_engine = AnswerSynthesisEngine(self.setup)
                    return synthesis_engine.generate_error_response(f"Query error: {e}")
        
        graph_engine = GraphQueryEngine(setup)
        
        logger.info("✅ GraphRAG system initialized")
        logger.info(f"✅ Using LLM: {MY_CONFIG.LLM_MODEL}")
        logger.info(f"✅ Using embedding: {MY_CONFIG.EMBEDDING_MODEL}")
        
        initialization_complete = True
        
    except Exception as e:
        initialization_complete = False
        logger.error(f"GraphRAG initialization error: {str(e)}")
        raise

def extract_thinking_section(response_text):
    """
    Extract thinking section from LLM response if present.
    
    Args:
        response_text (str): The full response from the LLM
        
    Returns:
        tuple: (thinking_content, cleaned_response)
            - thinking_content: Content within <think></think> tags or None if not found
            - cleaned_response: Response with thinking section removed
    """
    thinking_pattern = r'<think>(.*?)</think>'
    match = re.search(thinking_pattern, response_text, re.DOTALL)
    
    if match:
        thinking_content = match.group(1).strip()
        cleaned_response = re.sub(thinking_pattern, '', response_text, flags=re.DOTALL).strip()
        return thinking_content, cleaned_response
    else:
        return None, response_text

async def get_llm_response(message):
    """
    Process user message
    """
    global graph_engine, initialization_complete
    
    if not initialization_complete or graph_engine is None:
        return "System not initialized. Please try again later.", 0
    
    start_time = time.time()
    
    try:
        # Step 1: Query Preprocessing
        async with cl.Step(name="Query Analysis", type="tool") as step:
            step.input = message
            optimized_query = query_utils.tweak_query(message, MY_CONFIG.LLM_MODEL)
            step.output = f"Optimized query: {optimized_query}"
        
        # Execute GraphRAG query pipeline
        result = await graph_engine.run_query_async(message)
        
        # Step 2: Community Search
        if 'community_results' in result:
            async with cl.Step(name="Community Retrieval", type="retrieval") as step:
                communities = result['community_results'].get('communities', [])
                step.input = "Searching graph communities"
                step.output = f"Found {len(communities)} relevant communities"
        
        # Step 3: Follow-up Search
        if 'follow_up_results' in result:
            async with cl.Step(name="Entity Search", type="retrieval") as step:
                step.input = "Analyzing entity relationships"
                follow_up = result['follow_up_results']
    
                entities_found = len(follow_up.get('detailed_entities', []))
                relationships_found = sum(~
                    len(search.traversed_relationships) 
                    for search in follow_up.get('local_search_results', [])
                )
                step.output = f"Entities: {entities_found}, Relationships: {relationships_found}"
        
        # Step 4: Vector Augmentation
        if 'augmentation_results' in result:
            async with cl.Step(name="Document Augmentation", type="retrieval") as step:
                step.input = "Enriching with vector search"
                aug_results = result['augmentation_results']
                
                if hasattr(aug_results, 'vector_results'):
                    vector_results = aug_results.vector_results
                    step.output = f"Retrieved {len(vector_results)} relevant documents"
                else:
                    step.output = "Vector augmentation completed"
        
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
        
        # Extract thinking section if present
        thinking_content, cleaned_response = extract_thinking_section(response_text)
        
        # Step 5: Optional Thinking Process
        if thinking_content:
            async with cl.Step(name="Reasoning Process", type="run") as step:
                step.input = ""
                step.output = thinking_content
                logger.info(f"Thinking:\n{thinking_content[:200]}...")
        
        # Step 6: Final Answer
        async with cl.Step(name="Synthesis", type="llm") as step:
            step.input = "Generating comprehensive answer"
            step.output = cleaned_response if cleaned_response else response_text
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        return cleaned_response if cleaned_response else response_text, elapsed_time
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return f"Sorry, I encountered an error:\n{str(e)}", 0

# ====== CHAINLIT SPECIFIC CODE ======

@cl.set_starters
async def set_starters():
    starters = []
    for prompt in MY_CONFIG.STARTER_PROMPTS:
        starters.append(
            cl.Starter(
                label=prompt.strip(),
                message=prompt.strip(),
            )
        )
    return starters
## --- end: def set_starters(): ---

@cl.on_chat_start
async def start():
    """Initialize the chat session"""
    # Store initialization state in user session
    cl.user_session.set("chat_started", True)
    logger.info("User chat session started")
    init_error = None
    
    try:
        initialize()
        # await cl.Message(content="How can I assist you today?").send()
    except Exception as e:
        init_error = str(e)
        error_msg = f"""System Initialization Error

The system failed to initialize with the following error:

```
{init_error}
```

Please check your configuration and environment variables."""
        await cl.Message(content=error_msg).send()

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    user_message = message.content
    
    # Get response from LLM with RAG steps shown FIRST
    response_text, elapsed_time = await get_llm_response(user_message)
    # logger.info(f"LLM Response:\n{response_text[:200]}...")  # Log first 200 chars

    thinking_content, cleaned_response = extract_thinking_section(response_text)
    
    # Add timing stat to response
    full_response = cleaned_response + f"\n\n⏱️ *Total time: {elapsed_time:.1f} seconds*"
    
    # THEN create a new message for streaming
    msg = cl.Message(content="")
    await msg.send()
    
    # Stream the response character by character for better UX
    # This simulates streaming - in a real implementation you'd stream from the LLM
    for i in range(0, len(full_response), 5):  # Stream in chunks of 5 characters
        await msg.stream_token(full_response[i:i+5])
        await asyncio.sleep(0.01)  # Small delay for visual effect
    
    # Update the final message
    msg.content = full_response
    await msg.update()

## -------
if __name__ == '__main__':
    logger.info("App starting up...")
