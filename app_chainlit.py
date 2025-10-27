import chainlit as cl
import os
import logging
from dotenv import load_dotenv
import time
import asyncio
import re
import logging

# Import llama-index and related libraries
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.litellm import LiteLLM
from my_config import MY_CONFIG
import common.query_utils as query_utils

# Global variables for LLM and index
vector_index = None
initialization_complete = False

logging.basicConfig(level=logging.WARNING, 
                    # format='%(asctime)s - %(levelname)s - %(message)s',
                    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
                    force=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# Load environment variables from .env file
load_dotenv()

def initialize():
    """
    Initialize LLM and Milvus vector database using llama-index.
    This function sets up the necessary components for the chat application.
    """
    global vector_index, initialization_complete
    
    if initialization_complete:
        return
    
    logger.info("Initializing LLM and vector database...")
    
    # raise Exception ("init exception test") # debug
    
    try:
        ## embedding model
        Settings.embed_model = HuggingFaceEmbedding(
            model_name = MY_CONFIG.EMBEDDING_MODEL
        )
        logger.info(f"✅ Using embedding model: {MY_CONFIG.EMBEDDING_MODEL}")

        # Setup LLM
        logger.info(f"✅ Using LLM model : {MY_CONFIG.LLM_MODEL}")
        Settings.llm = LiteLLM(
            model=MY_CONFIG.LLM_MODEL,
            )
        
        # Initialize Milvus vector store
        vector_store = MilvusVectorStore(
            uri = MY_CONFIG.DB_URI ,
            dim = MY_CONFIG.EMBEDDING_LENGTH , 
            collection_name = MY_CONFIG.COLLECTION_NAME,
            overwrite=False  # so we load the index from db
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        logger.info  (f"✅ Connected to Milvus instance: {MY_CONFIG.DB_URI}")
        
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, storage_context=storage_context)
        logger.info  (f"✅ Loaded index from vector db: {MY_CONFIG.DB_URI}")

        logger.info("Successfully initialized LLM and vector database")

        initialization_complete = True
    except Exception as e:
        initialization_complete = False
        logger.error(f"Error initializing LLM and vector database: {str(e)}")
        raise (e)
        # return False
## -------------

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
    Process the user message and get a response from the LLM.
    Uses the initialized index for semantic search and LLM for response generation.
    Shows RAG process using Chainlit Steps.
    
    Returns tuple of (response_text, elapsed_time)
    """
    global vector_index, initialization_complete
    
    # Check if LLM and index are initialized
    if vector_index is None or  initialization_complete is None:
        return "System did not initialize. Please try again later.", 0
    
    start_time = time.time()
    response_text = ''
    
    try:
        # Step 1: Query preprocessing
        async with cl.Step(name="Query Preprocessing", type="tool") as step:
            logger.info("Start query preprocessing step...")
            step.input = message
            
            # Create a query engine from the index
            query_engine = vector_index.as_query_engine()
            
            # Preprocess the query
            original_message = message
            message = query_utils.tweak_query(message, MY_CONFIG.LLM_MODEL)
            
            step.output = f"Optimized query: {message}"
        ## --- end: Step 1 ---
        
        # Query the index
        logger.info("Calling LLM ...")
        t1 = time.time()
        response = query_engine.query(message)
        t2 = time.time()
        if response:
            response_text = str(response).strip()
        else:
            response_text = "No response from LLM."
        logger.info(f"LLM response received in {(t2 - t1):.2f} seconds:\n{response_text[:200]}")

        # Step 2: Vector search and retrieval
        async with cl.Step(name="Document Retrieval", type="retrieval") as step:
            step.input = message

            # Show retrieved documents
            if hasattr(response, 'source_nodes') and response.source_nodes:
                sources_output = []
                for i, node in enumerate(response.source_nodes[:3]):  # Show top 3 sources
                    score = node.score if hasattr(node, 'score') else 'N/A'
                    text_preview = node.text[:200] + "..." if len(node.text) > 200 else node.text
                    sources_output.append(f"Source {i+1} (Score: {score}): {text_preview}")
                step.output = "\n\n".join(sources_output)
            else:
                step.output = "No relevant documents found."
        
        
        # Extract thinking section if present
        thinking_content, cleaned_response = extract_thinking_section(response_text)
        # print (f"------ Thinking Content:-----\n{thinking_content}\n------")  # Debug print
        # print (f"------ Cleaned Response:-----\n{cleaned_response}\n------")  # Debug print
        
        # Step 3: Optional Thinking Process
        if thinking_content:
            async with cl.Step(name="💭 Thinking Process", type="run") as step:
                step.input = ""
                step.output = thinking_content
                logger.info(f"Thinking:\n{thinking_content[:200]}...")

        # Step 4: LLM Answer
        async with cl.Step(name="Response", type="llm") as step:
            step.input = f"Query: {message}\nContext: Retrieved from vector database"
            
            if cleaned_response:
                step.output = cleaned_response
                logger.info(f"Response:\n{cleaned_response[:200]}...")
            else:
                step.output = "No response from LLM."
                logger.info(f"Response:\nNo response from LLM.")

    except Exception as e:
        logger.error(f"Error getting LLM response: {str(e)}")
        response_text =  f"Sorry, I encountered an error while processing your request:\n{str(e)}"
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return response_text, elapsed_time
    
## --- end: def get_llm_response():

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
    