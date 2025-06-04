import chainlit as cl
import os
import logging
from dotenv import load_dotenv
import time
import asyncio

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Load environment variables from .env file
load_dotenv()

# Import llama-index and related libraries
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.replicate import Replicate
from llama_index.llms.ollama import Ollama
from my_config import MY_CONFIG
import query_utils

# Global variables for LLM and index
vector_index = None
initialization_complete = False

def initialize():
    """
    Initialize LLM and Milvus vector database using llama-index.
    This function sets up the necessary components for the chat application.
    """
    global vector_index, initialization_complete
    
    if initialization_complete:
        return
    
    logging.info("Initializing LLM and vector database...")
    
    # raise Exception ("init exception test") # debug
    
    try:
        ## embedding model
        Settings.embed_model = HuggingFaceEmbedding(
            model_name = MY_CONFIG.EMBEDDING_MODEL
        )
        print("✅ Using embedding model: ", MY_CONFIG.EMBEDDING_MODEL)
        
        # Setup LLM
        if MY_CONFIG.LLM_RUN_ENV == 'replicate':
            llm = Replicate(
                model=MY_CONFIG.LLM_MODEL,
                temperature=0.1
            )
        elif MY_CONFIG.LLM_RUN_ENV == 'local_ollama':
            llm = Ollama(
                model= MY_CONFIG.LLM_MODEL,
                request_timeout=30.0,
                temperature=0.1
            )
        else:
            raise ValueError("❌ Invalid LLM run environment. Please set it to 'replicate' or 'local_ollama'.")
        print("✅ LLM run environment: ", MY_CONFIG.LLM_RUN_ENV)    
        print("✅ Using LLM model : ", MY_CONFIG.LLM_MODEL)
        Settings.llm = llm
        
        # Initialize Milvus vector store
        vector_store = MilvusVectorStore(
            uri = MY_CONFIG.DB_URI ,
            dim = MY_CONFIG.EMBEDDING_LENGTH , 
            collection_name = MY_CONFIG.COLLECTION_NAME,
            overwrite=False  # so we load the index from db
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        print ("✅ Connected to Milvus instance: ", MY_CONFIG.DB_URI )
        
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, storage_context=storage_context)
        print ("✅ Loaded index from vector db:", MY_CONFIG.DB_URI )

        logging.info("Successfully initialized LLM and vector database")
    
        initialization_complete = True
    except Exception as e:
        initialization_complete = False
        logging.error(f"Error initializing LLM and vector database: {str(e)}")
        raise (e)
        # return False
## -------------

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
            logging.info("Start query preprocessing step...")
            step.input = message
            
            # Create a query engine from the index
            query_engine = vector_index.as_query_engine()
            
            # Preprocess the query
            original_message = message
            message = query_utils.tweak_query(message, MY_CONFIG.LLM_MODEL)
            
            step.output = f"Original: {original_message}\nOptimized: {message}"
        
        # Step 2: Vector search and retrieval
        async with cl.Step(name="Document Retrieval", type="retrieval") as step:
            logging.info("Start vector search and retrieval step...")
            step.input = message
            
            # Query the index
            response = query_engine.query(message)
            
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
        
        # Step 3: LLM generation
        async with cl.Step(name="Response Generation", type="llm") as step:
            logging.info("Start LLM response generation step...")
            step.input = f"Query: {message}\nContext: Retrieved from vector database"
            
            if response:
                response_text = str(response).strip()
                step.output = response_text[:500] + "..." if len(response_text) > 500 else response_text
            else:
                step.output = "No response generated."
        
    except Exception as e:
        logging.error(f"Error getting LLM response: {str(e)}")
        response_text =  f"Sorry, I encountered an error while processing your request:\n{str(e)}"
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return response_text, elapsed_time
    
## --- end: def get_llm_response():

# ====== CHAINLIT SPECIFIC CODE ======

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="What is the AI Alliance?",
            message="What is the AI Alliance?",
            ),
        cl.Starter(
            label="What are the main focus areas of AI Alliance?",
            message="What are the main focus areas of AI Alliance?",
            ),
        cl.Starter(
            label="What are some AI Alliance projects?",
            message="What are some AI Alliance projects?",
            ),
        cl.Starter(
            label="What are some upcoming AI Alliance events?",
            message="What are some upcoming AI Alliance events?",
            ),
        cl.Starter(
            label="How do I join the AI Alliance?",
            message="How do I join the AI Alliance?",
            )]

@cl.on_chat_start
async def start():
    """Initialize the chat session"""
    # Store initialization state in user session
    cl.user_session.set("chat_started", True)
    logging.info("User chat session started")
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
    logging.info(f"LLM Response: {response_text[:100]}...")  # Log first 100 chars
    
    # Add timing stat to response
    full_response = response_text + f"\n\n⏱️ *Total time: {elapsed_time:.1f} seconds*"
    
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
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.info("App starting up...")
    