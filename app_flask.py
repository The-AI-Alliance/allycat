from flask import Flask, g, render_template, request, jsonify
import os
import logging
from dotenv import load_dotenv
import time

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Load environment variables from .env file
load_dotenv()

# Import llama-index and related libraries
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.litellm import LiteLLM
from my_config import MY_CONFIG
import query_utils

app = Flask(__name__)

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
        
        # Setup LLM using LiteLLM
        llm = LiteLLM(
            model=MY_CONFIG.LLM_MODEL,
            temperature=0.1
        )
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

## ----
@app.route('/')
def index():
    init_error = app.config.get('INIT_ERROR', '')
    # init_error = g.get('init_error', None)
    return render_template('index.html', init_error=init_error)
## end --- def index():


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
    Process the user message and get a response from the LLM.
    Uses the initialized index for semantic search and LLM for response generation.
    """
    global vector_index, initialization_complete
    
    # Check if LLM and index are initialized
    if vector_index is None or  initialization_complete is None:
        return "System did not initialize. Please try again later."
    
    start_time = time.time()
    response_text = ''
    
    try:
        # raise Exception ("chat exception test") ## debug
        # Create a query engine from the index
        query_engine = vector_index.as_query_engine()
        
        # Query the index
        message = query_utils.tweak_query(message, MY_CONFIG.LLM_MODEL)
        response = query_engine.query(message)
        
        if response:
            response_text = str(response).strip()
        
    except Exception as e:
        logging.error(f"Error getting LLM response: {str(e)}")
        response_text =  f"Sorry, I encountered an error while processing your request:\n{str(e)}"
        
    end_time = time.time()
    
    # add timing stat
    response_text += f"\n(time taken: {(end_time - start_time):.1f} secs)"
    return response_text
    
## --- end: def get_llm_response():


    

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
        
    
    # app.run(debug=False)
    PORT = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", debug=False, port=PORT)
## -- end main ----