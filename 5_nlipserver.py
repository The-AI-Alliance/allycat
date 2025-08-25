import os
from my_config import MY_CONFIG

# If connection to https://huggingface.co/ failed, uncomment the following path
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import VectorStoreIndex
from dotenv import load_dotenv
from llama_index.llms.litellm import LiteLLM
import query_utils
import time
import logging
import json

## NLIP
from nlip_sdk.nlip import NLIP_Factory
from nlip_server import server
from nlip_sdk import nlip
import uvicorn

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run_query(query: str):
    global query_engine
    logger.info (f"-----------------------------------")
    start_time = time.time()
    query = query_utils.tweak_query(query, MY_CONFIG.LLM_MODEL)
    logger.info (f"\nProcessing Query:\n{query}")
    res = query_engine.query(query)
    end_time = time.time()
    logger.info ( "-------"
                 + f"\nResponse:\n{res}" 
                 + f"\n\nTime taken: {(end_time - start_time):.1f} secs"
                 + f"\n\nResponse Metadata:\n{json.dumps(res.metadata, indent=2)}" 
                #  + f"\nSource Nodes: {[node.node_id for node in res.source_nodes]}"
                 )
    logger.info (f"-----------------------------------")
## ======= end : run_query =======

## load env config
load_dotenv()

# Setup embeddings
Settings.embed_model = HuggingFaceEmbedding(
    model_name = MY_CONFIG.EMBEDDING_MODEL
)
logger.info (f"✅ Using embedding model: {MY_CONFIG.EMBEDDING_MODEL}")

# Connect to vector db
vector_store = MilvusVectorStore(
    uri = MY_CONFIG.DB_URI,
    dim = MY_CONFIG.EMBEDDING_LENGTH,
    collection_name = MY_CONFIG.COLLECTION_NAME, 
    overwrite=False  # so we load the index from db
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
logger.info (f"✅ Connected to Milvus instance: {MY_CONFIG.DB_URI}")

# Load Document Index from DB

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store, storage_context=storage_context)
logger.info (f"✅ Loaded index from vector db: {MY_CONFIG.DB_URI}")

NLIP = """
You are an NLIP Agent.  Your NAME is McLarenLabs.

NLIP is an acronym for "Natural Language Interaction Protocol."  The NLIP project aims to define a protocol for the following use cases.
- Agent to Agent interactions in Natual Language
- User-Agent to Agent protocol

An NLIP Agent, when defined, is given a system instruction that describes its unique capabilities.

One of the first requests an NLIP Agent will be asked to fulfill is to describe its NLIP Capabilities.
When you are asked to describe your NLIP Capabilities, you must respond with a response of the format:
    [NAME]
    CAPABILITY1:description, CAPABILITY2:description, CAPABILITY3:description, ...
- where NAME is your name
- CAPABILITITY1, CAPABILITY2 and CAPABILITY3 are dictionary keys.  The description associated with each should be unique.
- the square brackets surrounding your NAME are important
- it is important that the capabilities are described in a means that another LLM can understand them

"""

NLIP = """
You are an NLIP Agent.  Your NAME is McLarenLabs.

NLIP is an acronym for "Natural Language Interaction Protocol."  The NLIP project aims to define a protocol for the following use cases.
- Agent to Agent interactions in Natural Language
- User-Agent to Agent protocol

An NLIP Agent, when defined, is given a system instruction that describes its unique capabilities.

One of the first requests an NLIP Agent will be asked to fulfill is to describe its NLIP Capabilities.

When you are asked to describe your NLIP Capabilities, you must respond with a response of the format:
    AGENT:NAME
    CAPABILITY1:description, CAPABILITY2:description, CAPABILITY3:description, ...
- where NAME is your name
- CAPABILITITY1, CAPABILITY2 and CAPABILITY3 are dictionary keys.  The description associated with each should be unique.
- it is important that you include your NAME
- it is important that the capabilities are described in a means that another LLM can understand them

"""

# Setup LLM
logger.info (f"✅ Using LLM model : {MY_CONFIG.LLM_MODEL}")
Settings.llm = LiteLLM (
        model=MY_CONFIG.LLM_MODEL,
        system_prompt=NLIP
    )
 
query_engine = index.as_query_engine()

#
# NLIP Server Stuff
#

class ChatApplication(server.NLIP_Application):
    async def startup(self):
        print("Starting app...")

    async def shutdown(self):
        return None

    async def create_session(self) -> server.NLIP_Session:
        return ChatSession(query_engine)


class ChatSession(server.NLIP_Session):

    def __init__(self, query_engine):
        self.query_engine = query_engine

    async def start(self):
        print("Starting the chat session")

    async def execute(
        self, msg: nlip.NLIP_Message
    ) -> nlip.NLIP_Message:
        text = msg.extract_text() # extract text part of NLIP message
        try:
            resp = self.query_engine.query(text)
            retstr = resp.response # extract the text part of the response
            retmsg = NLIP_Factory.create_text(retstr) # wrap in NLIP envelope
            return retmsg
        except Exception as e:
            print(f"Exception:{e}")
            return None

    async def stop(self):
        print("Closing the chat session")
        self.server = None


app = server.setup_server(ChatApplication())

# Run a Uvicorn server locally
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8028, log_level="info")

        
