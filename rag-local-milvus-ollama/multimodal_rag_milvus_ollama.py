"""
multimodal_rag_milvus_ollama.py

End-to-end multimodal RAG:
- Uses Replicate granite-vision-3.3-2b to analyze an image -> text.
- Embeds that text with sentence-transformers.
- Stores embeddings in Milvus.
- Searches Milvus for similar items at query time.
- Uses LlamaIndex to wrap retrieved docs and sends them to Ollama gemma3:1b for final generation.
"""

import os
import time
import json
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()
import replicate
import httpx
from PIL import Image

import numpy as np
from sentence_transformers import SentenceTransformer

from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, Collection, utility
)

from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.llms import LLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import ollama

# -----------------------
# Configuration (edit)
# -----------------------
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN") # must be set
if not REPLICATE_API_TOKEN:
    raise RuntimeError("Set REPLICATE_API_TOKEN environment variable before running.")

MILVUS_HOST = os.environ.get("MILVUS_HOST", "localhost")
MILVUS_PORT = os.environ.get("MILVUS_PORT", "19530")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:1b")

COLLECTION_NAME = "vision_rag_collection"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # sentence-transformers
EMBED_DIM = 384  # all-MiniLM-L6-v2 default dim

IMAGE_PATH = "septoria-spot-tomato-plant.jpg"  # change to your image path

# -----------------------
# Initialize services
# -----------------------
# Replicate client (we'll use replicate.Client for model invocation)
rep = replicate.Client(api_token=REPLICATE_API_TOKEN)

# SentenceTransformer embedding model (text embeddings)
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# Connect to Milvus
connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

# Create collection if not exists
def ensure_milvus_collection(name: str, dim: int = EMBED_DIM):
    if utility.has_collection(name):
        col = Collection(name)
        # If existing collection uses different dim, raise error
        vector_field = None
        for f in col.schema.fields:
            if f.dtype == DataType.FLOAT_VECTOR:
                vector_field = f
                break
        if vector_field and vector_field.dtype == DataType.FLOAT_VECTOR:
            # can't easily check dim from FieldSchema object here portably; assume ok
            return col
    # define schema
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=2048),
    ]
    schema = CollectionSchema(fields, description="Vision RAG collection")
    col = Collection(name=name, schema=schema)
    # create index on embedding
    index_params = {"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}}
    col.create_index(field_name="embedding", index_params=index_params)
    # load collection for search
    col.load()
    return col

collection = ensure_milvus_collection(COLLECTION_NAME, EMBED_DIM)

# -----------------------
# Step 1: Run Granite Vision on image to get textual analysis
# -----------------------
def run_granite_vision(image_path: str, prompt_text: str) -> str:
    """
    Call replicate's ibm-granite/granite-vision-3.3-2b with image and prompt.
    Returns text output (string). If the model returns a list, join to text.
    """
    print("[granite] Running granite-vision on image...")
    with open(image_path, "rb") as f:
        # replicate.Client.run returns model output
        output = rep.run("ibm-granite/granite-vision-3.3-2b", input={"image": f, "prompt": prompt_text})
    # Output format varies; handle common cases:
    if isinstance(output, str):
        return output
    try:
        # If list of strings
        if isinstance(output, list):
            # join list parts
            text = " ".join([str(o) for o in output])
            return text
        # if dict-like
        if isinstance(output, dict):
            # try common keys
            for key in ("text", "output", "caption", "analysis", "description", "result"):
                if key in output:
                    val = output[key]
                    if isinstance(val, list):
                        return " ".join(map(str, val))
                    return str(val)
            # fallback to json
            return json.dumps(output)
    except Exception:
        pass
    # fallback to string conversion
    return str(output)

# -----------------------
# Step 2: Embed the text output
# -----------------------
def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Use sentence-transformers to embed a list of texts.
    Returns a 2D numpy array of shape (len(texts), EMBED_DIM)
    """
    embeddings = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings

# -----------------------
# Step 3: Upsert to Milvus
# -----------------------
def upsert_to_milvus(collection: Collection, texts: List[str], embeddings: np.ndarray, metadatas: List[str]=None):
    """
    Insert texts and corresponding embeddings into Milvus collection.
    """
    if metadatas is None:
        metadatas = ["{}"] * len(texts)
    # Milvus expects list-of-columns format
    entities = [
        embeddings.tolist(),  # embedding field
        texts,                # text field
        metadatas             # metadata field
    ]
    print(f"[milvus] Inserting {len(texts)} items into collection '{collection.name}'...")
    collection.insert(entities)
    # Optionally flush and load
    collection.flush()
    collection.load()

# -----------------------
# Step 4: Search Milvus
# -----------------------
def search_milvus(collection: Collection, query_embedding: np.ndarray, top_k: int = 3):
    """
    Search Milvus returning top_k hits: each hit returns {'score', 'text', 'metadata'}.
    """
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    results = collection.search(
        data=query_embedding.tolist(),
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["text", "metadata"]
    )
    hits = []
    if len(results) == 0:
        return hits
    for res in results[0]:
        hits.append({
            "score": res.score,
            "text": res.entity.get("text"),
            "metadata": res.entity.get("metadata")
        })
    return hits

# -----------------------
# Step 5: Use LlamaIndex + Ollama to generate final answer
# -----------------------
def generate_with_ollama(retrieved_texts: List[str], user_query: str, ollama_model=OLLAMA_MODEL) -> str:

    if len(retrieved_texts) == 0:
        context_text = ""
    else:
        context_text = "\n\n".join([f"Document {i+1}:\n{t}" for i, t in enumerate(retrieved_texts)])

    system_prompt = (
        "You are an expert plant pathologist. Use the retrieved document excerpts below as context. "
        "If the context answers the user's question, cite which Document number(s) you used. "
        "If the context is insufficient or missing, be explicit about missing info and give best-effort guidance."
    )

    full_prompt = (
        f"{system_prompt}\n\n=== Retrieved Documents ===\n{context_text}\n\n"
        f"=== User Query ===\n{user_query}\n\n"
        "Provide a concise, actionable answer and mention which documents you referenced."
    )

    print("[ollama] Sending prompt to Ollama model...")

    resp = ollama.generate(
        model=ollama_model,
        prompt=full_prompt,
    )

    return resp.get("response", str(resp))


# -----------------------
# Main flow
# -----------------------
def main():
    prompt_text = (
        "The plant has multiple leaves that have brown spots of differing sizes and some holes in the leaves. "
        "Identify the plant and its illness. Tell me how to treat it."
    )

    # 1) Run Granite Vision
    granite_text = run_granite_vision(IMAGE_PATH, prompt_text)
    print("\n[granite] Analysis text:\n", granite_text)

    # 2) Embed the granite output
    texts_to_index = [granite_text]
    embeddings = embed_texts(texts_to_index)
    print("[embed] embeddings shape:", embeddings.shape)

    # 3) Insert into Milvus
    upsert_to_milvus(collection, texts_to_index, embeddings, metadatas=["source:granite,image:"+os.path.basename(IMAGE_PATH)])

    # 4) Example query: ask how to treat (we could also embed query and search)
    user_query = "How should I treat the plant disease shown in the image?"
    # embed query with same embed model
    query_emb = embed_texts([user_query])
    hits = search_milvus(collection, query_emb, top_k=3)
    print("[milvus] Search hits:")
    for i, h in enumerate(hits):
        print(f"  Hit {i+1}: score={h['score']:.4f}, metadata={h.get('metadata')}\n    text={h['text'][:300]}")

    # 5) Build LlamaIndex docs from retrieved texts and call Ollama through LlamaIndex (we'll pass retrieved text as context)
    retrieved_texts = [h['text'] for h in hits]
    # Using LlamaIndex only to construct Documents and a light ServiceContext with LLMPredictor configured to Ollama
    # NOTE: llama_index expects an LLM wrapper; many versions accept a custom LLMPredictor. We'll use Ollama directly inside generate_with_ollama.
    final_answer = generate_with_ollama(retrieved_texts, user_query, ollama_model=OLLAMA_MODEL)

    print("\n=== FINAL ANSWER (from Ollama) ===\n")
    print(final_answer)


if __name__ == "__main__":
    main()
