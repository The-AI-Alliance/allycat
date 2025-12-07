"""
fancy_multimodal_rag_service.py

An upgraded multimodal RAG service:
- Replicate granite-vision -> image -> text
- Chunk & metadata
- Sentence-transformers embeddings
- Milvus vector store (collection management + indexing)
- Documents via LlamaIndex-style Document objects for context bookkeeping
- Ollama LLM wrapper used for generation and for query-expansion (agentic refinement)
- Agentic refinement loop: generate answer, if model asks for more context / low confidence, expand query and re-retrieve
- FastAPI endpoints for image upload / URL; returns JSON answer + retrieval trace

Run:    1. docker-compose up -d (starts Milvus)
        2. ollama pull gemma3:1b (ensure Ollama model is present)
        3. use API token from Replicate for granite-vision (enter it in the .env file or env var)
        4. uvicorn fancy_multimodal_rag_service:app
        5. Open browser to http://localhost:8000/ for web UI
"""

import os
import io
import time
import json
import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()
import httpx
from PIL import Image
import numpy as np
import replicate
import ollama
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, Collection, utility, Index
)
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# LlamaIndex Document class import - adapt per version
# The 'Document' type is used for storing text + metadata for building prompts.
try:
    from llama_index import Document
except Exception:
    # fallback for older/newer versions
    try:
        from llama_index.readers.schema.base import Document
    except Exception:
        # minimal shim if import fails (works for internal use)
        @dataclass
        class Document:
            text: str
            metadata: dict = None

# -----------------------
# Config
# -----------------------
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    raise RuntimeError("Set REPLICATE_API_TOKEN environment variable.")
rep = replicate.Client(api_token=REPLICATE_API_TOKEN)

MILVUS_HOST = os.environ.get("MILVUS_HOST", "localhost")
MILVUS_PORT = os.environ.get("MILVUS_PORT", "19530")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:1b")

EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "384"))

COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "vision_rag_collection_v2")

# chunking params
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 512))  # characters
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 64))  # characters overlap

# search params
DEFAULT_TOP_K = int(os.environ.get("DEFAULT_TOP_K", "5"))

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VisionRAG")

# -----------------------
# Helpers / small utils
# -----------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Naive character-level chunking with overlap (simple and robust)."""
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= L:
            break
        start = max(0, end - overlap)
    return chunks

# -----------------------
# Milvus collection helper
# -----------------------
def ensure_milvus_collection(name: str, dim: int = EMBED_DIM) -> Collection:
    """Create or return a Milvus collection with schema suitable for RAG text chunks."""
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    if utility.has_collection(name):
        col = Collection(name)
        # verify expected fields - best-effort
        try:
            col.load()
        except Exception:
            pass
        return col

    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=4096),
    ]
    schema = CollectionSchema(fields, description="Vision RAG collection v2")
    col = Collection(name=name, schema=schema)
    index_params = {"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}}
    col.create_index(field_name="embedding", index_params=index_params)
    col.load()
    return col

# -----------------------
# Ollama LLM wrapper
# -----------------------
class OllamaLLM:
    """Lightweight wrapper around ollama.generate for use in our pipeline.

    Methods:
        generate(prompt, max_tokens=512, temperature=0.0) -> dict with 'response' and raw fields
    """
    def __init__(self, model: str = OLLAMA_MODEL, host: str = OLLAMA_HOST, timeout: int = 60):
        self.model = model
        self.host = host
        # ollama python lib uses env OLLAMA_HOST or parameter.
        # If you want verbose control, pass additional kwargs to generate.
        # See ollama python library for more options.
        self.timeout = timeout

        # If the `ollama` module supports setting host globally, do it:
        try:
            ollama.config.host = host  # type: ignore
        except Exception:
            pass
 
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> Dict[str, Any]:
        """Call Ollama and return a dict with 'response' and raw output."""
        resp = ollama.generate(model=self.model, prompt=prompt, options={"num_predict": max_tokens,"temperature": temperature})

        # Extract raw text
        raw = resp.get("response", "")

        return {
            "response": raw
        }

# -----------------------
# Main service class
# -----------------------
class VisionRAGService:
    def __init__(self,
                 embed_model_name: str = EMBED_MODEL_NAME,
                 milvus_collection_name: str = COLLECTION_NAME,
                 ollama_model: str = OLLAMA_MODEL):
        # embed model
        logger.info("Loading embed model: %s", embed_model_name)
        self.embedder = SentenceTransformer(embed_model_name)

        # milvus collection
        logger.info("Connecting to Milvus at %s:%s", MILVUS_HOST, MILVUS_PORT)
        self.collection = ensure_milvus_collection(milvus_collection_name, dim=EMBED_DIM)

        # ollama LLM wrapper
        self.llm = OllamaLLM(model=ollama_model, host=OLLAMA_HOST)

    # ---- Step 1: image -> text via Granite Vision (Replicate)
    def analyze_image_with_granite(self, image_bytes: bytes, prompt_text: str) -> str:
        logger.info("Running granite vision...")
        with io.BytesIO(image_bytes) as f:
            f.seek(0)
            output = rep.run("ibm-granite/granite-vision-3.3-2b", input={"image": f, "prompt": prompt_text})

        # Extract text from output
        text = ""
        if isinstance(output, str):
            text = output
        elif isinstance(output, list):
            text = "".join(map(str, output))
        elif isinstance(output, dict):
            for key in ("text", "output", "caption", "analysis", "description", "result"):
                if key in output:
                    val = output[key]
                    if isinstance(val, list):
                        text = " ".join(map(str, val))
                    else:
                        text = str(val)
                    break
            else:
                text = json.dumps(output)

        return text


    # ---- Step 2: chunk and embed text
    def chunk_and_embed(self, text: str, source: str = "granite") -> Tuple[List[str], np.ndarray, List[dict]]:
        """Chunk text into smaller passages, produce embeddings and metadata entries."""
        chunks = chunk_text(text)
        if len(chunks) == 0:
            return [], np.zeros((0, EMBED_DIM)), []
        logger.info("Chunked text into %d chunks", len(chunks))
        embeddings = self.embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        metadatas = [{"source": source, "chunk_index": i, "chunk_len": len(chunks[i])} for i in range(len(chunks))]
        return chunks, embeddings, metadatas

    # ---- Step 3: upsert into Milvus
    def upsert_chunks(self, texts: List[str], embeddings: np.ndarray, metadatas: List[dict], source_label: str = "granite"):
        """Insert chunk rows into Milvus collection. Expects embeddings shape (N, dim)."""
        if len(texts) == 0:
            return
        # Convert metadata dicts to JSON strings to store in a single VARCHAR column
        metadata_strings = [json.dumps(m) for m in metadatas]
        logger.info("[milvus] inserting %d chunks", len(texts))
        entities = [
            embeddings.tolist(),  # embedding field
            texts,                # text field
            [source_label] * len(texts),  # source
            metadata_strings
        ]
        self.collection.insert(entities)
        # flush and load to ensure immediate availability
        self.collection.flush()
        self.collection.load()

    # ---- Step 4: search Milvus
    def search(self, query_emb: np.ndarray, top_k: int = DEFAULT_TOP_K) -> List[dict]:
        """Search Milvus with a single query embedding (shape (1, dim) or (dim,))."""
        if query_emb.ndim == 1:
            data = [query_emb.tolist()]
        else:
            data = query_emb.tolist()
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=data,
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text", "metadata", "source"]
        )
        hits = []
        if len(results) == 0:
            return hits
        # results[0] corresponds to the first query
        for res in results[0]:
            hits.append({
                "score": res.score,
                "text": res.entity.get("text"),
                "metadata": json.loads(res.entity.get("metadata")) if res.entity.get("metadata") else {},
                "source": res.entity.get("source")
            })
        return hits

    # ---- Step 5: build prompt from retrieved docs (LlamaIndex-style Documents)
    def build_context_and_prompt(self, hits: List[dict], user_query: str, system_role: Optional[str] = None) -> str:
        """Create a combined prompt using the retrieved chunks. Use LlamaIndex Documents for structure."""
        # build documents list
        docs = []
        for i, h in enumerate(hits):
            md = h.get("metadata", {})
            docs.append(Document(text=h["text"], metadata=md))

        # Combine retrieved documents into prompt text
        retrieved_text = "\n\n".join([f"Document {i+1} (score={h['score']:.4f}, src={h.get('source')}):\n{h['text']}"
                                     for i, h in enumerate(hits)])
        system_prompt = system_role or (
            "You are an expert plant pathologist. Use the retrieved document excerpts below as context. "
            "If the context answers the user's question, cite which Document number(s) you used. "
            "If the context is insufficient or missing, be explicit about missing info and give best-effort guidance."
        )
        full_prompt = (
            f"{system_prompt}\n\n=== Retrieved Documents ===\n{retrieved_text}\n\n"
            f"=== User Query ===\n{user_query}\n\n"
            "Provide a concise, actionable answer and mention which documents you referenced. "
            "Also include a short confidence estimate (high/medium/low) and whether further retrieval would help."
        )
        return full_prompt

    # ---- Step 6: call Ollama to get answer
    def generate_answer(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> Dict[str, Any]:
        try:
            resp = self.llm.generate(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
            text = resp.get("response") if isinstance(resp, dict) else str(resp)
            return {"text": text, "raw": resp}
        except Exception as e:
            logger.exception("Error calling Ollama: %s", e)
            return {"text": "", "raw": {"error": str(e)}}

    # ---- Agentic refinement loop
    def agentic_refinement(self, initial_query: str, initial_hits: List[dict],
                           max_iterations: int = 2, top_k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
        """
        Iteratively ask the LLM for an answer; if it suggests it needs more info or lower confidence,
        produce a query expansion using the LLM, re-embed the expanded query, re-retrieve, and retry.
        Returns final answer + trace of iterations.
        """
        trace = []
        current_hits = initial_hits
        current_query = initial_query

        for iteration in range(1, max_iterations + 1):
            prompt = self.build_context_and_prompt(current_hits, current_query)
            gen = self.generate_answer(prompt)
            ans_text = gen["text"]
            trace.append({"iteration": iteration, "query": current_query, "num_hits": len(current_hits), "answer": ans_text})

            # Heuristic: if model includes "confidence: low" or asks for more docs, do expansion.
            # We'll also look for keywords like "insufficient", "not enough", "more context".
            lower = ans_text.lower() if ans_text else ""
            needs_more = any(k in lower for k in ["insufficient", "not enough", "more context", "cannot", "don't have", "need more", "low confidence"])
            if not needs_more:
                # If the model responded and seems confident, return.
                return {"final_answer": ans_text, "trace": trace}
            # else ask LLM to expand query (query expansion)
            expansion_prompt = (
                f"You answered the user's query but said more context was needed. The original user query is:\n{initial_query}\n\n"
                f"Based on the retrieved documents and the answer you produced, propose a *concise* follow-up query or keywords "
                "that would likely retrieve more relevant documents from a vector DB. Output only the single-line query or keywords."
                f"\n\n=== Retrieved Documents ===\n" +
                "\n\n".join([f"- {h['text'][:200]}..." for h in current_hits])
            )
            expansion_resp = self.generate_answer(expansion_prompt)
            expanded_query = expansion_resp["text"].strip().splitlines()[0] if expansion_resp["text"] else current_query
            trace.append({"expansion_suggestion": expanded_query, "expansion_raw": expansion_resp})
            # embed expanded query, search Milvus
            query_emb = self.embedder.encode([expanded_query], convert_to_numpy=True, show_progress_bar=False)
            new_hits = self.search(query_emb, top_k=top_k)
            # If new_hits identical to current_hits (by text) break to avoid infinite loops
            old_texts = {h["text"] for h in current_hits}
            new_texts = {h["text"] for h in new_hits}
            if new_texts.issubset(old_texts):
                # not getting new info; stop
                trace.append({"note": "No new hits found from expansion; stopping refinement."})
                return {"final_answer": ans_text, "trace": trace}
            # else set current_hits to a merge of old+new (prefer higher scores)
            merged = (new_hits + [h for h in current_hits if h["text"] not in new_texts])[:top_k]
            current_hits = merged
            current_query = expanded_query
            # loop continues up to max_iterations

        # If loop finishes without confident answer, return last answer with trace
        return {"final_answer": trace[-1]["answer"] if trace else "", "trace": trace}

    # ---- Full pipeline orchestrator called by API
    def process_image_and_query(self, image_bytes: bytes, initial_prompt_text: str, user_query: str,
                                source_label: str = "granite", top_k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
        """Full pipeline: analyze image, chunk & index, search, agentic refinement, final answer."""
        t0 = time.time()
        # 1) analyze image
        analysis_text = self.analyze_image_with_granite(image_bytes, initial_prompt_text)

        # 2) chunk & embed
        chunks, embeddings, metadatas = self.chunk_and_embed(analysis_text, source=source_label)

        # 3) upsert into Milvus
        self.upsert_chunks(chunks, embeddings, metadatas, source_label=source_label)

        # 4) embed user query and search
        query_emb = self.embedder.encode([user_query], convert_to_numpy=True, show_progress_bar=False)[0]
        hits = self.search(query_emb, top_k=top_k)

        # 5) agentic refinement & final answer
        result = self.agentic_refinement(user_query, hits, max_iterations=2, top_k=top_k)
        duration = time.time() - t0

        # Build return payload (structured)
        payload = {
            "analysis_text": analysis_text,
            "num_chunks_indexed": len(chunks),
            "initial_hits_count": len(hits),
            "final_answer": result.get("final_answer"),
            "trace": result.get("trace"),
            "duration_seconds": duration
        }
        return payload

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title="Multimodal Plant Diagnostic Service ", version="v1")

# create single service instance (could be replaced by dependency injection)
service = VisionRAGService()

# -----------------------
# Web UI root page
# -----------------------
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Allycat Vision + Text RAG UI</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; max-width: 900px; }
            input, textarea { width: 100%; margin-bottom: 10px; padding: 8px; }
            button { padding: 10px 20px; margin-bottom: 20px; }
            pre { background: #f4f4f4; padding: 10px; overflow-x: auto; }
            img { max-width: 300px; margin-bottom: 10px; border: 1px solid #ccc; padding: 4px; }
            details { background: #e8f0fe; padding: 6px; margin: 4px 0; border-radius: 4px; }
            summary { font-weight: bold; cursor: pointer; }
            h2 { margin-top: 30px; }
        </style>
    </head>
    <body>
        <h1>Allycat Plant Diagnostic Service</h1>
        <form id="rag-form">
            <label>Upload Image:</label>
            <input type="file" id="file" name="file" accept="image/*"><br>
            
            <label>OR Image URL:</label>
            <input type="text" id="image_url" name="image_url" placeholder="https://example.com/image.jpg"><br>
            
            <img id="image_preview" src="" alt="" style="display:none;"><br>

            <label>User Query:</label>
            <textarea id="user_query" name="user_query" rows="2">What is wrong with this plant?</textarea><br>
            
            <label>Prompt Text (optional):</label>
            <textarea id="prompt_text" name="prompt_text" rows="3">
<Describe the plant and its appearance here>. Identify the plant and its illness. Tell me how to treat it.
            </textarea><br>
            
            <button type="submit">Analyze</button>
        </form>
        
        <h2>Result:</h2>
        <div id="result"></div>

        <script>
            const form = document.getElementById('rag-form');
            const preview = document.getElementById('image_preview');

            const updatePreview = (file, url) => {
                if (file) {
                    const reader = new FileReader();
                    reader.onload = e => { preview.src = e.target.result; preview.style.display = 'block'; };
                    reader.readAsDataURL(file);
                } else if (url) {
                    preview.src = url;
                    preview.style.display = 'block';
                } else {
                    preview.src = '';
                    preview.style.display = 'none';
                }
            };

            document.getElementById('file').addEventListener('change', e => updatePreview(e.target.files[0], ''));
            document.getElementById('image_url').addEventListener('input', e => updatePreview(null, e.target.value));

            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                const fileInput = document.getElementById('file');
                const image_url = document.getElementById('image_url').value;
                const user_query = document.getElementById('user_query').value;
                const prompt_text = document.getElementById('prompt_text').value;

                const formData = new FormData();
                if (fileInput.files.length > 0) {
                    formData.append('file', fileInput.files[0]);
                }
                formData.append('image_url', image_url);
                formData.append('user_query', user_query);
                formData.append('prompt_text', prompt_text);

                const resultEl = document.getElementById('result');
                resultEl.innerHTML = '<p>Processing...</p>';

                try {
                    const response = await fetch('/analyze', { method: 'POST', body: formData });
                    const data = await response.json();

                    let html = `<h3>Analysis Text:</h3>
                    <pre style="
                    white-space: pre-wrap;
                    word-wrap: break-word;
                    ">${data.analysis_text}</pre>`;
                    html += `<h3>Number of Chunks Indexed:</h3><p>${data.num_chunks_indexed}</p>`;
                    html += `<h3>Initial Hits Count:</h3><p>${data.initial_hits_count}</p>`;
                    html += `<h3>Final Answer:</h3>
                    <pre style="
                    white-space: pre-wrap;
                    word-wrap: break-word;
                    ">${data.final_answer}</pre>`;

                    // Collapsible Retrieved Chunks
                    html += `<h3>Retrieved Chunks:</h3>`;
                    if (data.trace && data.trace.length > 0) {
                        data.trace.forEach((t, i) => {
                            if (t.hits) {
                                t.hits.forEach((hit, j) => {
                                    html += `<details><summary>Chunk ${j+1} (score=${hit.score.toFixed(4)})</summary>
                                             <p><strong>Source:</strong> ${hit.source}</p>
                                             <pre>${hit.text}</pre>
                                             </details>`;
                                });
                            }
                        });
                    }

                    // Collapsible Agentic Refinement Trace
                    html += `<h3>Agentic Refinement Trace:</h3>`;
                    data.trace.forEach((t, i) => {
                        html += `<details><summary>Iteration ${t.iteration || i+1}</summary>
                                 <p><strong>Query:</strong> ${t.query || ''}</p>
                                 <p><strong>Number of Hits:</strong> ${t.num_hits || ''}</p>
                                 <p><strong>Answer:</strong><pre>${t.answer || ''}</pre>
                                 ${t.expansion_suggestion ? '<p><em>Expansion Suggestion:</em> '+t.expansion_suggestion+'</p>' : ''}
                                 </details>`;
                    });

                    html += `<h3>Processing Duration (s):</h3><p>${data.duration_seconds.toFixed(2)}</p>`;

                    resultEl.innerHTML = html;
                } catch (err) {
                    resultEl.innerHTML = '<p style="color:red;">Error: ' + err + '</p>';
                }
            });
        </script>
    </body>
    </html>
    """

# -----------------------
# Existing /analyze endpoint
# -----------------------

class AnalyzeResponse(BaseModel):
    analysis_text: str
    num_chunks_indexed: int
    initial_hits_count: int
    final_answer: Optional[str]
    trace: List[dict]
    duration_seconds: float

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image_endpoint(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    user_query: str = Form(...),
    prompt_text: str = Form(
        "<Describe the plant's appearance here.> Identify the plant and its illness. Tell me how to treat it."
    ),
    top_k: int = Form(DEFAULT_TOP_K)
):
    """
    Accept either an uploaded image or an image_url (one required).
    Returns JSON with analysis text, indexed chunks, and final answer from Ollama.
    """
    if (file is None) and (not image_url):
        raise HTTPException(status_code=400, detail="Upload a file or provide image_url.")

    # load image bytes
    image_bytes = None
    if file:
        image_bytes = await file.read()
    else:
        # fetch image_url
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.get(image_url)
                r.raise_for_status()
                image_bytes = r.content
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Unable to fetch image_url: {e}")

    try:
        result = service.process_image_and_query(image_bytes, prompt_text, user_query, top_k=top_k)
        return JSONResponse(content=result)
    except Exception as e:
        logger.exception("Processing error: %s", e)
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")

# lightweight health endpoint
@app.get("/health")
async def health():
    # confirm Milvus accessibility
    ok = True
    try:
        # attempt to list collections
        collections = utility.list_collections()
    except Exception as e:
        ok = False
        collections = str(e)
    return {"status": "ok" if ok else "error", "milvus_collections": collections}

# -----------------------
# Run guard for local dev (uvicorn recommended)
# -----------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fancy_multimodal_rag_service:app", host="0.0.0.0", port=11400, reload=True)
