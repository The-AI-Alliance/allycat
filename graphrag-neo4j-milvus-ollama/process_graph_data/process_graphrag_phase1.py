"""
Phase 1: Entity & Relationship Extraction for GraphRAG.
Input: workspace/processed/*.md    
Output: workspace/graph_data/phase1_output.json

"""

import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict
from datetime import datetime
import openai
from json_repair import repair_json
import orjson

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from my_config import MY_CONFIG

# Logging setup
LOG_DIR = Path("process_graphdata/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"phase1_extraction_{log_timestamp}.log"

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Log file paths
MAIN_LOG_FILE = log_file
FAILED_RESPONSES_LOG = LOG_DIR / "phase1_failed_responses.txt"
SUCCESSFUL_FILES_LOG = LOG_DIR / "phase1_processed_successfully.txt"
TOKEN_EXCEEDED_FILES_LOG = LOG_DIR / "phase1_token_limit_exceeded.txt"
QUOTA_EXCEEDED_FILES_LOG = LOG_DIR / "phase1_quota_exceeded.txt"
ERROR_FILES_LOG = LOG_DIR / "phase1_error_files.txt"


class GraphExtractor:
    """Phase 1: Entity and relationship extraction from markdown files."""
    
    def __init__(self, provider: str = None):
        # Use configured provider if not specified
        if provider is None:
            provider = MY_CONFIG.GRAPHRAG_LLM_PROVIDER
        
        if provider not in MY_CONFIG.GRAPHRAG_PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. Choose from: {list(MY_CONFIG.GRAPHRAG_PROVIDERS.keys())}")
        
        self.provider = provider
        config = MY_CONFIG.GRAPHRAG_PROVIDERS[provider]
        
        # Get API key from environment using the key name from config
        api_key = os.getenv(config["api_key_env"])
        if not api_key:
            raise ValueError(f"{config['api_key_env']} environment variable not set")
        
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=config["base_url"]
        )
        self.model_name = config["model"]
        
        # Token configuration
        self.max_tokens = config["max_tokens"]
        self.max_output = config["max_output"]
        self.rate_delay = config["rate_delay"]
        self.chars_per_token = config["chars_per_token"]
        
        # Calculate available tokens for input
        reserve_tokens = 2000
        safety_margin = 2000
        self.available_tokens = (
            self.max_tokens - self.max_output - reserve_tokens - safety_margin
        )
        
        # Graph data storage
        self.global_entity_registry = {}
        self.graph_data = {"nodes": [], "relationships": []}
        self.processed_files = 0
        
        # Extraction parameters from MY_CONFIG
        self.min_entities = MY_CONFIG.GRAPH_MIN_ENTITIES
        self.max_entities = MY_CONFIG.GRAPH_MAX_ENTITIES
        self.min_relationships = MY_CONFIG.GRAPH_MIN_RELATIONSHIPS
        self.max_relationships = MY_CONFIG.GRAPH_MAX_RELATIONSHIPS
        self.min_confidence = MY_CONFIG.GRAPH_MIN_CONFIDENCE
        
        # API statistics
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_api_calls = 0
        self.failed_calls = 0
        self.rate_limit_retries = 0
        
        logger.info(f"Initialized {self.model_name} ({provider}) - {self.max_tokens}tok input, {self.max_output}tok output")
        logger.info(f"Extraction: {self.min_entities}-{self.max_entities} entities, {self.min_relationships}-{self.max_relationships} relationships (confidence≥{self.min_confidence})")
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from character length."""
        return len(text) // self.chars_per_token
    
    def _check_token_limit(self, text: str, file_name: str) -> bool:
        """Validate text size against token limit."""
        estimated_tokens = self._estimate_tokens(text)
        system_prompt_tokens = self._estimate_tokens(self.get_entity_extraction_prompt())
        total_input_tokens = estimated_tokens + system_prompt_tokens + 100
        
        if total_input_tokens > self.max_tokens:
            logger.error(f"{file_name}: {estimated_tokens} tokens exceeds limit ({self.max_tokens})")
            return False
        
        logger.info(f"{file_name}: {estimated_tokens} tokens within limit")
        return True
    
    def get_entity_extraction_prompt(self) -> str:
        """Get system prompt for entity extraction."""
        return f"""You are a specialized knowledge graph extraction assistant. Your task is to analyze content and extract entities and relationships to build comprehensive knowledge graphs.

DYNAMIC EXTRACTION REQUIREMENTS (PER FILE):
- Extract {self.min_entities}-{self.max_entities} most important entities from THIS FILE
- Create {self.min_relationships}-{self.max_relationships} meaningful relationships between entities from THIS FILE
- Confidence threshold: {self.min_confidence} (only include high-confidence extractions)
- Focus on extracting diverse entity types relevant to the content domain
- IMPORTANT: Each file is processed independently - extract thoroughly from the provided content

EXTRACTION PRINCIPLES:
- Adapt entity types to content domain
- Focus on meaningful semantic relationships
- Prioritize quality over quantity

ENTITY EXTRACTION:
- Extract key concepts, people, organizations, technologies, events, locations
- Include explicit mentions and strong implications
- Assign types based on semantic role in content

SEMANTIC TAGS:
- Provide 2-5 conceptual tags per entity for semantic search
- Use lowercase with hyphens (e.g., "machine-learning", "healthcare-policy")
- Adapt tags to content domain (medical → "healthcare", legal → "law")

ENTITY FIELD REQUIREMENTS (ALL fields are MANDATORY):
1. "text": The exact name or title of the entity (string, 2-100 characters)
2. "type": The semantic category (e.g., "Person", "Organization", "Technology", "Concept", "Project", "Event", "Location")
3. "content": 1-3 sentence description of the entity and its significance (string, 50-300 characters)
4. "semantic_tags": Array of 2-5 lowercase conceptual tags for semantic search (array of strings)
5. "confidence": Extraction confidence score from 0.0 to 1.0 (float)
   - 0.95-1.0: Explicitly mentioned with clear definitions
   - 0.85-0.94: Clearly implied with strong contextual evidence
   - 0.80-0.84: Inferred from context with reasonable certainty
   - Never below {self.min_confidence}

RELATIONSHIP EXTRACTION:
- Capture semantic meaning, not just co-occurrence
- Use descriptive types expressing connection nature
- Include hierarchical, associative, and causal relationships

RELATIONSHIP FIELD REQUIREMENTS (ALL fields are MANDATORY):
1. "startNode": Exact "text" value of source entity (must match entity's "text" field)
2. "endNode": Exact "text" value of target entity (must match entity's "text" field)
3. "type": Relationship type in UPPERCASE_SNAKE_CASE (e.g., "BELONGS_TO", "CREATED_BY", "DEPENDS_ON")
4. "description": 1-2 sentence explanation of how startNode relates to endNode (string, 30-200 characters)
5. "evidence": Direct quote or paraphrase from text supporting this relationship (string, 20-300 characters)
6. "confidence": Relationship confidence score from 0.0 to 1.0 (float)
   - 0.95-1.0: Explicitly stated with direct evidence
   - 0.85-0.94: Clearly implied by context
   - 0.80-0.84: Reasonably inferred
   - Never below {self.min_confidence}

OUTPUT FORMAT (strict JSON):
{{
    "entities": [
        {{
            "text": "Entity Name",
            "type": "DynamicType",
            "content": "Comprehensive description of the entity",
            "semantic_tags": ["domain-tag", "topic-tag", "category-tag"],
            "confidence": 0.95
        }}
    ],
    "relationships": [
        {{
            "startNode": "Entity Name 1",
            "endNode": "Entity Name 2",
            "type": "DESCRIPTIVE_RELATIONSHIP_TYPE",
            "description": "How Entity Name 1 relates to Entity Name 2",
            "evidence": "Direct quote supporting this relationship",
            "confidence": 0.90
        }}
    ]
}}

CRITICAL REQUIREMENTS:
1. ALL fields are MANDATORY - never omit "semantic_tags", "confidence", "evidence", or "content"
2. Confidence values MUST be between {self.min_confidence} and 1.0
3. Evidence MUST be specific text from the source
4. Content MUST be meaningful, not just repeating the entity name
5. Semantic_tags MUST be 2-5 lowercase conceptual tags
6. Respond with ONLY valid JSON - no explanations, no markdown, no code blocks"""

    def _smart_json_parse(self, json_text: str) -> Dict:
        """Parse JSON with fallback repair strategies."""
        cleaned_text = json_text.strip()
        
        try:
            return orjson.loads(cleaned_text.encode('utf-8'))
        except Exception:
            pass
        
        try:
            repaired = repair_json(cleaned_text)
            return orjson.loads(repaired.encode('utf-8'))
        except Exception:
            pass
        
        try:
            return json.loads(cleaned_text)
        except Exception:
            pass
        
        try:
            repaired = repair_json(cleaned_text)
            return json.loads(repaired)
        except Exception:
            pass
        
        raise ValueError("JSON parsing failed")

    def _validate_complete_format(self, extraction_data: Dict) -> bool:
        """Validate extraction data structure and required fields."""
        if not isinstance(extraction_data, dict):
            return False

        if "entities" not in extraction_data or "relationships" not in extraction_data:
            return False

        entities = extraction_data.get("entities", [])
        relationships = extraction_data.get("relationships", [])
        
        if not isinstance(entities, list) or len(entities) == 0:
            return False
            
        for entity in entities:
            if not isinstance(entity, dict):
                return False
            
            required_fields = ["text", "type", "content", "semantic_tags", "confidence"]
            for field in required_fields:
                if field not in entity:
                    return False
                value = entity[field]
                if value is None or value == "" or (isinstance(value, str) and not value.strip()):
                    return False
            
            # Validate semantic_tags is a non-empty list
            if not isinstance(entity["semantic_tags"], list) or len(entity["semantic_tags"]) < 1:
                return False
                
            if not isinstance(entity["confidence"], (int, float)) or entity["confidence"] <= 0:
                return False
        
        if isinstance(relationships, list):
            for rel in relationships:
                if not isinstance(rel, dict):
                    return False
                
                required_fields = ["startNode", "endNode", "type", "description", "evidence", "confidence"]
                for field in required_fields:
                    if field not in rel:
                        return False
                    value = rel[field]
                    if value is None or value == "" or (isinstance(value, str) and not value.strip()):
                        return False
                
                if not isinstance(rel["confidence"], (int, float)) or rel["confidence"] <= 0:
                    return False
        
        return True

    def _save_failed_response(self, llm_response: str, file_name: str, _json_error: str, _repair_error: str):
        """Log failed API responses for analysis."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(FAILED_RESPONSES_LOG, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write("FAILED RESPONSE\n")
                f.write(f"File: {file_name}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Error: {_json_error}\n")
                f.write(f"{'='*80}\n")
                f.write(llm_response)
                f.write(f"\n{'='*80}\n\n")
                f.flush()
                
        except Exception as save_error:
            logger.error(f"Failed to log response from {file_name}: {save_error}")

    def extract_graph_from_full_text(self, text: str, source_file: str) -> Dict:
        """Extract knowledge graph entities and relationships from text."""
        
        if not self._check_token_limit(text, source_file):
            logger.error(f"{source_file}: Token limit exceeded")
            return {"entities": [], "relationships": [], "error": "token_limit_exceeded"}
        
        system_prompt = self.get_entity_extraction_prompt()
        user_prompt = f"""
Analyze the following content from file "{source_file}":

```
{text}
```

Extract all relevant entities, concepts, and their relationships from this content.
"""
        
        input_tokens_estimate = self._estimate_tokens(system_prompt + user_prompt)
        
        max_retries = 5 if self.provider == "cerebras" else 1
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                self.total_api_calls += 1
                call_start = time.time()
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=self.max_output,
                    response_format={"type": "json_object"}
                )
                
                call_duration = time.time() - call_start
                
                if not response or not response.choices or not response.choices[0].message.content:
                    self.failed_calls += 1
                    logger.error("Empty API response")
                    raise ValueError("Empty response from API")
                
                if hasattr(response, 'usage') and response.usage:
                    actual_input = response.usage.prompt_tokens
                    actual_output = response.usage.completion_tokens
                    self.total_input_tokens += actual_input
                    self.total_output_tokens += actual_output
                    logger.info(f"API #{self.total_api_calls}: {actual_input:,}tok in | {actual_output:,}tok out | {call_duration:.1f}s")
                else:
                    self.total_input_tokens += input_tokens_estimate
                    logger.info(f"API #{self.total_api_calls}: ~{input_tokens_estimate:,}tok | {call_duration:.1f}s")
                
                llm_response = response.choices[0].message.content.strip()
                
                cleaned_response = llm_response
                if "```json" in cleaned_response:
                    parts = cleaned_response.split("```json")
                    if len(parts) > 1:
                        cleaned_response = parts[1].split("```")[0].strip()
                elif "```" in cleaned_response:
                    parts = cleaned_response.split("```")
                    if len(parts) >= 3:
                        cleaned_response = parts[1].strip()
                
                extraction_data = self._smart_json_parse(cleaned_response)
                
                if not self._validate_complete_format(extraction_data):
                    self._save_failed_response(cleaned_response, source_file, "Format validation failed", "Missing required fields")
                    logger.error(f"{source_file}: Missing required fields")
                    return {"entities": [], "relationships": []}
                
                entities_count = len(extraction_data.get('entities', []))
                relationships_count = len(extraction_data.get('relationships', []))
                logger.info(f"{source_file}: Extracted {entities_count}E + {relationships_count}R")
                return extraction_data
                
            except Exception as e:
                error_msg = str(e)
                
                if "429" in error_msg or "rate" in error_msg.lower():
                    retry_count += 1
                    self.rate_limit_retries += 1
                    
                    if retry_count < max_retries:
                        wait_time = self.rate_delay * (retry_count + 1)
                        logger.warning(f"Rate limit: retry {retry_count}/{max_retries} in {wait_time:.0f}s")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"{source_file}: Rate limit exceeded")
                        self.failed_calls += 1
                        return {"entities": [], "relationships": [], "error": "rate_limit_exceeded"}
                
                if "quota" in error_msg.lower():
                    self.failed_calls += 1
                    logger.error(f"Quota exceeded: {source_file}")
                    return {"entities": [], "relationships": [], "error": "quota_exceeded"}
                
                self.failed_calls += 1
                logger.error(f"{source_file}: Extraction error - {error_msg}")
                return {"entities": [], "relationships": [], "error": str(e)}
        
        self.failed_calls += 1
        return {"entities": [], "relationships": [], "error": "max_retries_exceeded"}
    
    def process_entity(self, entity_data: Dict, source_file: str) -> str:
        """Process and deduplicate entity."""
        entity_text = entity_data.get("text", "").strip()
        if not entity_text:
            return None
        
        semantic_key = entity_text.lower().strip()
        
        if semantic_key in self.global_entity_registry:
            return self.global_entity_registry[semantic_key]["id"]
        
        entity_id = str(uuid.uuid4())
        entity_info = {
            "id": entity_id,
            "text": entity_text,
            "type": entity_data.get("type", "CONCEPT"),
            "content": entity_data.get("content", ""),
            "semantic_tags": entity_data.get("semantic_tags", []),
            "confidence": entity_data.get("confidence", 0.0),
            "source_file": source_file
        }
        
        self.global_entity_registry[semantic_key] = entity_info
        return entity_id
    
    def process_relationship(self, rel_data: Dict, source_file: str):
        """Process relationship between entities."""
        start_text = rel_data.get("startNode", "").strip().lower()
        end_text = rel_data.get("endNode", "").strip().lower()
        
        if start_text not in self.global_entity_registry or end_text not in self.global_entity_registry:
            return
        
        rel_confidence = rel_data.get("confidence", 0.0)
        if rel_confidence < self.min_confidence:
            return
        
        evidence = rel_data.get("evidence", "")
        evidence_length = len(evidence.split())
        
        evidence_multiplier = 1.0
        if evidence_length > 50:
            evidence_multiplier = 1.2
        elif evidence_length > 20:
            evidence_multiplier = 1.1
        
        weight = min(1.0, rel_confidence * evidence_multiplier)
        
        relationship = {
            "id": str(uuid.uuid4()),
            "startNode": self.global_entity_registry[start_text]["id"],
            "endNode": self.global_entity_registry[end_text]["id"],
            "type": rel_data.get("type", "RELATES_TO"),
            "description": rel_data.get("description", ""),
            "evidence": rel_data.get("evidence", ""),
            "confidence": rel_confidence,
            "weight": weight,
            "source_file": source_file
        }
        
        self.graph_data["relationships"].append(relationship)

    
    def process_md_file(self, md_file_path: str) -> Dict:
        """Process individual markdown file."""
        file_name = os.path.basename(md_file_path)
        
        try:
            with open(md_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            char_count = len(content)
            estimated_tokens = self._estimate_tokens(content)
            logger.info(f"{file_name}: {char_count:,} chars (~{estimated_tokens:,} tokens)")
            
            extraction = self.extract_graph_from_full_text(content, file_name)
            
            if "error" in extraction:
                error_type = extraction["error"]
                return {
                    "file": file_name,
                    "status": error_type,
                    "char_count": char_count,
                    "estimated_tokens": estimated_tokens,
                    "processed_at": datetime.now().isoformat()
                }
            
            entities_added = 0
            for entity_data in extraction.get("entities", []):
                entity_id = self.process_entity(entity_data, file_name)
                if entity_id:
                    entities_added += 1
            
            relationships_added = 0
            for rel_data in extraction.get("relationships", []):
                self.process_relationship(rel_data, file_name)
                relationships_added += 1
            
            result = {
                "file": file_name,
                "status": "success",
                "char_count": char_count,
                "estimated_tokens": estimated_tokens,
                "entities_extracted": len(extraction.get("entities", [])),
                "unique_entities_added": entities_added,
                "relationships_generated": relationships_added,
                "processed_at": datetime.now().isoformat()
            }
            
            self.processed_files += 1
            logger.info(f"{file_name}: {entities_added}E + {relationships_added}R")
            return result
            
        except Exception as e:
            logger.error(f"{file_name}: Processing error - {e}")
            return {
                "file": file_name,
                "status": "error",
                "error": str(e),
                "processed_at": datetime.now().isoformat()
            }
    
    def _save_processing_logs(self, results: list, elapsed_time: float):
        """Generate detailed processing logs."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        successful_files = [r for r in results if r.get("status") == "success"]
        token_exceeded_files = [r for r in results if r.get("status") == "token_limit_exceeded"]
        quota_exceeded_files = [r for r in results if r.get("status") == "quota_exceeded"]
        rate_limit_files = [r for r in results if r.get("status") == "rate_limit_exceeded"]
        error_files = [r for r in results if r.get("status") == "error"]
        
        if successful_files:
            with open(SUCCESSFUL_FILES_LOG, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"RUN: {run_id}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Successfully Processed: {len(successful_files)} files\n")
                f.write(f"{'='*80}\n")
                for r in successful_files:
                    f.write(f"{r['file']}\n")
                    f.write(f"  Chars: {r.get('char_count', 0)}, Tokens: {r.get('estimated_tokens', 0)}\n")
                    f.write(f"  Entities: {r.get('unique_entities_added', 0)}, Relationships: {r.get('relationships_generated', 0)}\n")
                f.write("\n")
        
        if token_exceeded_files:
            with open(TOKEN_EXCEEDED_FILES_LOG, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"RUN: {run_id}\n")
                f.write(f"Token Limit Exceeded: {len(token_exceeded_files)} files\n")
                f.write(f"{'='*80}\n")
                for r in token_exceeded_files:
                    f.write(f"{r['file']}\n")
                f.write("\n")
        
        if quota_exceeded_files or rate_limit_files:
            with open(QUOTA_EXCEEDED_FILES_LOG, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"RUN: {run_id}\n")
                f.write(f"Quota/Rate Limit: {len(quota_exceeded_files) + len(rate_limit_files)} files\n")
                f.write(f"{'='*80}\n")
                for r in quota_exceeded_files + rate_limit_files:
                    f.write(f"{r['file']}\n")
                f.write("\n")
        
        if error_files:
            with open(ERROR_FILES_LOG, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"RUN: {run_id}\n")
                f.write(f"Processing Errors: {len(error_files)} files\n")
                f.write(f"{'='*80}\n")
                for r in error_files:
                    f.write(f"{r['file']}: {r.get('error', 'Unknown')}\n")
                f.write("\n")
    
    def process_all_md_files(self, input_dir: str = None, output_path: str = None):
        """Process all markdown files individually (GraphRAG best practice)."""
        if input_dir is None:
            input_dir = "workspace/processed"
        if output_path is None:
            output_path = f"workspace/graph_data/{self.provider}_phase1_output.json"
        
        input_path = Path(input_dir)
        md_files = list(input_path.glob("**/*.md"))
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if not md_files:
            logger.warning(f"No markdown files found in {input_dir}")
            return {"status": "no_files", "message": "No markdown files found"}
        
        total_chars = 0
        total_estimated_tokens = 0
        
        logger.info(f"\n{'='*60}")
        logger.info("Scanning markdown files")
        logger.info(f"{'='*60}\n")
        
        for md_file in md_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                char_count = len(content)
                token_count = self._estimate_tokens(content)
                total_chars += char_count
                total_estimated_tokens += token_count
                logger.info(f"{md_file.name}: {char_count:,} chars (~{token_count:,} tokens)")
            except Exception as e:
                logger.error(f"{md_file.name}: Read error - {e}")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Total: {len(md_files)} files | {total_chars:,} chars | ~{total_estimated_tokens:,} tokens")
        logger.info("Processing mode: INDIVIDUAL FILES (GraphRAG best practice)")
        logger.info(f"API calls: {len(md_files)} (one per file)")
        logger.info(f"{'='*60}\n")
        
        results = []
        start_time = time.time()
        
        # Process each file individually (GraphRAG best practice)
        for i, md_file in enumerate(md_files, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"File {i}/{len(md_files)}: {md_file.name}")
            logger.info(f"{'='*60}")
            
            result = self.process_md_file(str(md_file))
            results.append(result)
            
            if result.get("status") in ["quota_exceeded", "rate_limit_exceeded"]:
                logger.error(f"Quota/rate limit exceeded at file {i}/{len(md_files)}")
                break
            
            # Rate limiting delay between files
            if i < len(md_files):
                time.sleep(self.rate_delay)
        
        elapsed = time.time() - start_time
        successful = sum(1 for r in results if r.get("status") == "success")
        token_exceeded = sum(1 for r in results if r.get("status") == "token_limit_exceeded")
        quota_exceeded_count = sum(1 for r in results if r.get("status") in ["quota_exceeded", "rate_limit_exceeded"])
        failed = len(results) - successful - token_exceeded - quota_exceeded_count
        
        self._save_processing_logs(results, elapsed)
        
        final_nodes = []
        for semantic_key, entity_info in self.global_entity_registry.items():
            node = {
                "id": entity_info["id"],
                "labels": [entity_info["type"]],
                "properties": {
                    "title": entity_info["text"],
                    "content": entity_info.get("content", ""),
                    "semantic_tags": entity_info.get("semantic_tags", []),
                    "source": entity_info.get("source_file", ""),
                    "confidence": entity_info["confidence"],
                    "created_date": datetime.now().strftime("%Y-%m-%d"),
                    "extraction_method": self.provider
                }
            }
            final_nodes.append(node)
        
        final_graph = {
            "nodes": final_nodes,
            "relationships": self.graph_data["relationships"],
            "metadata": {
                "node_count": len(final_nodes),
                "relationship_count": len(self.graph_data["relationships"]),
                "generated_at": datetime.now().isoformat(),
                "generator": "Allycat Graph Extractor - Full MD",
                "llm_provider": self.provider,
                "model": self.model_name,
                "format_version": "neo4j-2025",
                "token_usage": {
                    "total_input_tokens": self.total_input_tokens,
                    "total_output_tokens": self.total_output_tokens,
                    "total_api_calls": self.total_api_calls,
                    "failed_api_calls": self.failed_calls,
                    "rate_limit_retries": self.rate_limit_retries,
                    "total_chars_processed": total_chars,
                    "estimated_tokens_processed": total_estimated_tokens
                }
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_graph, f, indent=2, ensure_ascii=False)
        
        output_size = os.path.getsize(output_path)
        output_size_mb = output_size / (1024 * 1024)
        
        logger.info(f"\n{'='*60}")
        logger.info("COMPLETE")
        logger.info(f"Files: {successful}✓ {token_exceeded}⚠️ {quota_exceeded_count}⏸️  {failed}✗")
        logger.info(f"Graph: {len(final_nodes)} nodes | {len(self.graph_data['relationships'])} edges")
        logger.info(f"API: {self.total_api_calls} calls ({self.failed_calls} failed)")
        logger.info(f"Tokens: {self.total_input_tokens:,} in | {self.total_output_tokens:,} out")
        logger.info(f"Time: {elapsed:.1f}s | Size: {output_size_mb:.2f}MB")
        logger.info(f"Output: {output_path}")
        logger.info(f"{'='*60}\n")
        
        return {
            "status": "completed",
            "total_files": len(md_files),
            "successful": successful,
            "token_limit_exceeded": token_exceeded,
            "quota_exceeded": quota_exceeded_count,
            "failed": failed,
            "unique_entities": len(final_nodes),
            "total_relationships": len(self.graph_data["relationships"]),
            "processing_time_seconds": elapsed,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_api_calls": self.total_api_calls
        }


def main():
    """Execute graph extraction with configurable provider."""
    
    # Get configuration from MY_CONFIG
    PROVIDER = os.getenv("GRAPHRAG_LLM_PROVIDER", MY_CONFIG.GRAPHRAG_LLM_PROVIDER)
    INPUT_DIR = os.getenv("INPUT_DIR", MY_CONFIG.PROCESSED_DATA_DIR)
    OUTPUT_FILE = os.getenv("OUTPUT_FILE", os.path.join(MY_CONFIG.GRAPH_DATA_DIR, "phase1_output.json"))
    
    # Get provider config
    provider_config = MY_CONFIG.GRAPHRAG_PROVIDERS[PROVIDER]
    
    startup_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    banner = f"""
{'='*80}
PHASE 1: Entity & Relationship Extraction
{'='*80}
Time: {startup_time}
Provider: {PROVIDER.upper()} | Model: {provider_config['model']}
Context: {provider_config['max_tokens']:,} tokens
Input: {INPUT_DIR}
Output: {OUTPUT_FILE}
Log: {MAIN_LOG_FILE}
{'='*80}
"""
    logger.info(banner)
    
    try:
        extractor = GraphExtractor(provider=PROVIDER)
        summary = extractor.process_all_md_files(
            input_dir=INPUT_DIR,
            output_path=OUTPUT_FILE
        )
        
        if summary["status"] == "completed":
            end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ending_banner = f"""
{'='*80}
SUCCESS
{'='*80}
Time: {end_time}
Files: {summary['successful']}/{summary['total_files']}
Graph: {summary['unique_entities']} nodes | {summary['total_relationships']} edges
API: {summary['total_api_calls']} calls
Tokens: {summary['total_input_tokens']:,} in | {summary['total_output_tokens']:,} out
Duration: {summary['processing_time_seconds']:.1f}s
{'='*80}
"""
            logger.info(ending_banner)
            return 0
        else:
            logger.error("FAILED")
            return 1
            
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
