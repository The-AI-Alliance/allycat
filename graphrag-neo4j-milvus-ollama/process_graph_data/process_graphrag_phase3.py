"""
Phase 3: Community Summarization
Input: workspace/graph_data/phase2_output.json
Output: workspace/graph_data/phase3_output.json
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import openai

sys.path.insert(0, str(Path(__file__).parent.parent))
from my_config import MY_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CommunitySummarizer:
    def __init__(self, llm_provider: str = None, min_size: int = 2, 
                 max_communities: int = None):
        self.llm_provider = (llm_provider or MY_CONFIG.GRAPHRAG_LLM_PROVIDER).lower()
        self.min_size = min_size
        self.max_communities = max_communities
        
        if self.llm_provider == "cerebras":
            provider_config = MY_CONFIG.GRAPHRAG_PROVIDERS["cerebras"]
            api_key = getattr(MY_CONFIG, provider_config["api_key_env"], None)
            
            if not api_key:
                raise ValueError(f"{provider_config['api_key_env']} not set")
            
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url=provider_config["base_url"]
            )
            self.model = provider_config["model"]
            
        elif self.llm_provider == "nebius":
            provider_config = MY_CONFIG.GRAPHRAG_PROVIDERS["nebius"]
            api_key = getattr(MY_CONFIG, provider_config["api_key_env"], None)
            
            if not api_key:
                raise ValueError(f"{provider_config['api_key_env']} not set")
            
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url=provider_config["base_url"]
            )
            self.model = provider_config["model"]
            
        else:
            raise ValueError(f"Invalid provider: {self.llm_provider}. Choose: cerebras or nebius")
        
        logger.info(f"Initialized {self.llm_provider.upper()} with model {self.model}")
        logger.info(f"Min size: {min_size}, Max communities: {max_communities or 'all'}")
    
    def _llm_call(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert knowledge graph analyst specializing in community analysis and insight extraction."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=3000
        )
        return response.choices[0].message.content.strip()
    
    def _parse_json(self, text: str) -> Dict:
        text = text.strip()
        
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        try:
            return json.loads(text)
        except Exception as e:
            logger.error(f"JSON parse error: {e}")
            return {}
    
    def get_community_report_prompt(self, entities: List[Dict], relationships: List[Dict], 
                                   stats: Dict, community_id: str) -> str:
        entity_list = "\n".join([
            f"- {e['properties']['title']} ({', '.join(e['labels'])}): {e['properties']['content']}"
            for e in entities
        ])
        
        rel_list = "\n".join([
            f"- {r['type']}: {r['description']}"
            for r in relationships
        ])
        
        return f"""Analyze this knowledge graph community.

COMMUNITY: {community_id}
SIZE: {stats.get('size', 0)} entities
INTERNAL RELATIONSHIPS: {stats.get('internal_relationships', 0)}
EXTERNAL RELATIONSHIPS: {stats.get('external_relationships', 0)}
DENSITY: {stats.get('density', 0):.2f}
CONDUCTANCE: {stats.get('conductance', 0):.2f}

ENTITIES:
{entity_list}

RELATIONSHIPS:
{rel_list}

GENERATE JSON with these fields:
- title: Concise descriptive title (5-10 words)
- summary: 2-3 sentence overview
- full_content: Markdown report covering themes, entities, relationships, insights
- rank_explanation: Why this ranks at X/10
- rating_explanation: Impact level (CRITICAL/HIGH/MEDIUM/LOW) and reasoning
- findings: Array of 3-5 insights, each with:
  - summary: One sentence
  - explanation: 2-3 sentences with evidence
  - score: 0-10
  - finding_type: insight|trend|pattern|anomaly|contradiction

Return ONLY valid JSON."""
    
    def generate_community_report(
        self, 
        community_id: str,
        hierarchy_item: Dict,
        nodes: List[Dict],
        relationships: List[Dict],
        stats: Dict
    ) -> Dict:
        # Get entities in this community
        entity_ids = set(hierarchy_item['entity_ids'])
        community_entities = [n for n in nodes if n['id'] in entity_ids]
        
        # Get relationships in this community
        rel_ids = set(hierarchy_item['relationship_ids'])
        community_rels = [r for r in relationships if r['id'] in rel_ids]
        
        # Rank entities by centrality
        community_entities.sort(
            key=lambda e: e['properties'].get('degree_centrality', 0), 
            reverse=True
        )
        
        # Rank relationships by weight
        community_rels.sort(
            key=lambda r: r.get('weight', 1.0),
            reverse=True
        )
        
        # Generate report with LLM
        logger.info(f"Generating report for {community_id} ({len(community_entities)} entities)")
        prompt = self.get_community_report_prompt(
            community_entities, community_rels, stats, community_id
        )
        
        response = self._llm_call(prompt)
        report_data = self._parse_json(response)
        
        if not report_data:
            # Fallback if LLM fails
            report_data = {
                "title": f"Community {community_id}",
                "summary": f"A community of {len(community_entities)} entities.",
                "full_content": f"# Community {community_id}\n\nThis community contains {len(community_entities)} entities.",
                "rank_explanation": "Automated fallback report",
                "rating_explanation": "MEDIUM IMPACT: Standard community",
                "findings": []
            }
        
        # Calculate rank based on size and density
        rank = min(10.0, (len(community_entities) / 10.0) * 5 + stats.get('density', 0) * 5)
        
        # Build key entities ranked
        key_entities_ranked = []
        for i, entity in enumerate(community_entities):
            key_entities_ranked.append({
                "entity_id": entity['id'],
                "entity_title": entity['properties']['title'],
                "importance_score": float(max(0.0, 10.0 - i * 0.1))
            })
        
        # Build key relationships ranked
        key_relationships_ranked = []
        for i, rel in enumerate(community_rels):
            key_relationships_ranked.append({
                "relationship_id": rel['id'],
                "relationship_type": rel['type'],
                "importance_score": float(max(0.0, 10.0 - i * 0.1))
            })
        
        # Process findings
        findings = []
        for finding_data in report_data.get('findings', []):
            finding = {
                "summary": finding_data.get('summary', ''),
                "explanation": finding_data.get('explanation', ''),
                "score": float(finding_data.get('score', 5.0)),
                "confidence_score": 0.85,
                "source_entities": [e['id'] for e in community_entities[:5]],
                "source_relationships": [r['id'] for r in community_rels[:3]],
                "citations": [],
                "finding_type": finding_data.get('finding_type', 'insight'),
                "contradicting_evidence": [],
                "temporal_relevance": {
                    "time_period": "current",
                    "is_current": True
                }
            }
            findings.append(finding)
        
        # Build full_content_json
        full_content_json = {
            "title": report_data.get('title', f"Community {community_id}"),
            "overview": report_data.get('summary', ''),
            "key_insights": [f['summary'] for f in findings],
            "notable_entities": [e['properties']['title'] for e in community_entities]
        }
        
        # Token count estimation
        token_count = len(report_data.get('full_content', '').split())
        
        return {
            "community": community_id,
            "parent": hierarchy_item.get('parent'),
            "children": hierarchy_item.get('children', []),
            "level": hierarchy_item['level'],
            "title": report_data.get('title', f"Community {community_id}"),
            "summary": report_data.get('summary', ''),
            "full_content": report_data.get('full_content', ''),
            "rank": float(rank),
            "rank_explanation": report_data.get('rank_explanation', ''),
            "rating_explanation": report_data.get('rating_explanation', ''),
            "findings": findings,
            "full_content_json": full_content_json,
            "period": datetime.now().isoformat(),
            "size": len(community_entities),
            "generated_at": datetime.now().isoformat(),
            "report_embedding": None,
            "key_entities_ranked": key_entities_ranked,
            "key_relationships_ranked": key_relationships_ranked,
            "llm_model": self.model,
            "generation_confidence": 0.85,
            "token_count": token_count,
            "created_date": datetime.now().isoformat()
        }
    
    def run(self, input_file: str, output_file: str):
        logger.info(f"Loading {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        nodes = data['nodes']
        relationships = data['relationships']
        communities = data['communities']
        
        logger.info(f"Loaded {communities['total_communities']} total communities")
        
        # Filter by size (skip singletons)
        target_hierarchy = []
        skipped_count = 0
        
        for hierarchy_item in communities['hierarchy']:
            size = hierarchy_item.get('size', 0)
            
            if size < self.min_size:
                skipped_count += 1
                continue
            
            target_hierarchy.append(hierarchy_item)
        
        if not target_hierarchy:
            logger.error(f"No communities found with size >= {self.min_size}")
            logger.info("Try lowering min_size or check Phase 2 output")
            return
        
        logger.info(f"Processing {len(target_hierarchy)} communities (skipped {skipped_count} with size < {self.min_size})")
        
        # Limit total communities if specified
        if self.max_communities:
            target_hierarchy.sort(
                key=lambda h: h.get('size', 0),
                reverse=True
            )
            target_hierarchy = target_hierarchy[:self.max_communities]
            logger.info(f"Limited to top {self.max_communities} communities by size")
        
        # Generate reports for each community
        community_reports = []
        summaries = {}
        
        for i, hierarchy_item in enumerate(target_hierarchy, 1):
            community_id = hierarchy_item['community']
            
            logger.info(f"Processing {i}/{len(target_hierarchy)}: {community_id} (size={hierarchy_item.get('size', 0)})")
            
            report = self.generate_community_report(
                community_id,
                hierarchy_item,
                nodes,
                relationships,
                hierarchy_item
            )
            
            community_reports.append(report)
            summaries[community_id] = report['summary']
        
        # Update communities with summaries
        communities['summaries'] = summaries
        
        # Build output
        output = {
            "nodes": nodes,
            "relationships": relationships,
            "communities": communities,
            "community_reports": community_reports,
            "metadata": {
                "phase": "phase3_community_summarization",
                "total_reports": len(community_reports),
                "llm_model": self.model,
                "llm_provider": self.llm_provider,
                "created_date": datetime.now().isoformat()
            }
        }
        
        # Save output
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved to {output_file}")
        logger.info(f"Generated {len(community_reports)} community reports")


def main():
    import os
    
    llm_provider = os.getenv("GRAPH_LLM_PROVIDER", MY_CONFIG.GRAPHRAG_LLM_PROVIDER)
    min_size = int(os.getenv("GRAPH_MIN_SIZE", "2"))
    max_communities = os.getenv("GRAPH_MAX_COMMUNITIES", None)
    max_communities = int(max_communities) if max_communities else None
    
    summarizer = CommunitySummarizer(
        llm_provider=llm_provider,
        min_size=min_size,
        max_communities=max_communities
    )
    summarizer.run(
        input_file="workspace/graph_data/phase2_output.json",
        output_file="workspace/graph_data/phase3_output.json"
    )


if __name__ == "__main__":
    main()
