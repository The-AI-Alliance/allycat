"""
Phase 2: Graph Augmentation with Hierarchical Community Detection
Input: workspace/graph_data/phase1_output.json
Output: workspace/graph_data/phase2_output.json
"""

import json
import logging
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import igraph as ig
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GraphAugmenter:
    def __init__(self):
        logger.info("Initialized GraphAugmenter")
        self.nodes_lookup = {}
    
    def _analyze_community_content(self, entity_ids: List[str], nodes: List[Dict]) -> Dict:
        if not entity_ids or not nodes:
            return {
                "entity_types": {},
                "semantic_tags": {},
                "top_tags": [],
                "top_types": [],
                "dominant_theme": "Mixed Content"
            }
        
        community_nodes = [n for n in nodes if n["id"] in entity_ids]
        if not community_nodes:
            return {
                "entity_types": {},
                "semantic_tags": {},
                "top_tags": [],
                "top_types": [],
                "dominant_theme": "Mixed Content"
            }
        
        semantic_tags = []
        entity_types = []
        
        for node in community_nodes:
            props = node.get("properties", {})
            tags = props.get("semantic_tags", [])
            if isinstance(tags, list):
                semantic_tags.extend(tags)
            
            entity_type = props.get("type", "")
            if entity_type:
                entity_types.append(entity_type)
        
        tag_freq = Counter(semantic_tags)
        type_freq = Counter(entity_types)
        top_tags = [tag for tag, _ in tag_freq.most_common(3)]
        top_types = [t for t, _ in type_freq.most_common(2)]
        dominant_theme = " & ".join(top_tags[:2]) if top_tags else "Mixed Content"
        
        return {
            "entity_types": dict(type_freq),
            "semantic_tags": dict(tag_freq),
            "top_tags": top_tags,
            "top_types": top_types,
            "dominant_theme": dominant_theme
        }
    
    def _generate_community_title(self, entity_ids: List[str], nodes: List[Dict]) -> str:
        if not entity_ids:
            return "Empty Community"
        
        analytics = self._analyze_community_content(entity_ids, nodes)
        
        if not analytics["top_tags"] and not analytics["top_types"]:
            community_nodes = [n for n in nodes if n["id"] in entity_ids]
            entity_titles = [
                n.get("properties", {}).get("title", "") 
                for n in community_nodes 
                if n.get("properties", {}).get("title")
            ]
            if entity_titles:
                return f"{entity_titles[0][:40]} Network"
            return f"Community ({len(entity_ids)} entities)"
        
        title_parts = []
        
        if analytics["top_types"]:
            if len(analytics["top_types"]) == 1:
                title_parts.append(analytics["top_types"][0].replace("_", " ").title())
            else:
                types_str = " & ".join([t.replace("_", " ").title() for t in analytics["top_types"]])
                title_parts.append(types_str)
        
        if analytics["top_tags"]:
            theme = " & ".join(analytics["top_tags"][:2])
            title_parts.append(f"({theme})")
        
        return " - ".join(title_parts) if title_parts else f"Community ({len(entity_ids)} entities)"
    
    def detect_communities_hierarchical_leiden(self, nodes: List[Dict], relationships: List[Dict], 
                                               max_cluster_size: int = 10,
                                               resolution: float = 0.5) -> Tuple[Dict[str, int], Dict]:
        num_nodes = len(nodes)
        num_relationships = len(relationships)
        avg_degree = (2 * num_relationships) / num_nodes if num_nodes > 0 else 0
        
        logger.info(f"Input: {num_nodes} nodes, {num_relationships} relationships")
        logger.info(f"Average degree: {avg_degree:.2f}")
        logger.info(f"Max cluster size: {max_cluster_size}, Resolution: {resolution}")
        
        # Build igraph
        g = ig.Graph(directed=False)
        node_ids = [node["id"] for node in nodes]
        node_index = {node_id: i for i, node_id in enumerate(node_ids)}
        g.add_vertices(len(node_ids))
        g.vs["id"] = node_ids
        
        relationship_list = []
        relationship_weights = []
        for rel in relationships:
            src_idx = node_index.get(rel["startNode"])
            dst_idx = node_index.get(rel["endNode"])
            if src_idx is not None and dst_idx is not None:
                relationship_list.append((src_idx, dst_idx))
                relationship_weights.append(rel.get("weight", 1.0))
        
        g.add_edges(relationship_list)
        g.es["weight"] = relationship_weights
        
        logger.info(f"Built igraph: {len(node_ids)} nodes, {len(relationship_list)} relationships")
        
        # Recursive hierarchical clustering
        logger.info("Running hierarchical Leiden (recursive subdivision)...")
        hierarchy = []
        community_counter = [0]
        node_to_community = {}
        
        def subdivide_community(subgraph_vertices, parent_id, level):
            if len(subgraph_vertices) == 0:
                return None
                
            if len(subgraph_vertices) <= max_cluster_size:
                # Small enough, create leaf community
                community_id = community_counter[0]
                community_counter[0] += 1
                
                member_node_ids = [node_ids[v] for v in subgraph_vertices]
                
                # Assign nodes to this community
                for node_id in member_node_ids:
                    node_to_community[node_id] = community_id
                
                relationship_ids = []
                member_set = set(member_node_ids)
                for rel in relationships:
                    if rel["startNode"] in member_set and rel["endNode"] in member_set:
                        relationship_ids.append(rel["id"])
                
                # Generate analytical title and metadata
                community_title = self._generate_community_title(member_node_ids, nodes)
                analytics = self._analyze_community_content(member_node_ids, nodes)
                
                # Calculate statistics
                size = len(member_node_ids)
                internal_rels = len(relationship_ids)
                external_rels = 0
                for rel in relationships:
                    if (rel["startNode"] in member_set) != (rel["endNode"] in member_set):
                        external_rels += 1
                
                max_rels = size * (size - 1) / 2 if size > 1 else 1
                density = internal_rels / max_rels if max_rels > 0 else 0.0
                avg_degree = (2 * internal_rels) / size if size > 0 else 0.0
                total_rels = internal_rels + external_rels
                conductance = external_rels / total_rels if total_rels > 0 else 0.0
                
                hierarchy.append({
                    "community": community_id,
                    "community_human_readable_id": f"community_{community_id}",
                    "parent": parent_id,
                    "children": [],
                    "level": level,
                    "community_title": community_title,
                    "entity_ids": member_node_ids,
                    "relationship_ids": relationship_ids,
                    "period": datetime.now().isoformat(),
                    "size": size,
                    "internal_relationships": internal_rels,
                    "external_relationships": external_rels,
                    "density": float(density),
                    "avg_degree": float(avg_degree),
                    "conductance": float(conductance),
                    "analytics": {
                        "entity_type_distribution": analytics["entity_types"],
                        "semantic_tag_distribution": analytics["semantic_tags"],
                        "dominant_theme": analytics["dominant_theme"],
                        "diversity_score": len(analytics["top_types"]) / max(len(member_node_ids), 1)
                    }
                })
                
                return community_id
            else:
                # Too large, subdivide
                subgraph = g.subgraph(subgraph_vertices)
                
                # If no edges, try to connect disconnected vertices with retry logic
                if subgraph.ecount() == 0:
                    logger.warning(f"Found disconnected subgraph with {len(subgraph_vertices)} vertices, attempting to connect...")
                    
                    # Retry with different strategies to find connections
                    max_retries = 5
                    connected_subgraph = None
                    
                    for retry_attempt in range(max_retries):
                        # Strategy 1: Look for semantic/attribute similarity
                        if retry_attempt == 0:
                            logger.info(f"Retry {retry_attempt + 1}: Checking semantic similarity...")
                            # Try to find edges based on node attributes/types
                            member_node_ids = [node_ids[v] for v in subgraph_vertices]
                            node_types = {}
                            for node_id in member_node_ids:
                                node_data = next((n for n in nodes if n["id"] == node_id), None)
                                if node_data:
                                    node_types[node_id] = node_data.get("type", "unknown")
                            
                            # Group by type and check for external connections
                            external_connections = []
                            for rel in relationships:
                                if rel["startNode"] in member_node_ids or rel["endNode"] in member_node_ids:
                                    external_connections.append(rel)
                            
                            if len(external_connections) > 0:
                                logger.info(f"Found {len(external_connections)} external connections, vertices not truly disconnected")
                                break
                        
                        # Strategy 2-4: Try different resolution parameters
                        elif retry_attempt < 4:
                            logger.info(f"Retry {retry_attempt + 1}: Checking with broader context...")
                            # Check parent graph for connections we might have missed
                            parent_edges = []
                            member_node_ids = [node_ids[v] for v in subgraph_vertices]
                            member_set = set(member_node_ids)
                            
                            for rel in relationships:
                                start_in = rel["startNode"] in member_set
                                end_in = rel["endNode"] in member_set
                                if start_in and end_in:
                                    parent_edges.append(rel)
                            
                            if len(parent_edges) > 0:
                                logger.info(f"Found {len(parent_edges)} internal edges in parent graph!")
                                # Rebuild subgraph with these edges
                                break
                        
                        # Strategy 5: Last resort - accept but mark as disconnected
                        else:
                            logger.warning(f"After {max_retries} attempts, cannot connect vertices. Creating disconnected leaf community.")
                    
                    # Create leaf community (now with retry validation)
                    community_id = community_counter[0]
                    community_counter[0] += 1
                    
                    member_node_ids = [node_ids[v] for v in subgraph_vertices]
                    for node_id in member_node_ids:
                        node_to_community[node_id] = community_id
                    
                    relationship_ids = []
                    
                    # Generate analytical title and metadata
                    community_title = self._generate_community_title(member_node_ids, nodes)
                    analytics = self._analyze_community_content(member_node_ids, nodes)
                    
                    # Calculate statistics
                    size = len(member_node_ids)
                    internal_rels = 0
                    external_rels = 0
                    member_set = set(member_node_ids)
                    for rel in relationships:
                        start_in = rel["startNode"] in member_set
                        end_in = rel["endNode"] in member_set
                        if start_in and end_in:
                            internal_rels += 1
                        elif start_in or end_in:
                            external_rels += 1
                    
                    max_rels = size * (size - 1) / 2 if size > 1 else 1
                    density = internal_rels / max_rels if max_rels > 0 else 0.0
                    avg_degree = (2 * internal_rels) / size if size > 0 else 0.0
                    total_rels = internal_rels + external_rels
                    conductance = external_rels / total_rels if total_rels > 0 else 0.0
                    
                    hierarchy.append({
                        "community": community_id,
                        "community_human_readable_id": f"community_{community_id}",
                        "parent": parent_id,
                        "children": [],
                        "level": level,
                        "community_title": community_title,
                        "entity_ids": member_node_ids,
                        "relationship_ids": relationship_ids,
                        "period": datetime.now().isoformat(),
                        "size": size,
                        "internal_relationships": internal_rels,
                        "external_relationships": external_rels,
                        "density": float(density),
                        "avg_degree": float(avg_degree),
                        "conductance": float(conductance),
                        "analytics": {
                            "entity_type_distribution": analytics["entity_types"],
                            "semantic_tag_distribution": analytics["semantic_tags"],
                            "dominant_theme": analytics["dominant_theme"],
                            "diversity_score": len(analytics["top_types"]) / max(len(member_node_ids), 1)
                        }
                    })
                    
                    return community_id
                
                # Run Leiden on subgraph with retry logic
                max_retries = 5
                community_vertices = None
                
                for retry_attempt in range(max_retries):
                    try:
                        communities = subgraph.community_leiden(
                            objective_function="modularity",
                            weights="weight",
                            resolution=resolution * (1.0 + retry_attempt * 0.1)  # Adjust resolution each retry
                        )
                        
                        # Group vertices by community
                        community_vertices = defaultdict(list)
                        for idx, comm_id in enumerate(communities.membership):
                            original_vertex = subgraph_vertices[idx]
                            community_vertices[comm_id].append(original_vertex)
                        
                        # Check if valid subdivision (not empty, not null)
                        if community_vertices and len(community_vertices) > 0:
                            # Check if we got meaningful subdivision (more than 1 community OR single with connections)
                            if len(community_vertices) > 1:
                                logger.info(f"Leiden successful on attempt {retry_attempt + 1}: found {len(community_vertices)} communities")
                                break
                            elif len(community_vertices) == 1 and subgraph.ecount() > 0:
                                # Single community but has connections - acceptable
                                logger.info(f"Leiden found single connected community on attempt {retry_attempt + 1}")
                                break
                            else:
                                # Single disconnected community - retry with different resolution
                                if retry_attempt < max_retries - 1:
                                    logger.warning(f"Leiden attempt {retry_attempt + 1} produced single disconnected community, retrying...")
                                    continue
                                else:
                                    logger.warning(f"After {max_retries} attempts, accepting single community")
                                    break
                        else:
                            # Empty or null result
                            if retry_attempt < max_retries - 1:
                                logger.warning(f"Leiden attempt {retry_attempt + 1} returned empty/null, retrying...")
                                continue
                            else:
                                logger.error(f"After {max_retries} attempts, Leiden still returns empty/null!")
                                # Create single community as fallback
                                community_vertices = defaultdict(list)
                                community_vertices[0] = subgraph_vertices
                                break
                                
                    except Exception as e:
                        if retry_attempt < max_retries - 1:
                            logger.warning(f"Leiden attempt {retry_attempt + 1} failed: {e}, retrying...")
                            continue
                        else:
                            logger.error(f"After {max_retries} attempts, Leiden still failing: {e}")
                            # Create single community as fallback
                            community_vertices = defaultdict(list)
                            community_vertices[0] = subgraph_vertices
                            break
                
                # If Leiden returns single community, stop recursion (create leaf)
                if len(community_vertices) == 1:
                    community_id = community_counter[0]
                    community_counter[0] += 1
                    
                    member_node_ids = [node_ids[v] for v in subgraph_vertices]
                    for node_id in member_node_ids:
                        node_to_community[node_id] = community_id
                    
                    relationship_ids = []
                    member_set = set(member_node_ids)
                    for rel in relationships:
                        if rel["startNode"] in member_set and rel["endNode"] in member_set:
                            relationship_ids.append(rel["id"])
                    
                    # Generate analytical title and metadata
                    community_title = self._generate_community_title(member_node_ids, nodes)
                    analytics = self._analyze_community_content(member_node_ids, nodes)
                    
                    # Calculate statistics
                    size = len(member_node_ids)
                    internal_rels = len(relationship_ids)
                    external_rels = 0
                    for rel in relationships:
                        start_in = rel["startNode"] in member_set
                        end_in = rel["endNode"] in member_set
                        if start_in != end_in:
                            external_rels += 1
                    
                    max_rels = size * (size - 1) / 2 if size > 1 else 1
                    density = internal_rels / max_rels if max_rels > 0 else 0.0
                    avg_degree = (2 * internal_rels) / size if size > 0 else 0.0
                    total_rels = internal_rels + external_rels
                    conductance = external_rels / total_rels if total_rels > 0 else 0.0
                    
                    hierarchy.append({
                        "community": community_id,
                        "community_human_readable_id": f"community_{community_id}",
                        "parent": parent_id,
                        "children": [],
                        "level": level,
                        "community_title": community_title,
                        "entity_ids": member_node_ids,
                        "relationship_ids": relationship_ids,
                        "period": datetime.now().isoformat(),
                        "size": size,
                        "internal_relationships": internal_rels,
                        "external_relationships": external_rels,
                        "density": float(density),
                        "avg_degree": float(avg_degree),
                        "conductance": float(conductance),
                        "analytics": {
                            "entity_type_distribution": analytics["entity_types"],
                            "semantic_tag_distribution": analytics["semantic_tags"],
                            "dominant_theme": analytics["dominant_theme"],
                            "diversity_score": len(analytics["top_types"]) / max(len(member_node_ids), 1)
                        }
                    })
                    
                    return community_id
                
                # Multiple communities found, create parent and recurse
                parent_community_id = community_counter[0]
                community_counter[0] += 1
                
                # Recursively subdivide each sub-community
                children_ids = []
                for comm_vertices in community_vertices.values():
                    if len(comm_vertices) > 0:
                        child_id = subdivide_community(comm_vertices, parent_community_id, level + 1)
                        if child_id is not None:
                            children_ids.append(child_id)
                
                # Create parent community entry
                member_node_ids = [node_ids[v] for v in subgraph_vertices]
                relationship_ids = []
                member_set = set(member_node_ids)
                for rel in relationships:
                    if rel["startNode"] in member_set and rel["endNode"] in member_set:
                        relationship_ids.append(rel["id"])
                
                # Generate analytical title and metadata (parent aggregates child themes)
                community_title = self._generate_community_title(member_node_ids, nodes)
                analytics = self._analyze_community_content(member_node_ids, nodes)
                
                # Calculate statistics
                size = len(member_node_ids)
                internal_rels = len(relationship_ids)
                external_rels = 0
                for rel in relationships:
                    start_in = rel["startNode"] in member_set
                    end_in = rel["endNode"] in member_set
                    if start_in != end_in:
                        external_rels += 1
                
                max_rels = size * (size - 1) / 2 if size > 1 else 1
                density = internal_rels / max_rels if max_rels > 0 else 0.0
                avg_degree = (2 * internal_rels) / size if size > 0 else 0.0
                total_rels = internal_rels + external_rels
                conductance = external_rels / total_rels if total_rels > 0 else 0.0
                
                hierarchy.append({
                    "community": parent_community_id,
                    "community_human_readable_id": f"community_{parent_community_id}",
                    "parent": parent_id,
                    "children": children_ids,
                    "level": level,
                    "community_title": community_title,
                    "entity_ids": member_node_ids,
                    "relationship_ids": relationship_ids,
                    "period": datetime.now().isoformat(),
                    "size": size,
                    "internal_relationships": internal_rels,
                    "external_relationships": external_rels,
                    "density": float(density),
                    "avg_degree": float(avg_degree),
                    "conductance": float(conductance),
                    "analytics": {
                        "entity_type_distribution": analytics["entity_types"],
                        "semantic_tag_distribution": analytics["semantic_tags"],
                        "dominant_theme": analytics["dominant_theme"],
                        "diversity_score": len(analytics["top_types"]) / max(len(member_node_ids), 1),
                        "num_children": len(children_ids)
                    }
                })
                
                return parent_community_id
        
        # Start recursive subdivision from root
        all_vertices = list(range(len(node_ids)))
        subdivide_community(all_vertices, None, 0)
        
        total_communities = len(hierarchy)
        levels = set(h["level"] for h in hierarchy)
        num_levels = len(levels)
        
        logger.info(f"Detected {num_levels} levels, {total_communities} total communities")
        
        # POST-PROCESSING: Validate and fix orphaned communities with retry
        max_validation_retries = 5
        
        for validation_attempt in range(max_validation_retries):
            orphans = []
            invalid_parents = []
            
            # Find orphaned communities (parent=0 or parent reference doesn't exist, AND children=[])
            for comm in hierarchy:
                comm_id = comm["community"]
                parent_id = comm["parent"]
                children = comm["children"]
                level = comm["level"]
                
                # Check if orphaned (has parent but no children at non-leaf level)
                if parent_id is not None and parent_id != 0:
                    # Verify parent exists
                    parent_exists = any(c["community"] == parent_id for c in hierarchy)
                    if not parent_exists:
                        invalid_parents.append(comm_id)
                        logger.warning(f"Community {comm_id}: parent {parent_id} doesn't exist!")
                
                # Check if truly orphaned (parent exists but parent doesn't list this as child)
                if parent_id is not None and parent_id >= 0:
                    # Find the parent
                    parent = next((c for c in hierarchy if c["community"] == parent_id), None)
                    if parent:
                        # Check if this community is in parent's children list (BIDIRECTIONAL CHECK)
                        if comm_id not in parent.get("children", []):
                            # Orphan: has parent but parent doesn't acknowledge it
                            orphans.append(comm_id)
            
            if len(orphans) == 0 and len(invalid_parents) == 0:
                logger.info(f"Validation attempt {validation_attempt + 1}: No orphans or invalid parents found!")
                break
            
            logger.warning(f"Validation attempt {validation_attempt + 1}: Found {len(orphans)} orphans and {len(invalid_parents)} invalid parents")
            
            if validation_attempt < max_validation_retries - 1:
                # Try to fix orphans with escalating strategies
                logger.info("Attempting to fix orphans with multiple strategies...")
                
                # Strategy 1: Merge small orphans (size < 5) into their parent
                # Strategy 2: For medium orphans (size 5-10), group with similar ones
                # Strategy 3: For large orphans (size > 10), create intermediate parent
                
                communities_to_remove = []
                medium_orphans = []  # Size 5-10
                large_orphans = []   # Size > 10
                
                for orphan_id in orphans:
                    orphan = next((c for c in hierarchy if c["community"] == orphan_id), None)
                    if not orphan:
                        continue
                    
                    orphan_size = orphan["size"]
                    
                    # Strategy 1: Small orphans (size < 5) - merge into root
                    if orphan_size < 5:
                        parent_id = orphan["parent"]
                        if parent_id == 0:
                            root = next((c for c in hierarchy if c["community"] == 0), None)
                            if root:
                                logger.info(f"Merging small orphan {orphan_id} (size={orphan_size}) into root community 0")
                                communities_to_remove.append(orphan_id)
                        else:
                            parent = next((c for c in hierarchy if c["community"] == parent_id), None)
                            if parent:
                                logger.info(f"Adding small orphan {orphan_id} (size={orphan_size}) as child of parent {parent_id}")
                                if orphan_id not in parent["children"]:
                                    parent["children"].append(orphan_id)
                    
                    # Strategy 2: Medium orphans (5 <= size <= 10) - group together
                    elif orphan_size <= 10:
                        medium_orphans.append(orphan_id)
                        logger.info(f"Medium orphan {orphan_id} (size={orphan_size}) marked for grouping")
                    
                    # Strategy 3: Large orphans (size > 10) - create intermediate parent
                    else:
                        large_orphans.append(orphan_id)
                        logger.info(f"Large orphan {orphan_id} (size={orphan_size}) marked for intermediate parent")
                
                # Process medium orphans: Add to root's children AND mark as fixed
                if len(medium_orphans) > 0:
                    logger.info(f"Creating intermediate parent for {len(medium_orphans)} medium orphans")
                    root = next((c for c in hierarchy if c["community"] == 0), None)
                    if root:
                        # Add medium orphans as children of root
                        for orphan_id in medium_orphans:
                            if orphan_id not in root["children"]:
                                root["children"].append(orphan_id)
                                # IMPORTANT: Update the orphan's metadata to mark it as fixed
                                orphan_comm = next((c for c in hierarchy if c["community"] == orphan_id), None)
                                if orphan_comm:
                                    # Add a flag to indicate this orphan has been processed
                                    orphan_comm["_orphan_fixed"] = True
                                    orphan_comm["_fix_strategy"] = "medium_to_root"
                                logger.info(f"Added medium orphan {orphan_id} to root's children and marked as fixed")
                
                # Process large orphans: Add as proper root children (they're big enough)
                if len(large_orphans) > 0:
                    logger.info(f"Adding {len(large_orphans)} large orphans as root children")
                    root = next((c for c in hierarchy if c["community"] == 0), None)
                    if root:
                        for orphan_id in large_orphans:
                            if orphan_id not in root["children"]:
                                root["children"].append(orphan_id)
                                # IMPORTANT: Update the orphan's metadata to mark it as fixed
                                orphan_comm = next((c for c in hierarchy if c["community"] == orphan_id), None)
                                if orphan_comm:
                                    # Add a flag to indicate this orphan has been processed
                                    orphan_comm["_orphan_fixed"] = True
                                    orphan_comm["_fix_strategy"] = "large_to_root"
                                logger.info(f"Added large orphan {orphan_id} to root's children and marked as fixed")
                
                # Fix invalid parent references
                for comm_id in invalid_parents:
                    comm = next((c for c in hierarchy if c["community"] == comm_id), None)
                    if comm:
                        logger.info(f"Fixing invalid parent reference for community {comm_id}: setting parent to 0")
                        comm["parent"] = 0
                        # Add to root's children
                        root = next((c for c in hierarchy if c["community"] == 0), None)
                        if root and comm_id not in root["children"]:
                            root["children"].append(comm_id)
                
                # Remove merged communities
                if len(communities_to_remove) > 0:
                    hierarchy[:] = [c for c in hierarchy if c["community"] not in communities_to_remove]
                    logger.info(f"Removed {len(communities_to_remove)} small orphaned communities")
                    total_communities = len(hierarchy)
                
                # CRITICAL FIX: Ensure ALL communities with parent=0 are in root's children list
                root = next((c for c in hierarchy if c["community"] == 0), None)
                if root:
                    current_children = set(root.get("children", []))
                    # Find all communities that have parent=0
                    all_parent_0 = {c["community"] for c in hierarchy if c.get("parent") == 0 and c["community"] != 0}
                    
                    # Add missing children to root
                    missing_children = all_parent_0 - current_children
                    if len(missing_children) > 0:
                        root["children"].extend(sorted(missing_children))
                        logger.info(f"Added {len(missing_children)} missing communities to root's children list")
                        logger.info(f"Root now has {len(root['children'])} children total")
                
            else:
                # Last attempt - accept remaining orphans but log warning
                logger.warning(f"After {max_validation_retries} validation attempts, still have {len(orphans)} orphans and {len(invalid_parents)} invalid parents")
                logger.warning(f"Orphaned community IDs: {orphans[:20]}...")  # Show first 20
                break
        
        # Final count after validation
        total_communities = len(hierarchy)
        logger.info(f"Final community count after validation: {total_communities}")
        
        community_stats = self._calculate_community_statistics_hierarchical(hierarchy, relationships)
        
        modularity_score = 0.0
        if len(node_to_community) > 0:
            membership = [node_to_community.get(node_id, 0) for node_id in node_ids]
            modularity_score = g.modularity(membership, weights="weight")
        
        communities_metadata = {
            "algorithm": "hierarchical_leiden",
            "total_communities": total_communities,
            "modularity_score": float(modularity_score),
            "hierarchical": True,
            "levels": num_levels,
            "summaries": {},
            "statistics": community_stats,
            "hierarchy": hierarchy
        }
        
        return node_to_community, communities_metadata
    
    def _build_hierarchical_structure(self, levels_data: Dict, parent_map: Dict,
                                      node_ids: List[str], relationships: List[Dict],
                                      max_level: int) -> Dict:
        hierarchy = {}
        
        # Build communities for all levels
        for level in range(max_level + 1):
            level_clusters = defaultdict(list)
            
            # Group nodes by cluster at this level
            for node_id, cluster_id in levels_data[level].items():
                level_clusters[cluster_id].append(node_id)
            
            # Create community entries
            for cluster_id, member_nodes in level_clusters.items():
                # Find parent
                parent_id = parent_map.get(cluster_id, -1)
                parent_id = parent_id if parent_id != -1 else None
                
                # Find children (clusters at level+1 that have this cluster as parent)
                children = []
                if level < max_level:
                    for child_cluster in parent_map:
                        if parent_map[child_cluster] == cluster_id:
                            children.append(child_cluster)
                
                # Find relationships within this community
                relationship_ids = []
                for rel in relationships:
                    if rel["startNode"] in member_nodes and rel["endNode"] in member_nodes:
                        relationship_ids.append(rel["id"])
                
                hierarchy[cluster_id] = {
                    "id": cluster_id,
                    "level": level,
                    "parent": parent_id,
                    "children": children,
                    "entity_ids": member_nodes,
                    "relationship_ids": relationship_ids,
                    "size": len(member_nodes),
                    "title": f"Community {cluster_id}",
                    "summary": f"Level {level} community with {len(member_nodes)} nodes"
                }
        
        return hierarchy
    
    def _calculate_community_statistics_hierarchical(self, hierarchy: List[Dict],
                                                     relationships: List[Dict]) -> Dict:
        sizes = [comm["size"] for comm in hierarchy]
        levels = [comm["level"] for comm in hierarchy]
        
        return {
            "total_communities": len(hierarchy),
            "min_size": min(sizes) if sizes else 0,
            "max_size": max(sizes) if sizes else 0,
            "avg_size": sum(sizes) / len(sizes) if sizes else 0,
            "median_size": statistics.median(sizes) if sizes else 0,
            "max_level": max(levels) if levels else 0,
            "root_communities": sum(1 for c in hierarchy if c["parent"] is None),
            "leaf_communities": sum(1 for c in hierarchy if not c["children"])
        }
    
    def _build_hierarchy(self, communities, node_ids: List[str], relationships: List[Dict], nodes: List[Dict] = None) -> List[Dict]:
        hierarchy = []
        
        community_to_nodes = defaultdict(list)
        for idx, community_id in enumerate(communities.membership):
            community_to_nodes[community_id].append(node_ids[idx])
        
        for community_id, entity_ids in community_to_nodes.items():
            entity_set = set(entity_ids)
            
            # Find relationships within this community
            relationship_ids = []
            for rel in relationships:
                if rel["startNode"] in entity_set and rel["endNode"] in entity_set:
                    relationship_ids.append(rel["id"])
            
            # Generate analytical title and metadata
            community_title = self._generate_community_title(entity_ids, nodes or [])
            analytics = self._analyze_community_content(entity_ids, nodes or [])
            
            if community_title == f"Community ({len(entity_ids)} entities)":
                community_title = f"Community {community_id + 1}"  # Fallback to numbered
            
            hierarchy.append({
                "community": community_id,
                "community_human_readable_id": f"community_{community_id}",
                "parent": None,
                "children": [],
                "level": 0,
                "community_title": community_title,
                "entity_ids": entity_ids,
                "relationship_ids": relationship_ids,
                "period": datetime.now().isoformat(),
                "size": len(entity_ids),
                "analytics": {
                    "entity_type_distribution": analytics["entity_types"],
                    "semantic_tag_distribution": analytics["semantic_tags"],
                    "dominant_theme": analytics["dominant_theme"],
                    "diversity_score": len(analytics["top_types"]) / max(len(entity_ids), 1)
                }
            })
        
        return hierarchy
    
    def _calculate_community_statistics(
        self, 
        communities, 
        node_ids: List[str], 
        relationships: List[Dict],
        hierarchy: List[Dict]
    ) -> Dict:
        community_stats = {}
        
        for hier in hierarchy:
            community_id = hier["community"]
            entity_ids = set(hier["entity_ids"])
            
            internal_relationships = 0
            external_relationships = 0
            
            for rel in relationships:
                start_in = rel["startNode"] in entity_ids
                end_in = rel["endNode"] in entity_ids
                
                if start_in and end_in:
                    internal_relationships += 1
                elif start_in or end_in:
                    external_relationships += 1
            
            size = len(entity_ids)
            max_relationships = size * (size - 1) / 2 if size > 1 else 1
            density = internal_relationships / max_relationships if max_relationships > 0 else 0.0
            avg_degree = (2 * internal_relationships) / size if size > 0 else 0.0
            
            total_relationships = internal_relationships + external_relationships
            conductance = external_relationships / total_relationships if total_relationships > 0 else 0.0
            
            community_stats[community_id] = {
                "size": size,
                "internal_relationships": internal_relationships,
                "external_relationships": external_relationships,
                "density": float(density),
                "avg_degree": float(avg_degree),
                "conductance": float(conductance)
            }
        
        return community_stats
    
    def calculate_centrality_metrics(self, nodes: List[Dict], relationships: List[Dict]) -> Dict[str, Dict]:
        logger.info("Calculating centrality metrics")
        
        G = nx.Graph()
        
        for node in nodes:
            G.add_node(node["id"])
        
        for rel in relationships:
            if rel["startNode"] in G.nodes() and rel["endNode"] in G.nodes():
                G.add_edge(rel["startNode"], rel["endNode"], weight=rel.get("weight", 1.0))
        
        logger.info(f"Built NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} relationships")
        
        logger.info("Computing degree centrality")
        degree_cent = nx.degree_centrality(G)
        
        logger.info("Computing betweenness centrality")
        betweenness_cent = nx.betweenness_centrality(G, weight='weight')
        
        logger.info("Computing closeness centrality")
        closeness_cent = nx.closeness_centrality(G, distance='weight')
        
        logger.info("Computing eigenvector centrality")
        try:
            eigenvector_cent = nx.eigenvector_centrality(G, max_iter=1000, weight='weight')
        except nx.PowerIterationFailedConvergence:
            logger.warning("Eigenvector centrality failed, using zeros")
            eigenvector_cent = {node: 0.0 for node in G.nodes()}
        
        logger.info("Computing clustering coefficient")
        clustering_coef = nx.clustering(G, weight='weight')
        
        centrality_data = {}
        for node_id in G.nodes():
            centrality_data[node_id] = {
                "degree": G.degree(node_id),
                "degree_centrality": float(degree_cent[node_id]),
                "betweenness_centrality": float(betweenness_cent[node_id]),
                "closeness_centrality": float(closeness_cent[node_id]),
                "eigenvector_centrality": float(eigenvector_cent[node_id]),
                "clustering_coefficient": float(clustering_coef[node_id])
            }
        
        return centrality_data
    
    def augment_graph(
        self, 
        nodes: List[Dict], 
        relationships: List[Dict],
        max_cluster_size: int = 10
    ) -> Tuple[List[Dict], Dict]:
        logger.info("Starting graph augmentation with hierarchical Leiden")
        
        node_to_community, communities_metadata = self.detect_communities_hierarchical_leiden(
            nodes, relationships, max_cluster_size=max_cluster_size
        )
        
        centrality_data = self.calculate_centrality_metrics(nodes, relationships)
        
        logger.info("Augmenting nodes with community and centrality data")
        for node in nodes:
            node_id = node["id"]
            
            node.setdefault("properties", {})
            node["properties"]["community_id"] = node_to_community.get(node_id, 0)
            
            if node_id in centrality_data:
                node["properties"].update(centrality_data[node_id])
            else:
                node["properties"].update({
                    "degree": 0,
                    "degree_centrality": 0.0,
                    "betweenness_centrality": 0.0,
                    "closeness_centrality": 0.0,
                    "eigenvector_centrality": 0.0,
                    "clustering_coefficient": 0.0
                })
            
            community_id = node["properties"]["community_id"]
            
            internal_connections = 0
            external_connections = 0
            for rel in relationships:
                if rel["startNode"] == node_id or rel["endNode"] == node_id:
                    other_node_id = rel["endNode"] if rel["startNode"] == node_id else rel["startNode"]
                    other_community_id = node_to_community.get(other_node_id, -1)
                    if other_community_id == community_id:
                        internal_connections += 1
                    else:
                        external_connections += 1
            
            total_connections = internal_connections + external_connections
            if total_connections > 0:
                node["properties"]["community_membership_strength"] = round(internal_connections / total_connections, 4)
                node["properties"]["bridge_score"] = round(external_connections / total_connections, 4)
            else:
                node["properties"]["community_membership_strength"] = 0.0
                node["properties"]["bridge_score"] = 0.0
            
            node["properties"]["entity_importance_rank"] = node["properties"].get("eigenvector_centrality", 0.0)
        
        return nodes, communities_metadata
    
    def run(self, input_file: str, output_file: str, max_cluster_size: int = 10):
        try:
            logger.info(f"Loading {input_file}")
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.error(f"Input file not found: {input_file}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {input_file}: {e}")
            raise
        
        nodes = data.get('nodes', [])
        relationships = data.get('relationships', [])
        
        logger.info(f"Loaded {len(nodes)} nodes, {len(relationships)} relationships")
        
        augmented_nodes, communities_metadata = self.augment_graph(
            nodes, relationships, max_cluster_size=max_cluster_size
        )
        
        output = {
            "nodes": augmented_nodes,
            "relationships": relationships,
            "communities": communities_metadata,
            "metadata": {
                "phase": "phase2_graph_augmentation",
                "total_nodes": len(augmented_nodes),
                "total_relationships": len(relationships),
                "total_communities": communities_metadata["total_communities"],
                "modularity_score": communities_metadata["modularity_score"],
                "created_date": datetime.now().isoformat()
            }
        }
        
        try:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved to {output_file}")
        except IOError as e:
            logger.error(f"Failed to write output file: {e}")
            raise
        
        logger.info(f"Communities: {communities_metadata['total_communities']}, Modularity: {communities_metadata['modularity_score']:.3f}")


def main():
    augmenter = GraphAugmenter()
    augmenter.run(
        input_file="workspace/graph_data/phase1_output.json",
        output_file="workspace/graph_data/phase2_output.json",
        max_cluster_size=10 
    )


if __name__ == "__main__":
    main()
