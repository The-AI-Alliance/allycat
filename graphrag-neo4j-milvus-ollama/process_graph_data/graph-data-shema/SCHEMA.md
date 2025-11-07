# AllyCAT GraphRAG Data Schema

**Version:** 1.0 | **Type:** JSON | **Updated:** October 26, 2025

---

## 1. Nodes

| Attribute | Type | Description |
|-----------|------|-------------|
| **id** | string | UUID v4 identifier (required) |
| **labels** | array[string] | Entity type classification (required) |
| **properties** | object | Entity properties (required, see below) |

### Node Properties

| Attribute | Type | Description |
|-----------|------|-------------|
| **title** | string | Entity name (required) |
| **content** | string | Entity description (required) |
| semantic_tags | array[string] | Semantic classification tags |
| source | string | Source document reference |
| confidence | number | Extraction confidence (0-1) |
| created_date | string | ISO 8601 timestamp |
| extraction_method | string | LLM/method identifier |
| community_id | integer | Community membership ID |
| degree | integer | Node degree count |
| degree_centrality | number | Normalized degree centrality (0-1) |
| betweenness_centrality | number | Betweenness centrality (0-1) |
| closeness_centrality | number | Closeness centrality (0-1) |
| eigenvector_centrality | number | Eigenvector centrality (0-1) |
| clustering_coefficient | number | Local clustering coefficient (0-1) |
| community_membership_strength | number | Intra-community centrality (0-1) |
| bridge_score | number | Inter-community bridging (0-1) |
| entity_importance_rank | number | Global importance ranking |
| embedding_vector | array[number] | 384-dimensional embedding vector |
| embedding_model_version | string | Embedding model identifier |
| data_quality_score | number | Quality metric (0-1) |

---

## 2. Relationships

| Attribute | Type | Description |
|-----------|------|-------------|
| **id** | string | UUID v4 identifier (required) |
| **startNode** | string | Source entity UUID (required) |
| **endNode** | string | Target entity UUID (required) |
| **type** | string | Relationship type (required) |
| description | string | Relationship description |
| evidence | string | Supporting text evidence |
| confidence | number | Extraction confidence (0-1) |
| weight | number | Relationship strength for algorithms |
| source_file | string | Source document reference |

---

## 3. Communities

| Attribute | Type | Description |
|-----------|------|-------------|
| **algorithm** | string | Community detection algorithm (required) |
| **total_communities** | integer | Total community count (required) |
| modularity_score | number | Graph modularity quality (0-1) |
| **hierarchical** | boolean | Hierarchy built flag (required) |
| **levels** | integer | Hierarchy depth (required) |
| summaries | object | Community ID to title mapping |
| statistics | object | Community statistics |
| **hierarchy** | array[object] | Hierarchy items (required) |

### Communities Statistics

| Attribute | Type | Description |
|-----------|------|-------------|
| total_communities | integer | Total community count |
| min_size | integer | Smallest community size |
| max_size | integer | Largest community size |
| avg_size | number | Average community size |
| median_size | number | Median community size |
| max_level | integer | Maximum hierarchy level |
| root_communities | integer | Root-level community count |
| leaf_communities | integer | Leaf-level community count |

### Hierarchy Item

| Attribute | Type | Description |
|-----------|------|-------------|
| community | integer | Community ID |
| community_human_readable_id | string | Human-readable ID |
| parent | integer/null | Parent community ID |
| children | array[integer] | Child community IDs |
| level | integer | Hierarchy level (0=root) |
| community_title | string | Community title/theme |
| entity_ids | array[string] | Member entity UUIDs |
| relationship_ids | array[string] | Member relationship UUIDs |
| period | string | Timestamp |
| size | integer | Entity count |
| internal_relationships | integer | Internal edge count |
| external_relationships | integer | External edge count |
| density | number | Community density |
| avg_degree | number | Average node degree |
| conductance | number | Community conductance |
| analytics | object | Community analytics |

### Analytics Object

| Attribute | Type | Description |
|-----------|------|-------------|
| entity_type_distribution | object | Entity type frequency map |
| semantic_tag_distribution | object | Semantic tag frequency map |
| dominant_theme | string | Primary community theme |
| diversity_score | number | Content diversity metric |

---

## 4. Community Reports

| Attribute | Type | Description |
|-----------|------|-------------|
| community | integer | Community ID |
| parent | integer/null | Parent community ID |
| children | array[integer] | Child community IDs |
| level | integer | Hierarchy level |
| title | string | Report title |
| summary | string | Brief summary (2-3 sentences) |
| full_content | string | Detailed narrative analysis |
| rank | number | Importance score (0-10) |
| rank_explanation | string | Ranking justification |
| rating_explanation | string | Rating context |
| findings | array[object] | Key insights |
| full_content_json | object | Raw LLM response |
| period | string | Generation timestamp |
| size | integer | Community size |
| generated_at | string | Report timestamp |
| report_embedding | array[number]/null | Report embedding vector (384 dims) |
| key_entities_ranked | array[object] | Top entities with centrality |
| key_relationships_ranked | array[object] | Top relationships |
| llm_model | string | LLM model identifier |
| generation_confidence | number | Generation confidence (0-1) |
| token_count | integer | Token count |
| created_date | string | Creation date |

### Finding Object

| Attribute | Type | Description |
|-----------|------|-------------|
| summary | string | One-sentence finding |
| explanation | string | 2-3 sentence justification |

### Ranked Entity Object

| Attribute | Type | Description |
|-----------|------|-------------|
| entity_id | string | Entity UUID |
| title | string | Entity name |
| degree_centrality | number | Centrality score (0-1) |

### Ranked Relationship Object

| Attribute | Type | Description |
|-----------|------|-------------|
| source | string | Source entity name |
| target | string | Target entity name |
| type | string | Relationship type |
| weight | number | Relationship weight |

---

## 5. Metadata

| Attribute | Type | Description |
|-----------|------|-------------|
| **phase** | string | Pipeline phase identifier (required) |
| total_reports | integer | Community report count |
| llm_model | string | LLM model identifier |
| llm_provider | string | LLM provider name |
| created_date | string | Processing start timestamp (ISO 8601) |
| embedding_model | string | Embedding model identifier |
| embedding_dimensions | integer | Embedding vector dimensions |
| **generated_at** | string | File generation timestamp (ISO 8601, required) |

---

## 6. DRIFT Search Metadata

| Index Type | Description |
|------------|-------------|
| configuration | Core DRIFT settings |
| query_routing_config | 3-tier query routing (lightweight/standard/comprehensive) |
| community_search_index | Community lookup index |
| entity_search_index | Entity lookup by title/type/community |
| relationship_search_index | Relationship lookup index |
| vector_index_config | HNSW vector similarity index |
| fulltext_index_config | Fulltext search index |
| composite_index_config | Multi-field query index |
| range_index_config | Exact value lookup index |
| centrality_index_config | Top-k centrality query index |
| semantic_embedding_index | Pre-computed embedding cache |
| graph_topology_index | Global graph metrics cache |
| search_optimization | Search statistics |

### Semantic Embedding Index

| Attribute | Type | Description |
|-----------|------|-------------|
| entity_embeddings | object | Entity UUID to 384-dim vector map |
| community_report_embeddings | object | Community ID to 384-dim vector map |
| embedding_model | string | Embedding model identifier |
| embedding_dimensions | integer | Vector dimensions (384) |
| similarity_metric | string | Similarity metric (cosine) |

### Graph Topology Index

| Attribute | Type | Description |
|-----------|------|-------------|
| node_count | integer | Total node count |
| relationship_count | integer | Total relationship count |
| avg_degree | number | Average node degree |
| graph_density | number | Graph density metric |

### Search Optimization

| Attribute | Type | Description |
|-----------|------|-------------|
| total_entities | integer | Total entity count |
| total_relationships | integer | Total relationship count |
| total_communities | integer | Total community count |
| avg_community_size | number | Average community size |
| graph_modularity | number | Graph modularity score |

---
