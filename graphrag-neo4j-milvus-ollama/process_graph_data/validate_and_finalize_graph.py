""" Graph Data Validator and Finalizer """

import json
from pathlib import Path
from datetime import datetime
from jsonschema import Draft7Validator

class GraphDataValidator:
    def __init__(self, schema_path, data_path):
        self.schema_path = Path(schema_path)
        self.data_path = Path(data_path)
        self.schema = None
        self.data = None
        self.errors = []
        self.warnings = []
        
    def load_schema(self):
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            self.schema = json.load(f)
            
    def load_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
    def validate_schema(self):
        validator = Draft7Validator(self.schema)
        errors = list(validator.iter_errors(self.data))
        
        for error in errors:
            path = ".".join(str(p) for p in error.path)
            self.errors.append({'path': path, 'message': error.message})
            
        return len(errors) == 0
        
    def check_quality(self):
        nodes = self.data.get('nodes', [])
        relationships = self.data.get('relationships', [])
        communities = self.data.get('communities', {})
        reports = self.data.get('community_reports', [])
        drift = self.data.get('drift_search_metadata', {})
        
        quality = {
            'total_nodes': len(nodes),
            'total_relationships': len(relationships),
            'total_communities': communities.get('total_communities', 0),
            'total_reports': len(reports),
            'nodes_with_embeddings': sum(1 for n in nodes if n.get('properties', {}).get('embedding_vector')),
            'nodes_with_quality_scores': sum(1 for n in nodes if 'data_quality_score' in n.get('properties', {})),
            'nodes_with_centrality': sum(1 for n in nodes if 'degree_centrality' in n.get('properties', {})),
            'reports_with_embeddings': sum(1 for r in reports if r.get('report_embedding')),
            'drift_indexes': sum(1 for k in [
                'vector_index_config', 'fulltext_index_config', 'composite_index_config',
                'range_index_config', 'centrality_index_config', 'semantic_embedding_index',
                'graph_topology_index', 'search_optimization', 'community_search_index',
                'entity_search_index', 'relationship_search_index', 'query_routing_config'
            ] if k in drift and drift[k])
        }
        
        if quality['nodes_with_embeddings'] < quality['total_nodes']:
            self.warnings.append(f"{quality['total_nodes'] - quality['nodes_with_embeddings']} nodes missing embeddings")
            
        if quality['reports_with_embeddings'] < quality['total_reports']:
            self.warnings.append(f"{quality['total_reports'] - quality['reports_with_embeddings']} reports missing embeddings")
            
        if quality['drift_indexes'] < 12:
            self.warnings.append(f"{12 - quality['drift_indexes']} DRIFT indexes missing")
            
        return quality
        
    def save_final(self, output_path):
        final_data = self.data.copy()
        
        if 'metadata' not in final_data:
            final_data['metadata'] = {}
            
        final_data['metadata']['validation'] = {
            'validated_at': datetime.now().isoformat(),
            'validation_passed': len(self.errors) == 0,
            'warnings': len(self.warnings)
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
            
    def save_report(self, report_path, quality):
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'schema_file': str(self.schema_path),
            'input_file': str(self.data_path),
            'schema_validation': {
                'passed': len(self.errors) == 0,
                'errors': self.errors,
                'warnings': self.warnings
            },
            'data_quality': quality
        }
        
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


def main():
    schema_path = "process_graphdata/graph-data-shema/allycat-graph-data-json-schema.json"
    data_path = "workspace/graph_data/phase4_output.json"
    final_path = "workspace/graph_data/graph-data-final.json"
    report_path = "workspace/graph_data/validation_report.json"
    
    print("=" * 60)
    print("AllyCAT GraphRAG Validation")
    print("=" * 60)
    
    validator = GraphDataValidator(schema_path, data_path)
    
    print("Loading schema...")
    validator.load_schema()
    
    print("Loading data...")
    validator.load_data()
    
    print("Validating schema...")
    schema_valid = validator.validate_schema()
    
    print("Checking quality...")
    quality = validator.check_quality()
    
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Schema Valid: {'YES' if schema_valid else 'NO'}")
    print(f"Errors: {len(validator.errors)}")
    print(f"Warnings: {len(validator.warnings)}")
    
    if validator.errors:
        print("\nErrors:")
        for err in validator.errors[:5]:
            print(f"  {err['path']}: {err['message']}")
        if len(validator.errors) > 5:
            print(f"  ... and {len(validator.errors) - 5} more")
            
    if validator.warnings:
        print("\nWarnings:")
        for warn in validator.warnings:
            print(f"  {warn}")
    
    print("\n" + "=" * 60)
    print("DATA QUALITY")
    print("=" * 60)
    print(f"Nodes: {quality['total_nodes']}")
    print(f"  With embeddings: {quality['nodes_with_embeddings']}")
    print(f"  With quality scores: {quality['nodes_with_quality_scores']}")
    print(f"  With centrality: {quality['nodes_with_centrality']}")
    print(f"Relationships: {quality['total_relationships']}")
    print(f"Communities: {quality['total_communities']}")
    print(f"Reports: {quality['total_reports']}")
    print(f"  With embeddings: {quality['reports_with_embeddings']}")
    print(f"DRIFT indexes: {quality['drift_indexes']}/12")
    
    print("\n" + "=" * 60)
    print("Saving output...")
    validator.save_final(final_path)
    print(f"✓ Final output: {final_path}")
    
    validator.save_report(report_path, quality)
    print(f"✓ Validation report: {report_path}")
    print("=" * 60)
    
    return schema_valid


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
