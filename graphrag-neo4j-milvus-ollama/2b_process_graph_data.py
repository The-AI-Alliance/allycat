"""GraphRAG Processing Pipeline"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from process_graphdata import (
    phase1_main,
    phase2_main,
    phase3_main,
    phase4_main,
    validate_main
)


def main():
    """Run all processing phases sequentially"""
    print("=" * 80)
    print("GraphRAG Data Processing Pipeline")
    print("=" * 80)
    
    # Phase 1: Entity & Relationship Extraction
    print("\n[PHASE 1] Entity & Relationship Extraction")
    print("-" * 80)
    result = phase1_main()
    if result != 0:
        print("❌ Phase 1 failed")
        return 1
    print("✓ Phase 1 complete")
    
    # Phase 2: Community Detection & Centrality
    print("\n[PHASE 2] Community Detection & Centrality")
    print("-" * 80)
    phase2_main()
    print("✓ Phase 2 complete")
    
    # Phase 3: Community Report Generation
    print("\n[PHASE 3] Community Report Generation")
    print("-" * 80)
    phase3_main()
    print("✓ Phase 3 complete")
    
    # Phase 4: DRIFT Metadata & Embeddings
    print("\n[PHASE 4] DRIFT Metadata & Embeddings")
    print("-" * 80)
    phase4_main()
    print("✓ Phase 4 complete")
    
    # Validation
    print("\n[VALIDATION] Schema & Quality Check")
    print("-" * 80)
    result = validate_main()
    if result:  # True = schema valid
        print("✓ Validation complete")
    else:  # False = schema invalid
        print("⚠️  Validation failed")
    
    print("\n" + "=" * 80)
    print("✓ All phases complete")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    exit(main())
