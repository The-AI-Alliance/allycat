"""GraphRAG Data Processing Pipeline"""

from .process_graphrag_phase1 import main as phase1_main
from .process_graphrag_phase2 import main as phase2_main
from .process_graphrag_phase3 import main as phase3_main
from .process_graphrag_phase4 import main as phase4_main
from .validate_and_finalize_graph import main as validate_main

__all__ = [
    'phase1_main',
    'phase2_main', 
    'phase3_main',
    'phase4_main',
    'validate_main'
]
