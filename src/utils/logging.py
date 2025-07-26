# src/utils/logging.py - USE PYTHON'S STANDARD LOGGING
import logging
import sys
from pathlib import Path

def setup_logging():
    """Setup production logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "hybrid_rag.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger("hybrid_rag")
