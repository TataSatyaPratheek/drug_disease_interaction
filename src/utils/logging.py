# src/utils/logging.py - USE PYTHON'S STANDARD LOGGING
import logging
import sys
from pathlib import Path


def setup_logging():
    """Setup production logging configuration in a robust way."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Avoid adding handlers if they already exist
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # File handler
        file_handler = logging.FileHandler(log_dir / "hybrid_rag.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    
    return logging.getLogger("hybrid_rag_api") # Return a named logger for your app
