#!/usr/bin/env python3
# scripts/download_models.py - Download required models

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.model_manager import model_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Download all required models"""
    logger.info("🚀 Downloading required models...")
    
    # Download reranker model
    try:
        reranker = model_manager.ensure_reranker_model()
        logger.info("✅ Reranker model ready")
        
        # Test the model
        test_pairs = [["What is aspirin?", "Aspirin is a pain medication"]]
        scores = reranker.predict(test_pairs)
        logger.info(f"✅ Model test successful: {scores}")
        
    except Exception as e:
        logger.error(f"❌ Model download failed: {e}")
        return 1
    
    logger.info("🎉 All models downloaded successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
