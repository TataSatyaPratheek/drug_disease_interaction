# src/utils/model_manager.py - Model Download Manager
import os
import logging
from pathlib import Path
from sentence_transformers import CrossEncoder
import threading

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages downloading and caching of ML models"""
    
    def __init__(self):
        self.cache_dir = Path.home() / ".cache" / "drug_disease_interaction"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
    
    def ensure_reranker_model(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L6-v2') -> CrossEncoder:
        """Ensure reranker model is available, download if needed"""
        model_path = self.cache_dir / model_name.replace('/', '_')
        
        with self._lock:
            if model_path.exists():
                logger.info(f"Loading cached reranker model from {model_path}")
                try:
                    return CrossEncoder(str(model_path))
                except Exception:
                    logger.warning(f"Cached model corrupted, re-downloading...")
                    # Fall through to download
            
            logger.info(f"Downloading reranker model {model_name}...")
            try:
                # Download and cache the model
                model = CrossEncoder(model_name)
                model.save(str(model_path))
                logger.info(f"Model cached to {model_path}")
                return model
            except Exception as e:
                logger.error(f"Failed to download model {model_name}: {e}")
                # Return a mock model for testing
                return self._create_mock_reranker()
    
    def _create_mock_reranker(self):
        """Create a mock reranker for testing when model download fails"""
        class MockReranker:
            def predict(self, pairs):
                # Return random scores for testing
                import random
                return [random.uniform(0.5, 1.0) for _ in pairs]
        
        logger.warning("Using mock reranker for testing")
        return MockReranker()

# Global instance
model_manager = ModelManager()
