# src/graphrag/generators/llm_client.py
import ollama
from typing import Dict, List, Any, Optional
import time
import logging

class OllamaClient:
    """Local LLM client using Ollama"""
    
    def __init__(self, model_name: str = "llama3.1", host: str = "localhost", port: int = 11434):
        self.model_name = model_name
        self.host = host
        self.port = port
        self.client = ollama.Client(host=f"http://{host}:{port}")
        self.logger = logging.getLogger(__name__)
        
        # Test connection and pull model if needed
        self._ensure_model_available()
    
    def _ensure_model_available(self):
        """Ensure the model is available locally"""
        try:
            # Check if model exists
            models = self.client.list()['models']
            model_names = [model['name'] for model in models]
            
            if self.model_name not in model_names:
                self.logger.info(f"Pulling model {self.model_name}...")
                self.client.pull(self.model_name)
                self.logger.info(f"Model {self.model_name} pulled successfully")
            else:
                self.logger.info(f"Model {self.model_name} is available")
        
        except Exception as e:
            self.logger.error(f"Error with Ollama setup: {e}")
            raise
    
    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.1) -> str:
        """Generate response using local Ollama model"""
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'num_predict': max_tokens,
                    'temperature': temperature,
                    'top_p': 0.9,
                    'top_k': 40
                }
            )
            return response['response']
        
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings for text (if model supports it)"""
        try:
            response = self.client.embeddings(
                model=self.model_name,
                prompt=text
            )
            return response['embedding']
        except:
            # Fallback to sentence transformers if Ollama embeddings not available
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            return model.encode(text).tolist()
    
    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 2048) -> str:
        """Chat interface for conversation-style interactions"""
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={
                    'num_predict': max_tokens,
                    'temperature': 0.1
                }
            )
            return response['message']['content']
        
        except Exception as e:
            self.logger.error(f"Error in chat: {e}")
            return f"Error in chat: {str(e)}"

class EmbeddingClient:
    """Separate client for generating embeddings using sentence transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts to embeddings"""
        return self.model.encode(texts).tolist()
    
    def encode_single(self, text: str) -> List[float]:
        """Encode single text to embedding"""
        return self.model.encode([text])[0].tolist()
