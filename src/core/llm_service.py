# src/core/llm_service.py - USE LLAMAINDEX + OLLAMA (DON'T REINVENT LLM INTERFACE)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings

import logging
from typing import List

logger = logging.getLogger(__name__)

class LLMService:
    """LLM service using LlamaIndex + Ollama - optimized for GTX 1650Ti"""
    
    def __init__(self, hardware_config: dict):
        # Use your existing hardware optimization
        model_name = hardware_config['model_recommendations']['primary_model']
        
        # Configure for your GTX 1650Ti (4GB VRAM)
        self.llm = Ollama(
            model=model_name,
            base_url="http://localhost:11434",
            temperature=hardware_config['llm_config']['temperature'],
            request_timeout=60.0
        )
        
        # Configure embeddings
        self.embedding = OllamaEmbedding(
            model_name="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        
        # Set global LlamaIndex settings
        Settings.llm = self.llm
        Settings.embed_model = self.embedding
        
        logger.info(f"Initialized LLM service with model: {model_name}")
    
    async def generate_response(self, context: str, query: str) -> str:
        """Generate response using retrieved context"""
        prompt = f"""
        Based on the following drug-disease interaction context, answer the question:
        
        Context: {context}
        
        Question: {query}
        
        Answer:"""
        
        response = await self.llm.acomplete(prompt)
        return str(response)
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for text"""
        embedding = await self.embedding.aget_text_embedding(text)
        return embedding
