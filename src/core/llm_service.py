# src/core/llm_service.py - USE LLAMAINDEX + OLLAMA (DON'T REINVENT LLM INTERFACE)

# src/core/llm_service.py - ENHANCED
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
import httpx
import logging
from typing import List

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, hardware_config: dict, ollama_url: str, request_timeout: int):
        model_name = hardware_config['model_recommendations']['primary_model']
        self.base_url = ollama_url
        self.llm = Ollama(
            model=model_name,
            base_url=self.base_url,
            temperature=hardware_config['llm_config']['temperature'],
            request_timeout=request_timeout
        )
        self.embedding = OllamaEmbedding(
            model_name="nomic-embed-text",
            base_url=self.base_url
        )
        Settings.llm = self.llm
        Settings.embed_model = self.embedding
        logger.info(f"LLM service configured for model: {model_name} at {self.base_url}")

    async def check_connection(self) -> bool:
        """Verifies connection to the Ollama server."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.base_url)
                response.raise_for_status()
            logger.info("Successfully connected to Ollama server.")
            return True
        except (httpx.ConnectError, httpx.HTTPStatusError) as e:
            logger.error(f"Failed to connect to Ollama server at {self.base_url}: {e}")
            return False
    
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
