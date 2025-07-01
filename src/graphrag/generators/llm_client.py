# src/graphrag/generators/llm_client.py
import ollama
from typing import Dict, List, Any, Optional, Tuple
import time
import logging
import re

class OllamaClient:
    """Enhanced Ollama client optimized for qwen3 reasoning output"""
    
    def __init__(self, model_name: str = "qwen3:1.7b", host: str = "localhost", port: int = 11434):
        self.model_name = model_name
        self.host = host
        self.port = port
        self.client = ollama.Client(host=f"http://{host}:{port}")
        self.logger = logging.getLogger(__name__)
        self.temperature = 0.1  # Lower for more focused reasoning
        
        # Test connection
        self._ensure_model_available()
    
    def _ensure_model_available(self):
        """Ensure the model is available locally"""
        try:
            # Get models list from Ollama
            models_response = self.client.list()
            
            # Handle different response formats
            if hasattr(models_response, 'models'):
                models_list = models_response.models
            elif isinstance(models_response, dict) and 'models' in models_response:
                models_list = models_response['models']
            else:
                models_list = models_response
            
            # Extract model names correctly
            model_names = []
            for model in models_list:
                if hasattr(model, 'model'):
                    # Ollama Model object with .model attribute
                    model_names.append(model.model)
                elif isinstance(model, dict):
                    # Dictionary format (fallback)
                    name = model.get('name') or model.get('model') or str(model)
                    model_names.append(name)
                else:
                    model_names.append(str(model))
            
            if self.model_name not in model_names:
                self.logger.info(f"Pulling model {self.model_name}...")
                self.client.pull(self.model_name)
                self.logger.info(f"Model {self.model_name} pulled successfully")
            else:
                self.logger.info(f"Model {self.model_name} is available")
                
        except Exception as e:
            self.logger.error(f"Error with Ollama setup: {e}")
            raise

    
    def generate_with_reasoning(self, prompt: str, max_tokens: int = 2048) -> Tuple[str, str]:
        """Generate response with explicit reasoning using qwen3's capabilities"""
        
        # Enhanced prompt to trigger qwen3's reasoning mode
        reasoning_prompt = f"""You are a biomedical AI research assistant with expertise in drug discovery and molecular biology. 

For the following query, I want you to show your reasoning process step-by-step, then provide your final answer.

Structure your response as:
🧠 **REASONING:**
[Show your step-by-step thinking process, evidence consideration, and logical connections]

🎯 **FINAL ANSWER:**
[Your concise, definitive response]

QUERY: {prompt}

Please think through this systematically and show your reasoning:"""
        
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=reasoning_prompt,
                options={
                    'num_predict': max_tokens,
                    'temperature': self.temperature,
                    'top_p': 0.9,
                    'top_k': 40,
                    'repeat_penalty': 1.1
                }
            )
            
            full_response = response['response']
            
            # Parse reasoning vs final answer
            reasoning, final_answer = self._parse_reasoning_response(full_response)
            
            return reasoning, final_answer
        
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return f"Error in reasoning: {str(e)}", f"Error generating response: {str(e)}"
    
    def _parse_reasoning_response(self, response: str) -> Tuple[str, str]:
        """Parse qwen3 response to extract reasoning and final answer"""
        
        # Look for the structured format
        reasoning_match = re.search(r'🧠.*?REASONING.*?:(.*?)🎯.*?FINAL ANSWER.*?:(.*)', response, re.DOTALL | re.IGNORECASE)
        
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
            final_answer = reasoning_match.group(2).strip()
            return reasoning, final_answer
        
        # Fallback: Look for other common patterns
        if "REASONING:" in response.upper() and "FINAL ANSWER:" in response.upper():
            parts = re.split(r'FINAL ANSWER.*?:', response, flags=re.IGNORECASE)
            if len(parts) >= 2:
                reasoning_part = parts[0].replace("REASONING:", "").strip()
                final_answer = parts[1].strip()
                return reasoning_part, final_answer
        
        # If no clear structure, split roughly in half
        if len(response) > 100:
            mid_point = len(response) // 2
            reasoning = response[:mid_point] + "..."
            final_answer = response[mid_point:]
            return reasoning, final_answer
        
        # Fallback: treat entire response as final answer
        return "Direct response without explicit reasoning steps.", response
    
    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = None) -> str:
        """Standard generate method (backward compatibility)"""
        if temperature:
            self.temperature = temperature
        
        _, final_answer = self.generate_with_reasoning(prompt, max_tokens)
        return final_answer
    
    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 2048) -> str:
        """Chat interface for conversation-style interactions"""
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={
                    'num_predict': max_tokens,
                    'temperature': self.temperature
                }
            )
            return response['message']['content']
        
        except Exception as e:
            self.logger.error(f"Error in chat: {e}")
            return f"Error in chat: {str(e)}"

class EmbeddingClient:
    """Separate client for generating embeddings"""
    
    def __init__(self, model_name: str = "nomic-embed-text:latest"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Fallback to local model
        self.ollama_model = model_name
        self.client = ollama.Client()
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts to embeddings"""
        return self.model.encode(texts).tolist()
    
    def encode_single(self, text: str) -> List[float]:
        """Encode single text to embedding"""
        return self.model.encode([text])[0].tolist()
