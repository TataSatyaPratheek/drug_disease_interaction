# src/graphrag/generators/llm_client.py
"""Direct local model access without any server connections."""

import logging
import re
import subprocess
import tempfile
import os
from typing import Dict, List, Tuple
from pathlib import Path

LOGGER = logging.getLogger(__name__)

class LocalOllamaClient:
    """Direct local model access for maximum privacy."""

    def __init__(
        self,
        model_name: str = "qwen3:1.7b",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Ensure model is available locally
        self._ensure_model_available()

    def _ensure_model_available(self) -> None:
        """Ensure model is available locally without server calls."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Ollama not accessible: {result.stderr}")
            
            if self.model_name not in result.stdout:
                LOGGER.info(f"Pulling model {self.model_name}...")
                pull_result = subprocess.run(
                    ["ollama", "pull", self.model_name],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if pull_result.returncode != 0:
                    raise RuntimeError(f"Failed to pull model: {pull_result.stderr}")
                    
                LOGGER.info("Model pulled successfully")
            else:
                LOGGER.info(f"Model {self.model_name} is available")
                
        except subprocess.TimeoutExpired:
            raise RuntimeError("Model operation timed out")
        except FileNotFoundError:
            raise RuntimeError("Ollama binary not found. Please install ollama.")

    def generate_with_reasoning(self, prompt: str) -> Tuple[str, str]:
        """Generate response with structured reasoning - improved error handling."""
        
        # Simplified prompt that works better with qwen3:1.7b
        structured_prompt = f"""You are a medical research assistant. Answer this question about drug treatments.

    Question: {prompt}

    Please provide:
    1. Your analysis and reasoning
    2. Your final answer

    Format your response clearly with your reasoning first, then your conclusion."""

        try:
            LOGGER.info("Starting Ollama generation...")
            
            # Use simpler subprocess call with better error handling
            result = subprocess.run([
                "ollama", "run", self.model_name
            ], 
            input=structured_prompt, 
            capture_output=True, 
            text=True, 
            timeout=60,  # Reduced timeout
            encoding='utf-8'
            )
            
            LOGGER.info(f"Ollama returncode: {result.returncode}")
            LOGGER.info(f"Ollama stdout length: {len(result.stdout)}")
            LOGGER.info(f"Ollama stderr: {result.stderr}")
            
            if result.returncode != 0:
                error_msg = f"Ollama failed with return code {result.returncode}: {result.stderr}"
                LOGGER.error(error_msg)
                return "Ollama execution failed", error_msg
            
            response = result.stdout.strip()
            
            if not response:
                error_msg = "Ollama returned empty response"
                LOGGER.error(error_msg)
                return "Empty response from Ollama", error_msg
            
            LOGGER.info(f"Generated response length: {len(response)}")
            LOGGER.info(f"Response preview: {response[:200]}...")
            
            # Simple parsing - split response roughly in half
            if len(response) > 100:
                # Find natural split point
                lines = response.split('\n')
                mid_point = len(lines) // 2
                reasoning = '\n'.join(lines[:mid_point]).strip()
                final_answer = '\n'.join(lines[mid_point:]).strip()
            else:
                reasoning = "Brief response generated"
                final_answer = response
            
            # Ensure we have content
            if not final_answer:
                final_answer = response
            if not reasoning:
                reasoning = "Analysis provided in response"
            
            LOGGER.info(f"Parsed reasoning length: {len(reasoning)}")
            LOGGER.info(f"Parsed answer length: {len(final_answer)}")
            
            return reasoning, final_answer
            
        except subprocess.TimeoutExpired:
            error_msg = "Ollama generation timed out after 60 seconds"
            LOGGER.error(error_msg)
            return "Generation timeout", error_msg
        except FileNotFoundError:
            error_msg = "Ollama binary not found. Please ensure ollama is installed and in PATH"
            LOGGER.error(error_msg)
            return "Ollama not found", error_msg
        except Exception as e:
            error_msg = f"Unexpected error in generation: {str(e)}"
            LOGGER.error(error_msg)
            return "Generation error", error_msg


    def _parse_structured_response(self, response: str) -> Tuple[str, str]:
        """Parse structured response with reliable header detection."""
        
        # Look for structured headers
        reasoning_start = response.find("=== REASONING ===")
        answer_start = response.find("=== FINAL ANSWER ===")
        
        if reasoning_start != -1 and answer_start != -1:
            # Extract sections
            reasoning = response[reasoning_start + len("=== REASONING ==="):answer_start].strip()
            final_answer = response[answer_start + len("=== FINAL ANSWER ==="):].strip()
            
            if reasoning and final_answer:
                return reasoning, final_answer
        
        # Fallback: try other common patterns
        if "REASONING:" in response.upper() and "FINAL ANSWER:" in response.upper():
            parts = response.split("FINAL ANSWER:")
            if len(parts) >= 2:
                reasoning = parts[0].replace("REASONING:", "").strip()
                final_answer = parts[1].strip()
                return reasoning, final_answer
        
        # Last resort: split response roughly
        if len(response) > 200:
            mid_point = len(response) // 2
            return response[:mid_point].strip(), response[mid_point:].strip()
        
        # Return full response as answer if parsing fails
        return "Response parsing incomplete", response

    def generate(self, prompt: str) -> str:
        """Simple generation interface."""
        _, answer = self.generate_with_reasoning(prompt)
        return answer

# Alias for compatibility
OllamaClient = LocalOllamaClient
