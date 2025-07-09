# src/graphrag/generators/llm_client.py
"""Direct local model access without any server connections."""

import logging
import re
from typing import Dict, List, Tuple
from pathlib import Path
import json
import subprocess
import os

LOGGER = logging.getLogger(__name__)

class LocalOllamaClient:
    """
    Direct local model access without HTTP connections.
    Uses ollama binary directly for maximum privacy.
    """

    def __init__(
        self,
        model_name: str = "qwen2.5:1.5b",
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
            # Use ollama binary directly
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Ollama binary not accessible: {result.stderr}")
            
            # Check if model exists
            if self.model_name not in result.stdout:
                LOGGER.info(f"Pulling model {self.model_name} locally...")
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
                LOGGER.info(f"Model {self.model_name} is available locally")
                
        except subprocess.TimeoutExpired:
            raise RuntimeError("Model operation timed out")
        except FileNotFoundError:
            raise RuntimeError("Ollama binary not found. Please install ollama locally.")

    def generate_with_reasoning(self, prompt: str) -> Tuple[str, str]:
        """
        Generate response with reasoning using direct local model access.
        """
        sys_prompt = (
            "You are a biomedical AI assistant with access to a drug-disease knowledge graph. "
            "Analyze the provided context and answer step-by-step.\n\n"
            "RESPONSE FORMAT:\n"
            "🧠 REASONING:\n"
            "[your step-by-step analysis]\n\n"
            "🎯 FINAL ANSWER:\n"
            "[your concise answer]\n\n"
        )

        full_prompt = f"{sys_prompt}QUERY: {prompt}"
        
        try:
            # Create temporary prompt file for privacy
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(full_prompt)
                prompt_file = f.name
            
            # Use ollama binary directly - removed unsupported flags
            result = subprocess.run([
                "ollama", "run", self.model_name,
                "--verbose=false",
            ], capture_output=True, text=True, timeout=120)
            
            # Clean up temp file
            os.unlink(prompt_file)
            
            if result.returncode != 0:
                raise RuntimeError(f"Model generation failed: {result.stderr}")
            
            response = result.stdout.strip()
            reasoning, final_answer = self._split_response(response)
            return reasoning, final_answer
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Model generation timed out")
        except Exception as e:
            LOGGER.error(f"Local model generation failed: {e}")
            raise


    def generate(self, prompt: str) -> str:
        """Simple generation interface."""
        _, answer = self.generate_with_reasoning(prompt)
        return answer

    @staticmethod
    def _split_response(response: str) -> Tuple[str, str]:
        """Extract reasoning and final answer from response."""
        # Look for structured format
        reasoning_match = re.search(
            r'🧠.*?REASONING.*?:\s*(.*?)\s*🎯.*?FINAL ANSWER.*?:\s*(.*)',
            response,
            re.DOTALL | re.IGNORECASE
        )
        
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
            final_answer = reasoning_match.group(2).strip()
            return reasoning, final_answer
        
        # Fallback parsing
        if "FINAL ANSWER:" in response.upper():
            parts = re.split(r'FINAL ANSWER.*?:', response, flags=re.IGNORECASE)
            if len(parts) >= 2:
                reasoning = parts[0].replace("REASONING:", "").strip()
                return reasoning, parts[1].strip()
        
        return "Direct response provided", response

# Alias for backward compatibility
OllamaClient = LocalOllamaClient
