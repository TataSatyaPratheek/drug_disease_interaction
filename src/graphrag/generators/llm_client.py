# src/graphrag/generators/llm_client.py
"""Direct local model access without any server connections."""

import logging
import re
import subprocess
import tempfile
import os
import concurrent.futures
import threading
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path

LOGGER = logging.getLogger(__name__)

class LocalOllamaClient:
    """Direct local model access optimized for Ryzen 4800H + GTX 1650Ti."""

    def __init__(
        self,
        model_name: str = "qwen3:1.7b",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        num_threads: int = 8,  # Utilize Ryzen 4800H cores
        gpu_layers: int = 32,  # Offload to GTX 1650Ti
        context_size: int = 4096,  # Use more RAM
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_threads = num_threads
        self.gpu_layers = gpu_layers
        self.context_size = context_size
        
        # Thread pool for concurrent operations
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self._generation_lock = threading.Lock()
        
        # Ensure model is available locally
        self._ensure_model_available()
        self._configure_ollama_performance()

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

    def _configure_ollama_performance(self) -> None:
        """Configure Ollama for optimal performance on this hardware."""
        try:
            # Set environment variables for better performance
            env = os.environ.copy()
            env.update({
                'OLLAMA_NUM_PARALLEL': '4',  # Parallel requests
                'OLLAMA_MAX_LOADED_MODELS': '2',  # Keep models in memory
                'OLLAMA_FLASH_ATTENTION': '1',  # Enable flash attention
                'CUDA_VISIBLE_DEVICES': '0',  # Use GTX 1650Ti
                'OMP_NUM_THREADS': str(self.num_threads),  # CPU threads
            })
            
            # Create a modelfile for optimized settings
            modelfile_content = f"""FROM {self.model_name}
PARAMETER num_thread {self.num_threads}
PARAMETER num_gpu {self.gpu_layers}
PARAMETER num_ctx {self.context_size}
PARAMETER temperature {self.temperature}
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
"""
            
            # Apply optimized settings
            with tempfile.NamedTemporaryFile(mode='w', suffix='.modelfile', delete=False) as f:
                f.write(modelfile_content)
                f.flush()
                
                optimized_name = f"{self.model_name}-optimized"
                
                # Check if optimized model already exists
                result = subprocess.run(
                    ["ollama", "list"],
                    capture_output=True,
                    text=True,
                    env=env
                )
                
                if optimized_name not in result.stdout:
                    LOGGER.info(f"Creating optimized model: {optimized_name}")
                    subprocess.run([
                        "ollama", "create", optimized_name, "-f", f.name
                    ], env=env, check=True)
                    
                    # Use the optimized model
                    self.model_name = optimized_name
                
                os.unlink(f.name)
                
        except Exception as e:
            LOGGER.warning(f"Performance optimization failed: {e}, using default settings")

    def generate_batch(self, prompts: List[str]) -> List[Tuple[str, str]]:
        """Generate responses for multiple prompts concurrently."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(prompts))) as executor:
            futures = [executor.submit(self.generate_with_reasoning, prompt) for prompt in prompts]
            results = []
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=120)
                    results.append(result)
                except Exception as e:
                    LOGGER.error(f"Batch generation error: {e}")
                    results.append(("Generation failed", f"Error: {e}"))
            
            return results

    def generate_with_reasoning(self, prompt: str) -> Tuple[str, str]:
        """Generate response with structured reasoning - optimized for GPU/CPU."""
        
        # Enhanced prompt for better structured output
        structured_prompt = f"""You are a medical research assistant. Analyze this question systematically.

Question: {prompt}

Please structure your response as follows:
ANALYSIS:
[Your step-by-step reasoning and analysis]

CONCLUSION:
[Your final answer and recommendations]

Be thorough but concise in your analysis."""

        try:
            LOGGER.info("Starting optimized Ollama generation...")
            
            # Use optimized environment variables
            env = os.environ.copy()
            env.update({
                'CUDA_VISIBLE_DEVICES': '0',
                'OMP_NUM_THREADS': str(self.num_threads),
                'OLLAMA_NUM_PARALLEL': '1',  # Single request optimization
            })
            
            with self._generation_lock:  # Prevent resource conflicts
                # Use JSON mode for more structured output
                result = subprocess.run([
                    "ollama", "run", self.model_name,
                    "--verbose"  # More detailed output
                ], 
                input=structured_prompt, 
                capture_output=True, 
                text=True, 
                timeout=90,  # Increased timeout for complex queries
                encoding='utf-8',
                env=env
                )
            
            LOGGER.info(f"Ollama returncode: {result.returncode}")
            LOGGER.info(f"Ollama stdout length: {len(result.stdout)}")
            
            if result.stderr:
                LOGGER.debug(f"Ollama stderr: {result.stderr}")
            
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
            
            # Enhanced parsing for structured output
            reasoning, final_answer = self._parse_structured_response_enhanced(response)
            
            LOGGER.info(f"Parsed reasoning length: {len(reasoning)}")
            LOGGER.info(f"Parsed answer length: {len(final_answer)}")
            
            return reasoning, final_answer
            
        except subprocess.TimeoutExpired:
            error_msg = "Ollama generation timed out after 90 seconds"
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


    def _parse_structured_response_enhanced(self, response: str) -> Tuple[str, str]:
        """Enhanced parsing for structured responses with multiple fallbacks."""
        
        # Try the new ANALYSIS/CONCLUSION format first
        analysis_start = response.upper().find("ANALYSIS:")
        conclusion_start = response.upper().find("CONCLUSION:")
        
        if analysis_start != -1 and conclusion_start != -1:
            reasoning = response[analysis_start + 9:conclusion_start].strip()
            final_answer = response[conclusion_start + 11:].strip()
            
            if reasoning and final_answer:
                return reasoning, final_answer
        
        # Fall back to original parsing logic
        return self._parse_structured_response(response)

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

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        try:
            import psutil
            process = psutil.Process()
            return {
                "memory_percent": process.memory_percent(),
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "cpu_percent": process.cpu_percent(),
            }
        except ImportError:
            return {"status": "psutil not available"}

# Alias for compatibility
OllamaClient = LocalOllamaClient
