# src/graphrag/generators/llm_client.py
"""Thin wrapper around the Ollama HTTP API with safe SSL-handling."""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Tuple

import ollama
from httpx import HTTPError, TimeoutException

LOGGER = logging.getLogger(__name__)


class OllamaClient:
    """
    Wrapper that connects over plain HTTP and provides helpers for
    step-by-step reasoning extraction.
    """

    def __init__(
        self,
        model_name: str = "qwen3:1.7b",
        host: str = "localhost",
        port: int = 11434,
        *,
        temperature: float = 0.1,
        timeout: int = 300,
    ) -> None:
        self.model_name = model_name
        self.host = host
        self.port = port
        self.temperature = temperature

        # Simplified client initialization for broader compatibility.
        # The http:// prefix ensures a non-SSL connection.
        self.client = ollama.Client(
            host=f"http://{host}:{port}",
            timeout=timeout,
        )

        # Verify that the server is reachable and the model exists
        self._ensure_model_available()

    def _ensure_model_available(self) -> None:
        """Pull `self.model_name` if it is not present locally."""
        try:
            models_resp = self.client.list()
            raw_list = (
                models_resp.get("models", models_resp)
            )
            available = {
                m.get("name", str(m)).split(":")[0]
                for m in raw_list
            }

            if self.model_name.split(":")[0] not in available:
                LOGGER.info("Pulling Ollama model '%s'...", self.model_name)
                self.client.pull(self.model_name)
                LOGGER.info("Model pulled successfully.")
        except (HTTPError, TimeoutException) as exc:
            LOGGER.error("Failed to communicate with Ollama: %s", exc)
            raise

    def generate_with_reasoning(
        self, prompt: str, *, max_tokens: int = 2048
    ) -> Tuple[str, str]:
        """
        Return `(reasoning, final_answer)` using qwen-style step-by-step output.
        """
        sys_prompt = (
            "You are a biomedical AI assistant. "
            "First think step-by-step, then give a concise final answer.\n\n"
            "RESPONSE FORMAT:\n"
            "🧠 REASONING:\n"
            "[your reasoning]\n\n"
            "🎯 FINAL ANSWER:\n"
            "[your short answer]"
        )

        full_prompt = f"{sys_prompt}\n\nQUERY: {prompt}"
        try:
            rsp = self.client.generate(
                model=self.model_name,
                prompt=full_prompt,
                options={
                    "num_predict": max_tokens,
                    "temperature": self.temperature,
                    "top_p": 0.9,
                    "top_k": 40,
                    "repeat_penalty": 1.1,
                },
            )
        except Exception as exc:
            LOGGER.error("Generation failed: %s", exc)
            raise

        reasoning, final_answer = self._split_response(rsp["response"])
        return reasoning, final_answer

    def generate(
        self, prompt: str, *, max_tokens: int = 2048, temperature: float | None = None
    ) -> str:
        """Compatibility shim: return only the final answer."""
        if temperature is not None:
            self.temperature = temperature
        return self.generate_with_reasoning(prompt, max_tokens=max_tokens)[1]

    def chat(
        self, messages: List[Dict[str, str]], *, max_tokens: int = 2048
    ) -> str:
        """Thin wrapper around Ollama’s /chat endpoint."""
        rsp = self.client.chat(
            model=self.model_name,
            messages=messages,
            options={"num_predict": max_tokens, "temperature": self.temperature},
        )
        return rsp["message"]["content"]

    @staticmethod
    def _split_response(response: str) -> Tuple[str, str]:
        """
        Extract reasoning & final answer blocks from qwen-style output.
        """
        m = re.search(
            r"🧠.*?REASONING.*?:\s*(.*?)\s*🎯.*?FINAL ANSWER.*?:\s*(.*)",
            response,
            re.S | re.I,
        )
        if m:
            return m.group(1).strip(), m.group(2).strip()

        # Fallback heuristic
        if "FINAL ANSWER:" in response.upper():
            parts = re.split(r'FINAL ANSWER.*?:', response, flags=re.IGNORECASE)
            if len(parts) >= 2:
                reasoning_part = parts[0].replace("REASONING:", "").strip()
                return reasoning_part, parts[1].strip()
        
        return "Direct response without explicit reasoning.", response


class EmbeddingClient:
    """Uses sentence-transformers locally; avoids Ollama for embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    def encode_single(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()

