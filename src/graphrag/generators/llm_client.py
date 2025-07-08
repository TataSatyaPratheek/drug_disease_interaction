# src/graphrag/generators/llm_client.py
"""Thin wrapper around the Ollama HTTP API with safe SSL-handling."""

from __future__ import annotations

import logging
import re
import ssl
from typing import Dict, List, Tuple

import ollama
from httpx import HTTPError, TimeoutException

LOGGER = logging.getLogger(__name__)


class OllamaClient:
    """
    Wrapper that:

    1. Connects over plain HTTP by default.
    2. Retries once with `verify=False` if the local CA bundle is missing.
    3. Provides helpers for step-by-step reasoning extraction.
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

        # 1st attempt – normal verify
        scheme = "http"  # use http, TLS off
        try:
            self.client = ollama.Client(
                host=f"{scheme}://{host}:{port}",
                timeout=timeout,
                tls=False,
                verify=True,      # may fail on mis-installed root CAs
            )
        except ssl.SSLError as e:  # pragma: no cover
            LOGGER.warning(
                "SSL verification failed when contacting Ollama (%s). "
                "Retrying with verify=False. NOTE: connection is local-host only.",
                e,
            )
            self.client = ollama.Client(
                host=f"{scheme}://{host}:{port}",
                timeout=timeout,
                tls=False,
                verify=False,    # ← skips CA lookup, fixes NO_CERTIFICATE_OR_CRL_FOUND
            )

        # Verify that the server is reachable and the model exists
        self._ensure_model_available()

    # --------------------------------------------------------------------- #
    #                           Private helpers                             #
    # --------------------------------------------------------------------- #
    def _ensure_model_available(self) -> None:
        """Pull `self.model_name` if it is not present locally."""
        try:
            models_resp = self.client.list()
            # The response shape differs between client versions; normalise:
            raw_list = (
                models_resp.models
                if hasattr(models_resp, "models")
                else models_resp.get("models", models_resp)
            )
            available = {
                (m.model if hasattr(m, "model") else m.get("name", str(m))).split(":")[0]
                for m in raw_list
            }

            if self.model_name.split(":")[0] not in available:
                LOGGER.info("Pulling Ollama model '%s'…", self.model_name)
                self.client.pull(self.model_name)
                LOGGER.info("Model pulled successfully.")
        except (HTTPError, TimeoutException, ssl.SSLError) as exc:  # pragma: no cover
            LOGGER.error("Failed to communicate with Ollama: %s", exc)
            raise

    # --------------------------------------------------------------------- #
    #                         Public generation APIs                        #
    # --------------------------------------------------------------------- #
    def generate_with_reasoning(  # noqa: D401
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
        except Exception as exc:  # pragma: no cover
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

    # ------------------------------------------------------------------ #
    #                        Parsing utilities                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _split_response(response: str) -> Tuple[str, str]:
        """
        Extract reasoning & final answer blocks from qwen-style output.
        Falls back gracefully if the pattern is absent.
        """
        m = re.search(
            r"🧠.*?REASONING.*?:\s*(.*?)\s*🎯.*?FINAL ANSWER.*?:\s*(.*)",
            response,
            re.S | re.I,
        )
        if m:
            return m.group(1).strip(), m.group(2).strip()

        # Fallback heuristics
        up = response.upper()
        if "REASONING:" in up and "FINAL ANSWER:" in up:
            reasoning, _, final_ans = response.partition("FINAL ANSWER:")
            return reasoning.replace("REASONING:", "").strip(), final_ans.strip()

        half = len(response) // 2
        return response[:half].strip(), response[half:].strip()


# --------------------------------------------------------------------- #
#                     Lightweight embedding helper                      #
# --------------------------------------------------------------------- #
class EmbeddingClient:
    """Uses sentence-transformers locally; avoids Ollama for embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    def encode_single(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()
