"""
core/inference.py — Ollama + ExLlamaV2 inference layer (Cluster 03: Sovereign Developer OS)

Unified inference interface. Uses Ollama for managed models,
falls back to ExLlamaV2 for GGUF/EXL2 models loaded directly.

SDKs: Ollama, llama-cpp-python (GGUF fallback)
"""
import json
import logging
import time
import urllib.request
import urllib.error
from typing import Optional, Generator

logger = logging.getLogger(__name__)


class LocalInference:
    """
    Unified local inference. Tries Ollama first, falls back to llama-cpp-python.
    """

    def __init__(self, config: dict):
        self._cfg = config
        ollama = config.get("ollama", {})
        self._base_url  = ollama.get("base_url", "http://localhost:11434")
        self._model     = ollama.get("model", "llama3")
        self._embed_model = ollama.get("embed_model", "nomic-embed-text")
        self._backend   = None  # "ollama" | "llamacpp"
        self._llama_model = None

    def load(self) -> str:
        """Detect available backend. Returns "ollama" or "llamacpp"."""
        if self._try_ollama():
            self._backend = "ollama"
            logger.info("Inference backend: Ollama (%s @ %s)", self._model, self._base_url)
        else:
            logger.warning("Ollama unavailable, trying llama-cpp-python...")
            if self._try_llamacpp():
                self._backend = "llamacpp"
                logger.info("Inference backend: llama-cpp-python")
            else:
                logger.error("No inference backend available")
                self._backend = "none"
        return self._backend

    def _try_ollama(self) -> bool:
        try:
            req = urllib.request.Request(f"{self._base_url}/api/tags",
                headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=3) as r:
                data = json.loads(r.read())
                models = [m["name"] for m in data.get("models", [])]
                logger.info("Ollama models available: %s", models)
                return True
        except Exception as e:
            logger.debug("Ollama not reachable: %s", e)
            return False

    def _try_llamacpp(self) -> bool:
        try:
            from llama_cpp import Llama
            model_path = self._cfg.get("finetune", {}).get("output_dir", "~/.sovereign/finetuned")
            import os, glob
            expanded = os.path.expanduser(model_path)
            gguf_files = glob.glob(os.path.join(expanded, "**/*.gguf"), recursive=True)
            if gguf_files:
                self._llama_model = Llama(model_path=gguf_files[0], n_ctx=4096, verbose=False)
                return True
            return False
        except Exception:
            return False

    def complete(self, prompt: str, system: Optional[str] = None,
                 max_tokens: int = 1024, temperature: float = 0.2) -> dict:
        """Single completion. Returns {text, latency_ms, backend, tokens}."""
        t0 = time.perf_counter()
        if self._backend == "ollama":
            result = self._ollama_complete(prompt, system, max_tokens, temperature)
        elif self._backend == "llamacpp":
            result = self._llamacpp_complete(prompt, max_tokens, temperature)
        else:
            result = {"text": "No inference backend available", "tokens": 0}
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        result["backend"] = self._backend
        return result

    def stream(self, prompt: str, system: Optional[str] = None,
               max_tokens: int = 1024, temperature: float = 0.2) -> Generator[str, None, None]:
        """Streaming completion. Yields token strings."""
        if self._backend == "ollama":
            yield from self._ollama_stream(prompt, system, max_tokens, temperature)
        elif self._backend == "llamacpp":
            yield from self._llamacpp_stream(prompt, max_tokens, temperature)
        else:
            yield "No inference backend available"

    def embed(self, text: str) -> list:
        """Generate embeddings via Ollama."""
        payload = json.dumps({"model": self._embed_model, "prompt": text}).encode()
        req = urllib.request.Request(f"{self._base_url}/api/embeddings",
            data=payload, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read()).get("embedding", [])
        except Exception as e:
            logger.error("Embedding failed: %s", e)
            return []

    def _ollama_complete(self, prompt, system, max_tokens, temperature) -> dict:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = json.dumps({"model": self._model, "messages": messages,
                              "stream": False, "options": {"temperature": temperature,
                              "num_predict": max_tokens}}).encode()
        req = urllib.request.Request(f"{self._base_url}/api/chat",
            data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
            msg = data.get("message", {})
            return {"text": msg.get("content", ""), "tokens": data.get("eval_count", 0)}

    def _ollama_stream(self, prompt, system, max_tokens, temperature):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = json.dumps({"model": self._model, "messages": messages, "stream": True,
                              "options": {"temperature": temperature, "num_predict": max_tokens}}).encode()
        req = urllib.request.Request(f"{self._base_url}/api/chat",
            data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=120) as r:
            for line in r:
                try:
                    data = json.loads(line.decode())
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break
                except Exception:
                    pass

    def _llamacpp_complete(self, prompt, max_tokens, temperature) -> dict:
        result = self._llama_model(prompt, max_tokens=max_tokens, temperature=temperature, stream=False)
        return {"text": result["choices"][0]["text"], "tokens": result["usage"]["completion_tokens"]}

    def _llamacpp_stream(self, prompt, max_tokens, temperature):
        for chunk in self._llama_model(prompt, max_tokens=max_tokens, temperature=temperature, stream=True):
            yield chunk["choices"][0]["text"]

    def get_metadata(self) -> dict:
        return {"backend": self._backend, "model": self._model,
                "embed_model": self._embed_model, "base_url": self._base_url}
