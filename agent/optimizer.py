"""
agent/optimizer.py — DSPy prompt optimization (Cluster 03: Sovereign Developer OS)

Uses DSPy to automatically optimize prompts for the code retrieval
and suggestion tasks based on accepted suggestion feedback.

SDKs: DSPy, Ollama
"""
import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class CodeSuggestionSignature:
    """DSPy signature for code suggestion task."""
    pass


class SovereignOptimizer:
    """
    DSPy-based prompt optimizer.
    Periodically recompiles prompts using accepted suggestions as training data.
    """

    def __init__(self, config: dict, store=None):
        self._cfg = config
        self._store = store
        self._lm = None
        self._optimized_program = None
        self._min_examples = config.get("finetune", {}).get("min_accepted_suggestions", 50)

    def load(self) -> bool:
        try:
            import dspy
            ollama_cfg = self._cfg.get("ollama", {})
            self._lm = dspy.OllamaLocal(
                model=ollama_cfg.get("model", "llama3"),
                base_url=ollama_cfg.get("base_url", "http://localhost:11434"),
                max_tokens=512,
                temperature=0.1,
            )
            dspy.settings.configure(lm=self._lm)
            logger.info("DSPy optimizer loaded with Ollama")
            return True
        except ImportError:
            logger.warning("dspy-ai not installed — optimizer disabled")
            return False
        except Exception as e:
            logger.warning("DSPy load failed: %s", e)
            return False

    def _build_examples(self) -> list:
        """Build DSPy training examples from accepted suggestions."""
        if not self._store:
            return []
        suggestions = self._store.get_accepted_suggestions(limit=200)
        examples = []
        try:
            import dspy
            for s in suggestions:
                if s.get("original") and s.get("suggested"):
                    examples.append(dspy.Example(
                        original_code=s["original"],
                        file_context=s.get("file_path", ""),
                        improved_code=s["suggested"],
                    ).with_inputs("original_code", "file_context"))
        except ImportError:
            pass
        return examples

    def should_optimize(self) -> bool:
        """Check if we have enough data to run optimization."""
        if not self._store:
            return False
        count = self._store.accepted_count()
        logger.info("Accepted suggestions: %d / %d needed", count, self._min_examples)
        return count >= self._min_examples

    def optimize(self) -> bool:
        """Run DSPy optimization. Returns True if successful."""
        if not self._lm:
            logger.warning("DSPy not loaded")
            return False
        examples = self._build_examples()
        if len(examples) < self._min_examples:
            logger.info("Not enough examples (%d < %d)", len(examples), self._min_examples)
            return False
        try:
            import dspy
            from dspy.teleprompt import BootstrapFewShot

            class CodeImprover(dspy.Signature):
                """Improve the given code based on context."""
                original_code = dspy.InputField(desc="Original code to improve")
                file_context  = dspy.InputField(desc="File path and surrounding context")
                improved_code = dspy.OutputField(desc="Improved version of the code")

            program = dspy.Predict(CodeImprover)
            optimizer = BootstrapFewShot(metric=self._acceptance_metric, max_bootstrapped_demos=4)
            self._optimized_program = optimizer.compile(program, trainset=examples[:100])
            logger.info("DSPy optimization complete with %d examples", len(examples))
            return True
        except Exception as e:
            logger.error("Optimization failed: %s", e)
            return False

    def _acceptance_metric(self, example, pred, trace=None) -> bool:
        """Simple metric: generated code is non-empty and different from input."""
        return bool(pred.improved_code) and pred.improved_code != example.original_code

    def suggest(self, code: str, context: str = "") -> Optional[str]:
        """Use optimized program to suggest code improvement."""
        if not self._optimized_program:
            return None
        try:
            result = self._optimized_program(original_code=code, file_context=context)
            return result.improved_code
        except Exception as e:
            logger.error("Suggestion failed: %s", e)
            return None

    def stats(self) -> dict:
        return {
            "loaded": self._lm is not None,
            "optimized": self._optimized_program is not None,
            "min_examples_needed": self._min_examples,
            "accepted_count": self._store.accepted_count() if self._store else 0,
            "ready_to_optimize": self.should_optimize(),
        }
