"""
core/finetune.py — Axolotl + Unsloth periodic fine-tuning (Cluster 03)

Periodically fine-tunes the local model on accepted suggestions
using Axolotl config + Unsloth for fast LoRA training.

SDKs: Axolotl, Unsloth, Weights & Biases
"""
import json
import logging
import os
import subprocess
import tempfile
import time
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


AXOLOTL_CONFIG_TEMPLATE = """
base_model: {base_model}
model_type: LlamaForCausalLM
tokenizer_type: LlamaTokenizer

load_in_8bit: false
load_in_4bit: true
strict: false

datasets:
  - path: {dataset_path}
    type: alpaca

dataset_prepared_path: {prepared_path}
val_set_size: 0.05

output_dir: {output_dir}

adapter: lora
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - v_proj

sequence_len: 2048
sample_packing: true

num_epochs: 3
micro_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 0.0002
optimizer: adamw_bnb_8bit
lr_scheduler: cosine
warmup_steps: 10

logging_steps: 10
save_steps: 100
eval_steps: 50

use_wandb: {use_wandb}
wandb_project: sovereign-developer-os
wandb_run_name: finetune-{timestamp}

unsloth: true
"""


class FineTuner:
    """
    Periodic fine-tuning on accepted suggestions.
    Uses Axolotl YAML config + Unsloth for 2x faster LoRA training.
    """

    def __init__(self, config: dict, store=None):
        self._cfg = config
        self._ft_cfg = config.get("finetune", {})
        self._store = store
        self._enabled = self._ft_cfg.get("enabled", False)
        self._interval_h = self._ft_cfg.get("interval_hours", 24)
        self._min_suggestions = self._ft_cfg.get("min_accepted_suggestions", 50)
        self._output_dir = os.path.expanduser(self._ft_cfg.get("output_dir", "~/.sovereign/finetuned"))
        self._last_run = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _build_dataset(self, suggestions: list) -> str:
        """Convert accepted suggestions to Alpaca-format JSONL."""
        records = []
        for s in suggestions:
            if not s.get("suggested"):
                continue
            instruction = f"Improve this code"
            if s.get("file_path"):
                instruction += f" from {s['file_path']}"
            records.append({
                "instruction": instruction,
                "input": s.get("original", ""),
                "output": s["suggested"],
            })
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        for r in records:
            tmp.write(json.dumps(r) + "
")
        tmp.close()
        return tmp.name

    def _write_axolotl_config(self, dataset_path: str) -> str:
        ollama_cfg = self._cfg.get("ollama", {})
        base_model = ollama_cfg.get("model", "llama3")
        prepared = os.path.join(self._output_dir, "prepared")
        os.makedirs(self._output_dir, exist_ok=True)

        config_str = AXOLOTL_CONFIG_TEMPLATE.format(
            base_model=base_model,
            dataset_path=dataset_path,
            prepared_path=prepared,
            output_dir=self._output_dir,
            use_wandb="true",
            timestamp=int(time.time()),
        )
        cfg_path = os.path.join(self._output_dir, "axolotl_config.yaml")
        Path(cfg_path).write_text(config_str)
        return cfg_path

    def should_run(self) -> bool:
        if not self._enabled:
            return False
        if time.time() - self._last_run < self._interval_h * 3600:
            return False
        if not self._store:
            return False
        count = self._store.accepted_count()
        return count >= self._min_suggestions

    def run(self) -> dict:
        """Trigger fine-tuning. Runs axolotl in subprocess."""
        if not self.should_run():
            return {"skipped": True, "reason": "conditions not met"}

        suggestions = self._store.get_accepted_suggestions(limit=500)
        if len(suggestions) < self._min_suggestions:
            return {"skipped": True, "reason": f"only {len(suggestions)} suggestions"}

        logger.info("Starting fine-tune on %d suggestions", len(suggestions))
        dataset_path = self._build_dataset(suggestions)
        config_path = self._write_axolotl_config(dataset_path)

        try:
            import wandb
            wandb.init(project="sovereign-developer-os", name=f"finetune-{int(time.time())}")
        except ImportError:
            pass

        t0 = time.time()
        try:
            result = subprocess.run(
                ["python", "-m", "axolotl.cli.train", config_path],
                capture_output=True, text=True, timeout=3600
            )
            elapsed = time.time() - t0
            self._last_run = time.time()
            success = result.returncode == 0
            logger.info("Fine-tune %s in %.0fs", "succeeded" if success else "failed", elapsed)
            return {
                "success": success,
                "elapsed_s": round(elapsed, 1),
                "output_dir": self._output_dir,
                "suggestions_used": len(suggestions),
                "stdout_tail": result.stdout[-500:] if result.stdout else "",
                "stderr_tail": result.stderr[-500:] if result.stderr else "",
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Timeout after 3600s"}
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            os.unlink(dataset_path)

    def start_scheduler(self) -> None:
        """Start background thread that checks and runs fine-tuning on schedule."""
        def _loop():
            while self._running:
                if self.should_run():
                    logger.info("Fine-tune scheduler: running...")
                    result = self.run()
                    logger.info("Fine-tune result: %s", result)
                time.sleep(3600)

        self._running = True
        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        logger.info("Fine-tune scheduler started (interval=%dh)", self._interval_h)

    def stop_scheduler(self) -> None:
        self._running = False

    def stats(self) -> dict:
        return {
            "enabled": self._enabled,
            "interval_hours": self._interval_h,
            "min_suggestions": self._min_suggestions,
            "output_dir": self._output_dir,
            "last_run": self._last_run,
            "should_run": self.should_run(),
            "accepted_count": self._store.accepted_count() if self._store else 0,
        }
