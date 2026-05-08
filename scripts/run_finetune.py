"""
scripts/run_finetune.py — Trigger fine-tuning from CLI (Cluster 03)
"""
import argparse, json, logging
from memory.store import SovereignStore
from core.finetune import FineTuner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.json")
parser.add_argument("--force", action="store_true")
args = parser.parse_args()

with open(args.config) as f:
    cfg = json.load(f)

if args.force:
    cfg["finetune"]["enabled"] = True

store = SovereignStore(cfg)
ft = FineTuner(cfg, store=store)
print("Fine-tune stats:", ft.stats())
result = ft.run()
print("Result:", result)
