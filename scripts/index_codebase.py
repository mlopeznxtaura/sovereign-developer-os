"""
scripts/index_codebase.py — One-shot full codebase index (Cluster 03)
"""
import argparse, json, logging
from core.rag import CodebaseRAG

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.json")
parser.add_argument("--path", nargs="+")
args = parser.parse_args()

with open(args.config) as f:
    cfg = json.load(f)

rag = CodebaseRAG(cfg)
paths = args.path or cfg.get("workspace", {}).get("paths", ["."])
n = rag.index(paths)
print(f"Indexed {n} documents from {paths}")
print(rag.get_stats())
