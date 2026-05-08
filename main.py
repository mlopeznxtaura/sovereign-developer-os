"""
main.py — Sovereign Developer OS entrypoint (Cluster 03)

Wires: RAG + memory + agent + inference + fine-tuner + API + file watcher.
Run: python main.py --config config.json
"""
import json, logging, argparse, signal, os
from memory.store import SovereignStore
from core.rag import CodebaseRAG
from core.inference import LocalInference
from core.session import DeveloperSession
from core.finetune import FineTuner
from agent.graph import SovereignAgent
from agent.optimizer import SovereignOptimizer
from memory.indexer import IncrementalIndexer
from api.server import start_api_server
from telemetry.metrics import SovereignMetrics

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


class SovereignApp:
    def __init__(self, config):
        self._cfg = config
        self.store     = SovereignStore(config)
        self.rag       = CodebaseRAG(config)
        self.inference = LocalInference(config)
        self.agent     = SovereignAgent(config, rag=self.rag, store=self.store)
        self.finetuner = FineTuner(config, store=self.store)
        self.optimizer = SovereignOptimizer(config, store=self.store)
        self.indexer   = IncrementalIndexer(self.rag, config)
        self.metrics   = SovereignMetrics(config)

    def start(self):
        logger.info("Starting Sovereign Developer OS")
        self.inference.load()
        self.rag.load_existing() or self.rag.index(
            self._cfg.get("workspace", {}).get("paths", ["."])
        )
        self.agent.load()
        self.optimizer.load()
        self.indexer.start_watching()
        self.finetuner.start_scheduler()
        self.metrics.start_server()
        start_api_server(self._cfg, sovereign_app=self)
        logger.info("Sovereign Developer OS ready. API at http://0.0.0.0:%d",
                    self._cfg.get("api", {}).get("port", 8001))
        signal.pause()

    def stop(self):
        self.indexer.stop_watching()
        self.finetuner.stop_scheduler()
        logger.info("Sovereign Developer OS stopped")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    app = SovereignApp(cfg)
    signal.signal(signal.SIGINT,  lambda s,f: app.stop())
    signal.signal(signal.SIGTERM, lambda s,f: app.stop())
    app.start()

if __name__ == "__main__":
    main()
