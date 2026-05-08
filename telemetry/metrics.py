"""
telemetry/metrics.py — Prometheus + OpenTelemetry (Cluster 03: Sovereign Developer OS)
"""
import logging
from prometheus_client import Counter, Histogram, Gauge, start_http_server

logger = logging.getLogger(__name__)


class SovereignMetrics:
    def __init__(self, config: dict):
        self._port = config.get("telemetry", {}).get("prometheus_port", 9091)
        self._started = False

        self.queries_total      = Counter("sovereign_queries_total", "Total RAG queries")
        self.query_latency_ms   = Histogram("sovereign_query_latency_ms", "Query latency",
                                            buckets=[50,100,200,500,1000,2000,5000])
        self.chat_turns_total   = Counter("sovereign_chat_turns_total", "Total chat turns")
        self.suggestions_total  = Counter("sovereign_suggestions_total", "Suggestions generated")
        self.accepted_total     = Counter("sovereign_accepted_total", "Suggestions accepted")
        self.finetune_runs      = Counter("sovereign_finetune_runs_total", "Fine-tune runs",
                                          labelnames=["status"])
        self.index_docs         = Gauge("sovereign_indexed_docs", "Documents in index")
        self.accepted_count     = Gauge("sovereign_accepted_count", "Total accepted suggestions")

    def start_server(self):
        if not self._started:
            try:
                start_http_server(self._port)
                self._started = True
                logger.info("Prometheus metrics at http://0.0.0.0:%d/metrics", self._port)
            except Exception as e:
                logger.warning("Could not start Prometheus: %s", e)
