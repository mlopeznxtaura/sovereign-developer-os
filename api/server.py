"""
api/server.py — FastAPI app wiring (Cluster 03: Sovereign Developer OS)
"""
import logging
import json
import threading
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router, set_app

logger = logging.getLogger(__name__)

app = FastAPI(title="Sovereign Developer OS API", version="1.0.0",
              description="Local AI pair programmer — RAG + LangGraph + fine-tuning")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router, prefix="/api/v1")


def start_api_server(config: dict, sovereign_app=None) -> threading.Thread:
    if sovereign_app:
        set_app(sovereign_app)
    api_cfg = config.get("api", {})
    host = api_cfg.get("host", "0.0.0.0")
    port = api_cfg.get("port", 8001)
    def _run():
        logger.info("API server at http://%s:%d", host, port)
        uvicorn.run(app, host=host, port=port, log_level="warning")
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t
