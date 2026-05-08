"""
api/routes.py — FastAPI routes (Cluster 03: Sovereign Developer OS)

Endpoints: chat, index, suggestions, memory, fine-tune status.
"""
import time
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

router = APIRouter()
_app_ref = None

def set_app(app): global _app_ref; _app_ref = app


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class IndexRequest(BaseModel):
    paths: Optional[list] = None
    force: bool = False

class SuggestionAccept(BaseModel):
    suggestion_id: int
    feedback: Optional[str] = None

class MemorySetRequest(BaseModel):
    key: str
    value: object


@router.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}

@router.get("/status")
def status():
    if not _app_ref: return {"loaded": False}
    return {
        "loaded": True,
        "inference_backend": _app_ref.inference.get_metadata() if _app_ref.inference else None,
        "rag_stats": _app_ref.rag.get_stats() if _app_ref.rag else None,
        "store_stats": _app_ref.store.stats() if _app_ref.store else None,
        "finetune_stats": _app_ref.finetuner.stats() if _app_ref.finetuner else None,
        "optimizer_stats": _app_ref.optimizer.stats() if _app_ref.optimizer else None,
    }

@router.post("/chat")
def chat(req: ChatRequest):
    if not _app_ref or not _app_ref.agent:
        raise HTTPException(503, "Agent not loaded")
    session_id = req.session_id or "default"
    result = _app_ref.agent.chat(session_id, req.message)
    return result

@router.post("/index")
def index_codebase(req: IndexRequest, background_tasks: BackgroundTasks):
    if not _app_ref or not _app_ref.rag:
        raise HTTPException(503, "RAG not loaded")
    paths = req.paths or _app_ref._cfg.get("workspace", {}).get("paths", ["."])
    background_tasks.add_task(_app_ref.rag.index, paths, req.force)
    return {"queued": True, "paths": paths}

@router.get("/history/{session_id}")
def get_history(session_id: str, limit: int = 20):
    if not _app_ref or not _app_ref.store:
        raise HTTPException(503, "Store not loaded")
    return _app_ref.store.get_history(session_id, limit=limit)

@router.get("/suggestions")
def get_suggestions(limit: int = 50):
    if not _app_ref or not _app_ref.store:
        raise HTTPException(503)
    return _app_ref.store.get_accepted_suggestions(limit=limit)

@router.post("/suggestions/accept")
def accept_suggestion(req: SuggestionAccept):
    if not _app_ref or not _app_ref.store:
        raise HTTPException(503)
    _app_ref.store.accept_suggestion(req.suggestion_id, req.feedback)
    return {"ok": True, "suggestion_id": req.suggestion_id}

@router.post("/finetune/trigger")
def trigger_finetune(background_tasks: BackgroundTasks):
    if not _app_ref or not _app_ref.finetuner:
        raise HTTPException(503)
    background_tasks.add_task(_app_ref.finetuner.run)
    return {"queued": True}

@router.get("/rag/stats")
def rag_stats():
    if not _app_ref or not _app_ref.rag:
        raise HTTPException(503)
    return _app_ref.rag.get_stats()
