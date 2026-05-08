# Cluster 03 — Sovereign Developer OS

> A fully local, self-improving dev environment — your own AI pair programmer with memory

## Overview

A local, persistent, self-improving developer assistant that runs entirely on your machine.
Knows your codebase, remembers your decisions, and gets better over time through fine-tuning
on your own accepted suggestions.

## Architecture

```
Workspace files → memory/indexer.py → Chroma (vector) + SQLite (metadata)
                                              ↓
User query → agent/graph.py (LangGraph) → memory/retriever.py → Ollama LLM
                                              ↓
                                    api/server.py (FastAPI)
                                              ↓
                                    ui/app.py (Tauri desktop)
                                              ↓
                              core/finetune.py (Axolotl/Unsloth periodic)
```

## 20 SDKs in this cluster

| SDK | Role |
|---|---|
| Ollama | Local LLM inference (llama3/mistral/phi) |
| LangGraph | Multi-step reasoning chains |
| LlamaIndex | Codebase-aware RAG pipeline |
| Chroma | Vector store for code embeddings |
| Unsloth | Fast LoRA fine-tuning on accepted suggestions |
| Axolotl | Fine-tuning config and training loop |
| ExLlamaV2 | Fast local GGUF/EXL2 inference |
| Tauri | Desktop app shell |
| SQLAlchemy | Conversation + suggestion history |
| FastAPI | REST API backend |
| Pydantic | Structured LLM output validation |
| Instructor | Typed LLM responses |
| DSPy | Prompt optimization |
| GGUF Tools | Model loading (llama-cpp-python) |
| Weights & Biases | Fine-tuning experiment tracking |
| GitHub Actions SDK | CI integration |
| Prometheus Client | Metrics |
| OpenTelemetry | Tracing |
| Nx Build System | Monorepo build tooling |
| Nix SDK | Reproducible dev environment |

## Build order

1. `core/rag.py` — LlamaIndex pipeline indexing local code to Chroma via Ollama embeddings
2. `memory/indexer.py` — file watcher + incremental index updates
3. `memory/store.py` — Chroma + SQLAlchemy metadata store
4. `memory/retriever.py` — hybrid BM25 + vector retrieval
5. `agent/tools.py` — code execution, file read/write, shell tools
6. `agent/graph.py` — LangGraph reasoning graph
7. `agent/optimizer.py` — DSPy prompt optimization
8. `core/inference.py` — Ollama + ExLlamaV2 inference layer
9. `core/session.py` — conversation + suggestion history
10. `core/finetune.py` — Axolotl + Unsloth periodic fine-tuning
11. `api/routes.py` — FastAPI routes
12. `api/server.py` — FastAPI app wiring
13. `telemetry/metrics.py` — Prometheus + OpenTelemetry
14. `ui/app.py` — Tauri desktop app entry

## Quickstart

```bash
pip install -r requirements.txt
# Start Ollama: ollama serve && ollama pull llama3 && ollama pull nomic-embed-text
python scripts/index_codebase.py --config config.json --path ~/code
python -m api.server --config config.json
```
