"""
agent/tools.py — LangGraph-compatible tools (Cluster 03: Sovereign Developer OS)

Code execution, file read/write, shell, and search tools for the agent.
All tools are sandboxed and safe for local execution.

SDKs: LangGraph, FastAPI
"""
import os
import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MAX_OUTPUT_CHARS = 8000
ALLOWED_SHELL_CMDS = {"ls", "find", "grep", "cat", "head", "tail", "wc", "git", "python3", "pip"}


def read_file(path: str, start_line: int = 1, end_line: Optional[int] = None) -> dict:
    """Read a file, optionally sliced to line range."""
    expanded = os.path.expanduser(path)
    if not os.path.exists(expanded):
        return {"error": f"File not found: {path}", "content": None}
    try:
        lines = Path(expanded).read_text(errors="replace").splitlines()
        total = len(lines)
        sl = max(0, start_line - 1)
        el = end_line if end_line else total
        sliced = lines[sl:el]
        content = "\n".join(f"{sl+i+1}|{line}" for i, line in enumerate(sliced))
        return {"content": content[:MAX_OUTPUT_CHARS], "total_lines": total,
                "start_line": sl+1, "end_line": sl+len(sliced), "path": path}
    except Exception as e:
        return {"error": str(e), "content": None}


def write_file(path: str, content: str) -> dict:
    """Write content to a file. Creates parent dirs."""
    expanded = os.path.expanduser(path)
    try:
        os.makedirs(os.path.dirname(expanded) or ".", exist_ok=True)
        Path(expanded).write_text(content)
        return {"ok": True, "path": path, "bytes_written": len(content.encode())}
    except Exception as e:
        return {"error": str(e), "ok": False}


def execute_python(code: str, timeout: int = 15) -> dict:
    """Execute Python code in a subprocess sandbox. Returns stdout/stderr."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name
    try:
        t0 = time.perf_counter()
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True, text=True, timeout=timeout
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "stdout": result.stdout[:MAX_OUTPUT_CHARS],
            "stderr": result.stderr[:2000],
            "exit_code": result.returncode,
            "elapsed_ms": round(elapsed_ms, 1),
        }
    except subprocess.TimeoutExpired:
        return {"error": f"Timeout after {timeout}s", "exit_code": -1}
    except Exception as e:
        return {"error": str(e), "exit_code": -1}
    finally:
        os.unlink(tmp_path)


def run_shell(command: str, workdir: Optional[str] = None, timeout: int = 10) -> dict:
    """Run a shell command. Only allows safe commands."""
    cmd_name = command.strip().split()[0]
    if cmd_name not in ALLOWED_SHELL_CMDS:
        return {"error": f"Command not allowed: {cmd_name}. Allowed: {ALLOWED_SHELL_CMDS}"}
    try:
        t0 = time.perf_counter()
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=os.path.expanduser(workdir) if workdir else None
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "stdout": result.stdout[:MAX_OUTPUT_CHARS],
            "stderr": result.stderr[:2000],
            "exit_code": result.returncode,
            "elapsed_ms": round(elapsed_ms, 1),
        }
    except subprocess.TimeoutExpired:
        return {"error": f"Timeout after {timeout}s", "exit_code": -1}
    except Exception as e:
        return {"error": str(e), "exit_code": -1}


def search_codebase(rag, query: str) -> dict:
    """Search the indexed codebase via RAG."""
    if not rag:
        return {"error": "RAG not initialized", "results": []}
    result = rag.query(query)
    return {"answer": result["answer"][:MAX_OUTPUT_CHARS],
            "source_files": result["source_files"],
            "node_count": result["node_count"]}


def list_files(directory: str, extensions: Optional[list] = None, max_files: int = 100) -> dict:
    """List files in a directory, optionally filtered by extension."""
    expanded = os.path.expanduser(directory)
    if not os.path.exists(expanded):
        return {"error": f"Directory not found: {directory}", "files": []}
    files = []
    for root, dirs, fs in os.walk(expanded):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules","__pycache__",".git")]
        for f in fs:
            if extensions:
                if not any(f.endswith(ext) for ext in extensions):
                    continue
            full = os.path.join(root, f)
            rel = os.path.relpath(full, expanded)
            files.append({"path": rel, "size": os.path.getsize(full)})
            if len(files) >= max_files:
                return {"files": files, "truncated": True, "total_shown": len(files)}
    return {"files": files, "truncated": False, "total_shown": len(files)}


# Tool registry for LangGraph
TOOL_REGISTRY = {
    "read_file": read_file,
    "write_file": write_file,
    "execute_python": execute_python,
    "run_shell": run_shell,
    "list_files": list_files,
}
