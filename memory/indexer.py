"""
memory/indexer.py — File watcher + incremental index updates (Cluster 03: Sovereign Developer OS)

Watches workspace directories for file changes and incrementally
updates the Chroma index without full re-indexing.

SDKs: watchdog, LlamaIndex, Chroma
"""
import os
import time
import json
import logging
import threading
from pathlib import Path
from typing import Set

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent

logger = logging.getLogger(__name__)


class CodebaseEventHandler(FileSystemEventHandler):
    """Debounced file event handler — batches rapid changes."""

    def __init__(self, indexer, extensions: list, debounce_s: float = 2.0):
        self._indexer = indexer
        self._extensions = set(extensions)
        self._debounce_s = debounce_s
        self._pending: Set[str] = set()
        self._pending_deletes: Set[str] = set()
        self._lock = threading.Lock()
        self._timer: threading.Timer = None

    def _is_tracked(self, path: str) -> bool:
        return Path(path).suffix in self._extensions

    def on_modified(self, event):
        if not event.is_directory and self._is_tracked(event.src_path):
            with self._lock:
                self._pending.add(event.src_path)
            self._schedule_flush()

    def on_created(self, event):
        if not event.is_directory and self._is_tracked(event.src_path):
            with self._lock:
                self._pending.add(event.src_path)
            self._schedule_flush()

    def on_deleted(self, event):
        if not event.is_directory and self._is_tracked(event.src_path):
            with self._lock:
                self._pending_deletes.add(event.src_path)
            self._schedule_flush()

    def _schedule_flush(self):
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(self._debounce_s, self._flush)
        self._timer.daemon = True
        self._timer.start()

    def _flush(self):
        with self._lock:
            to_update = set(self._pending)
            to_delete = set(self._pending_deletes)
            self._pending.clear()
            self._pending_deletes.clear()

        if to_update:
            logger.info("Incremental update: %d files changed", len(to_update))
            self._indexer.update_files(list(to_update))

        if to_delete:
            logger.info("Removing %d deleted files from index", len(to_delete))
            self._indexer.remove_files(list(to_delete))


class IncrementalIndexer:
    """
    Watches workspace and incrementally keeps the Chroma index fresh.
    Uses LlamaIndex refresh_ref_docs for incremental updates.
    """

    def __init__(self, rag, config: dict):
        self._rag = rag
        self._cfg = config
        self._extensions = config.get("workspace", {}).get(
            "extensions", [".py", ".ts", ".js", ".go", ".rs", ".md", ".json"]
        )
        self._paths = [os.path.expanduser(p) for p in
                       config.get("workspace", {}).get("paths", ["."])]
        self._observer = None
        self._running = False
        self._update_count = 0

    def update_files(self, file_paths: list) -> int:
        """Re-index specific files. Returns number successfully updated."""
        from llama_index.core import SimpleDirectoryReader

        updated = 0
        for path in file_paths:
            if not os.path.exists(path):
                continue
            try:
                reader = SimpleDirectoryReader(input_files=[path])
                docs = reader.load_data()
                if docs and self._rag._index:
                    for doc in docs:
                        self._rag._index.refresh_ref_docs([doc])
                    updated += 1
                    self._update_count += 1
                    logger.debug("Updated index for: %s", path)
            except Exception as e:
                logger.warning("Failed to update %s: %s", path, e)

        return updated

    def remove_files(self, file_paths: list) -> int:
        """Remove deleted files from the index."""
        removed = 0
        if not self._rag._index:
            return 0
        for path in file_paths:
            try:
                # LlamaIndex uses doc_id based on file path
                doc_id = path
                self._rag._index.delete_ref_doc(doc_id, delete_from_docstore=True)
                removed += 1
                logger.debug("Removed from index: %s", path)
            except Exception as e:
                logger.debug("Could not remove %s (may not exist in index): %s", path, e)
        return removed

    def start_watching(self) -> None:
        """Start the file system watcher in a background thread."""
        self._observer = Observer()
        handler = CodebaseEventHandler(self, self._extensions, debounce_s=2.0)

        for path in self._paths:
            if os.path.exists(path):
                self._observer.schedule(handler, path, recursive=True)
                logger.info("Watching: %s", path)
            else:
                logger.warning("Watch path does not exist: %s", path)

        self._observer.start()
        self._running = True
        logger.info("File watcher started. Watching %d paths.", len(self._paths))

    def stop_watching(self) -> None:
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
        self._running = False
        logger.info("File watcher stopped. Total updates: %d", self._update_count)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def update_count(self) -> int:
        return self._update_count

    def get_stats(self) -> dict:
        return {
            "running": self._running,
            "watched_paths": self._paths,
            "extensions": self._extensions,
            "update_count": self._update_count,
        }
