"""
File watcher: monitor raw/ for new/changed .md files → auto-pipeline.

Flow on file event (debounced):
  new/modified .md in raw/ → wait debounce_secs → ingest → compile
  → if auto_approve: approve + git commit

Uses watchdog for cross-platform filesystem events.
Uses threading.Timer for debounce: events collected for debounce_secs,
then processed as a batch (avoids hammering Ollama on rapid saves).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

log = logging.getLogger(__name__)

_WATCHED_SUFFIXES = frozenset({".md", ".xlsx", ".docx", ".pptx", ".csv", ".pdf"})


class _DebounceHandler(FileSystemEventHandler):
    """
    Collects raw/ file-change events and fires callback after debounce_secs of silence.
    Non-.md sources are converted to markdown before the callback fires so callers
    always receive a list of .md paths.
    Thread-safe: events arrive on watchdog thread, callback fires on timer thread.
    """

    def __init__(
        self,
        callback: Callable[[list[str]], None],
        debounce_secs: float,
        config=None,
    ) -> None:
        super().__init__()
        self._callback = callback
        self._debounce_secs = debounce_secs
        self._config = config
        self._pending: set[str] = set()
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    # ── watchdog hooks ────────────────────────────────────────────────────────

    def on_created(self, event: FileSystemEvent) -> None:
        self._handle(event)

    def on_modified(self, event: FileSystemEvent) -> None:
        self._handle(event)

    def on_moved(self, event: FileSystemEvent) -> None:
        # Handle renames into raw/ (e.g. Obsidian saves via temp rename)
        dest = getattr(event, "dest_path", None)
        if dest and Path(dest).suffix.lower() in _WATCHED_SUFFIXES:
            self._enqueue(dest)

    # ── internal ──────────────────────────────────────────────────────────────

    def _handle(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        src = getattr(event, "src_path", "")
        if Path(src).suffix.lower() in _WATCHED_SUFFIXES:
            self._enqueue(src)

    def _enqueue(self, path: str) -> None:
        with self._lock:
            self._pending.add(path)
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce_secs, self._fire)
            self._timer.daemon = True
            self._timer.start()

    def _fire(self) -> None:
        with self._lock:
            paths = list(self._pending)
            self._pending.clear()
            self._timer = None
        if not paths:
            return

        md_paths: list[str] = []
        for p in paths:
            path = Path(p)
            suffix = path.suffix.lower()
            if suffix == ".md":
                md_paths.append(p)
            elif self._config is not None:
                self._convert_and_collect(path, suffix, md_paths)

        if md_paths:
            self._callback(md_paths)

    def _convert_and_collect(self, path: Path, suffix: str, out: list[str]) -> None:
        """Run the appropriate converter for a changed non-.md file, appending .md output paths."""
        try:
            from .pipeline.ingest import (  # lazy import keeps watcher.py lightweight
                convert_csv_to_markdown,
                convert_docx_to_markdown,
                convert_pdf_to_markdown,
                convert_pptx_to_markdown,
                convert_xlsx_to_markdown,
            )
        except ImportError as e:
            log.warning("Watcher: could not import converters: %s", e)
            return

        converters = {
            ".xlsx": convert_xlsx_to_markdown,
            ".docx": convert_docx_to_markdown,
            ".pptx": convert_pptx_to_markdown,
            ".csv": convert_csv_to_markdown,
            ".pdf": convert_pdf_to_markdown,
        }
        converter = converters.get(suffix)
        if converter is None:
            return
        try:
            converted = converter(path, overwrite=True, config=self._config)
            out.extend(str(cp) for cp in converted)
        except Exception as e:
            log.warning("Watcher: conversion failed for %s: %s", path.name, e)

    def flush(self) -> None:
        """Force-fire any pending events immediately (used in tests / shutdown)."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            paths = list(self._pending)
            self._pending.clear()
        if paths:
            self._callback(paths)


# ── Public API ────────────────────────────────────────────────────────────────


def watch(
    config,
    client,
    db,
    on_event: Callable[[list[str]], None],
    debounce_secs: float | None = None,
) -> None:
    """
    Block until KeyboardInterrupt. Calls on_event(paths) after each debounced batch.

    config       — Config (uses config.raw_dir, config.pipeline.watch_debounce)
    client       — OllamaClient
    db           — StateDB
    on_event     — callback(changed_paths: list[str]) — runs on timer thread
    debounce_secs — override config.pipeline.watch_debounce
    """
    if debounce_secs is None:
        debounce_secs = config.pipeline.watch_debounce

    handler = _DebounceHandler(on_event, debounce_secs, config=config)
    observer = Observer()
    observer.schedule(handler, str(config.raw_dir), recursive=True)
    observer.start()
    log.info("Watching %s (debounce=%.1fs)", config.raw_dir, debounce_secs)

    try:
        while observer.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        handler.flush()
        observer.stop()
        observer.join()
