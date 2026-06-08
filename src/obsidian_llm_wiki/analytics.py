"""
Analytics layer for the pipeline.

Tracks per-document and per-run token usage, timing, and machine context.
Data is stored in the state.db SQLite (machine_profiles, pipeline_runs, doc_metrics)
and optionally exported to a JSONL sidecar file.

Usage pattern:
    collector = AnalyticsCollector(db, config, "ingest")
    set_active_collector(collector)
    try:
        run_pipeline(...)
    finally:
        set_active_collector(None)
        collector.flush(jsonl_path=Path("analytics.jsonl"))

Thread safety: DocRecord accumulation uses a lock. Per-doc token tracking
uses thread-local storage so parallel workers don't interfere.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .state import StateDB

# ── Per-doc thread-local context ──────────────────────────────────────────────

_tls = threading.local()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def start_doc(doc_path: str, step: str) -> None:
    """Call at the start of processing a document or concept."""
    _tls.doc_path = doc_path
    _tls.step = step
    _tls.t0 = time.monotonic()
    _tls.started_at = _now_iso()
    _tls.input_tokens = 0
    _tls.output_tokens = 0
    _tls.chunks = 0
    _tls.model = ""
    _tls.provider = ""


def accumulate_tokens(
    input_tokens: int, output_tokens: int, model: str = "", provider: str = ""
) -> None:
    """Called from emit_event on each successful provider_request to accumulate tokens."""
    if not hasattr(_tls, "doc_path"):
        return
    _tls.input_tokens = getattr(_tls, "input_tokens", 0) + input_tokens
    _tls.output_tokens = getattr(_tls, "output_tokens", 0) + output_tokens
    _tls.chunks = getattr(_tls, "chunks", 0) + 1
    if model:
        _tls.model = model
    if provider:
        _tls.provider = provider


def end_doc(status: str = "ok", error: str | None = None) -> None:
    """
    End current doc context and record to active collector.

    Only records when at least one LLM chunk was processed (chunks > 0),
    so early-continue paths that never call the LLM are silently ignored.
    """
    if not hasattr(_tls, "doc_path"):
        return
    doc_path = _tls.doc_path
    del _tls.doc_path  # clear so future calls know there's no active doc

    if getattr(_tls, "chunks", 0) == 0 and status not in ("failed", "ok"):
        return  # skipped before any LLM call — not interesting

    collector = get_active_collector()
    if collector is None:
        return

    duration_ms = int((time.monotonic() - _tls.t0) * 1000)
    record = DocRecord(
        run_id=collector.run_id,
        doc_path=doc_path,
        pipeline_step=_tls.step,
        started_at=_tls.started_at,
        duration_ms=duration_ms,
        input_tokens=getattr(_tls, "input_tokens", 0),
        output_tokens=getattr(_tls, "output_tokens", 0),
        chunk_count=getattr(_tls, "chunks", 0),
        model=getattr(_tls, "model", ""),
        provider=getattr(_tls, "provider", ""),
        status=status,
        error=error,
    )
    collector.record_doc(record)


# ── Module-level active collector (set by CLI commands) ───────────────────────

_active_collector: AnalyticsCollector | None = None
_collector_lock = threading.Lock()


def set_active_collector(c: AnalyticsCollector | None) -> None:
    global _active_collector
    with _collector_lock:
        _active_collector = c


def get_active_collector() -> AnalyticsCollector | None:
    with _collector_lock:
        return _active_collector


# ── Data types ────────────────────────────────────────────────────────────────


@dataclass
class DocRecord:
    run_id: str
    doc_path: str
    pipeline_step: str
    started_at: str
    duration_ms: int
    input_tokens: int
    output_tokens: int
    chunk_count: int
    model: str
    provider: str
    status: str
    error: str | None = None


# ── Collector ─────────────────────────────────────────────────────────────────


class AnalyticsCollector:
    """
    Thread-safe in-memory accumulator for one pipeline run.
    Call flush() at end to persist to SQLite + optional JSONL.
    """

    def __init__(self, db: StateDB, config: Any, pipeline_step: str) -> None:
        self.run_id = str(uuid.uuid4())
        self._db = db
        self._config = config
        self._step = pipeline_step
        self._lock = threading.Lock()
        self._docs: list[DocRecord] = []
        self._started_mono = time.monotonic()
        self._started_at = _now_iso()

    def record_doc(self, record: DocRecord) -> None:
        with self._lock:
            self._docs.append(record)

    def flush(self, jsonl_path: Path | None = None) -> None:
        """Write aggregated run to SQLite analytics tables and optional JSONL."""
        from .machine_info import get_machine_profile

        duration_ms = int((time.monotonic() - self._started_mono) * 1000)
        finished_at = _now_iso()

        with self._lock:
            docs = list(self._docs)

        total_input = sum(d.input_tokens for d in docs)
        total_output = sum(d.output_tokens for d in docs)
        docs_processed = sum(1 for d in docs if d.pipeline_step == "ingest")
        concepts_compiled = sum(1 for d in docs if d.pipeline_step == "compile")

        machine = get_machine_profile()
        config_snapshot = _config_snapshot(self._config)

        conn: sqlite3.Connection = self._db._conn
        now = _now_iso()

        # Upsert machine profile
        existing = conn.execute(
            "SELECT id FROM machine_profiles WHERE id = ?", (machine["id"],)
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE machine_profiles SET last_seen_at = ? WHERE id = ?",
                (now, machine["id"]),
            )
        else:
            conn.execute(
                """INSERT INTO machine_profiles
                   (id, hostname, cpu_model, cpu_cores, ram_gb,
                    gpu_model, gpu_vram_gb, os_platform, python_version,
                    first_seen_at, last_seen_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    machine["id"],
                    machine["hostname"],
                    machine["cpu_model"],
                    machine["cpu_cores"],
                    machine["ram_gb"],
                    machine["gpu_model"],
                    machine["gpu_vram_gb"],
                    machine["os_platform"],
                    machine["python_version"],
                    now,
                    now,
                ),
            )

        fast_model = getattr(getattr(self._config, "models", None), "fast", None)
        heavy_model = getattr(getattr(self._config, "models", None), "heavy", None)
        provider = getattr(
            getattr(self._config, "effective_provider", None), "name", None
        )

        conn.execute(
            """INSERT OR REPLACE INTO pipeline_runs
               (run_id, machine_id, pipeline_step, started_at, finished_at,
                duration_ms, docs_processed, concepts_compiled,
                total_input_tokens, total_output_tokens,
                fast_model, heavy_model, provider, config_snapshot)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                self.run_id,
                machine["id"],
                self._step,
                self._started_at,
                finished_at,
                duration_ms,
                docs_processed,
                concepts_compiled,
                total_input,
                total_output,
                fast_model,
                heavy_model,
                provider,
                config_snapshot,
            ),
        )

        if docs:
            conn.executemany(
                """INSERT INTO doc_metrics
                   (run_id, doc_path, pipeline_step, started_at, duration_ms,
                    input_tokens, output_tokens, chunk_count, model, provider,
                    status, error)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        d.run_id,
                        d.doc_path,
                        d.pipeline_step,
                        d.started_at,
                        d.duration_ms,
                        d.input_tokens,
                        d.output_tokens,
                        d.chunk_count,
                        d.model,
                        d.provider,
                        d.status,
                        d.error,
                    )
                    for d in docs
                ],
            )

        conn.commit()

        if jsonl_path is not None:
            _export_jsonl(conn, jsonl_path)

    def summary_line(self) -> str:
        """One-line summary for end-of-run log message."""
        with self._lock:
            docs = list(self._docs)
        tin = sum(d.input_tokens for d in docs)
        tout = sum(d.output_tokens for d in docs)
        n = len(docs)
        elapsed = time.monotonic() - self._started_mono
        return (
            f"{n} doc(s) | {tin:,} in + {tout:,} out tokens"
            f" | {elapsed:.1f}s total"
        )


# ── Summary queries (for `wiki analytics` command) ────────────────────────────


def get_summary(db: StateDB) -> dict:
    """
    Return structured summary data for the analytics CLI command.

    Includes:
    - all_time: cumulative token totals and doc/concept counts
    - recent_runs: last 15 pipeline_runs rows
    - top_slow_docs: 10 slowest doc_metrics rows
    - efficiency_trend: tokens_per_doc and ms_per_doc per run (last 30)
    - machines: all machine_profiles rows
    """
    conn = db._conn

    all_time = conn.execute(
        """SELECT
               COUNT(DISTINCT run_id)       AS total_runs,
               SUM(docs_processed)          AS total_docs,
               SUM(concepts_compiled)       AS total_concepts,
               SUM(total_input_tokens)      AS total_input_tokens,
               SUM(total_output_tokens)     AS total_output_tokens,
               SUM(total_input_tokens + total_output_tokens) AS total_tokens,
               SUM(duration_ms) / 1000.0    AS total_seconds
           FROM pipeline_runs"""
    ).fetchone()

    recent_runs = conn.execute(
        """SELECT run_id, pipeline_step, started_at, duration_ms,
                  docs_processed, concepts_compiled,
                  total_input_tokens, total_output_tokens,
                  fast_model, provider
           FROM pipeline_runs
           ORDER BY started_at DESC
           LIMIT 15"""
    ).fetchall()

    top_slow = conn.execute(
        """SELECT doc_path, pipeline_step, duration_ms,
                  input_tokens, output_tokens, chunk_count,
                  model, status, started_at
           FROM doc_metrics
           WHERE status = 'ok'
           ORDER BY duration_ms DESC
           LIMIT 10"""
    ).fetchall()

    efficiency = conn.execute(
        """SELECT run_id, started_at, pipeline_step,
                  docs_processed, concepts_compiled,
                  total_input_tokens, total_output_tokens,
                  duration_ms
           FROM pipeline_runs
           WHERE docs_processed + concepts_compiled > 0
           ORDER BY started_at DESC
           LIMIT 30"""
    ).fetchall()

    machines = conn.execute(
        """SELECT id, hostname, cpu_model, cpu_cores, ram_gb,
                  gpu_model, gpu_vram_gb, os_platform, python_version,
                  first_seen_at, last_seen_at
           FROM machine_profiles
           ORDER BY last_seen_at DESC"""
    ).fetchall()

    return {
        "all_time": dict(all_time) if all_time else {},
        "recent_runs": [dict(r) for r in recent_runs],
        "top_slow_docs": [dict(r) for r in top_slow],
        "efficiency_trend": [dict(r) for r in efficiency],
        "machines": [dict(r) for r in machines],
    }


# ── JSONL export ──────────────────────────────────────────────────────────────


def _export_jsonl(conn: sqlite3.Connection, path: Path) -> None:
    """Overwrite path with a full JSONL export of all pipeline_runs + doc_metrics."""
    runs = conn.execute(
        "SELECT * FROM pipeline_runs ORDER BY started_at"
    ).fetchall()
    docs = conn.execute(
        "SELECT * FROM doc_metrics ORDER BY run_id, id"
    ).fetchall()

    docs_by_run: dict[str, list[dict]] = {}
    for d in docs:
        d_dict = dict(d)
        docs_by_run.setdefault(d_dict["run_id"], []).append(d_dict)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for run in runs:
                r_dict = dict(run)
                r_dict["doc_metrics"] = docs_by_run.get(r_dict["run_id"], [])
                f.write(json.dumps(r_dict, ensure_ascii=True))
                f.write("\n")
    except Exception:
        pass


# ── Helpers ───────────────────────────────────────────────────────────────────


def _config_snapshot(config: Any) -> str:
    """Capture key config fields as compact JSON string."""
    try:
        snap: dict[str, Any] = {}
        if hasattr(config, "models"):
            snap["fast"] = config.models.fast
            snap["heavy"] = config.models.heavy
        if hasattr(config, "effective_provider"):
            snap["provider"] = getattr(config.effective_provider, "name", None)
            snap["fast_ctx"] = getattr(config.effective_provider, "fast_ctx", None)
            snap["heavy_ctx"] = getattr(config.effective_provider, "heavy_ctx", None)
        return json.dumps(snap)
    except Exception:
        return "{}"
