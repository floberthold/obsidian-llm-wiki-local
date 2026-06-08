"""
Pipeline orchestrator — runs the full ingest → compile → lint → approve sequence.

Used by `olw run` and `olw watch`. Handles:
  - Selective compile (only concepts linked to changed sources)
  - Transient-failure retry (one additional round)
  - Optional stub creation after lint
  - Timing instrumentation per step
  - Dry-run mode (no LLM calls, no file writes)
"""

from __future__ import annotations

import os
import logging
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Callable

from ..config import Config
from ..protocols import LLMClientProtocol
from ..state import StateDB

log = logging.getLogger(__name__)


class FailureReason(StrEnum):
    TRANSIENT = "transient"  # timeout, connection reset — retry
    LLM_OUTPUT = "llm_output"  # bad JSON / schema mismatch — structured_output already retried 3×
    MISSING_SOURCES = "missing_sources"  # no readable source files
    UNKNOWN = "unknown"


@dataclass
class FailureRecord:
    concept: str
    reason: FailureReason
    error_msg: str = ""


@dataclass
class PipelineReport:
    ingested: int = 0
    compiled: int = 0
    failed: list[FailureRecord] = field(default_factory=list)
    published: int = 0
    bundles_created: int = 0
    lint_issues: int = 0
    stubs_created: int = 0
    rounds: int = 0
    timings: dict[str, float] = field(default_factory=dict)
    concept_timings: dict[str, float] = field(default_factory=dict)

    @property
    def failed_names(self) -> list[str]:
        return [f.concept for f in self.failed]


class PipelineOrchestrator:
    """
    Orchestrates the full pipeline. Caller is responsible for acquiring the
    pipeline lock before calling run() — this class does NOT lock internally.
    """

    def __init__(self, config: Config, client: LLMClientProtocol, db: StateDB) -> None:
        self.config = config
        self.client = client
        self.db = db

    def run(
        self,
        paths: list[str] | None = None,
        auto_approve: bool = False,
        build_bundles: bool = True,
        fix: bool = False,
        max_rounds: int = 2,
        dry_run: bool = False,
        on_progress: Callable[[str, int, int, float | None, str], None] | None = None,
    ) -> PipelineReport:
        """
        Run full pipeline: ingest → compile → lint → [stubs] → [approve] → [bundles].

        paths: specific raw note paths to ingest (None = ingest all changed notes)
        auto_approve: publish drafts immediately without manual review
        build_bundles: generate project/client/app/idea bundle notes under wiki/projects/
        fix: create stubs for broken wikilinks after lint
        max_rounds: maximum compile rounds (round 2 retries transient failures only)
        dry_run: report what would happen; no LLM calls, no file writes
        """
        from ..git_ops import git_commit
        from ..indexer import append_log, generate_index
        from ..pipeline.compile import approve_drafts
        from ..pipeline.ingest import (
            _find_document_source_dirs,
            _write_document_aggregate,
            collect_ingest_paths,
            ingest_note,
        )
        from ..pipeline.lint import run_lint
        from ..pipeline.maintain import create_stubs

        config = self.config
        client = self.client
        db = self.db
        report = PipelineReport()

        # ── Round 1: Ingest ────────────────────────────────────────────────────
        t0 = time.monotonic()
        ingested_paths: list[str] = []

        if paths is not None:
            md_paths = [str(p) for p in collect_ingest_paths(config, [Path(p) for p in paths])]
        else:
            md_paths = [str(p) for p in collect_ingest_paths(config)]

        log.info("── Ingest (%d note(s)) ──────────────────────────────────", len(md_paths))
        ingest_durations: list[float] = []
        ingest_processed_durations: list[float] = []
        ingest_total = len(md_paths)
        # Snapshot concept names once per ingest run to avoid repeated full-table scans.
        existing_topics = db.list_all_concept_names()
        if dry_run or ingest_total <= 1:
            for idx, raw_path_str in enumerate(md_paths, 1):
                step_t0 = time.monotonic()
                p = Path(raw_path_str)
                processed = False
                if not p.exists():
                    continue
                if dry_run:
                    log.info("[dry-run] would ingest: %s", p.name)
                    ingested_paths.append(raw_path_str)
                    report.ingested += 1
                    if on_progress:
                        eta = None
                        if idx < ingest_total:
                            eta = float(ingest_total - idx)
                        on_progress("ingest", idx, ingest_total, eta, p.name)
                    continue
                try:
                    result = ingest_note(
                        path=p,
                        config=config,
                        client=client,
                        db=db,
                        existing_topics=existing_topics,
                    )
                    if result is not None:
                        processed = True
                        report.ingested += 1
                        ingested_paths.append(raw_path_str)
                except Exception as e:
                    log.error("Ingest failed for %s: %s", p.name, e)
                finally:
                    elapsed = time.monotonic() - step_t0
                    ingest_durations.append(elapsed)
                    if processed:
                        ingest_processed_durations.append(elapsed)
                    if on_progress and ingest_total:
                        eta = None
                        if idx < ingest_total:
                            basis = ingest_processed_durations or ingest_durations
                            if basis:
                                avg = sum(basis) / len(basis)
                                eta = avg * (ingest_total - idx)
                        on_progress("ingest", idx, ingest_total, eta, p.name)
        else:
            env_parallel = os.getenv("OLLAMA_NUM_PARALLEL", "").strip()
            try:
                configured_workers = int(env_parallel) if env_parallel else 0
            except ValueError:
                configured_workers = 0
            if configured_workers <= 0:
                configured_workers = 4
            max_workers = max(1, min(ingest_total, configured_workers))
            log.info(
                "Parallel ingest enabled: %d worker(s) for %d note(s)",
                max_workers,
                ingest_total,
            )

            def _worker_index(worker_name: str) -> int:
                parts = worker_name.split("_")
                if len(parts) >= 2:
                    try:
                        return int(parts[-1])
                    except ValueError:
                        pass
                return 0

            def _make_lane(worker_name: str) -> str:
                idx = _worker_index(worker_name)
                cells = [f"W{i:02d}" if i == idx else "   " for i in range(max_workers)]
                return "|" + "|".join(cells) + "|"

            def _ingest_one(raw_path_str: str) -> tuple[str, str, bool, float]:
                step_t0 = time.monotonic()
                p = Path(raw_path_str)
                worker_name = threading.current_thread().name
                lane = _make_lane(worker_name)
                if not p.exists():
                    log.info("%s SKIP missing %s", lane, p.name)
                    return raw_path_str, p.name, False, time.monotonic() - step_t0

                log.info("%s START %s", lane, p.name)

                worker_db = StateDB(config.state_db_path)
                try:
                    result = ingest_note(
                        path=p,
                        config=config,
                        client=client,
                        db=worker_db,
                        existing_topics=existing_topics,
                    )
                    elapsed = time.monotonic() - step_t0
                    if result is not None:
                        log.info("%s DONE ingest %s (%.1fs)", lane, p.name, elapsed)
                    else:
                        log.info("%s DONE skip  %s (%.1fs)", lane, p.name, elapsed)
                    return raw_path_str, p.name, result is not None, elapsed
                except Exception as e:
                    log.error("Ingest failed for %s: %s", p.name, e)
                    elapsed = time.monotonic() - step_t0
                    log.info("%s DONE FAIL  %s (%.1fs)", lane, p.name, elapsed)
                    return raw_path_str, p.name, False, elapsed
                finally:
                    worker_db.close()

            completed = 0
            executor = ThreadPoolExecutor(max_workers=max_workers)
            interrupted = False
            futures = []
            try:
                futures = [executor.submit(_ingest_one, raw_path_str) for raw_path_str in md_paths]
                for future in as_completed(futures):
                    raw_path_str, note_name, success, duration_s = future.result()
                    completed += 1
                    ingest_durations.append(duration_s)
                    if success:
                        ingest_processed_durations.append(duration_s)
                        report.ingested += 1
                        ingested_paths.append(raw_path_str)

                    if on_progress and ingest_total:
                        eta = None
                        if completed < ingest_total:
                            basis = ingest_processed_durations or ingest_durations
                            if basis:
                                avg = sum(basis) / len(basis)
                                eta = avg * (ingest_total - completed)
                        on_progress("ingest", completed, ingest_total, eta, note_name)
            except KeyboardInterrupt:
                interrupted = True
                # Avoid waiting for all worker threads when user cancels.
                for future in futures:
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            finally:
                if not interrupted:
                    executor.shutdown(wait=True)

        report.timings["ingest"] = time.monotonic() - t0

        if not dry_run:
            for doc_dir in _find_document_source_dirs(config):
                try:
                    _write_document_aggregate(doc_dir, config, db)
                except Exception as e:
                    log.warning("Document aggregate failed for %s: %s", doc_dir.name, e)

        if not dry_run and report.ingested > 0:
            generate_index(config, db)
            append_log(config, f"run | ingested {report.ingested} note(s)")

        # ── Round 1: Compile ──────────────────────────────────────────────────
        # Normalize ingested paths to vault-relative for DB lookup (watchdog supplies
        # absolute paths; DB stores relative paths like raw/note.md).
        priority_concepts: list[str] | None = None
        if ingested_paths:
            relative_ingested = []
            for p_str in ingested_paths:
                try:
                    relative_ingested.append(Path(p_str).relative_to(config.vault).as_posix())
                except ValueError:
                    relative_ingested.append(Path(p_str).as_posix())  # already relative
            priority_concepts = db.get_concepts_for_sources(relative_ingested) or None

        n_concepts = len(priority_concepts) if priority_concepts else "all"
        log.info("── Compile round 1 (%s concept(s)) ─────────────────────────", n_concepts)
        t1 = time.monotonic()

        def _on_round1_progress(completed: int, total: int, name: str, eta: float | None) -> None:
            if on_progress:
                on_progress("compile_r1", completed, total, eta, name)

        draft_paths, round1_failed, r1_timings = _run_compile(
            config,
            client,
            db,
            concepts=priority_concepts,
            dry_run=dry_run,
            on_progress=_on_round1_progress,
        )
        if on_progress and priority_concepts:
            on_progress("compile_r1", len(priority_concepts), len(priority_concepts), 0.0, "done")
        report.timings["compile_r1"] = time.monotonic() - t1
        report.compiled += len(draft_paths)
        report.failed.extend(round1_failed)
        report.concept_timings.update(r1_timings)
        report.rounds = 1

        # ── Lint ──────────────────────────────────────────────────────────────
        log.info("── Lint ─────────────────────────────────────────────────────")
        if not dry_run:
            lint_result = run_lint(config, db)
            report.lint_issues = len(lint_result.issues)
            broken_links = [i for i in lint_result.issues if i.issue_type == "broken_link"]

            if fix and broken_links:
                stubs = create_stubs(config, db, broken_link_issues=broken_links, max_stubs=3)
                report.stubs_created = len(stubs)

        # ── Round 2: Retry transient failures ─────────────────────────────────
        transient = [f for f in round1_failed if f.reason == FailureReason.TRANSIENT]
        if transient and report.rounds < max_rounds:
            log.info("── Compile round 2 (%d retries) ────────────────────────────", len(transient))
            transient_concepts = [f.concept for f in transient]
            t2 = time.monotonic()

            def _on_round2_progress(completed: int, total: int, name: str, eta: float | None) -> None:
                if on_progress:
                    on_progress("compile_r2", completed, total, eta, name)

            r2_drafts, r2_failed, r2_timings = _run_compile(
                config,
                client,
                db,
                concepts=transient_concepts,
                dry_run=dry_run,
                on_progress=_on_round2_progress,
            )
            if on_progress and transient_concepts:
                on_progress(
                    "compile_r2",
                    len(transient_concepts),
                    len(transient_concepts),
                    0.0,
                    "done",
                )
            report.timings["compile_r2"] = time.monotonic() - t2
            report.compiled += len(r2_drafts)
            draft_paths = draft_paths + r2_drafts
            report.concept_timings.update(r2_timings)
            # Replace transient failures with round-2 results
            report.failed = [f for f in report.failed if f.reason != FailureReason.TRANSIENT]
            report.failed.extend(r2_failed)
            report.rounds = 2

        # ── Approve ────────────────────────────────────────────────────────────
        if auto_approve and draft_paths and not dry_run:
            log.info(
                "── Auto-approve (%d draft(s)) ───────────────────────────────", len(draft_paths)
            )  # noqa: E501
            published = approve_drafts(config, db, draft_paths)
            report.published = len(published)
            generate_index(config, db)
            append_log(config, f"run | {report.published} articles published")

        # ── Bundles ────────────────────────────────────────────────────────────
        if build_bundles and not dry_run:
            report.bundles_created = _generate_bundles(config)
            if report.bundles_created:
                generate_index(config, db)
                append_log(config, f"run | {report.bundles_created} bundle article(s) generated")

        # ── Commit ─────────────────────────────────────────────────────────────
        if config.pipeline.auto_commit and not dry_run and (
            report.compiled or report.published or report.bundles_created
        ):
            msg = f"run: {report.compiled} compiled"
            if report.published:
                msg += f", {report.published} published"
            if report.bundles_created:
                msg += f", {report.bundles_created} bundles"
            git_commit(config.vault, msg, paths=["wiki/", ".olw/"])

        return report


def _run_compile(
    config: Config,
    client: LLMClientProtocol,
    db: StateDB,
    concepts: list[str] | None,
    dry_run: bool,
    on_progress: Callable[[int, int, str, float | None], None] | None = None,
) -> tuple[list[Path], list[FailureRecord], dict[str, float]]:
    """Run compile_concepts and classify failures by reason."""
    from ..openai_compat_client import LLMBadRequestError, LLMError
    from ..pipeline.compile import compile_concepts

    try:
        compile_t0 = time.monotonic()

        def _on_compile_progress(idx: int, total: int, name: str) -> None:
            if not on_progress:
                return
            completed = max(idx - 1, 0)
            eta = None
            if completed > 0 and total > completed:
                elapsed = time.monotonic() - compile_t0
                eta = (elapsed / completed) * (total - completed)
            on_progress(completed, total, name, eta)

        draft_paths, failed_names, concept_timings = compile_concepts(
            config=config,
            client=client,
            db=db,
            dry_run=dry_run,
            concepts=concepts,
            on_progress=_on_compile_progress,
        )
    except LLMBadRequestError as e:
        # Bad request (HTTP 400) — non-retryable; mark all as UNKNOWN not TRANSIENT
        log.error("LLM bad request during compile: %s", e)
        all_concepts = concepts or db.concepts_needing_compile()
        return (
            [],
            [
                FailureRecord(concept=c, reason=FailureReason.UNKNOWN, error_msg=str(e))
                for c in all_concepts
            ],
            {},
        )
    except LLMError as e:
        # Connection-level failure — all concepts are transient
        log.error("LLM connection error during compile: %s", e)
        all_concepts = concepts or db.concepts_needing_compile()
        return (
            [],
            [
                FailureRecord(concept=c, reason=FailureReason.TRANSIENT, error_msg=str(e))
                for c in all_concepts
            ],
            {},
        )

    # Classify individual concept failures
    # compile_concepts returns bare names — we can't know the exact reason
    # per-concept without changing its return type. Use UNKNOWN for now;
    # transient failures (timeouts) will bubble up as LLMError above.
    failure_records = [
        FailureRecord(concept=name, reason=FailureReason.UNKNOWN) for name in failed_names
    ]
    return draft_paths, failure_records, concept_timings


def _generate_bundles(config: Config) -> int:
    """Run local bundle generation and return total generated bundle count."""
    bundle_script = (
        config.vault.parent / "local-llm-wiki-query" / "scripts" / "build_all_bundles.py"
    )
    if not bundle_script.exists():
        log.warning("Bundle generation skipped; missing script: %s", bundle_script)
        return 0

    cmd = [sys.executable, str(bundle_script), "--vault", str(config.vault)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if stdout:
        for line in stdout.splitlines():
            log.info("bundle: %s", line)
    if stderr:
        for line in stderr.splitlines():
            log.warning("bundle: %s", line)

    if proc.returncode != 0:
        log.warning("Bundle generation failed with exit code %s", proc.returncode)
        return 0

    generated_clients = 0
    for line in stdout.splitlines():
        if line.startswith("Client bundles generated:"):
            try:
                generated_clients = int(line.split(":", 1)[1].strip())
            except ValueError:
                generated_clients = 0
            break

    # Aggregate bundles are always generated when the script succeeds.
    return generated_clients + 4
