"""
Ingest pipeline: raw note → chunk → analyze → embed → update state.

Uses fast model (gemma4:e4b) for analysis.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path

from ..config import Config
from ..models import AnalysisResult, Concept, RawNoteRecord
from ..protocols import LLMClientProtocol
from ..state import StateDB
from ..structured_output import request_structured
from ..telemetry import emit_event
from ..vault import (
    chunk_text,
    generate_aliases,
    parse_note,
    sanitize_filename,
    sanitize_wikilink_target,
    write_note,
)

log = logging.getLogger(__name__)

_SYSTEM = (
    "You are a knowledge analyst. Read the provided note and extract structured information. "
    "Be concise and accurate. Do not invent information not present in the note. "
    "Detect the primary language of the note and return its ISO 639-1 code in the 'language' field "
    "(e.g. 'en', 'fr', 'de'). Use null if uncertain."
)


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_text_with_fallback(path: Path) -> str:
    """Read text files with practical encoding fallbacks for user-authored notes."""
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    # Latin-1 should always succeed, but keep a final safe guard.
    return raw.decode("utf-8", errors="replace")


def _build_analysis_prompt(
    body: str,
    existing_concepts: list[str],
    path_name: str = "",
    chunk_label: str = "",
) -> str:
    concepts_hint = ", ".join(existing_concepts[:30]) if existing_concepts else "none yet"
    label = f" {chunk_label}" if chunk_label else ""
    return (
        f"Analyze this note{label} and extract structured metadata.\n\n"
        f"Existing wiki concepts (reuse these names where applicable): {concepts_hint}\n\n"
        f"For each concept, provide 3-5 short surface forms used in running text "
        f"(abbreviations, short names). Example: name='Program Counter (PC)', "
        f"aliases=['PC', 'program counter']. Use empty list if no natural aliases exist.\n\n"
        f"NOTE CONTENT:\n{body}"
    )


def _merge_chunk_results(results: list[AnalysisResult]) -> AnalysisResult:
    """Merge AnalysisResults from multiple chunks into one.

    Concepts and topics: union (deduplicated, insertion order preserved).
    Aliases for the same concept are merged across chunks.
    Summary: first chunk's (intro is most representative).
    Quality: minimum across chunks (conservative).
    """
    if len(results) == 1:
        return results[0]

    # Dedup concepts by canonical name (case-insensitive), merge aliases
    seen: dict[str, list[str]] = {}  # lower(name) -> accumulated aliases
    order: list[str] = []  # canonical names in insertion order
    canonical_by_lower: dict[str, str] = {}

    for r in results:
        for c in r.concepts:
            key = c.name.lower()
            if key not in seen:
                seen[key] = list(c.aliases)
                order.append(key)
                canonical_by_lower[key] = c.name
            else:
                existing_lower = {a.lower() for a in seen[key]}
                for a in c.aliases:
                    if a.lower() not in existing_lower:
                        seen[key].append(a)
                        existing_lower.add(a.lower())

    all_concepts = [Concept(name=canonical_by_lower[k], aliases=seen[k]) for k in order][:8]

    seen_topics: set[str] = set()
    all_topics: list[str] = []
    for r in results:
        for t in r.suggested_topics:
            if t.lower() not in seen_topics:
                seen_topics.add(t.lower())
                all_topics.append(t)

    quality_rank = {"high": 2, "medium": 1, "low": 0}
    min_result = min(results, key=lambda r: quality_rank.get(r.quality, 1))

    merged_language = next((r.language for r in results if r.language), None)

    return AnalysisResult(
        summary=results[0].summary,
        concepts=all_concepts,
        suggested_topics=all_topics[:5],
        quality=min_result.quality,
        language=merged_language,
    )


def _analyze_body(
    body: str,
    existing_concepts: list[str],
    path_name: str,
    client: LLMClientProtocol,
    config: Config,
) -> AnalysisResult:
    """Analyze note body, splitting into chunks when body exceeds configured chunk size."""
    ratio = max(0.25, min(config.pipeline.ingest_chunk_ratio, 0.9))
    chunk_size = max(1, int(config.effective_provider.fast_ctx * ratio))

    if len(body) <= chunk_size:
        prompt = _build_analysis_prompt(body, existing_concepts, path_name)
        return request_structured(
            client=client,
            prompt=prompt,
            model_class=AnalysisResult,
            model=config.models.fast,
            system=_SYSTEM,
            num_ctx=config.effective_provider.fast_ctx,
            max_retries=config.pipeline.ingest_max_retries,
            telemetry_config=config,
            telemetry_stage="ingest_analysis",
        )

    # Split into chunks — no overlap needed for concept extraction
    chunks = [body[i : i + chunk_size] for i in range(0, len(body), chunk_size)]
    log.info(
        "Note %s split into %d chunks for analysis (%d chars, chunk_size=%d)",
        path_name or "unknown",
        len(chunks),
        len(body),
        chunk_size,
    )

    def _analyze_chunk(chunk: str, idx: int) -> AnalysisResult:
        label = f"[part {idx + 1}/{len(chunks)}]"
        log.info("Analyzing %s %s …", path_name or "note", label)
        t0 = time.monotonic()
        prompt = _build_analysis_prompt(chunk, existing_concepts, path_name, chunk_label=label)
        result = request_structured(
            client=client,
            prompt=prompt,
            model_class=AnalysisResult,
            model=config.models.fast,
            system=_SYSTEM,
            num_ctx=config.effective_provider.fast_ctx,
            max_retries=config.pipeline.ingest_max_retries,
            telemetry_config=config,
            telemetry_stage="ingest_analysis_chunk",
        )
        log.info("Analyzed %s %s (%.1fs)", path_name or "note", label, time.monotonic() - t0)
        return result

    if config.pipeline.ingest_parallel:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        env_parallel = os.getenv("OLLAMA_NUM_PARALLEL", "").strip()
        try:
            configured_workers = int(env_parallel) if env_parallel else 0
        except ValueError:
            configured_workers = 0
        if configured_workers <= 0:
            configured_workers = 4

        max_workers = max(1, min(len(chunks), configured_workers))
        chunk_results: list[AnalysisResult | None] = [None] * len(chunks)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_analyze_chunk, chunk, i): i for i, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                chunk_results[futures[future]] = future.result()
        results = [r for r in chunk_results if r is not None]
    else:
        results = [_analyze_chunk(chunk, i) for i, chunk in enumerate(chunks)]

    return _merge_chunk_results(results)


_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "is",
        "it",
        "in",
        "on",
        "at",
        "to",
        "by",
        "for",
        "of",
        "as",
        "from",
        "with",
        "this",
        "that",
        "these",
        "those",
        "be",
        "are",
    }
)

_CONCEPT_JUNK_RE = re.compile(
    r"^(?:page|p\.?|figure|fig|section|sec|chapter|ch|slide)\s*[-:#.]?\s*\d+(?:\.\d+)*\b",
    re.IGNORECASE,
)

_SINGULAR_EXCEPTIONS = frozenset({"analysis", "news", "series", "species", "status"})


def _is_valid_concept_name(name: str) -> bool:
    """Drop obviously low-information concepts like page/figure labels."""
    stripped = name.strip()
    if not stripped:
        return False
    if _CONCEPT_JUNK_RE.match(stripped):
        return False
    if re.fullmatch(r"\d+(?:\.\d+)*", stripped):
        return False
    return True


def _singularize_token(token: str) -> str:
    """Apply lightweight plural reduction for simple concept deduping."""
    lower = token.lower()
    if len(lower) <= 3 or lower in _SINGULAR_EXCEPTIONS:
        return token
    if lower.endswith("ies") and len(lower) > 4:
        return token[:-3] + "y"
    if lower.endswith("ses") and len(lower) > 4:
        return token[:-2]
    if lower.endswith("s") and not lower.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def _singularize_phrase(name: str) -> str:
    words = name.split()
    if not words:
        return name
    words[-1] = _singularize_token(words[-1])
    return " ".join(words)


def _concept_singular_key(name: str) -> str:
    return re.sub(r"\s+", " ", _singularize_phrase(name).strip()).lower()


def _validate_aliases(canonical: str, raw_aliases: list[str]) -> list[str]:
    """Filter LLM-produced aliases: remove too-short, stopwords, self-matches, duplicates."""
    seen = {canonical.lower()}
    valid: list[str] = []
    for alias in raw_aliases:
        a = alias.strip()
        if not a or a.lower() in seen:
            continue
        if len(a) < 2:
            continue
        if len(a) <= 3 and not a.isupper():
            continue
        if a.lower() in _STOPWORDS:
            continue
        seen.add(a.lower())
        valid.append(a)
    return valid[:5]


def _normalize_concepts(raw_concepts: list[Concept], db: StateDB) -> list[tuple[str, list[str]]]:
    """Case-insensitive dedup against existing canonical concept names.

    Returns (canonical_name, validated_aliases) pairs.
    """
    existing_names = db.list_all_concept_names()
    existing = {n.lower(): n for n in existing_names}
    existing_singular = {_concept_singular_key(n): n for n in existing_names}
    seen: set[str] = set()
    result: list[tuple[str, list[str]]] = []
    for concept in raw_concepts:
        name = concept.name.strip()
        if not _is_valid_concept_name(name):
            continue
        canonical = existing.get(name.lower()) or existing_singular.get(_concept_singular_key(name))
        if canonical is None:
            canonical = name

        canonical_key = _concept_singular_key(canonical)
        if canonical_key in seen:
            continue
        seen.add(canonical_key)
        aliases = _validate_aliases(canonical, concept.aliases)
        result.append((canonical, aliases))
    return result


_HEADER_SCAN_LINES = 30  # only strip short lines from the opening section

# Media reference patterns for source page preservation
_OBSIDIAN_EMBED_RE = re.compile(
    r"!\[\[([^\]]+\.(?:png|jpg|jpeg|gif|svg|webp|bmp|tiff|avif|pdf|mp4|webm|mov|mp3|wav|ogg))\]\]",
    re.IGNORECASE,
)
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def _preprocess_web_clip(content: str) -> str:
    """Clean common Obsidian Web Clipper artifacts (nav bars, cookie banners, HTML tags).

    HTML stripping is scoped to the first _HEADER_SCAN_LINES only — body HTML
    (<details>, <kbd>, <sup>, etc.) is intentional and preserved.
    """
    _MD_STARTS = ("#", "-", "*", ">", "[", "!")  # markdown structural chars — always keep
    lines = content.splitlines()

    cleaned = []
    for i, line in enumerate(lines):
        if i < _HEADER_SCAN_LINES:
            # Strip HTML only in header region (nav/banner cleanup)
            line = re.sub(r"<[^>]+>", "", line)
            stripped = line.strip()
            # Skip short non-empty non-markdown lines (nav/banner heuristic)
            if stripped and len(stripped.split()) <= 5 and not stripped.startswith(_MD_STARTS):
                continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _is_ingest_candidate(path: Path) -> bool:
    return path.is_file() and "processed" not in path.parts and not path.name.startswith(".")


def _page_output_dir(pdf_path: Path) -> Path:
    return pdf_path.parent / sanitize_filename(pdf_path.stem)


def convert_pdf_to_markdown(pdf_path: Path, overwrite: bool = False) -> list[Path]:
    """Convert a PDF into one markdown file per page inside a sibling folder."""
    try:
        from pypdf import PdfReader
    except Exception as e:
        log.warning("PDF conversion unavailable for %s: %s", pdf_path.name, e)
        return []

    output_dir = _page_output_dir(pdf_path)
    existing_pages = sorted(output_dir.glob("page-*.md")) if output_dir.exists() else []
    if existing_pages and not overwrite:
        return existing_pages

    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:
        log.warning("Failed to read PDF %s: %s", pdf_path.name, e)
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for existing in output_dir.glob("page-*.md"):
            existing.unlink(missing_ok=True)

    rel_pdf = pdf_path.as_posix()
    written_paths: list[Path] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            log.debug("Skipping image-only PDF page %d in %s", page_number, pdf_path.name)
            continue

        out_path = output_dir / f"page-{page_number:03d}.md"
        write_note(
            out_path,
            {
                "title": f"{pdf_path.stem} - Page {page_number}",
                "source_pdf": rel_pdf,
                "source_page": page_number,
                "tags": ["pdf-page"],
            },
            f"## Page {page_number}\n\n{text}\n",
        )
        written_paths.append(out_path)

    log.info(
        "Converted PDF %s into %d markdown page(s) under %s",
        pdf_path.name,
        len(written_paths),
        output_dir.name,
    )
    emit_event(
        config=None,
        event_type="pdf_converted",
        source_pdf=pdf_path.name,
        page_count=len(written_paths),
    )
    return written_paths


def collect_ingest_paths(config: Config, paths: list[Path] | None = None) -> list[Path]:
    """Collect markdown paths for ingest, auto-converting PDFs into per-page notes."""
    if paths is None:
        candidates = list(config.raw_dir.rglob("*")) if config.raw_dir.exists() else []
    else:
        candidates = [Path(path) for path in paths]

    md_paths: list[Path] = []
    seen: set[str] = set()

    for path in sorted(candidates):
        if not _is_ingest_candidate(path):
            continue

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            for page_path in convert_pdf_to_markdown(path):
                key = page_path.resolve().as_posix()
                if key not in seen:
                    seen.add(key)
                    md_paths.append(page_path)
        elif suffix == ".md":
            key = path.resolve().as_posix()
            if key not in seen:
                seen.add(key)
                md_paths.append(path)

    return sorted(md_paths)


def _collect_media_refs(body: str) -> list[str]:
    """Extract media references from note body for preservation in source pages."""
    refs: list[str] = []
    for m in _OBSIDIAN_EMBED_RE.finditer(body):
        refs.append(f"- ![[{m.group(1)}]]")
    for m in _MD_IMAGE_RE.finditer(body):
        alt, url = m.group(1), m.group(2)
        refs.append(f"- ![{alt}]({url})")
    return refs


def _create_source_summary_page(
    path: Path,
    src_meta: dict,
    result: AnalysisResult,
    config: Config,
    body: str = "",
) -> Path:
    """
    Generate wiki/sources/{Title}.md from AnalysisResult. No extra LLM call.
    Returns the path written.
    """
    # Derive title from note frontmatter > file stem
    title = src_meta.get("title") or path.stem.replace("-", " ").title()
    # Mirror the raw folder hierarchy: raw/subdir/note.md -> sources/subdir/note.md
    out_path = config.sources_dir / path.relative_to(config.raw_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d")
    rel_raw = path.relative_to(config.vault).as_posix()
    source_url = src_meta.get("source") or src_meta.get("url") or ""
    aliases = generate_aliases(title, "")  # source pages rarely have abbreviations

    # Build concept list as [[wikilinks]]
    concept_lines = "\n".join(
        f"- [[{sanitize_wikilink_target(c.name)}]]" for c in result.concepts[:8] if c.name.strip()
    )

    out_meta: dict = {
        "title": title,
        "aliases": aliases,
        "tags": ["source"],
        "status": "published",
        "source_file": rel_raw,
        "quality": result.quality,
        "created": now,
    }
    if source_url:
        out_meta["source_url"] = source_url

    body_parts = [
        f"# {title}",
        "",
        "## Summary",
        result.summary,
        "",
        "## Concepts",
        concept_lines,
        "",
        "## Source Info",
        f"- **Quality:** {result.quality}",
        f"- **Raw file:** [[{rel_raw}]]",
        f"- **Ingested:** {now}",
    ]
    if source_url:
        body_parts.append(f"- **URL:** {source_url}")

    media_refs = _collect_media_refs(body)
    if media_refs:
        body_parts += ["", "## Media"] + media_refs

    write_note(out_path, out_meta, "\n".join(body_parts))
    log.info("Source summary written: %s", out_path.name)
    return out_path


def ingest_note(
    path: Path,
    config: Config,
    client: LLMClientProtocol,
    db: StateDB,
    rag=None,  # Optional RAGStore, injected in Phase 2
    existing_topics: list[str] | None = None,  # existing concept names for prompt
    force: bool = False,
) -> AnalysisResult | None:
    """
    Ingest a single raw note.

    Returns AnalysisResult or None if skipped (duplicate / already ingested).
    """
    fn_t0 = time.monotonic()
    try:
        meta, body = parse_note(path)
    except Exception:
        meta, body = {}, _read_text_with_fallback(path)

    # Hash body only (strip frontmatter) so copies are detected as duplicates
    # even after ingest has updated the original's frontmatter (olw_status etc.).
    # Exception: when source_pdf is set (PDF-extracted pages), include it in the
    # hash so pages from *different* PDFs with identical text are not falsely
    # flagged as duplicates.
    source_pdf = meta.get("source_pdf", "")
    body_for_hash = body
    hash_input = (source_pdf + "\x00" + body_for_hash) if source_pdf else body_for_hash
    h = _content_hash(hash_input)

    # Dedup check
    rel_path = path.relative_to(config.vault).as_posix()

    existing = db.get_raw_by_hash(h)
    if existing and existing.path != rel_path:
        log.info("Duplicate of %s, skipping %s", existing.path, path.name)
        emit_event(
            config,
            event_type="function_timing",
            function_name="ingest_note",
            stage="ingest",
            model=config.models.fast,
            success=True,
            outcome="skipped_duplicate",
            duration_ms=round((time.monotonic() - fn_t0) * 1000.0, 2),
            note=path.name,
        )
        return None

    record = db.get_raw(rel_path)

    if record and record.status == "ingested" and not force:
        log.info("Already ingested: %s", path.name)
        emit_event(
            config,
            event_type="function_timing",
            function_name="ingest_note",
            stage="ingest",
            model=config.models.fast,
            success=True,
            outcome="skipped_already_ingested",
            duration_ms=round((time.monotonic() - fn_t0) * 1000.0, 2),
            note=path.name,
        )
        return None

    # Pre-process web clips
    if meta.get("source") or meta.get("url"):  # web clipper adds these
        body = _preprocess_web_clip(body)

    # Skip notes with no usable content (e.g. image-only PDF pages already on disk)
    _EMPTY_BODY_MARKERS = (
        "(No extractable text found on this page. The PDF may be image-only.)",
    )
    stripped_body = body.strip()
    if not stripped_body or stripped_body in _EMPTY_BODY_MARKERS:
        log.info("Skipping empty/image-only note: %s", path.name)
        emit_event(
            config,
            event_type="function_timing",
            function_name="ingest_note",
            stage="ingest",
            model=config.models.fast,
            success=True,
            outcome="skipped_empty",
            duration_ms=round((time.monotonic() - fn_t0) * 1000.0, 2),
            note=path.name,
        )
        return None

    # Chunk + embed only when RAG store is wired in (Phase 2)
    if rag is not None:
        chunks = chunk_text(
            body, chunk_size=config.rag.chunk_size, overlap=config.rag.chunk_overlap
        )
        embeddings = client.embed_batch(chunks, model=config.models.embed)
        rag.add_document(
            doc_id=rel_path,
            chunks=chunks,
            embeddings=embeddings,
            metadata={"source": rel_path, "type": "raw"},
        )

    # LLM analysis — use existing concept names so model can reuse canonical names
    if existing_topics is None:
        existing_topics = db.list_all_concept_names()
    try:
        result: AnalysisResult = _analyze_body(
            body=body,
            existing_concepts=existing_topics,
            path_name=path.name,
            client=client,
            config=config,
        )
    except Exception as e:
        log.error("Analysis failed for %s: %s", path.name, e)
        db.upsert_raw(
            RawNoteRecord(
                path=rel_path,
                content_hash=h,
                status="failed",
                error=str(e),
            )
        )
        emit_event(
            config,
            event_type="function_timing",
            function_name="ingest_note",
            stage="ingest",
            model=config.models.fast,
            success=False,
            outcome="failed",
            duration_ms=round((time.monotonic() - fn_t0) * 1000.0, 2),
            note=path.name,
            error_class=e.__class__.__name__,
            error_message=str(e),
        )
        return None

    # Update state DB (raw files stay immutable — metadata lives in state.db only)
    db.upsert_raw(
        RawNoteRecord(
            path=rel_path,
            content_hash=h,
            status="ingested",
            summary=result.summary,
            quality=result.quality,
            language=result.language,
            ingested_at=datetime.now(),
        )
    )

    # Normalize concept names against existing canonical names, store linkages
    max_concepts = config.pipeline.max_concepts_per_source
    normalized = _normalize_concepts(result.concepts[:max_concepts], db)
    canonical_names = [name for name, _ in normalized]
    db.upsert_concepts(rel_path, canonical_names)
    for canonical, aliases in normalized:
        if aliases:
            db.upsert_aliases(canonical, aliases)

    # Create source summary page in wiki/sources/ (no extra LLM call)
    try:
        _create_source_summary_page(path, meta, result, config, body=body)
    except Exception as e:
        log.warning("Source summary page failed for %s: %s", path.name, e)

    log.info(
        "Ingested: %s (quality=%s, concepts=%s)",
        path.name,
        result.quality,
        [c.name for c in result.concepts[:3]],
    )
    emit_event(
        config,
        event_type="function_timing",
        function_name="ingest_note",
        stage="ingest",
        model=config.models.fast,
        success=True,
        outcome="ingested",
        duration_ms=round((time.monotonic() - fn_t0) * 1000.0, 2),
        note=path.name,
    )
    return result


def ingest_all(
    config: Config,
    client: LLMClientProtocol,
    db: StateDB,
    rag=None,
    force: bool = False,
) -> list[tuple[Path, AnalysisResult | None]]:
    """Ingest all markdown files in raw/, including per-page PDF conversions."""
    raw_files = collect_ingest_paths(config)
    # Snapshot concept names once before loop (for consistent prompt context)
    existing_topics = db.list_all_concept_names()
    results = []
    for path in sorted(raw_files):
        result = ingest_note(
            path=path,
            config=config,
            client=client,
            db=db,
            rag=rag,
            existing_topics=existing_topics,
            force=force,
        )
        results.append((path, result))
    return results
