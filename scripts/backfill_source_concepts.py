from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from obsidian_llm_wiki.client_factory import build_client
from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.pipeline.ingest import ingest_note
from obsidian_llm_wiki.state import StateDB
from obsidian_llm_wiki.vault import parse_note

NONE_CONCEPTS_RE = re.compile(r"## Concepts\r?\n- \(none\)")


def _raw_body_letter_count(path: Path) -> int:
    try:
        _, body = parse_note(path)
    except Exception:
        body = path.read_text(encoding="utf-8", errors="ignore")
    return sum(ch.isalpha() for ch in body)


def _concepts_missing(source_path: Path) -> bool:
    try:
        _, body = parse_note(source_path)
    except Exception:
        return False
    return bool(NONE_CONCEPTS_RE.search(body))


def _find_candidates(config: Config, min_letters: int, limit: int) -> list[Path]:
    candidates: list[Path] = []
    for source_path in sorted(config.sources_dir.rglob("group-*.md")):
        if not _concepts_missing(source_path):
            continue
        rel = source_path.relative_to(config.sources_dir)
        raw_path = config.raw_dir / rel
        if not raw_path.exists():
            continue
        if _raw_body_letter_count(raw_path) < min_letters:
            continue
        candidates.append(raw_path)
        if len(candidates) >= limit:
            break
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Incrementally backfill source Concepts sections for grouped raw notes."
    )
    parser.add_argument("--vault", required=True, help="Path to the Obsidian vault")
    parser.add_argument("--limit", type=int, default=5, help="Max notes to re-ingest in one run")
    parser.add_argument(
        "--min-letters",
        type=int,
        default=180,
        help="Minimum alphabetic characters in raw body to consider it worth ingesting",
    )
    args = parser.parse_args()

    config = Config.from_vault(Path(args.vault))
    client = build_client(config)
    client.require_healthy()
    db = StateDB(config.state_db_path)

    candidates = _find_candidates(config, min_letters=args.min_letters, limit=args.limit)
    print(f"candidate_batch={len(candidates)}")
    if not candidates:
        return 0

    updated = 0
    improved = 0
    still_none = 0

    for idx, raw_path in enumerate(candidates, start=1):
        result = ingest_note(raw_path, config=config, client=client, db=db, force=True)
        status = "updated" if result is not None else "skipped"
        rel = raw_path.relative_to(config.raw_dir)
        source_path = config.sources_dir / rel
        has_concepts = source_path.exists() and not _concepts_missing(source_path)
        if result is not None:
            updated += 1
        if has_concepts:
            improved += 1
            concept_status = "has-concepts"
        else:
            still_none += 1
            concept_status = "none"
        print(f"[{idx}/{len(candidates)}] {status} | {concept_status} | {raw_path.as_posix()}")

    print(f"updated={updated}")
    print(f"improved={improved}")
    print(f"still_none={still_none}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
