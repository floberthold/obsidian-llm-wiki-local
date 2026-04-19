from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import Config

log = logging.getLogger(__name__)
_WRITE_LOCK = threading.Lock()


def resolve_metrics_path(config: Config) -> Path:
    rel = Path(config.pipeline.telemetry_jsonl_path)
    if rel.is_absolute():
        return rel
    return config.vault / rel


def emit_event(config: Config | None, **event: Any) -> dict[str, Any]:
    """Emit telemetry event to JSONL when enabled and return the full payload."""
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "telemetry_version": 1,
        **event,
    }
    if config is None or not config.pipeline.telemetry_enabled:
        return payload

    out_path = resolve_metrics_path(config)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, ensure_ascii=True)
        with _WRITE_LOCK:
            with out_path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.write("\n")
    except Exception as e:
        log.debug("telemetry write failed: %s", e)

    return payload
