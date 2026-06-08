"""
Hardware profile detection for analytics.

Builds a stable machine fingerprint (short SHA256 hash) from hostname + CPU + GPU
so runs on the same machine can be grouped across time.
"""

from __future__ import annotations

import hashlib
import platform
import socket


def get_machine_profile() -> dict:
    """Return a dict describing the current machine's hardware."""
    cpu = platform.processor() or platform.machine() or "unknown"
    gpu_model, gpu_vram_gb = _detect_gpu()

    raw = f"{socket.gethostname()}|{cpu}|{gpu_model or ''}"
    machine_id = hashlib.sha256(raw.encode()).hexdigest()[:16]

    try:
        import psutil

        cores = psutil.cpu_count(logical=False) or 0
        ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    except Exception:
        cores = 0
        ram_gb = 0.0

    return {
        "id": machine_id,
        "hostname": socket.gethostname(),
        "cpu_model": cpu,
        "cpu_cores": cores,
        "ram_gb": ram_gb,
        "gpu_model": gpu_model,
        "gpu_vram_gb": gpu_vram_gb,
        "os_platform": platform.platform(),
        "python_version": platform.python_version(),
    }


def _detect_gpu() -> tuple[str | None, float | None]:
    try:
        import GPUtil  # type: ignore[import-untyped]

        gpus = GPUtil.getGPUs()
        if gpus:
            return gpus[0].name, round(gpus[0].memoryTotal / 1024, 1)
    except Exception:
        pass
    return None, None
