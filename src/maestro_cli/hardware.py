"""Zero-dependency local-hardware detection for model routing + diagnostics.

Detects what a machine can actually run locally so that ``model: auto`` routing
to a local engine (``ollama``/``llama``) lands on a model the user has installed
and that fits available VRAM — and so ``maestro doctor --hardware`` can report
it.  Everything degrades gracefully: a missing GPU, a stopped Ollama server, or
an unset ``LLAMA_MODEL_DIR`` simply yields an empty section, never an error.

Detection sources (all stdlib, no new dependencies):
- **GPU / VRAM** — ``nvidia-smi`` if it is on PATH.
- **Ollama models** — the local Ollama HTTP API (``GET {OLLAMA_HOST}/api/tags``).
- **llama.cpp models** — ``*.gguf`` files under ``LLAMA_MODEL_DIR``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_NVIDIA_SMI_TIMEOUT = 5
_OLLAMA_TIMEOUT = 3
_DEFAULT_OLLAMA_HOST = "http://localhost:11434"
_VRAM_HEADROOM = 0.9  # leave 10% headroom over the model's on-disk size
_LOCAL_ENGINES = frozenset({"ollama", "llama"})
_TIER_ORDER = ("high", "medium", "low")


@dataclass
class GpuInfo:
    name: str
    vram_total_mb: int | None
    vram_free_mb: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "vram_total_mb": self.vram_total_mb,
            "vram_free_mb": self.vram_free_mb,
        }


@dataclass
class LocalModel:
    name: str
    size_bytes: int | None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "size_bytes": self.size_bytes}


@dataclass
class HardwareInfo:
    gpus: list[GpuInfo] = field(default_factory=list)
    ollama_models: list[LocalModel] = field(default_factory=list)
    llama_models: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def free_vram_mb(self) -> int | None:
        """Largest single-GPU free VRAM (a model loads onto one GPU)."""
        values = [g.vram_free_mb for g in self.gpus if g.vram_free_mb is not None]
        return max(values) if values else None

    @property
    def total_vram_mb(self) -> int | None:
        values = [g.vram_total_mb for g in self.gpus if g.vram_total_mb is not None]
        return max(values) if values else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpus": [g.to_dict() for g in self.gpus],
            "ollama_models": [m.to_dict() for m in self.ollama_models],
            "llama_models": list(self.llama_models),
            "notes": list(self.notes),
            "free_vram_mb": self.free_vram_mb,
            "total_vram_mb": self.total_vram_mb,
        }


def _safe_int(value: str) -> int | None:
    try:
        return int(float(value.strip()))
    except (ValueError, AttributeError):
        return None


def _detect_gpus() -> list[GpuInfo]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        proc = subprocess.run(
            [
                exe,
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    if proc.returncode != 0:
        return []

    gpus: list[GpuInfo] = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0]:
            continue
        gpus.append(
            GpuInfo(
                name=parts[0],
                vram_total_mb=_safe_int(parts[1]),
                vram_free_mb=_safe_int(parts[2]),
            )
        )
    return gpus


def _ollama_host() -> str:
    host = (os.environ.get("OLLAMA_HOST") or "").strip() or _DEFAULT_OLLAMA_HOST
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host.rstrip("/")


def _detect_ollama_models() -> list[LocalModel]:
    url = _ollama_host() + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=_OLLAMA_TIMEOUT) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return []

    if not isinstance(payload, dict):
        return []
    models: list[LocalModel] = []
    for entry in payload.get("models", []):
        if not isinstance(entry, dict):
            continue
        name = entry.get("name") or entry.get("model")
        if not isinstance(name, str) or not name:
            continue
        size = entry.get("size")
        models.append(
            LocalModel(
                name=name,
                size_bytes=int(size) if isinstance(size, (int, float)) else None,
            )
        )
    return models


def _detect_llama_models() -> list[str]:
    model_dir = os.environ.get("LLAMA_MODEL_DIR")
    if not model_dir:
        return []
    try:
        path = Path(model_dir)
        if not path.is_dir():
            return []
        return sorted(p.name for p in path.glob("*.gguf"))
    except OSError:
        return []


def detect_hardware() -> HardwareInfo:
    """Detect local GPUs and installed local models (graceful on every source)."""
    info = HardwareInfo()
    info.gpus = _detect_gpus()
    info.ollama_models = _detect_ollama_models()
    info.llama_models = _detect_llama_models()

    if not info.gpus:
        info.notes.append("no NVIDIA GPU detected (nvidia-smi unavailable)")
    if not info.ollama_models:
        info.notes.append("no Ollama models found (server down or none pulled)")
    if not info.llama_models and os.environ.get("LLAMA_MODEL_DIR"):
        info.notes.append("no .gguf files found under LLAMA_MODEL_DIR")
    return info


# ---------------------------------------------------------------------------
# Hardware-aware model selection for `model: auto` on local engines
# ---------------------------------------------------------------------------

def _name_matches(tier_model: str, installed_name: str) -> bool:
    """Whether an installed model name corresponds to a routing tier model."""
    base = installed_name.split(":", 1)[0]
    return (
        installed_name == tier_model
        or base == tier_model
        or installed_name.startswith(tier_model + ":")
        or tier_model in installed_name
    )


def _select_from_tiers(
    tier: str,
    engine_tiers: dict[str, str],
    usable: Callable[[str], bool],
) -> str | None:
    """Pick a usable model at or below *tier*, else any usable, else None.

    Returns ``None`` when the desired tier model is already usable (no change)
    or when nothing is usable (keep the caller's default).
    """
    desired = engine_tiers.get(tier)
    if desired is None or usable(desired):
        return None
    candidates = (
        list(_TIER_ORDER[_TIER_ORDER.index(tier):])
        if tier in _TIER_ORDER
        else list(_TIER_ORDER)
    )
    for level in candidates + [t for t in _TIER_ORDER if t not in candidates]:
        model = engine_tiers.get(level)
        if model and usable(model):
            return model
    return None


def select_local_model(
    engine: str,
    tier: str,
    engine_tiers: dict[str, str],
    hardware: HardwareInfo,
) -> str | None:
    """Adjust an auto-routed local model to what is installed and fits VRAM.

    Returns a replacement model name, or ``None`` to keep the routing default
    (the desired model is already usable, or nothing better is available).
    """
    if engine not in _LOCAL_ENGINES:
        return None

    if engine == "ollama":
        installed = hardware.ollama_models
        if not installed:
            return None
        free_mb = hardware.free_vram_mb

        def _size_of(tier_model: str) -> int | None:
            for model in installed:
                if _name_matches(tier_model, model.name):
                    return model.size_bytes
            return None

        def _usable(tier_model: str) -> bool:
            if not any(_name_matches(tier_model, m.name) for m in installed):
                return False
            if free_mb is None:
                return True
            size = _size_of(tier_model)
            if size is None:
                return True
            return size <= free_mb * 1024 * 1024 * _VRAM_HEADROOM

        return _select_from_tiers(tier, engine_tiers, _usable)

    # llama.cpp: best-effort file-name match; no reliable size/VRAM signal.
    files = hardware.llama_models
    if not files:
        return None

    def _usable_llama(tier_model: str) -> bool:
        return any(tier_model in name for name in files)

    return _select_from_tiers(tier, engine_tiers, _usable_llama)


# ---------------------------------------------------------------------------
# Formatting (maestro doctor --hardware)
# ---------------------------------------------------------------------------

def format_hardware(info: HardwareInfo) -> str:
    lines: list[str] = ["Local hardware:"]
    if info.gpus:
        for gpu in info.gpus:
            total = f"{gpu.vram_total_mb} MB" if gpu.vram_total_mb else "?"
            free = f"{gpu.vram_free_mb} MB free" if gpu.vram_free_mb else "? free"
            lines.append(f"  GPU: {gpu.name} ({total}, {free})")
    else:
        lines.append("  GPU: none detected")

    if info.ollama_models:
        lines.append(f"  Ollama models ({len(info.ollama_models)}):")
        for model in info.ollama_models:
            size = (
                f"{model.size_bytes / 1e9:.1f} GB"
                if model.size_bytes
                else "size unknown"
            )
            lines.append(f"    - {model.name} ({size})")
    else:
        lines.append("  Ollama models: none")

    if info.llama_models:
        lines.append(f"  llama.cpp models ({len(info.llama_models)}):")
        for name in info.llama_models:
            lines.append(f"    - {name}")

    for note in info.notes:
        lines.append(f"  note: {note}")
    return "\n".join(lines)


def format_hardware_json(info: HardwareInfo) -> str:
    return json.dumps(info.to_dict(), indent=2)
