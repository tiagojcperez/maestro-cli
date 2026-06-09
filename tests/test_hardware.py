from __future__ import annotations

import json
from pathlib import Path

import pytest

from maestro_cli import hardware
from maestro_cli.hardware import (
    GpuInfo,
    HardwareInfo,
    LocalModel,
    detect_hardware,
    format_hardware,
    format_hardware_json,
    select_local_model,
)

_OLLAMA_TIERS = {"low": "phi3", "medium": "llama3", "high": "mixtral"}
_LLAMA_TIERS = {
    "low": "llama-3.2-3b",
    "medium": "llama-3-8b",
    "high": "codellama-13b",
}


class _FakeProc:
    def __init__(self, stdout: str, returncode: int = 0) -> None:
        self.stdout = stdout
        self.returncode = returncode


class _FakeResp:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> _FakeResp:
        return self

    def __exit__(self, *_args: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._payload


class TestDetectGpus:
    def test_parses_nvidia_smi(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(hardware.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
        monkeypatch.setattr(
            hardware.subprocess,
            "run",
            lambda *a, **k: _FakeProc(
                "NVIDIA RTX 4090, 24576, 20000\nNVIDIA RTX 3090, 24576, 8000\n"
            ),
        )
        gpus = hardware._detect_gpus()
        assert len(gpus) == 2
        assert gpus[0].name == "NVIDIA RTX 4090"
        assert gpus[0].vram_total_mb == 24576
        assert gpus[0].vram_free_mb == 20000

    def test_empty_when_no_nvidia_smi(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(hardware.shutil, "which", lambda _: None)
        assert hardware._detect_gpus() == []

    def test_empty_on_subprocess_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(hardware.shutil, "which", lambda _: "/usr/bin/nvidia-smi")

        def _raise(*_a: object, **_k: object) -> object:
            raise OSError("boom")

        monkeypatch.setattr(hardware.subprocess, "run", _raise)
        assert hardware._detect_gpus() == []

    def test_empty_on_nonzero_returncode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(hardware.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
        monkeypatch.setattr(
            hardware.subprocess, "run", lambda *a, **k: _FakeProc("", returncode=9)
        )
        assert hardware._detect_gpus() == []


class TestDetectOllamaModels:
    def test_parses_api_tags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = json.dumps(
            {
                "models": [
                    {"name": "llama3:latest", "size": 4_700_000_000},
                    {"name": "phi3:mini", "size": 2_300_000_000},
                ]
            }
        ).encode("utf-8")
        monkeypatch.setattr(
            hardware.urllib.request, "urlopen", lambda *a, **k: _FakeResp(payload)
        )
        models = hardware._detect_ollama_models()
        assert {m.name for m in models} == {"llama3:latest", "phi3:mini"}
        assert models[0].size_bytes == 4_700_000_000

    def test_graceful_on_url_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(*_a: object, **_k: object) -> object:
            raise hardware.urllib.error.URLError("connection refused")

        monkeypatch.setattr(hardware.urllib.request, "urlopen", _raise)
        assert hardware._detect_ollama_models() == []

    def test_graceful_on_bad_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            hardware.urllib.request,
            "urlopen",
            lambda *a, **k: _FakeResp(b"not json"),
        )
        assert hardware._detect_ollama_models() == []


class TestDetectLlamaModels:
    def test_lists_gguf_files(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        (tmp_path / "model-b.gguf").write_text("x", encoding="utf-8")
        (tmp_path / "model-a.gguf").write_text("x", encoding="utf-8")
        (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
        monkeypatch.setenv("LLAMA_MODEL_DIR", str(tmp_path))
        assert hardware._detect_llama_models() == ["model-a.gguf", "model-b.gguf"]

    def test_empty_without_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LLAMA_MODEL_DIR", raising=False)
        assert hardware._detect_llama_models() == []


class TestDetectHardware:
    def test_combines_and_notes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(hardware.shutil, "which", lambda _: None)

        def _raise(*_a: object, **_k: object) -> object:
            raise hardware.urllib.error.URLError("down")

        monkeypatch.setattr(hardware.urllib.request, "urlopen", _raise)
        monkeypatch.delenv("LLAMA_MODEL_DIR", raising=False)

        info = detect_hardware()
        assert info.gpus == []
        assert info.ollama_models == []
        assert any("GPU" in n for n in info.notes)
        assert any("Ollama" in n for n in info.notes)

    def test_free_vram_uses_largest_gpu(self) -> None:
        info = HardwareInfo(
            gpus=[
                GpuInfo("a", 24576, 8000),
                GpuInfo("b", 24576, 20000),
            ]
        )
        assert info.free_vram_mb == 20000
        assert info.total_vram_mb == 24576


class TestSelectLocalModel:
    def _hw(
        self, ollama: list[LocalModel] | None = None, free_mb: int | None = None
    ) -> HardwareInfo:
        info = HardwareInfo(ollama_models=ollama or [])
        if free_mb is not None:
            info.gpus = [GpuInfo("gpu", free_mb * 2, free_mb)]
        return info

    def test_keeps_installed_and_fitting(self) -> None:
        hw = self._hw([LocalModel("mixtral:latest", 20_000_000_000)], free_mb=40_000)
        assert select_local_model("ollama", "high", _OLLAMA_TIERS, hw) is None

    def test_downgrades_to_installed(self) -> None:
        hw = self._hw([LocalModel("phi3:mini", 2_000_000_000)])
        assert select_local_model("ollama", "high", _OLLAMA_TIERS, hw) == "phi3"

    def test_vram_forces_downgrade(self) -> None:
        hw = self._hw(
            [
                LocalModel("mixtral:latest", 30_000_000_000),
                LocalModel("phi3:mini", 2_000_000_000),
            ],
            free_mb=4_000,
        )
        # mixtral (30 GB) does not fit 4 GB VRAM; phi3 (2 GB) does.
        assert select_local_model("ollama", "high", _OLLAMA_TIERS, hw) == "phi3"

    def test_none_when_nothing_installed(self) -> None:
        assert select_local_model("ollama", "high", _OLLAMA_TIERS, self._hw([])) is None

    def test_unknown_vram_assumes_fits(self) -> None:
        hw = self._hw([LocalModel("mixtral:latest", None)])
        assert select_local_model("ollama", "high", _OLLAMA_TIERS, hw) is None

    def test_non_local_engine_is_noop(self) -> None:
        assert select_local_model("claude", "high", {"high": "opus"}, self._hw()) is None

    def test_llama_file_name_match(self) -> None:
        info = HardwareInfo(llama_models=["llama-3.2-3b-instruct.gguf"])
        assert select_local_model("llama", "high", _LLAMA_TIERS, info) == "llama-3.2-3b"

    def test_llama_none_without_files(self) -> None:
        assert select_local_model("llama", "high", _LLAMA_TIERS, HardwareInfo()) is None


class TestFormatHardware:
    def test_text_with_gpus_and_models(self) -> None:
        info = HardwareInfo(
            gpus=[GpuInfo("RTX 4090", 24576, 20000)],
            ollama_models=[LocalModel("llama3:latest", 4_700_000_000)],
        )
        text = format_hardware(info)
        assert "RTX 4090" in text
        assert "llama3:latest" in text
        assert "GB" in text

    def test_text_empty(self) -> None:
        text = format_hardware(HardwareInfo())
        assert "none detected" in text

    def test_json_round_trips(self) -> None:
        info = HardwareInfo(gpus=[GpuInfo("g", 100, 50)])
        payload = json.loads(format_hardware_json(info))
        assert payload["free_vram_mb"] == 50
        assert payload["gpus"][0]["name"] == "g"


class TestRoutingIntegration:
    def test_resolve_auto_model_adjusts_to_installed(self, tmp_path: Path) -> None:
        from maestro_cli.loader import load_plan
        from maestro_cli.routing import resolve_auto_model

        plan_path = tmp_path / "p.yaml"
        plan_path.write_text(
            """
version: 1
name: hw-demo
tasks:
  - id: t1
    engine: ollama
    model: auto
    tags: [security, architecture]
    prompt: "Audit the authentication subsystem for injection and broken access control issues across all endpoints and middleware layers."
""",
            encoding="utf-8",
        )
        plan = load_plan(str(plan_path))
        task = plan.tasks[0]
        info = HardwareInfo(ollama_models=[LocalModel("phi3:mini", 2_000_000_000)])

        model = resolve_auto_model(
            task, plan, "ollama", dag_metadata={"hardware": info}
        )
        # high-tier default (mixtral) is not installed → falls to the only
        # installed model.
        assert model == "phi3"

    def test_resolve_auto_model_noop_without_hardware(self, tmp_path: Path) -> None:
        from maestro_cli.loader import load_plan
        from maestro_cli.routing import resolve_auto_model

        plan_path = tmp_path / "p.yaml"
        plan_path.write_text(
            """
version: 1
name: hw-demo
tasks:
  - id: t1
    engine: ollama
    model: auto
    tags: [trivial]
    prompt: "fix typo"
""",
            encoding="utf-8",
        )
        plan = load_plan(str(plan_path))
        task = plan.tasks[0]
        # No hardware in metadata → tier default stands (low → phi3).
        model = resolve_auto_model(task, plan, "ollama", dag_metadata={})
        assert model == "phi3"
