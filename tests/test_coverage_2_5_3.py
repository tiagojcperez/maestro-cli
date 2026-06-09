"""Targeted coverage for the v2.5.3 tranche (estimate / fts prefix / scip /
hardware / routing) — exercises the branches the feature tests did not reach so
the package stays at its ~100% coverage standard."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

import maestro_cli.hardware as hardware
import maestro_cli.scheduler as scheduler_mod
from maestro_cli.estimate import (
    _fmt_cost,
    _fmt_tokens,
    estimate_plan,
)
from maestro_cli.fts import rank_documents
from maestro_cli.hardware import (
    GpuInfo,
    HardwareInfo,
    LocalModel,
    detect_hardware,
    format_hardware,
    format_hardware_json,
    select_local_model,
)
from maestro_cli.loader import load_plan
from maestro_cli.models import TaskResult
from maestro_cli.routing import resolve_auto_model
from maestro_cli.scheduler import run_plan
from maestro_cli.scip import (
    _documentation_text,
    _symbol_name,
    format_scip_map,
    load_scip_index,
    parse_scip_index,
)

_OLLAMA_TIERS = {"low": "phi3", "medium": "llama3", "high": "mixtral"}


def _write_plan(tmp_path: Path, body: str) -> str:
    path = tmp_path / "plan.yaml"
    path.write_text(body, encoding="utf-8")
    return str(path)


class _FakeResp:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> _FakeResp:
        return self

    def __exit__(self, *_a: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._payload


class _FakeProc:
    def __init__(self, stdout: str, returncode: int = 0) -> None:
        self.stdout = stdout
        self.returncode = returncode


# ---------------------------------------------------------------------------
# estimate.py
# ---------------------------------------------------------------------------

class TestEstimateCoverage:
    def test_manifest_edge_cases_are_skipped(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "runs"
        (run_dir / "r0_est").mkdir(parents=True)  # no manifest -> skip (111)
        (run_dir / "r1_est").mkdir(parents=True)
        (run_dir / "r1_est" / "run_manifest.json").write_text("{bad", encoding="utf-8")  # 114-115
        (run_dir / "r2_est").mkdir(parents=True)
        (run_dir / "r2_est" / "run_manifest.json").write_text(
            json.dumps({"task_results": "nope"}), encoding="utf-8"  # 118
        )
        (run_dir / "r3_est").mkdir(parents=True)
        (run_dir / "r3_est" / "run_manifest.json").write_text(
            json.dumps({"task_results": {"design": "nope"}}), encoding="utf-8"  # 121
        )
        plan = load_plan(
            _write_plan(
                tmp_path,
                "version: 1\nname: est\ntasks:\n  - id: design\n    engine: claude\n    model: sonnet\n    prompt: design the thing\n",
            )
        )
        report = estimate_plan(plan, run_dir)
        # No usable history anywhere -> heuristic.
        assert report.task_estimates[0].source == "heuristic"

    def test_group_task(self, tmp_path: Path) -> None:
        (tmp_path / "sub.yaml").write_text(
            "version: 1\nname: sub\ntasks:\n  - id: s1\n    command: echo s1\n",
            encoding="utf-8",
        )
        plan = load_plan(
            _write_plan(
                tmp_path,
                "version: 1\nname: est\ntasks:\n  - id: g\n    group: sub.yaml\n",
            )
        )
        report = estimate_plan(plan, tmp_path / "runs")
        g = report.task_estimates[0]
        assert g.kind == "group"
        assert g.cost_usd is None
        assert report.unpriced_tasks == 1

    def test_prompt_md_file(self, tmp_path: Path) -> None:
        (tmp_path / "p.md").write_text(
            "## Task\n\n```text\nImplement the secure login module with hashing.\n```\n",
            encoding="utf-8",
        )
        plan = load_plan(
            _write_plan(
                tmp_path,
                "version: 1\nname: est\ntasks:\n  - id: t1\n    engine: claude\n    model: sonnet\n    prompt_md_file: p.md\n    prompt_md_heading: Task\n",
            )
        )
        report = estimate_plan(plan, tmp_path / "runs")
        assert report.task_estimates[0].source == "heuristic"
        assert report.task_estimates[0].input_tokens > 0

    def test_long_prompt_token_formatting(self, tmp_path: Path) -> None:
        long_prompt = "word " * 1200  # > 1000 tokens after /4
        plan = load_plan(
            _write_plan(
                tmp_path,
                f"version: 1\nname: est\ntasks:\n  - id: t1\n    engine: claude\n    model: sonnet\n    prompt: {long_prompt}\n",
            )
        )
        from maestro_cli.estimate import format_estimate

        text = format_estimate(estimate_plan(plan, tmp_path / "runs"))
        assert "k/" in text or "/0" in text  # _fmt_tokens emitted a 'Xk' value

    def test_fmt_helpers_edges(self) -> None:
        assert _fmt_tokens(0) == "—"
        assert _fmt_tokens(1500) == "1.5k"
        assert _fmt_tokens(42) == "42"
        assert _fmt_cost(None) == "—"


# ---------------------------------------------------------------------------
# fts.py — prefix term cap
# ---------------------------------------------------------------------------

class TestFtsCoverage:
    def test_query_term_cap(self) -> None:
        # > _MAX_QUERY_TERMS (64) unique tokens exercises the break.
        query = " ".join(f"term{n}" for n in range(80))
        hits = rank_documents(["term1 term2 term75 content"], query)
        assert hits  # still matches; no crash


# ---------------------------------------------------------------------------
# scip.py
# ---------------------------------------------------------------------------

class TestScipCoverage:
    def test_symbol_name_without_identifiers(self) -> None:
        assert _symbol_name("scip py a b 1 ###", "") == "###"

    def test_documentation_non_list(self) -> None:
        idx = parse_scip_index(
            {"documents": [{"relative_path": "a.py", "symbols": [
                {"symbol": "s", "display_name": "S", "documentation": "not-a-list"}
            ]}]}
        )
        assert idx.symbols[0].documentation == ""

    def test_occurrence_branches(self) -> None:
        idx = parse_scip_index(
            {"documents": [
                {"relative_path": "a.py", "occurrences": "not-a-list"},  # 154
                {"relative_path": "b.py", "occurrences": [
                    None,  # 157
                    {"symbol": "", "symbol_roles": 1},  # 161 (empty)
                    {"symbol": "dup", "symbol_roles": 1},
                    {"symbol": "dup", "symbol_roles": 1},  # 161 (seen)
                ]},
            ]}
        )
        assert [s.symbol_id for s in idx.symbols] == ["dup"]

    def test_load_index_not_a_dict(self, tmp_path: Path) -> None:
        (tmp_path / "index.scip.json").write_text("[1, 2, 3]", encoding="utf-8")
        assert load_scip_index(tmp_path) is None

    def test_score_with_empty_query(self) -> None:
        idx = parse_scip_index(
            {"documents": [{"relative_path": "a.py", "symbols": [
                {"symbol": "s", "display_name": "Thing"}
            ]}]}
        )
        out = format_scip_map(idx, "", 6000)  # empty query -> overview fallback
        assert "Thing" in out

    def test_documentation_text_direct(self) -> None:
        assert _documentation_text(42) == ""
        assert _documentation_text(["a", 5, "b"]) == "a b"

    def test_parse_skips_non_dict_symbol_entry(self) -> None:
        idx = parse_scip_index(
            {"documents": [{"relative_path": "a.py", "symbols": [
                None, {"symbol": "s", "display_name": "S"}
            ]}]}
        )
        assert [s.name for s in idx.symbols] == ["S"]


# ---------------------------------------------------------------------------
# hardware.py
# ---------------------------------------------------------------------------

class TestHardwareCoverage:
    def test_json_full_data(self) -> None:
        info = HardwareInfo(
            gpus=[GpuInfo("g", 100, 50)],
            ollama_models=[LocalModel("m:latest", 200)],
            llama_models=["x.gguf"],
            notes=["n"],
        )
        payload = json.loads(format_hardware_json(info))
        assert payload["ollama_models"][0] == {"name": "m:latest", "size_bytes": 200}
        assert payload["llama_models"] == ["x.gguf"]

    def test_detect_gpus_skips_malformed_lines(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(hardware.shutil, "which", lambda _: "/x/nvidia-smi")
        monkeypatch.setattr(
            hardware.subprocess,
            "run",
            lambda *a, **k: _FakeProc("good gpu, 100, 50\nbadline\n, 100, 50\n"),
        )
        gpus = hardware._detect_gpus()
        assert [g.name for g in gpus] == ["good gpu"]

    def test_ollama_host_prepends_scheme(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OLLAMA_HOST", "myhost:1234")
        captured: dict[str, str] = {}

        def _fake(url: str, **_k: object) -> object:
            captured["url"] = url
            raise hardware.urllib.error.URLError("x")

        monkeypatch.setattr(hardware.urllib.request, "urlopen", _fake)
        assert hardware._detect_ollama_models() == []
        assert captured["url"].startswith("http://myhost:1234")

    def test_ollama_payload_not_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            hardware.urllib.request, "urlopen", lambda *a, **k: _FakeResp(b"[1,2,3]")
        )
        assert hardware._detect_ollama_models() == []

    def test_ollama_skips_bad_entries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = json.dumps(
            {"models": [None, {"size": 5}, {"name": "", "size": 5}, {"name": "ok:latest", "size": 100}]}
        ).encode("utf-8")
        monkeypatch.setattr(
            hardware.urllib.request, "urlopen", lambda *a, **k: _FakeResp(payload)
        )
        models = hardware._detect_ollama_models()
        assert [m.name for m in models] == ["ok:latest"]

    def test_llama_dir_is_a_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        f = tmp_path / "notadir.txt"
        f.write_text("x", encoding="utf-8")
        monkeypatch.setenv("LLAMA_MODEL_DIR", str(f))
        assert hardware._detect_llama_models() == []

    def test_detect_hardware_llama_note(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(hardware.shutil, "which", lambda _: None)

        def _raise(*_a: object, **_k: object) -> object:
            raise hardware.urllib.error.URLError("down")

        monkeypatch.setattr(hardware.urllib.request, "urlopen", _raise)
        monkeypatch.setenv("LLAMA_MODEL_DIR", str(tmp_path))  # empty dir
        info = detect_hardware()
        assert any("gguf" in n for n in info.notes)

    def test_select_none_when_no_tier_match(self) -> None:
        hw = HardwareInfo(ollama_models=[LocalModel("totally-different:latest", 100)])
        assert select_local_model("ollama", "high", _OLLAMA_TIERS, hw) is None

    def test_select_unknown_size_with_known_vram(self) -> None:
        hw = HardwareInfo(
            ollama_models=[LocalModel("mixtral:latest", None)],
            gpus=[GpuInfo("g", 8000, 4000)],
        )
        assert select_local_model("ollama", "high", _OLLAMA_TIERS, hw) is None

    def test_format_hardware_with_llama_and_notes(self) -> None:
        info = HardwareInfo(llama_models=["model-a.gguf"], notes=["a note"])
        text = format_hardware(info)
        assert "llama.cpp models" in text
        assert "model-a.gguf" in text
        assert "note: a note" in text

    def test_detect_gpus_non_numeric_vram(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(hardware.shutil, "which", lambda _: "/x/nvidia-smi")
        monkeypatch.setattr(
            hardware.subprocess, "run", lambda *a, **k: _FakeProc("gpu x, abc, def\n")
        )
        gpus = hardware._detect_gpus()
        assert gpus[0].vram_total_mb is None
        assert gpus[0].vram_free_mb is None


# ---------------------------------------------------------------------------
# routing.py — hardware adjustment records evidence
# ---------------------------------------------------------------------------

class TestRoutingEvidence:
    def test_evidence_records_hardware_adjustment(self, tmp_path: Path) -> None:
        plan = load_plan(
            _write_plan(
                tmp_path,
                "version: 1\nname: r\ntasks:\n  - id: t1\n    engine: ollama\n    model: auto\n    tags: [security, architecture]\n    prompt: audit the authentication and access control across endpoints thoroughly\n",
            )
        )
        info = HardwareInfo(ollama_models=[LocalModel("phi3:mini", 2_000_000_000)])
        evidence: dict[str, object] = {}
        model = resolve_auto_model(
            plan.tasks[0], plan, "ollama",
            dag_metadata={"hardware": info}, evidence=evidence,
        )
        assert model == "phi3"
        assert evidence.get("hardware_adjusted_from") == "mixtral"


# ---------------------------------------------------------------------------
# runners.py — selective FTS path, non-hit chunk carried by upstream boost
# ---------------------------------------------------------------------------

class TestSelectiveBoost:
    def test_non_hit_chunk_rides_on_upstream_boost(self) -> None:
        from maestro_cli.runners import _build_selective_context

        upstreams = {
            "hit": "alpha beta gamma matching keywords here. ",
            "miss": "zzz qqq unrelated lines with no keyword overlap at all. ",
        }
        # 'miss' has no keyword hit but a strong upstream boost -> included via boost.
        result = _build_selective_context(
            upstreams, 5000, {"alpha", "beta"}, {"miss": 5.0}
        )
        assert "miss" in result


# ---------------------------------------------------------------------------
# Integration: context_mode dispatch + hardware detection through run_plan
# ---------------------------------------------------------------------------

def _mock_execute_factory(capture: dict[str, str] | None = None) -> object:
    def mock_execute(
        plan: object,
        task: object,
        run_path: Path,
        dry_run: bool = False,
        execution_profile: str = "plan",
        upstream_results: object = None,
        context_synthesis: str = "",
        workspace_brief: str = "",
        **kwargs: object,
    ) -> TaskResult:
        if capture is not None:
            capture["cs"] = context_synthesis
        now = datetime.now(timezone.utc)
        result = TaskResult(
            task_id=task.id,  # type: ignore[attr-defined]
            status="success",
            exit_code=0,
            started_at=now,
            finished_at=now,
            duration_sec=0.01,
            command="echo ok",
            log_path=run_path / f"{task.id}.log",  # type: ignore[attr-defined]
            result_path=run_path / f"{task.id}.result.json",  # type: ignore[attr-defined]
            message="ok",
        )
        result.log_path.write_text("status=success\n", encoding="utf-8")
        result.result_path.write_text(
            json.dumps(result.to_dict(), indent=2), encoding="utf-8"
        )
        return result

    return mock_execute


class TestContextDispatchIntegration:
    def test_scip_dispatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "index.scip.json").write_text(
            json.dumps({
                "metadata": {"tool_info": {"name": "scip-python"}},
                "documents": [{"relative_path": "auth.py", "symbols": [
                    {"symbol": "s", "display_name": "authenticate",
                     "documentation": ["Authenticate a user."]}
                ]}],
            }),
            encoding="utf-8",
        )
        # The workspace-derived dispatch fires inside `if task.context_from`, so
        # the consuming task lists an upstream (its output is ignored; the map
        # comes from the workspace index).
        plan = load_plan(
            _write_plan(
                tmp_path,
                f"version: 1\nname: scip-run\nworkspace_root: {ws.as_posix()}\ntasks:\n  - id: a\n    command: echo up\n  - id: t1\n    engine: claude\n    model: sonnet\n    context_mode: scip\n    context_from: [a]\n    depends_on: [a]\n    prompt: authentication\n",
            )
        )
        captured: dict[str, str] = {}
        monkeypatch.setattr(
            scheduler_mod, "execute_task", _mock_execute_factory(captured)
        )
        monkeypatch.setattr(scheduler_mod, "_preflight_checks", lambda *a, **kw: None)
        run_plan(plan, run_dir_override=str(tmp_path / "runs"))
        assert "authenticate" in captured.get("cs", "")

    def test_codebase_map_dispatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ws = tmp_path / "ws"
        (ws / ".understand-anything").mkdir(parents=True)
        (ws / ".understand-anything" / "knowledge-graph.json").write_text(
            json.dumps({"nodes": [
                {"id": "n1", "type": "file", "name": "login", "summary": "auth login"}
            ], "edges": []}),
            encoding="utf-8",
        )
        plan = load_plan(
            _write_plan(
                tmp_path,
                f"version: 1\nname: cbm-run\nworkspace_root: {ws.as_posix()}\ntasks:\n  - id: a\n    command: echo up\n  - id: t1\n    engine: claude\n    model: sonnet\n    context_mode: codebase_map\n    context_from: [a]\n    depends_on: [a]\n    prompt: login\n",
            )
        )
        monkeypatch.setattr(scheduler_mod, "execute_task", _mock_execute_factory())
        monkeypatch.setattr(scheduler_mod, "_preflight_checks", lambda *a, **kw: None)
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))
        assert result.task_results["t1"].status == "success"

    def test_hardware_detect_for_auto_local_engine(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # nvidia-smi / ollama absent -> detection is graceful; the point is to
        # exercise the scheduler's once-per-run hardware-detect block.
        monkeypatch.setattr(hardware.shutil, "which", lambda _: None)
        monkeypatch.setattr(
            hardware.urllib.request,
            "urlopen",
            lambda *a, **k: (_ for _ in ()).throw(hardware.urllib.error.URLError("x")),
        )
        plan = load_plan(
            _write_plan(
                tmp_path,
                "version: 1\nname: hw-run\ntasks:\n  - id: t1\n    engine: ollama\n    model: auto\n    prompt: lint the code\n",
            )
        )
        monkeypatch.setattr(scheduler_mod, "execute_task", _mock_execute_factory())
        monkeypatch.setattr(scheduler_mod, "_preflight_checks", lambda *a, **kw: None)
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))
        assert result.task_results["t1"].status == "success"
