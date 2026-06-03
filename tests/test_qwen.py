from __future__ import annotations

from pathlib import Path

from maestro_cli.doctor import run_doctor
from maestro_cli.loader import load_plan
from maestro_cli.models import EngineName, PlanDefaults, PlanSpec, QWEN_MODELS, TaskSpec
from maestro_cli.runners import _ENV_ALLOWLIST, build_command


def _write_plan(tmp_path: Path, content: str) -> Path:
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(content, encoding="utf-8")
    return plan_file


class TestQwenModels:
    def test_qwen_in_engine_name(self) -> None:
        assert "qwen" in EngineName.__args__

    def test_qwen_models_aliases(self) -> None:
        assert "coder" in QWEN_MODELS
        assert "coder-turbo" in QWEN_MODELS
        assert "max" in QWEN_MODELS
        assert "plus" in QWEN_MODELS
        assert "qwq" in QWEN_MODELS

    def test_plan_defaults_has_qwen(self) -> None:
        defaults = PlanDefaults()
        assert hasattr(defaults, "qwen")


class TestQwenLoader:
    def test_qwen_engine_valid(self, tmp_path: Path) -> None:
        plan_file = _write_plan(
            tmp_path,
            """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: qwen
    prompt: "Do something"
""",
        )
        plan = load_plan(plan_file)
        assert plan.tasks[0].engine == "qwen"

    def test_qwen_defaults_parsed(self, tmp_path: Path) -> None:
        plan_file = _write_plan(
            tmp_path,
            """\
version: 1
name: test-plan
defaults:
  qwen:
    model: coder
tasks:
  - id: t1
    engine: qwen
    prompt: "Do something"
""",
        )
        plan = load_plan(plan_file)
        assert plan.defaults.qwen.model == "coder"

    def test_qwen_defaults_args(self, tmp_path: Path) -> None:
        plan_file = _write_plan(
            tmp_path,
            """\
version: 1
name: test-plan
defaults:
  qwen:
    args: ["--yolo", "--foo"]
tasks:
  - id: t1
    engine: qwen
    prompt: "Do something"
""",
        )
        plan = load_plan(plan_file)
        assert isinstance(plan.defaults.qwen.args, list)
        assert plan.defaults.qwen.args == ["--yolo", "--foo"]


class TestQwenRunner:
    def _make_plan(self, **kwargs) -> PlanSpec:
        return PlanSpec(version=1, name="p", defaults=PlanDefaults(**kwargs), tasks=[])

    def test_build_command_qwen(self, monkeypatch) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        plan = self._make_plan()
        task = TaskSpec(id="t", engine="qwen", prompt="Do stuff")
        cmd, shell = build_command(plan, task, Path("/tmp"))
        assert cmd[0] == "qwen-code"
        assert "--prompt" in cmd
        assert not shell

    def test_qwen_model_alias_resolved(self, monkeypatch) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        plan = self._make_plan()
        task = TaskSpec(id="t", engine="qwen", model="coder", prompt="Do stuff")
        cmd, _ = build_command(plan, task, Path("/tmp"))
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "qwen-coder-plus"

    def test_qwen_plus_alias_resolved(self, monkeypatch) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        plan = self._make_plan()
        task = TaskSpec(id="t", engine="qwen", model="plus", prompt="Do stuff")
        cmd, _ = build_command(plan, task, Path("/tmp"))
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "qwen-plus"

    def test_qwen_yolo_profile(self, monkeypatch) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        plan = self._make_plan()
        task = TaskSpec(id="t", engine="qwen", prompt="Do stuff")
        cmd, _ = build_command(plan, task, Path("/tmp"), execution_profile="yolo")
        assert "--yolo" in cmd

    def test_qwen_env_allowlist(self) -> None:
        assert "DASHSCOPE_API_KEY" in _ENV_ALLOWLIST


class TestQwenDoctor:
    def test_doctor_checks_qwen(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr("maestro_cli.doctor.shutil.which", lambda _name: None)
        results = run_doctor(run_dir=str(tmp_path / "runs"), json_output=False)
        check_names = [name for name, _, _ in results]
        assert any("qwen" in name for name in check_names)
