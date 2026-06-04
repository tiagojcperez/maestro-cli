from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from maestro_cli.blame import _load_events_evidence, blame_run
from maestro_cli.ci import _require_non_empty, get_ci_provider
from maestro_cli.contracts import normalize_task_contract
from maestro_cli.cost_backfill import discover_run_roots
from maestro_cli.council import CouncilParticipant, _call_participant
from maestro_cli.models import PlanSpec, PolicySpec, TaskSpec
from maestro_cli.policy import compile_policy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task(task_id: str, **kwargs: object) -> TaskSpec:
    return TaskSpec(id=task_id, **kwargs)  # type: ignore[arg-type]


def _policy(rule: str) -> PolicySpec:
    return PolicySpec(name="p1", rule=rule, action="warn", message="")  # type: ignore[arg-type]


def _write_log(path: Path, body: str) -> None:
    """Write a log file with header + blank line + body (so the contract
    extractor takes the body after the first blank line)."""
    path.write_text(f"header\n\n{body}\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# blame.py
# ---------------------------------------------------------------------------


class TestBlameEventsEvidenceOSError:
    def test_read_text_oserror_is_swallowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An OSError while reading events.jsonl is caught and yields an empty
        evidence map rather than propagating."""
        events_path = tmp_path / "events.jsonl"
        events_path.write_text("{}\n", encoding="utf-8")

        original_read_text = Path.read_text

        def _boom(self: Path, *args: object, **kwargs: object) -> str:
            if self.name == "events.jsonl":
                raise OSError("simulated read failure")
            return original_read_text(self, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(Path, "read_text", _boom)

        result = _load_events_evidence(tmp_path)
        assert result == {}


class TestBlameRunNoNodes:
    def test_failed_tasks_present_still_builds_a_chain(self, tmp_path: Path) -> None:
        """Sanity: a failed task always produces a node, so blame_run returns a
        populated chain (the empty-nodes guard is defensive)."""
        manifest = {
            "task_results": {
                "t1": {"task_id": "t1", "status": "failed", "exit_code": 1,
                       "message": "boom", "depends_on": []},
            }
        }
        (tmp_path / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        chain = blame_run(tmp_path)
        assert chain.root_task_id == "t1"
        assert chain.nodes


# ---------------------------------------------------------------------------
# ci.py
# ---------------------------------------------------------------------------


class TestCiRequireNonEmpty:
    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            _require_non_empty("   ", "workflow_name")

    def test_field_name_included(self) -> None:
        with pytest.raises(ValueError, match="python_version"):
            _require_non_empty("", "python_version")

    def test_non_empty_returns_stripped(self) -> None:
        assert _require_non_empty("  Maestro CI  ", "workflow_name") == "Maestro CI"


class TestCiGetProvider:
    def test_unsupported_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported CI provider"):
            get_ci_provider("jenkins")

    def test_unsupported_provider_lists_supported(self) -> None:
        with pytest.raises(ValueError, match="github_actions"):
            get_ci_provider("does-not-exist")

    def test_known_alias_returns_provider(self) -> None:
        provider = get_ci_provider("github")
        assert provider.name == "github_actions"


# ---------------------------------------------------------------------------
# council.py
# ---------------------------------------------------------------------------


class TestCouncilCostExtraction:
    @patch("maestro_cli.runners.build_command")
    @patch("maestro_cli.council.subprocess.run")
    def test_cost_line_is_extracted_and_loop_breaks(
        self, mock_run: MagicMock, mock_build: MagicMock, tmp_path: Path
    ) -> None:
        """A recognizable cost line in the output drives the cost-extraction
        loop to assign the value and break."""
        mock_build.return_value = (["codex", "exec", "-p", "x"], False)
        mock_run.return_value = SimpleNamespace(
            stdout="some answer text\ncost: $0.0420\n",
            stderr="",
            returncode=0,
        )

        p = CouncilParticipant(engine="codex", model="5.4", role="coder")
        text, cost = _call_participant(p, "code", tmp_path)

        assert "some answer text" in text
        assert cost == pytest.approx(0.042)

    @patch("maestro_cli.runners.build_command")
    @patch("maestro_cli.council.subprocess.run")
    def test_no_cost_line_returns_none(
        self, mock_run: MagicMock, mock_build: MagicMock, tmp_path: Path
    ) -> None:
        mock_build.return_value = (["codex", "exec", "-p", "x"], False)
        mock_run.return_value = SimpleNamespace(
            stdout="plain answer with no cost marker\n",
            stderr="",
            returncode=0,
        )

        p = CouncilParticipant(engine="codex", model="5.4", role="coder")
        text, cost = _call_participant(p, "code", tmp_path)

        assert "plain answer" in text
        assert cost is None


# ---------------------------------------------------------------------------
# policy.py
# ---------------------------------------------------------------------------


class TestPolicyComputedFields:
    def test_contract_type_field(self) -> None:
        plan = PlanSpec(name="p")
        task = _make_task("t1", contract_type="sql-schema")
        ev = compile_policy(_policy('task.contract_type == "sql-schema"'))
        assert ev(task, plan) is True

    def test_contract_type_defaults_to_empty(self) -> None:
        plan = PlanSpec(name="p")
        task = _make_task("t1")  # no contract_type
        ev = compile_policy(_policy('task.contract_type == ""'))
        assert ev(task, plan) is True

    def test_has_consistency_group_true(self) -> None:
        plan = PlanSpec(name="p")
        task = _make_task("t1", consistency_group="grp-a")
        ev = compile_policy(_policy("task.has_consistency_group"))
        assert ev(task, plan) is True

    def test_has_consistency_group_false(self) -> None:
        plan = PlanSpec(name="p")
        task = _make_task("t1")  # no consistency_group
        ev = compile_policy(_policy("task.has_consistency_group == False"))
        assert ev(task, plan) is True


# ---------------------------------------------------------------------------
# contracts.py
# ---------------------------------------------------------------------------


class TestContractApiSchemaPreviewTruncation:
    def test_more_than_five_schemas_truncates_preview(self, tmp_path: Path) -> None:
        """An OpenAPI body with > 5 component schemas truncates the preview
        with an ellipsis."""
        schemas = {f"Model{i}": {"type": "object"} for i in range(7)}
        payload = {
            "openapi": "3.0.0",
            "paths": {"/a": {}, "/b": {}},
            "components": {"schemas": schemas},
        }
        log = tmp_path / "t.log"
        _write_log(log, json.dumps(payload))
        task = _make_task("t1", contract_type="api-schema")

        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "..." in contract.summary
        assert contract.metadata["schema_count"] == 7


class TestContractTestManifestZeroTotal:
    def test_explicit_zero_total_is_recomputed_from_parts(
        self, tmp_path: Path
    ) -> None:
        """When JSON declares total=0 but has non-zero passed/failed, total is
        recomputed from the parts."""
        payload = {"passed": 3, "failed": 2, "skipped": 1, "total": 0}
        log = tmp_path / "t.log"
        _write_log(log, json.dumps(payload))
        task = _make_task("t1", contract_type="test-manifest")

        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert contract.metadata["total"] == 6
        assert contract.metadata["passed"] == 3


# ---------------------------------------------------------------------------
# cost_backfill.py
# ---------------------------------------------------------------------------


class TestDiscoverRunRootsResolveOSError:
    def test_resolve_oserror_falls_back_to_original_path(self) -> None:
        """If project_root.resolve() raises OSError, the original path is used;
        a non-existent path then yields an empty list."""
        fake_root = MagicMock(spec=Path)
        fake_root.resolve.side_effect = OSError("resolve failed")
        fake_root.exists.return_value = False

        roots = discover_run_roots(fake_root)

        assert roots == []
        fake_root.resolve.assert_called_once()
        fake_root.exists.assert_called_once()
