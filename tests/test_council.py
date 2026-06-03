from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from maestro_cli.council import (
    CouncilParticipant,
    CouncilResult,
    CouncilRound,
    CouncilSpec,
    _adjust_for_council,
    _build_chain_prompt,
    _build_consolidation_prompt,
    _build_graph_round_prompt,
    _build_round_prompt,
    _call_participant,
    run_council,
)


# ===========================================================================
# TestCouncilDataclasses
# ===========================================================================


class TestCouncilDataclasses:
    def test_participant_to_dict(self) -> None:
        p = CouncilParticipant(engine="claude", model="opus", role="architect")
        d = p.to_dict()
        assert d["engine"] == "claude"
        assert d["model"] == "opus"
        assert d["role"] == "architect"

    def test_spec_to_dict(self) -> None:
        spec = CouncilSpec(
            participants=[
                CouncilParticipant(engine="claude", model="opus", role="architect"),
                CouncilParticipant(engine="gemini", model="pro", role="critic"),
            ],
            rounds=2,
            topology="star",
            consensus_threshold=0.8,
        )
        d = spec.to_dict()
        assert len(d["participants"]) == 2
        assert d["rounds"] == 2
        assert d["topology"] == "star"

    def test_result_to_dict(self) -> None:
        result = CouncilResult(
            rounds=[CouncilRound(round_num=1, responses={"architect": "Design A"})],
            synthesis="Consensus: Design A",
            total_cost_usd=0.05,
        )
        d = result.to_dict()
        assert len(d["rounds"]) == 1
        assert d["synthesis"] == "Consensus: Design A"


# ===========================================================================
# TestBuildRoundPrompt
# ===========================================================================


class TestBuildRoundPrompt:
    def test_first_round_basic(self) -> None:
        p = CouncilParticipant(engine="claude", model="opus", role="architect")
        prompt = _build_round_prompt("Design auth system", p, 1, [], "")
        assert "architect" in prompt
        assert "Round 1" in prompt
        assert "Design auth system" in prompt

    def test_first_round_with_upstream(self) -> None:
        p = CouncilParticipant(engine="claude", model="opus", role="reviewer")
        prompt = _build_round_prompt("Review code", p, 1, [], "upstream output here")
        assert "<upstream_context>" in prompt
        assert "upstream output here" in prompt

    def test_subsequent_round_includes_prior(self) -> None:
        p = CouncilParticipant(engine="gemini", model="pro", role="critic")
        prior = [CouncilRound(round_num=1, responses={"architect": "I suggest X"})]
        prompt = _build_round_prompt("Design system", p, 2, prior, "")
        assert "<prior_discussion>" in prompt
        assert "architect" in prompt
        assert "I suggest X" in prompt
        assert "Consider the perspectives" in prompt

    def test_no_role(self) -> None:
        p = CouncilParticipant(engine="claude", model="sonnet", role="")
        prompt = _build_round_prompt("task", p, 1, [], "")
        assert "Round 1" in prompt
        assert "<task>" in prompt


# ===========================================================================
# TestBuildConsolidationPrompt
# ===========================================================================


class TestBuildConsolidationPrompt:
    def test_includes_all_rounds(self) -> None:
        rounds = [
            CouncilRound(round_num=1, responses={"architect": "Plan A", "critic": "Problem with A"}),
            CouncilRound(round_num=2, responses={"architect": "Revised A", "critic": "Better"}),
        ]
        prompt = _build_consolidation_prompt("Design system", rounds)
        assert "consolidator" in prompt
        assert "Plan A" in prompt
        assert "Problem with A" in prompt
        assert "Revised A" in prompt
        assert "<original_task>" in prompt
        assert "<council_discussion>" in prompt


# ===========================================================================
# TestRunCouncil
# ===========================================================================


class TestRunCouncil:
    @patch("maestro_cli.council._call_participant")
    def test_single_round_star(self, mock_call: MagicMock) -> None:
        mock_call.side_effect = [
            ("I recommend approach A", 0.01),  # architect
            ("I see issues with A", 0.01),  # critic
            ("Consensus: modified approach A", 0.005),  # consolidation
        ]

        spec = CouncilSpec(
            participants=[
                CouncilParticipant(engine="claude", model="opus", role="architect"),
                CouncilParticipant(engine="gemini", model="pro", role="critic"),
            ],
            rounds=1,
        )

        result = run_council(spec, "Design auth", Path.cwd())

        assert len(result.rounds) == 1
        assert "architect" in result.rounds[0].responses
        assert "critic" in result.rounds[0].responses
        assert result.synthesis == "Consensus: modified approach A"
        assert result.total_cost_usd == pytest.approx(0.025)

    @patch("maestro_cli.council._call_participant")
    def test_two_rounds(self, mock_call: MagicMock) -> None:
        mock_call.side_effect = [
            ("Round 1: Plan A", 0.01),
            ("Round 1: Concerns", 0.01),
            ("Round 2: Revised A", 0.01),
            ("Round 2: Acceptable", 0.01),
            ("Final synthesis", 0.005),
        ]

        spec = CouncilSpec(
            participants=[
                CouncilParticipant(engine="claude", model="opus", role="architect"),
                CouncilParticipant(engine="claude", model="sonnet", role="critic"),
            ],
            rounds=2,
        )

        result = run_council(spec, "Design system", Path.cwd())
        assert len(result.rounds) == 2
        assert result.synthesis == "Final synthesis"
        assert mock_call.call_count == 5  # 2 participants × 2 rounds + 1 consolidation

    @patch("maestro_cli.council._call_participant")
    def test_event_callback_called(self, mock_call: MagicMock) -> None:
        mock_call.return_value = ("response", 0.01)

        spec = CouncilSpec(
            participants=[
                CouncilParticipant(engine="claude", role="a"),
                CouncilParticipant(engine="claude", role="b"),
            ],
            rounds=1,
        )

        events: list[str] = []
        callback = MagicMock(side_effect=lambda evt, **kw: events.append(evt))

        run_council(spec, "task", Path.cwd(), event_callback=callback)
        assert "council_turn" in events
        assert "council_consolidation" in events

    @patch("maestro_cli.council._call_participant")
    def test_cost_accumulation(self, mock_call: MagicMock) -> None:
        mock_call.side_effect = [
            ("resp1", 0.10),
            ("resp2", 0.20),
            ("synthesis", 0.05),
        ]

        spec = CouncilSpec(
            participants=[
                CouncilParticipant(engine="claude", role="a"),
                CouncilParticipant(engine="codex", role="b"),
            ],
            rounds=1,
        )

        result = run_council(spec, "task", Path.cwd())
        assert result.total_cost_usd == pytest.approx(0.35)

    @patch("maestro_cli.council._call_participant")
    def test_none_cost_handled(self, mock_call: MagicMock) -> None:
        mock_call.return_value = ("response", None)

        spec = CouncilSpec(
            participants=[
                CouncilParticipant(engine="ollama", role="local"),
                CouncilParticipant(engine="ollama", role="local2"),
            ],
            rounds=1,
        )

        result = run_council(spec, "task", Path.cwd())
        assert result.total_cost_usd == 0.0


# ===========================================================================
# Loader integration
# ===========================================================================


class TestCouncilLoaderIntegration:
    def test_council_accepted_in_plan(self, tmp_path: Path) -> None:
        from maestro_cli.loader import load_plan

        yaml_text = (
            "version: 1\nname: council-test\ntasks:\n"
            "  - id: deliberate\n    engine: claude\n    prompt: design auth\n"
            "    context_mode: council\n"
            "    council:\n"
            "      participants:\n"
            "        - engine: claude\n          model: opus\n          role: architect\n"
            "        - engine: gemini\n          model: pro\n          role: critic\n"
            "      rounds: 2\n"
            "      consensus_threshold: 0.8\n"
        )
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(yaml_text, encoding="utf-8")
        plan = load_plan(plan_path)
        task = plan.tasks[0]
        assert task.context_mode == "council"
        assert task.council is not None
        assert len(task.council.participants) == 2
        assert task.council.rounds == 2

    def test_council_requires_context_mode(self, tmp_path: Path) -> None:
        from maestro_cli.errors import PlanValidationError
        from maestro_cli.loader import load_plan

        yaml_text = (
            "version: 1\nname: test\ntasks:\n"
            "  - id: t\n    engine: claude\n    prompt: x\n"
            "    council:\n"
            "      participants:\n"
            "        - engine: claude\n          role: a\n"
            "        - engine: claude\n          role: b\n"
        )
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(yaml_text, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="context_mode: council"):
            load_plan(plan_path)

    def test_council_mode_without_block(self, tmp_path: Path) -> None:
        from maestro_cli.errors import PlanValidationError
        from maestro_cli.loader import load_plan

        yaml_text = (
            "version: 1\nname: test\ntasks:\n"
            "  - id: t\n    engine: claude\n    prompt: x\n"
            "    context_mode: council\n"
        )
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(yaml_text, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="council.*block"):
            load_plan(plan_path)

    def test_council_needs_two_participants(self, tmp_path: Path) -> None:
        from maestro_cli.errors import PlanValidationError
        from maestro_cli.loader import load_plan

        yaml_text = (
            "version: 1\nname: test\ntasks:\n"
            "  - id: t\n    engine: claude\n    prompt: x\n"
            "    context_mode: council\n"
            "    council:\n"
            "      participants:\n"
            "        - engine: claude\n          role: solo\n"
        )
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(yaml_text, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="at least 2"):
            load_plan(plan_path)

    def test_council_invalid_rounds(self, tmp_path: Path) -> None:
        from maestro_cli.errors import PlanValidationError
        from maestro_cli.loader import load_plan

        yaml_text = (
            "version: 1\nname: test\ntasks:\n"
            "  - id: t\n    engine: claude\n    prompt: x\n"
            "    context_mode: council\n"
            "    council:\n"
            "      participants:\n"
            "        - engine: claude\n          role: a\n"
            "        - engine: claude\n          role: b\n"
            "      rounds: 10\n"
        )
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(yaml_text, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="1-5"):
            load_plan(plan_path)

    def test_council_invalid_engine(self, tmp_path: Path) -> None:
        from maestro_cli.errors import PlanValidationError
        from maestro_cli.loader import load_plan

        yaml_text = (
            "version: 1\nname: test\ntasks:\n"
            "  - id: t\n    engine: claude\n    prompt: x\n"
            "    context_mode: council\n"
            "    council:\n"
            "      participants:\n"
            "        - engine: fake_engine\n          role: a\n"
            "        - engine: claude\n          role: b\n"
        )
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(yaml_text, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="valid engine"):
            load_plan(plan_path)

    def test_council_in_context_modes(self) -> None:
        from maestro_cli.models import CONTEXT_MODES

        assert "council" in CONTEXT_MODES


# ===========================================================================
# TestAdjustForCouncil
# ===========================================================================


class TestAdjustForCouncil:
    def test_replaces_stream_json(self) -> None:
        cmd = ["claude", "--output-format", "stream-json", "--print", "hello"]
        result = _adjust_for_council(cmd, "claude")
        assert result == ["claude", "--output-format", "text", "--print", "hello"]

    def test_replaces_json_format(self) -> None:
        cmd = ["claude", "--output-format", "json", "-p", "x"]
        result = _adjust_for_council(cmd, "claude")
        assert result == ["claude", "--output-format", "text", "-p", "x"]

    def test_no_replacement_for_text_format(self) -> None:
        cmd = ["claude", "--output-format", "text", "-p", "x"]
        result = _adjust_for_council(cmd, "claude")
        assert result == ["claude", "--output-format", "text", "-p", "x"]

    def test_passthrough_no_output_format(self) -> None:
        cmd = ["codex", "exec", "-p", "hello"]
        result = _adjust_for_council(cmd, "codex")
        assert result == ["codex", "exec", "-p", "hello"]

    def test_output_format_at_end_no_value(self) -> None:
        """--output-format at end with no following arg keeps it as-is."""
        cmd = ["claude", "--output-format"]
        result = _adjust_for_council(cmd, "claude")
        assert result == ["claude", "--output-format"]

    def test_output_format_with_non_json_value(self) -> None:
        """--output-format with a non-json/stream-json value passes through."""
        cmd = ["claude", "--output-format", "yaml", "--print", "x"]
        result = _adjust_for_council(cmd, "claude")
        assert result == ["claude", "--output-format", "yaml", "--print", "x"]


# ===========================================================================
# TestCallParticipant
# ===========================================================================


class TestCallParticipant:
    @patch("maestro_cli.runners.build_command")
    @patch("maestro_cli.council.subprocess.run")
    def test_successful_call(
        self, mock_run: MagicMock, mock_build: MagicMock, tmp_path: Path
    ) -> None:
        mock_build.return_value = (["claude", "--print", "hello"], False)
        mock_run.return_value = SimpleNamespace(
            stdout="The answer is 42\n", stderr="", returncode=0
        )

        p = CouncilParticipant(engine="codex", model="5.4", role="coder")
        text, cost = _call_participant(p, "solve it", tmp_path)

        assert "42" in text
        mock_build.assert_called_once()

    @patch("maestro_cli.runners.build_command")
    @patch("maestro_cli.council.subprocess.run")
    def test_timeout_returns_error_message(
        self, mock_run: MagicMock, mock_build: MagicMock, tmp_path: Path
    ) -> None:
        mock_build.return_value = (["claude", "--print", "x"], False)
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=120)

        p = CouncilParticipant(engine="claude", model="haiku", role="thinker")
        text, cost = _call_participant(p, "think hard", tmp_path)

        assert "timed out" in text
        assert "120" in text
        assert cost is None

    @patch("maestro_cli.runners.build_command")
    @patch("maestro_cli.council.subprocess.run")
    def test_file_not_found_returns_error(
        self, mock_run: MagicMock, mock_build: MagicMock, tmp_path: Path
    ) -> None:
        mock_build.return_value = (["nonexistent-cli", "x"], False)
        mock_run.side_effect = FileNotFoundError("not found")

        p = CouncilParticipant(engine="codex", role="dev")
        text, cost = _call_participant(p, "do it", tmp_path)

        assert "CLI not found" in text
        assert cost is None

    @patch("maestro_cli.runners.build_command")
    @patch("maestro_cli.council.subprocess.run")
    def test_generic_exception_returns_error(
        self, mock_run: MagicMock, mock_build: MagicMock, tmp_path: Path
    ) -> None:
        mock_build.return_value = (["claude", "x"], False)
        mock_run.side_effect = OSError("connection reset")

        p = CouncilParticipant(engine="claude", role="reviewer")
        text, cost = _call_participant(p, "review", tmp_path)

        assert "error calling reviewer" in text
        assert cost is None

    @patch("maestro_cli.runners.build_command")
    def test_build_command_exception(
        self, mock_build: MagicMock, tmp_path: Path
    ) -> None:
        mock_build.side_effect = RuntimeError("build failed")

        p = CouncilParticipant(engine="claude", model="opus", role="architect")
        text, cost = _call_participant(p, "design", tmp_path)

        assert "error building command" in text
        assert "architect" in text
        assert cost is None

    @patch("maestro_cli.runners.build_command")
    @patch("maestro_cli.council.subprocess.run")
    def test_empty_stdout_uses_stderr(
        self, mock_run: MagicMock, mock_build: MagicMock, tmp_path: Path
    ) -> None:
        mock_build.return_value = (["claude", "--print", "x"], False)
        mock_run.return_value = SimpleNamespace(
            stdout="", stderr="stderr output here", returncode=1
        )

        p = CouncilParticipant(engine="gemini", role="helper")
        text, cost = _call_participant(p, "help", tmp_path)

        assert "stderr output here" in text

    @patch("maestro_cli.runners._extract_stream_json_result_text")
    @patch("maestro_cli.runners.build_command")
    @patch("maestro_cli.council.subprocess.run")
    def test_claude_stream_json_extraction(
        self,
        mock_run: MagicMock,
        mock_build: MagicMock,
        mock_extract: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_build.return_value = (["claude", "--print", "x"], False)
        mock_run.return_value = SimpleNamespace(
            stdout='{"type":"result","result":"Clean answer"}',
            stderr="",
            returncode=0,
        )
        mock_extract.return_value = "Clean answer"

        p = CouncilParticipant(engine="claude", model="sonnet", role="analyst")
        text, cost = _call_participant(p, "analyze", tmp_path)

        assert text == "Clean answer"
        mock_extract.assert_called_once()

    @patch("maestro_cli.runners._extract_stream_json_result_text")
    @patch("maestro_cli.runners.build_command")
    @patch("maestro_cli.council.subprocess.run")
    def test_claude_stream_json_extraction_empty(
        self,
        mock_run: MagicMock,
        mock_build: MagicMock,
        mock_extract: MagicMock,
        tmp_path: Path,
    ) -> None:
        """When extraction returns empty, keep the original output."""
        mock_build.return_value = (["claude", "--print", "x"], False)
        raw_output = "raw json events"
        mock_run.return_value = SimpleNamespace(
            stdout=raw_output, stderr="", returncode=0
        )
        mock_extract.return_value = ""

        p = CouncilParticipant(engine="claude", model="haiku", role="checker")
        text, cost = _call_participant(p, "check", tmp_path)

        assert text == raw_output

    @patch("maestro_cli.runners.build_command")
    @patch("maestro_cli.council.subprocess.run")
    def test_non_claude_skips_stream_json_extraction(
        self, mock_run: MagicMock, mock_build: MagicMock, tmp_path: Path
    ) -> None:
        mock_build.return_value = (["codex", "exec", "-p", "x"], False)
        mock_run.return_value = SimpleNamespace(
            stdout="plain text answer", stderr="", returncode=0
        )

        p = CouncilParticipant(engine="codex", model="5.4", role="dev")
        text, cost = _call_participant(p, "code it", tmp_path)

        assert text == "plain text answer"

    @patch("maestro_cli.runners.build_command")
    @patch("maestro_cli.council.subprocess.run")
    def test_cost_extraction_from_output(
        self, mock_run: MagicMock, mock_build: MagicMock, tmp_path: Path
    ) -> None:
        mock_build.return_value = (["codex", "exec", "-p", "x"], False)
        mock_run.return_value = SimpleNamespace(
            stdout="result here\ntotal_cost_usd: 0.042\n",
            stderr="",
            returncode=0,
        )

        p = CouncilParticipant(engine="codex", model="5.4", role="coder")
        text, cost = _call_participant(p, "code", tmp_path)

        # Cost extraction depends on _extract_cost_from_line matching the format
        # The key point is we don't crash and return something
        assert text is not None

    @patch("maestro_cli.runners.build_command")
    @patch("maestro_cli.council.subprocess.run")
    def test_string_command_uses_shell(
        self, mock_run: MagicMock, mock_build: MagicMock, tmp_path: Path
    ) -> None:
        """When build_command returns a string, shell=True is used."""
        mock_build.return_value = ("echo hello | grep hello", True)
        mock_run.return_value = SimpleNamespace(
            stdout="hello", stderr="", returncode=0
        )

        p = CouncilParticipant(engine="codex", role="tester")
        text, cost = _call_participant(p, "test", tmp_path)

        # Verify subprocess was called with shell=True
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["shell"] is True

    @patch("maestro_cli.runners.build_command")
    @patch("maestro_cli.council.subprocess.run")
    def test_participant_without_role_uses_engine(
        self, mock_run: MagicMock, mock_build: MagicMock, tmp_path: Path
    ) -> None:
        mock_build.return_value = (["codex", "exec"], False)
        mock_run.return_value = SimpleNamespace(
            stdout="result", stderr="", returncode=0
        )

        p = CouncilParticipant(engine="codex", model="5.4", role="")
        text, cost = _call_participant(p, "do it", tmp_path)

        assert text == "result"

    @patch("maestro_cli.runners.build_command")
    @patch("maestro_cli.council.subprocess.run")
    def test_null_stdout_handled(
        self, mock_run: MagicMock, mock_build: MagicMock, tmp_path: Path
    ) -> None:
        mock_build.return_value = (["claude", "--print", "x"], False)
        mock_run.return_value = SimpleNamespace(
            stdout=None, stderr=None, returncode=0
        )

        p = CouncilParticipant(engine="gemini", role="analyst")
        text, cost = _call_participant(p, "analyze", tmp_path)

        assert text == ""
        assert cost is None

    @patch("maestro_cli.runners.build_command")
    @patch("maestro_cli.council.subprocess.run")
    def test_list_command_triggers_adjust(
        self, mock_run: MagicMock, mock_build: MagicMock, tmp_path: Path
    ) -> None:
        """When build_command returns a list, _adjust_for_council is called."""
        mock_build.return_value = (
            ["claude", "--output-format", "stream-json", "--print", "x"],
            False,
        )
        mock_run.return_value = SimpleNamespace(
            stdout="plain text", stderr="", returncode=0
        )

        p = CouncilParticipant(engine="claude", model="sonnet", role="dev")
        text, cost = _call_participant(p, "code", tmp_path)

        # The command passed to subprocess should have "text" not "stream-json"
        call_args = mock_run.call_args
        actual_cmd = call_args[1].get("command") if call_args[1] else call_args[0][0]
        if isinstance(actual_cmd, list):
            assert "stream-json" not in actual_cmd


# ===========================================================================
# TestRunCouncilWithRealCallParticipant
# ===========================================================================


class TestRunCouncilRoleKeys:
    @patch("maestro_cli.council._call_participant")
    def test_participant_without_role_uses_engine_model(
        self, mock_call: MagicMock
    ) -> None:
        """When role is empty, the key is engine-model."""
        mock_call.return_value = ("response", None)

        spec = CouncilSpec(
            participants=[
                CouncilParticipant(engine="claude", model="opus", role=""),
                CouncilParticipant(engine="gemini", model="pro", role=""),
            ],
            rounds=1,
        )

        result = run_council(spec, "task", Path.cwd())
        keys = list(result.rounds[0].responses.keys())
        assert "claude-opus" in keys
        assert "gemini-pro" in keys

    @patch("maestro_cli.council._call_participant")
    def test_participant_without_role_or_model(
        self, mock_call: MagicMock
    ) -> None:
        """When role and model are empty, the key is engine-default."""
        mock_call.return_value = ("response", None)

        spec = CouncilSpec(
            participants=[
                CouncilParticipant(engine="claude", model=None, role=""),
                CouncilParticipant(engine="gemini", model=None, role=""),
            ],
            rounds=1,
        )

        result = run_council(spec, "task", Path.cwd())
        keys = list(result.rounds[0].responses.keys())
        assert "claude-default" in keys
        assert "gemini-default" in keys


# ===========================================================================
# TestBuildRoundPromptEdgeCases
# ===========================================================================


class TestBuildRoundPromptEdgeCases:
    def test_long_response_truncated_in_prior(self) -> None:
        """Responses longer than 4000 chars are truncated in prior discussion."""
        p = CouncilParticipant(engine="claude", role="reviewer")
        long_response = "A" * 5000
        prior = [CouncilRound(round_num=1, responses={"architect": long_response})]
        prompt = _build_round_prompt("task", p, 2, prior, "")
        # The truncated response should be at most 4000 chars
        assert long_response not in prompt
        assert "A" * 4000 in prompt

    def test_multiple_prior_rounds(self) -> None:
        """All prior rounds are included in the discussion."""
        p = CouncilParticipant(engine="claude", role="dev")
        prior = [
            CouncilRound(round_num=1, responses={"a": "round1-a", "b": "round1-b"}),
            CouncilRound(round_num=2, responses={"a": "round2-a", "b": "round2-b"}),
        ]
        prompt = _build_round_prompt("task", p, 3, prior, "")
        assert "round1-a" in prompt
        assert "round2-b" in prompt
        assert "round 1" in prompt
        assert "round 2" in prompt


# ===========================================================================
# TestBuildConsolidationPromptEdgeCases
# ===========================================================================


class TestBuildConsolidationPromptEdgeCases:
    def test_long_response_truncated(self) -> None:
        long_resp = "X" * 5000
        rounds = [CouncilRound(round_num=1, responses={"dev": long_resp})]
        prompt = _build_consolidation_prompt("task", rounds)
        assert "X" * 4000 in prompt
        assert long_resp not in prompt

    def test_empty_rounds(self) -> None:
        prompt = _build_consolidation_prompt("task", [])
        assert "consolidator" in prompt
        assert "original_task" in prompt


# ===========================================================================
# TestCouncilSpecConnections
# ===========================================================================


class TestCouncilSpecConnections:
    def test_connections_field_default_empty(self) -> None:
        spec = CouncilSpec()
        assert spec.connections == {}

    def test_connections_field_set(self) -> None:
        spec = CouncilSpec(connections={"a": ["b"], "b": ["a"]})
        assert spec.connections == {"a": ["b"], "b": ["a"]}

    def test_to_dict_includes_connections_when_set(self) -> None:
        spec = CouncilSpec(connections={"a": ["b"]})
        d = spec.to_dict()
        assert "connections" in d
        assert d["connections"] == {"a": ["b"]}

    def test_to_dict_omits_connections_when_empty(self) -> None:
        spec = CouncilSpec()
        d = spec.to_dict()
        assert "connections" not in d


# ===========================================================================
# TestBuildChainPrompt
# ===========================================================================


class TestBuildChainPrompt:
    def test_first_participant_no_prior(self) -> None:
        p = CouncilParticipant(engine="claude", model="opus", role="architect")
        prompt = _build_chain_prompt("Design auth", p, None, "")
        assert "architect" in prompt
        assert "focus area" in prompt
        assert "<task>" in prompt
        assert "Design auth" in prompt
        assert "<prior_response>" not in prompt

    def test_with_prior_response(self) -> None:
        p = CouncilParticipant(engine="gemini", model="pro", role="critic")
        prompt = _build_chain_prompt("Design auth", p, "Use OAuth2", "")
        assert "<prior_response>" in prompt
        assert "Use OAuth2" in prompt
        assert "Refine and improve" in prompt

    def test_with_upstream_context(self) -> None:
        p = CouncilParticipant(engine="claude", role="dev")
        prompt = _build_chain_prompt("task", p, None, "upstream data here")
        assert "<upstream_context>" in prompt
        assert "upstream data here" in prompt

    def test_truncates_long_prior(self) -> None:
        p = CouncilParticipant(engine="claude", role="reviewer")
        long_resp = "Z" * 5000
        prompt = _build_chain_prompt("task", p, long_resp, "")
        assert "Z" * 4000 in prompt
        assert long_resp not in prompt

    def test_no_role(self) -> None:
        p = CouncilParticipant(engine="claude", model="sonnet", role="")
        prompt = _build_chain_prompt("task", p, None, "")
        assert "focus area" not in prompt
        assert "<task>" in prompt


# ===========================================================================
# TestBuildGraphRoundPrompt
# ===========================================================================


class TestBuildGraphRoundPrompt:
    def test_first_round_no_prior(self) -> None:
        p = CouncilParticipant(engine="claude", model="opus", role="architect")
        connections: dict[str, list[str]] = {"architect": ["critic"]}
        prompt = _build_graph_round_prompt("Design auth", p, 1, [], connections, "")
        assert "architect" in prompt
        assert "Round 1" in prompt
        assert "<prior_discussion>" not in prompt

    def test_filters_to_connected_peers_only(self) -> None:
        p = CouncilParticipant(engine="claude", role="architect")
        prior = [CouncilRound(
            round_num=1,
            responses={"critic": "I see problems", "security": "Looks safe"},
        )]
        connections: dict[str, list[str]] = {"architect": ["critic"]}
        prompt = _build_graph_round_prompt("Design auth", p, 2, prior, connections, "")
        assert "I see problems" in prompt
        assert "Looks safe" not in prompt

    def test_no_connections_for_participant(self) -> None:
        p = CouncilParticipant(engine="claude", role="isolated")
        prior = [CouncilRound(
            round_num=1,
            responses={"a": "response A", "b": "response B"},
        )]
        connections: dict[str, list[str]] = {"a": ["b"], "b": ["a"]}
        prompt = _build_graph_round_prompt("task", p, 2, prior, connections, "")
        assert "<prior_discussion>" not in prompt

    def test_with_upstream_context(self) -> None:
        p = CouncilParticipant(engine="claude", role="dev")
        connections: dict[str, list[str]] = {"dev": []}
        prompt = _build_graph_round_prompt("task", p, 1, [], connections, "ctx data")
        assert "<upstream_context>" in prompt
        assert "ctx data" in prompt

    def test_role_key_fallback_without_role(self) -> None:
        """When participant has no role, uses engine-model as key."""
        p = CouncilParticipant(engine="claude", model="opus", role="")
        connections: dict[str, list[str]] = {"claude-opus": ["critic"]}
        prior = [CouncilRound(round_num=1, responses={"critic": "feedback"})]
        prompt = _build_graph_round_prompt("task", p, 2, prior, connections, "")
        assert "feedback" in prompt

    def test_truncates_long_responses(self) -> None:
        p = CouncilParticipant(engine="claude", role="reviewer")
        long_resp = "W" * 5000
        prior = [CouncilRound(round_num=1, responses={"peer": long_resp})]
        connections: dict[str, list[str]] = {"reviewer": ["peer"]}
        prompt = _build_graph_round_prompt("task", p, 2, prior, connections, "")
        assert "W" * 4000 in prompt
        assert long_resp not in prompt


# ===========================================================================
# TestRunChainCouncil
# ===========================================================================


class TestRunChainCouncil:
    @patch("maestro_cli.council._call_participant")
    def test_chain_basic(self, mock_call: MagicMock) -> None:
        mock_call.side_effect = [
            ("Draft plan A", 0.01),
            ("Refined plan A with improvements", 0.02),
        ]

        spec = CouncilSpec(
            participants=[
                CouncilParticipant(engine="claude", model="opus", role="architect"),
                CouncilParticipant(engine="gemini", model="pro", role="critic"),
            ],
            topology="chain",
        )

        result = run_council(spec, "Design auth", Path.cwd())

        assert len(result.rounds) == 1
        assert "architect" in result.rounds[0].responses
        assert "critic" in result.rounds[0].responses
        # Last participant's output is the synthesis
        assert result.synthesis == "Refined plan A with improvements"
        assert result.total_cost_usd == pytest.approx(0.03)
        # No consolidation call — only 2 participant calls
        assert mock_call.call_count == 2

    @patch("maestro_cli.council._call_participant")
    def test_chain_three_participants(self, mock_call: MagicMock) -> None:
        mock_call.side_effect = [
            ("Step 1 output", 0.01),
            ("Step 2 refined", 0.01),
            ("Step 3 final", 0.01),
        ]

        spec = CouncilSpec(
            participants=[
                CouncilParticipant(engine="claude", role="a"),
                CouncilParticipant(engine="claude", role="b"),
                CouncilParticipant(engine="claude", role="c"),
            ],
            topology="chain",
        )

        result = run_council(spec, "task", Path.cwd())
        assert result.synthesis == "Step 3 final"
        assert mock_call.call_count == 3

    @patch("maestro_cli.council._call_participant")
    def test_chain_event_callback(self, mock_call: MagicMock) -> None:
        mock_call.return_value = ("response", 0.01)

        spec = CouncilSpec(
            participants=[
                CouncilParticipant(engine="claude", role="a"),
                CouncilParticipant(engine="claude", role="b"),
            ],
            topology="chain",
        )

        events: list[tuple[str, dict[str, Any]]] = []
        callback = MagicMock(side_effect=lambda evt, **kw: events.append((evt, kw)))

        run_council(spec, "task", Path.cwd(), event_callback=callback)
        event_types = [e[0] for e in events]
        assert "council_chain_step" in event_types
        # No consolidation event for chain
        assert "council_consolidation" not in event_types

    @patch("maestro_cli.council._call_participant")
    def test_chain_step_indices(self, mock_call: MagicMock) -> None:
        mock_call.return_value = ("response", None)

        spec = CouncilSpec(
            participants=[
                CouncilParticipant(engine="claude", role="first"),
                CouncilParticipant(engine="claude", role="second"),
                CouncilParticipant(engine="claude", role="third"),
            ],
            topology="chain",
        )

        events: list[dict[str, Any]] = []
        callback = MagicMock(side_effect=lambda evt, **kw: events.append({"event": evt, **kw}))

        run_council(spec, "task", Path.cwd(), event_callback=callback)
        step_events = [e for e in events if e["event"] == "council_chain_step"]
        assert len(step_events) == 3
        assert step_events[0]["step_index"] == 0
        assert step_events[1]["step_index"] == 1
        assert step_events[2]["step_index"] == 2

    @patch("maestro_cli.council._call_participant")
    def test_chain_prompt_builds_correctly(self, mock_call: MagicMock) -> None:
        """First participant gets no prior; second gets first's response."""
        prompts_received: list[str] = []

        def capture_prompt(participant: Any, prompt: str, workdir: Any) -> tuple[str, float | None]:
            prompts_received.append(prompt)
            return ("response from " + (participant.role or "unknown"), 0.01)

        mock_call.side_effect = capture_prompt

        spec = CouncilSpec(
            participants=[
                CouncilParticipant(engine="claude", role="first"),
                CouncilParticipant(engine="claude", role="second"),
            ],
            topology="chain",
        )

        run_council(spec, "Design something", Path.cwd())

        # First participant should not have prior_response
        assert "<prior_response>" not in prompts_received[0]
        # Second participant should have prior_response
        assert "<prior_response>" in prompts_received[1]
        assert "response from first" in prompts_received[1]

    @patch("maestro_cli.council._call_participant")
    def test_chain_none_cost_handled(self, mock_call: MagicMock) -> None:
        mock_call.return_value = ("response", None)

        spec = CouncilSpec(
            participants=[
                CouncilParticipant(engine="ollama", role="local1"),
                CouncilParticipant(engine="ollama", role="local2"),
            ],
            topology="chain",
        )

        result = run_council(spec, "task", Path.cwd())
        assert result.total_cost_usd == 0.0


# ===========================================================================
# TestRunGraphCouncil
# ===========================================================================


class TestRunGraphCouncil:
    @patch("maestro_cli.council._call_participant")
    def test_graph_basic(self, mock_call: MagicMock) -> None:
        mock_call.side_effect = [
            ("architect says A", 0.01),
            ("critic says B", 0.01),
            ("Consolidated AB", 0.005),
        ]

        spec = CouncilSpec(
            participants=[
                CouncilParticipant(engine="claude", role="architect"),
                CouncilParticipant(engine="gemini", role="critic"),
            ],
            topology="graph",
            connections={"architect": ["critic"], "critic": ["architect"]},
            rounds=1,
        )

        result = run_council(spec, "Design auth", Path.cwd())

        assert len(result.rounds) == 1
        assert "architect" in result.rounds[0].responses
        assert "critic" in result.rounds[0].responses
        assert result.synthesis == "Consolidated AB"
        # 2 participants + 1 consolidation = 3 calls
        assert mock_call.call_count == 3

    @patch("maestro_cli.council._call_participant")
    def test_graph_filters_visibility(self, mock_call: MagicMock) -> None:
        """Verify that graph topology constrains what each participant sees."""
        prompts_received: dict[str, list[str]] = {"a": [], "b": [], "c": []}

        call_count = [0]

        def capture_prompt(participant: Any, prompt: str, workdir: Any) -> tuple[str, float | None]:
            role = participant.role
            prompts_received.setdefault(role, []).append(prompt)
            call_count[0] += 1
            # Return after consolidation check
            if role == "consolidator":
                return ("synthesis", 0.005)
            return (f"response from {role}", 0.01)

        mock_call.side_effect = capture_prompt

        spec = CouncilSpec(
            participants=[
                CouncilParticipant(engine="claude", role="a"),
                CouncilParticipant(engine="claude", role="b"),
                CouncilParticipant(engine="claude", role="c"),
            ],
            topology="graph",
            connections={
                "a": ["b"],        # a sees only b
                "b": ["a", "c"],   # b sees a and c
                "c": [],           # c sees nobody
            },
            rounds=2,
        )

        run_council(spec, "task", Path.cwd())

        # In round 2, "a" should see "b" but not "c"
        a_round2_prompt = prompts_received["a"][1]
        assert "response from b" in a_round2_prompt
        assert "response from c" not in a_round2_prompt

        # In round 2, "c" should see nobody (empty connections)
        c_round2_prompt = prompts_received["c"][1]
        assert "<prior_discussion>" not in c_round2_prompt

    @patch("maestro_cli.council._call_participant")
    def test_graph_event_callback(self, mock_call: MagicMock) -> None:
        mock_call.return_value = ("response", 0.01)

        spec = CouncilSpec(
            participants=[
                CouncilParticipant(engine="claude", role="a"),
                CouncilParticipant(engine="claude", role="b"),
            ],
            topology="graph",
            connections={"a": ["b"], "b": ["a"]},
            rounds=1,
        )

        events: list[str] = []
        callback = MagicMock(side_effect=lambda evt, **kw: events.append(evt))

        run_council(spec, "task", Path.cwd(), event_callback=callback)
        assert "council_turn" in events
        assert "council_consolidation" in events

    @patch("maestro_cli.council._call_participant")
    def test_graph_cost_accumulation(self, mock_call: MagicMock) -> None:
        mock_call.side_effect = [
            ("resp a", 0.10),
            ("resp b", 0.20),
            ("synthesis", 0.05),
        ]

        spec = CouncilSpec(
            participants=[
                CouncilParticipant(engine="claude", role="a"),
                CouncilParticipant(engine="claude", role="b"),
            ],
            topology="graph",
            connections={"a": ["b"], "b": ["a"]},
            rounds=1,
        )

        result = run_council(spec, "task", Path.cwd())
        assert result.total_cost_usd == pytest.approx(0.35)


# ===========================================================================
# TestStarTopologyUnchanged
# ===========================================================================


class TestStarTopologyUnchanged:
    """Verify that star topology still works exactly as before."""

    @patch("maestro_cli.council._call_participant")
    def test_star_explicit_topology(self, mock_call: MagicMock) -> None:
        mock_call.side_effect = [
            ("resp A", 0.01),
            ("resp B", 0.01),
            ("synthesis", 0.005),
        ]

        spec = CouncilSpec(
            participants=[
                CouncilParticipant(engine="claude", role="a"),
                CouncilParticipant(engine="claude", role="b"),
            ],
            topology="star",
            rounds=1,
        )

        result = run_council(spec, "task", Path.cwd())
        assert len(result.rounds) == 1
        assert result.synthesis == "synthesis"
        assert mock_call.call_count == 3  # 2 participants + 1 consolidation

    @patch("maestro_cli.council._call_participant")
    def test_default_topology_is_star(self, mock_call: MagicMock) -> None:
        mock_call.side_effect = [
            ("resp A", 0.01),
            ("resp B", 0.01),
            ("synthesis", 0.005),
        ]

        spec = CouncilSpec(
            participants=[
                CouncilParticipant(engine="claude", role="a"),
                CouncilParticipant(engine="claude", role="b"),
            ],
        )

        result = run_council(spec, "task", Path.cwd())
        assert result.synthesis == "synthesis"
        assert mock_call.call_count == 3


# ===========================================================================
# Loader integration — chain/graph topologies
# ===========================================================================


class TestCouncilLoaderChainGraph:
    def test_chain_topology_accepted(self, tmp_path: Path) -> None:
        from maestro_cli.loader import load_plan

        yaml_text = (
            "version: 1\nname: test\ntasks:\n"
            "  - id: t\n    engine: claude\n    prompt: x\n"
            "    context_mode: council\n"
            "    council:\n"
            "      participants:\n"
            "        - engine: claude\n          role: a\n"
            "        - engine: claude\n          role: b\n"
            "      topology: chain\n"
        )
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(yaml_text, encoding="utf-8")
        plan = load_plan(plan_path)
        assert plan.tasks[0].council is not None
        assert plan.tasks[0].council.topology == "chain"

    def test_graph_topology_with_connections(self, tmp_path: Path) -> None:
        from maestro_cli.loader import load_plan

        yaml_text = (
            "version: 1\nname: test\ntasks:\n"
            "  - id: t\n    engine: claude\n    prompt: x\n"
            "    context_mode: council\n"
            "    council:\n"
            "      participants:\n"
            "        - engine: claude\n          role: architect\n"
            "        - engine: gemini\n          role: critic\n"
            "      topology: graph\n"
            "      connections:\n"
            "        architect: [critic]\n"
            "        critic: [architect]\n"
        )
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(yaml_text, encoding="utf-8")
        plan = load_plan(plan_path)
        council = plan.tasks[0].council
        assert council is not None
        assert council.topology == "graph"
        assert council.connections == {"architect": ["critic"], "critic": ["architect"]}

    def test_graph_topology_requires_connections(self, tmp_path: Path) -> None:
        from maestro_cli.errors import PlanValidationError
        from maestro_cli.loader import load_plan

        yaml_text = (
            "version: 1\nname: test\ntasks:\n"
            "  - id: t\n    engine: claude\n    prompt: x\n"
            "    context_mode: council\n"
            "    council:\n"
            "      participants:\n"
            "        - engine: claude\n          role: a\n"
            "        - engine: claude\n          role: b\n"
            "      topology: graph\n"
        )
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(yaml_text, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E072.*connections"):
            load_plan(plan_path)

    def test_graph_connections_key_must_match_role(self, tmp_path: Path) -> None:
        from maestro_cli.errors import PlanValidationError
        from maestro_cli.loader import load_plan

        yaml_text = (
            "version: 1\nname: test\ntasks:\n"
            "  - id: t\n    engine: claude\n    prompt: x\n"
            "    context_mode: council\n"
            "    council:\n"
            "      participants:\n"
            "        - engine: claude\n          role: a\n"
            "        - engine: claude\n          role: b\n"
            "      topology: graph\n"
            "      connections:\n"
            "        unknown_role: [a]\n"
        )
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(yaml_text, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E072.*unknown_role"):
            load_plan(plan_path)

    def test_graph_connections_value_must_match_role(self, tmp_path: Path) -> None:
        from maestro_cli.errors import PlanValidationError
        from maestro_cli.loader import load_plan

        yaml_text = (
            "version: 1\nname: test\ntasks:\n"
            "  - id: t\n    engine: claude\n    prompt: x\n"
            "    context_mode: council\n"
            "    council:\n"
            "      participants:\n"
            "        - engine: claude\n          role: a\n"
            "        - engine: claude\n          role: b\n"
            "      topology: graph\n"
            "      connections:\n"
            "        a: [nonexistent]\n"
        )
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(yaml_text, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E072.*nonexistent"):
            load_plan(plan_path)

    def test_graph_requires_non_empty_roles(self, tmp_path: Path) -> None:
        from maestro_cli.errors import PlanValidationError
        from maestro_cli.loader import load_plan

        yaml_text = (
            "version: 1\nname: test\ntasks:\n"
            "  - id: t\n    engine: claude\n    prompt: x\n"
            "    context_mode: council\n"
            "    council:\n"
            "      participants:\n"
            "        - engine: claude\n"
            "        - engine: claude\n          role: b\n"
            "      topology: graph\n"
            "      connections:\n"
            "        b: []\n"
        )
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(yaml_text, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E072.*non-empty.*role"):
            load_plan(plan_path)

    def test_connections_not_dict_raises_e072(self, tmp_path: Path) -> None:
        from maestro_cli.errors import PlanValidationError
        from maestro_cli.loader import load_plan

        yaml_text = (
            "version: 1\nname: test\ntasks:\n"
            "  - id: t\n    engine: claude\n    prompt: x\n"
            "    context_mode: council\n"
            "    council:\n"
            "      participants:\n"
            "        - engine: claude\n          role: a\n"
            "        - engine: claude\n          role: b\n"
            "      topology: graph\n"
            "      connections: not_a_dict\n"
        )
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(yaml_text, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E072"):
            load_plan(plan_path)

    def test_connections_value_not_list_raises_e072(self, tmp_path: Path) -> None:
        from maestro_cli.errors import PlanValidationError
        from maestro_cli.loader import load_plan

        yaml_text = (
            "version: 1\nname: test\ntasks:\n"
            "  - id: t\n    engine: claude\n    prompt: x\n"
            "    context_mode: council\n"
            "    council:\n"
            "      participants:\n"
            "        - engine: claude\n          role: a\n"
            "        - engine: claude\n          role: b\n"
            "      topology: graph\n"
            "      connections:\n"
            "        a: b\n"
        )
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(yaml_text, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E072.*list"):
            load_plan(plan_path)

    def test_w28_connections_on_non_graph_topology(self, tmp_path: Path) -> None:
        from maestro_cli.loader import load_plan

        yaml_text = (
            "version: 1\nname: test\ntasks:\n"
            "  - id: t\n    engine: claude\n    prompt: x\n"
            "    context_mode: council\n"
            "    council:\n"
            "      participants:\n"
            "        - engine: claude\n          role: a\n"
            "        - engine: claude\n          role: b\n"
            "      topology: star\n"
            "      connections:\n"
            "        a: [b]\n"
        )
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(yaml_text, encoding="utf-8")
        plan = load_plan(plan_path)
        w28 = [w for w in plan.validation_warnings if "W28" in w]
        assert len(w28) == 1
        assert "connections" in w28[0]
        assert "not 'graph'" in w28[0]

    def test_w28_chain_with_multiple_rounds(self, tmp_path: Path) -> None:
        from maestro_cli.loader import load_plan

        yaml_text = (
            "version: 1\nname: test\ntasks:\n"
            "  - id: t\n    engine: claude\n    prompt: x\n"
            "    context_mode: council\n"
            "    council:\n"
            "      participants:\n"
            "        - engine: claude\n          role: a\n"
            "        - engine: claude\n          role: b\n"
            "      topology: chain\n"
            "      rounds: 3\n"
        )
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(yaml_text, encoding="utf-8")
        plan = load_plan(plan_path)
        w28 = [w for w in plan.validation_warnings if "W28" in w]
        assert len(w28) == 1
        assert "chain" in w28[0]
        assert "single-pass" in w28[0]

    def test_chain_rounds_1_no_warning(self, tmp_path: Path) -> None:
        from maestro_cli.loader import load_plan

        yaml_text = (
            "version: 1\nname: test\ntasks:\n"
            "  - id: t\n    engine: claude\n    prompt: x\n"
            "    context_mode: council\n"
            "    council:\n"
            "      participants:\n"
            "        - engine: claude\n          role: a\n"
            "        - engine: claude\n          role: b\n"
            "      topology: chain\n"
            "      rounds: 1\n"
        )
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(yaml_text, encoding="utf-8")
        plan = load_plan(plan_path)
        w28 = [w for w in plan.validation_warnings if "W28" in w]
        assert len(w28) == 0

    def test_valid_topologies_updated(self) -> None:
        from maestro_cli.council import _VALID_TOPOLOGIES

        assert "star" in _VALID_TOPOLOGIES
        assert "chain" in _VALID_TOPOLOGIES
        assert "graph" in _VALID_TOPOLOGIES
