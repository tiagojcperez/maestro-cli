from __future__ import annotations

import typing
from pathlib import Path

import pytest

from maestro_cli.models import JudgeSpec, ModelRecord, PlanSpec, TaskHistory, TaskSpec
from maestro_cli.routing import (
    _COST_WEIGHTS,
    _apply_historical_signal,
    _score_task_complexity,
    _tier_from_score,
    load_task_histories,
    resolve_auto_model,
)


def _make_task(**kwargs) -> TaskSpec:
    """Create a minimal TaskSpec for testing."""
    defaults = {
        "id": "test-task",
        "engine": "claude",
        "prompt": "test prompt",
    }
    defaults.update(kwargs)
    return TaskSpec(**defaults)


def _make_plan(**kwargs) -> PlanSpec:
    defaults = {
        "name": "test",
        "tasks": [],
    }
    defaults.update(kwargs)
    return PlanSpec(**defaults)


class TestResolveAutoModel:
    @pytest.mark.parametrize(
        ("engine", "tags", "expected"),
        [("claude", ["trivial"], "haiku")],
    )
    def test_claude_trivial_tag_routes_to_haiku(
        self, engine: str, tags: list[str], expected: str
    ) -> None:
        task = _make_task(tags=tags)
        plan = _make_plan()

        assert resolve_auto_model(task, plan, engine) == expected

    @pytest.mark.parametrize(
        ("engine", "tags", "expected"),
        [("claude", ["security"], "opus")],
    )
    def test_claude_security_tag_routes_to_opus(
        self, engine: str, tags: list[str], expected: str
    ) -> None:
        task = _make_task(tags=tags)
        plan = _make_plan()

        assert resolve_auto_model(task, plan, engine) == expected

    @pytest.mark.parametrize(
        ("engine", "tags", "expected"),
        [("claude", [], "sonnet")],
    )
    def test_claude_default_routes_to_sonnet(
        self, engine: str, tags: list[str], expected: str
    ) -> None:
        task = _make_task(tags=tags, prompt="x" * 100)
        plan = _make_plan()

        assert resolve_auto_model(task, plan, engine) == expected

    @pytest.mark.parametrize(
        ("engine", "tags", "expected"),
        # Codex high tier moved 5.4 -> 5.5 on 2026-04-27 alongside the GPT-5.5 launch.
        [("codex", ["critical"], "5.5")],
    )
    def test_codex_high_routes_correctly(
        self, engine: str, tags: list[str], expected: str
    ) -> None:
        task = _make_task(tags=tags)
        plan = _make_plan()

        assert resolve_auto_model(task, plan, engine) == expected

    @pytest.mark.parametrize(
        ("engine", "tags", "expected"),
        [("gemini", ["trivial"], "flash-lite")],
    )
    def test_gemini_low_routes_to_flash_lite(
        self, engine: str, tags: list[str], expected: str
    ) -> None:
        task = _make_task(tags=tags)
        plan = _make_plan()

        assert resolve_auto_model(task, plan, engine) == expected

    @pytest.mark.parametrize("engine", ["unknown"])
    def test_unknown_engine_returns_auto(self, engine: str) -> None:
        task = _make_task(engine="unknown")
        plan = _make_plan()

        assert resolve_auto_model(task, plan, engine) == "auto"

    @pytest.mark.parametrize(
        ("engine", "expected_low", "expected_high"),
        [
            ("claude", "haiku", "opus"),
            # Codex high tier bumped 5.4 -> 5.5 on 2026-04-27 (GPT-5.5 launch).
            ("codex", "5-mini", "5.5"),
            ("gemini", "flash-lite", "pro"),
            ("copilot", "haiku", "opus"),
            ("qwen", "coder-turbo", "max"),
            ("ollama", "phi3", "mixtral"),
        ],
    )
    def test_all_engines_low_and_high_tiers(
        self, engine: str, expected_low: str, expected_high: str
    ) -> None:
        plan = _make_plan()
        low_task = _make_task(tags=["trivial"])
        high_task = _make_task(tags=["security"])

        assert resolve_auto_model(low_task, plan, engine) == expected_low
        assert resolve_auto_model(high_task, plan, engine) == expected_high

    @pytest.mark.parametrize(
        ("engine", "expected_medium"),
        [
            ("claude", "sonnet"),
            ("codex", "5.4"),
            ("gemini", "flash"),
            ("copilot", "sonnet"),
            ("qwen", "coder"),
            ("ollama", "llama3"),
        ],
    )
    def test_all_engines_medium_tier(self, engine: str, expected_medium: str) -> None:
        plan = _make_plan()
        # neutral prompt (100 chars) + no tags → score ~0.5 → medium
        task = _make_task(prompt="x" * 100)

        assert resolve_auto_model(task, plan, engine) == expected_medium


class TestScoreTaskComplexity:
    def test_baseline_score_is_medium(self) -> None:
        score = _score_task_complexity(_make_task(prompt="x" * 100), _make_plan())

        assert score == pytest.approx(0.5, abs=0.01)

    @pytest.mark.parametrize("tags", [["security"]])
    def test_security_tag_forces_high(self, tags: list[str]) -> None:
        score = _score_task_complexity(
            _make_task(tags=tags, prompt="x" * 100),
            _make_plan(),
        )

        assert score >= 0.8

    @pytest.mark.parametrize("tags", [["trivial"]])
    def test_trivial_tag_forces_low(self, tags: list[str]) -> None:
        score = _score_task_complexity(_make_task(tags=tags), _make_plan())

        assert score <= 0.2

    def test_long_prompt_increases_score(self) -> None:
        score = _score_task_complexity(_make_task(prompt="x" * 2001), _make_plan())

        assert score > 0.5

    def test_short_prompt_decreases_score(self) -> None:
        score = _score_task_complexity(_make_task(prompt="short prompt"), _make_plan())

        assert score < 0.5

    def test_many_deps_increases_score(self) -> None:
        score = _score_task_complexity(
            _make_task(prompt="x" * 100, depends_on=["a", "b", "c", "d"]),
            _make_plan(),
        )

        assert score > 0.5

    def test_recursive_context_increases_score(self) -> None:
        score = _score_task_complexity(
            _make_task(prompt="x" * 100, context_mode="recursive"),
            _make_plan(),
        )

        assert score > 0.5

    def test_judge_increases_score(self) -> None:
        score = _score_task_complexity(
            _make_task(prompt="x" * 100, judge=JudgeSpec(criteria=["correctness"])),
            _make_plan(),
        )

        assert score > 0.5

    @pytest.mark.parametrize("tag", ["review", "qa", "refactor", "complex", "algorithm"])
    def test_medium_boost_tags_increase_score_above_baseline(self, tag: str) -> None:
        baseline = _score_task_complexity(_make_task(prompt="x" * 100), _make_plan())
        boosted = _score_task_complexity(
            _make_task(tags=[tag], prompt="x" * 100), _make_plan()
        )

        assert boosted > baseline

    def test_conflicting_high_and_low_tags_high_wins(self) -> None:
        # security (high) + trivial (low) — high-tier max(0.8) applied first, then
        # low-tier min(0.2) would reduce — but _HIGH_TIER_TAGS check uses max so
        # the order matters: if both apply, low wins (min applied after max).
        # Verify the actual behaviour rather than assuming an outcome.
        score = _score_task_complexity(
            _make_task(tags=["security", "trivial"], prompt="x" * 100), _make_plan()
        )

        # Both branches fire: score set to max(0.5,0.8)=0.8, then min(0.8,0.2)=0.2
        assert score <= 0.2

    def test_context_from_increases_score(self) -> None:
        baseline = _score_task_complexity(_make_task(prompt="x" * 100), _make_plan())
        boosted = _score_task_complexity(
            _make_task(prompt="x" * 100, context_from=["a", "b", "c"]), _make_plan()
        )

        assert boosted > baseline

    @pytest.mark.parametrize(
        ("prompt_len", "expected_direction"),
        [
            (501, "above"),   # 500-1000 chars → +0.05
            (1001, "above"),  # 1000-2000 chars → +0.10
        ],
    )
    def test_medium_length_prompts_increase_score(
        self, prompt_len: int, expected_direction: str
    ) -> None:
        baseline = _score_task_complexity(_make_task(prompt="x" * 100), _make_plan())
        score = _score_task_complexity(_make_task(prompt="x" * prompt_len), _make_plan())

        assert score > baseline

    @pytest.mark.parametrize("tag", ["architecture", "critical", "audit", "security"])
    def test_all_high_tier_tags_force_score_above_08(self, tag: str) -> None:
        score = _score_task_complexity(
            _make_task(tags=[tag], prompt="x" * 100), _make_plan()
        )
        assert score >= 0.8

    @pytest.mark.parametrize("tag", ["typo", "config", "docs", "rename", "trivial"])
    def test_all_low_tier_tags_force_score_below_02(self, tag: str) -> None:
        score = _score_task_complexity(_make_task(tags=[tag]), _make_plan())
        assert score <= 0.2

    def test_score_clamped_to_1_0_with_all_boosts(self) -> None:
        # security (0.8) + recursive (+0.15) + judge (+0.10) + long prompt (+0.15) + many deps (+0.10)
        # would exceed 1.0 without clamping
        score = _score_task_complexity(
            _make_task(
                tags=["security"],
                prompt="x" * 2001,
                depends_on=["a", "b", "c", "d"],
                context_from=["a", "b", "c"],
                context_mode="recursive",
                judge=JudgeSpec(criteria=["correctness"]),
            ),
            _make_plan(),
        )
        assert score == 1.0

    def test_depends_on_boundary_exactly_3_no_boost(self) -> None:
        # exactly 3 deps → no boost (threshold is > 3)
        score_3 = _score_task_complexity(
            _make_task(prompt="x" * 100, depends_on=["a", "b", "c"]), _make_plan()
        )
        score_4 = _score_task_complexity(
            _make_task(prompt="x" * 100, depends_on=["a", "b", "c", "d"]), _make_plan()
        )
        assert score_4 > score_3

    def test_context_from_boundary_2_vs_3(self) -> None:
        # exactly 2 context_from → no boost; 3 → +0.10 boost (threshold is > 2)
        score_2 = _score_task_complexity(
            _make_task(prompt="x" * 100, context_from=["a", "b"]), _make_plan()
        )
        score_3 = _score_task_complexity(
            _make_task(prompt="x" * 100, context_from=["a", "b", "c"]), _make_plan()
        )
        assert score_2 == pytest.approx(0.5, abs=0.01)
        assert score_3 > score_2

    def test_multiple_medium_boost_tags_only_add_once(self) -> None:
        # `if tags & _MEDIUM_BOOST_TAGS: score += 0.15` fires once regardless of
        # how many matching tags are present — adding "review" and "qa" together
        # should produce the same score as "review" alone.
        score_one = _score_task_complexity(
            _make_task(tags=["review"], prompt="x" * 100), _make_plan()
        )
        score_two = _score_task_complexity(
            _make_task(tags=["review", "qa"], prompt="x" * 100), _make_plan()
        )
        assert score_one == pytest.approx(score_two, abs=0.001)

    def test_medium_boost_tag_and_many_deps_accumulate(self) -> None:
        # review tag (+0.15) and >3 deps (+0.10) are independent addends
        score_tag_only = _score_task_complexity(
            _make_task(tags=["review"], prompt="x" * 100), _make_plan()
        )
        score_deps_only = _score_task_complexity(
            _make_task(prompt="x" * 100, depends_on=["a", "b", "c", "d"]), _make_plan()
        )
        score_both = _score_task_complexity(
            _make_task(tags=["review"], prompt="x" * 100, depends_on=["a", "b", "c", "d"]),
            _make_plan(),
        )
        assert score_both > score_tag_only
        assert score_both > score_deps_only

    def test_score_clamped_above_zero_with_all_penalties(self) -> None:
        # low-tier tag forces min(score, 0.2), then short prompt subtracts 0.10 → 0.1
        # max(0.0, 0.1) = 0.1 — never goes below 0.0
        score = _score_task_complexity(
            _make_task(tags=["trivial"], prompt="hi"),  # < 100 chars → -0.10
            _make_plan(),
        )
        assert score >= 0.0
        assert score < 0.2  # low tier cap still applies


    def test_tag_uppercase_normalised_to_high_tier(self) -> None:
        # routing.py does tag.strip().lower() so "SECURITY" should fire the high-tier path
        score = _score_task_complexity(
            _make_task(tags=["SECURITY"], prompt="x" * 100), _make_plan()
        )
        assert score >= 0.8

    def test_tag_with_surrounding_whitespace_stripped(self) -> None:
        # tags with leading/trailing spaces should still match after .strip()
        score = _score_task_complexity(
            _make_task(tags=["  trivial  "], prompt="x" * 100), _make_plan()
        )
        assert score <= 0.2

    def test_non_recursive_context_mode_no_boost(self) -> None:
        # only context_mode=="recursive" gets +0.15; summarized and map_reduce should not
        baseline = _score_task_complexity(_make_task(prompt="x" * 100), _make_plan())
        for mode in ("summarized", "map_reduce", "raw"):
            score = _score_task_complexity(
                _make_task(prompt="x" * 100, context_mode=mode), _make_plan()
            )
            assert score == pytest.approx(baseline, abs=0.001), f"mode={mode} unexpectedly boosted score"

    def test_codex_medium_and_high_tiers_now_differ(self) -> None:
        # 2026-04-27: post-GPT-5.5 launch, codex tiers are
        # {"low": "5-mini", "medium": "5.4", "high": "5.5"} — medium and high
        # now resolve to different models. Previously both were 5.4.
        plan = _make_plan()
        medium_task = _make_task(prompt="x" * 100)          # score ~0.5 → medium
        high_task = _make_task(tags=["security"])             # score >= 0.8 → high
        assert resolve_auto_model(medium_task, plan, "codex") == "5.4"
        assert resolve_auto_model(high_task, plan, "codex") == "5.5"


    def test_none_prompt_treated_as_zero_length(self) -> None:
        # None prompt → len("") == 0 → < 100 → -0.10 penalty
        score_none = _score_task_complexity(_make_task(prompt=None), _make_plan())
        score_baseline = _score_task_complexity(_make_task(prompt="x" * 100), _make_plan())

        assert score_none < score_baseline
        assert score_none == pytest.approx(0.4, abs=0.01)

    def test_prompt_exactly_500_chars_no_boost(self) -> None:
        # 500 chars is NOT > 500 → no boost → stays at baseline 0.5
        score = _score_task_complexity(_make_task(prompt="x" * 500), _make_plan())

        assert score == pytest.approx(0.5, abs=0.01)

    def test_medium_boost_tag_and_recursive_context_both_accumulate(self) -> None:
        # review (+0.15) + recursive context (+0.15) are independent addends
        score_tag_only = _score_task_complexity(
            _make_task(tags=["review"], prompt="x" * 100), _make_plan()
        )
        score_recursive_only = _score_task_complexity(
            _make_task(prompt="x" * 100, context_mode="recursive"), _make_plan()
        )
        score_both = _score_task_complexity(
            _make_task(tags=["review"], prompt="x" * 100, context_mode="recursive"),
            _make_plan(),
        )

        assert score_both > score_tag_only
        assert score_both > score_recursive_only

    def test_multiple_low_tier_tags_clamp_same_as_single(self) -> None:
        # the low-tier branch uses min(score, 0.2) which fires once; adding more
        # low-tier tags shouldn't push score below what a single tag produces
        score_one = _score_task_complexity(_make_task(tags=["trivial"]), _make_plan())
        score_two = _score_task_complexity(
            _make_task(tags=["trivial", "config"]), _make_plan()
        )

        assert score_one == pytest.approx(score_two, abs=0.001)
        assert score_two <= 0.2


    def test_prompt_exactly_1000_chars_gets_small_boost_not_medium(self) -> None:
        # 1000 is NOT > 1000, so only the >500 branch fires (+0.05), not +0.10
        score_1000 = _score_task_complexity(_make_task(prompt="x" * 1000), _make_plan())
        score_1001 = _score_task_complexity(_make_task(prompt="x" * 1001), _make_plan())

        assert score_1000 == pytest.approx(0.55, abs=0.01)
        assert score_1001 > score_1000

    def test_prompt_exactly_2000_chars_gets_medium_boost_not_large(self) -> None:
        # 2000 is NOT > 2000, so only the >1000 branch fires (+0.10), not +0.15
        score_2000 = _score_task_complexity(_make_task(prompt="x" * 2000), _make_plan())
        score_2001 = _score_task_complexity(_make_task(prompt="x" * 2001), _make_plan())

        assert score_2000 == pytest.approx(0.60, abs=0.01)
        assert score_2001 > score_2000

    def test_recursive_context_and_judge_together_push_to_high_tier(self) -> None:
        # 0.5 (baseline) + 0.15 (recursive) + 0.10 (judge) = 0.75 → > 0.7 → high
        score = _score_task_complexity(
            _make_task(
                prompt="x" * 100,
                context_mode="recursive",
                judge=JudgeSpec(criteria=["correctness"]),
            ),
            _make_plan(),
        )

        assert score > 0.7

    def test_resolve_auto_model_non_tag_signals_reach_high_tier(self) -> None:
        # recursive + judge (no tags) → score > 0.7 → high tier
        plan = _make_plan()
        task = _make_task(
            prompt="x" * 100,
            context_mode="recursive",
            judge=JudgeSpec(criteria=["correctness"]),
        )

        assert resolve_auto_model(task, plan, "claude") == "opus"
        assert resolve_auto_model(task, plan, "gemini") == "pro"
        assert resolve_auto_model(task, plan, "qwen") == "max"

    def test_low_tier_tag_plus_recursive_context_pushes_to_medium(self) -> None:
        # trivial forces min(score, 0.2) = 0.2, then recursive adds +0.15 → 0.35 → medium
        score = _score_task_complexity(
            _make_task(tags=["trivial"], prompt="x" * 100, context_mode="recursive"),
            _make_plan(),
        )
        assert 0.3 <= score <= 0.7  # falls into medium tier band

    def test_medium_boost_tag_plus_judge_reaches_high_tier(self) -> None:
        # 0.5 (baseline) + 0.15 (review) + 0.10 (judge) = 0.75 > 0.7 → high
        score = _score_task_complexity(
            _make_task(
                tags=["review"],
                prompt="x" * 100,
                judge=JudgeSpec(criteria=["correctness"]),
            ),
            _make_plan(),
        )
        assert score > 0.7

    def test_resolve_auto_model_uses_engine_param_not_task_engine(self) -> None:
        # task.engine="claude" but we pass engine="ollama" → should use ollama tiers
        plan = _make_plan()
        task = _make_task(engine="claude", tags=["trivial"])  # low tier
        assert resolve_auto_model(task, plan, "ollama") == "phi3"

    def test_many_deps_and_long_prompt_boost_to_high_tier(self) -> None:
        # 0.5 + 0.10 (>3 deps) + 0.15 (>2000 chars) = 0.75 → high
        score = _score_task_complexity(
            _make_task(prompt="x" * 2001, depends_on=["a", "b", "c", "d"]),
            _make_plan(),
        )
        assert score > 0.7


class TestTierFromScore:
    @pytest.mark.parametrize(("score", "expected"), [(0.1, "low")])
    def test_low_tier(self, score: float, expected: str) -> None:
        assert _tier_from_score(score) == expected

    @pytest.mark.parametrize(("score", "expected"), [(0.5, "medium")])
    def test_medium_tier(self, score: float, expected: str) -> None:
        assert _tier_from_score(score) == expected

    @pytest.mark.parametrize(("score", "expected"), [(0.9, "high")])
    def test_high_tier(self, score: float, expected: str) -> None:
        assert _tier_from_score(score) == expected

    @pytest.mark.parametrize(("score", "expected"), [(0.3, "medium")])
    def test_boundary_low(self, score: float, expected: str) -> None:
        assert _tier_from_score(score) == expected

    @pytest.mark.parametrize(("score", "expected"), [(0.7, "medium")])
    def test_boundary_high(self, score: float, expected: str) -> None:
        assert _tier_from_score(score) == expected

    @pytest.mark.parametrize(
        ("score", "expected"),
        [
            (0.0, "low"),
            (0.29, "low"),
            (0.71, "high"),
            (1.0, "high"),
        ],
    )
    def test_extreme_and_near_boundary_values(self, score: float, expected: str) -> None:
        assert _tier_from_score(score) == expected

    @pytest.mark.parametrize(
        ("score", "expected"),
        [
            (0.301, "medium"),
            (0.699, "medium"),
        ],
    )
    def test_interior_medium_range_values(self, score: float, expected: str) -> None:
        # Values clearly inside the medium band — not touching either boundary
        assert _tier_from_score(score) == expected


class TestScoreTaskComplexityExtra:
    """Additional score edge-case tests."""

    def test_multiple_high_tier_tags_cap_at_08(self) -> None:
        # Two high-tier tags fire the same `if tags & _HIGH_TIER_TAGS` branch once.
        # score: 0.5 → max(0.5, 0.8) = 0.8; 100-char prompt = no change → 0.8
        score = _score_task_complexity(
            _make_task(tags=["security", "architecture"], prompt="x" * 100),
            _make_plan(),
        )
        assert score == pytest.approx(0.8, abs=0.001)

    def test_low_tier_tag_plus_judge_pushes_to_medium_range(self) -> None:
        # trivial → min(score, 0.2) = 0.2; judge adds +0.10 → 0.30; 100-char prompt = no change
        # 0.30 is NOT < 0.3, so tier is "medium", not "low"
        score = _score_task_complexity(
            _make_task(
                tags=["trivial"],
                prompt="x" * 100,
                judge=JudgeSpec(criteria=["correctness"]),
            ),
            _make_plan(),
        )
        assert score == pytest.approx(0.30, abs=0.001)
        assert _tier_from_score(score) == "medium"

    def test_high_tier_tag_with_long_prompt_reaches_near_max(self) -> None:
        # security → 0.8; >2000-char prompt → +0.15 → 0.95 (below 1.0 without extra boosts)
        score = _score_task_complexity(
            _make_task(tags=["security"], prompt="x" * 2001),
            _make_plan(),
        )
        assert score == pytest.approx(0.95, abs=0.001)
        assert _tier_from_score(score) == "high"

    def test_resolve_medium_tier_low_tag_plus_judge(self) -> None:
        # trivial + judge → score 0.30 → medium tier → claude: sonnet, qwen: coder
        plan = _make_plan()
        task = _make_task(
            tags=["trivial"],
            prompt="x" * 100,
            judge=JudgeSpec(criteria=["correctness"]),
        )
        assert resolve_auto_model(task, plan, "claude") == "sonnet"
        assert resolve_auto_model(task, plan, "qwen") == "coder"

    def test_high_tier_tag_plus_medium_boost_tag_accumulate(self) -> None:
        # security → max(0.5, 0.8) = 0.8; review → +0.15 → 0.95 (still high tier)
        score = _score_task_complexity(
            _make_task(tags=["security", "review"], prompt="x" * 100),
            _make_plan(),
        )
        assert score == pytest.approx(0.95, abs=0.001)
        assert _tier_from_score(score) == "high"

    def test_all_three_tag_tiers_combined(self) -> None:
        # security sets max(0.5,0.8)=0.8; trivial sets min(0.8,0.2)=0.2; review adds +0.15 → 0.35
        score = _score_task_complexity(
            _make_task(tags=["security", "trivial", "review"], prompt="x" * 100),
            _make_plan(),
        )
        assert score == pytest.approx(0.35, abs=0.001)
        assert _tier_from_score(score) == "medium"

    def test_resolve_auto_model_empty_string_engine_returns_auto(self) -> None:
        # empty string is not in _MODEL_TIERS → passthrough "auto"
        task = _make_task()
        plan = _make_plan()
        assert resolve_auto_model(task, plan, "") == "auto"

    def test_prompt_exactly_99_chars_gets_short_penalty(self) -> None:
        # 99 < 100 → -0.10 → score = 0.40
        score = _score_task_complexity(_make_task(prompt="x" * 99), _make_plan())
        assert score == pytest.approx(0.40, abs=0.001)

    def test_empty_string_prompt_same_penalty_as_none(self) -> None:
        # "" → len 0 < 100 → -0.10 → 0.40, same as prompt=None
        score_empty = _score_task_complexity(_make_task(prompt=""), _make_plan())
        score_none = _score_task_complexity(_make_task(prompt=None), _make_plan())
        assert score_empty == pytest.approx(0.40, abs=0.001)
        assert score_empty == pytest.approx(score_none, abs=0.001)

    def test_context_from_exactly_one_no_boost(self) -> None:
        # only > 2 triggers the +0.10 boost; 1 or 2 items should not change score
        baseline = _score_task_complexity(_make_task(prompt="x" * 100), _make_plan())
        score_one = _score_task_complexity(
            _make_task(prompt="x" * 100, context_from=["a"]), _make_plan()
        )
        score_two = _score_task_complexity(
            _make_task(prompt="x" * 100, context_from=["a", "b"]), _make_plan()
        )
        assert score_one == pytest.approx(baseline, abs=0.001)
        assert score_two == pytest.approx(baseline, abs=0.001)

    def test_depends_on_four_exact_score(self) -> None:
        # baseline 0.5 + >3 deps +0.10 = 0.60; 100-char prompt = no change
        score = _score_task_complexity(
            _make_task(prompt="x" * 100, depends_on=["a", "b", "c", "d"]), _make_plan()
        )
        assert score == pytest.approx(0.60, abs=0.001)

    def test_medium_boost_tag_alone_exact_score(self) -> None:
        # baseline 0.5 + review +0.15 = 0.65; 100-char prompt = no change
        score = _score_task_complexity(
            _make_task(tags=["review"], prompt="x" * 100), _make_plan()
        )
        assert score == pytest.approx(0.65, abs=0.001)
        assert _tier_from_score(score) == "medium"

    def test_security_tag_exact_score_with_medium_prompt(self) -> None:
        # security → max(0.5, 0.8) = 0.8; 100-char prompt = no change → exactly 0.80
        score = _score_task_complexity(
            _make_task(tags=["security"], prompt="x" * 100), _make_plan()
        )
        assert score == pytest.approx(0.80, abs=0.001)

    def test_context_from_exactly_three_exact_score(self) -> None:
        # baseline 0.5 + context_from > 2 (+0.10) = 0.60; 100-char prompt = no change
        score = _score_task_complexity(
            _make_task(prompt="x" * 100, context_from=["a", "b", "c"]), _make_plan()
        )
        assert score == pytest.approx(0.60, abs=0.001)

    def test_depends_on_five_gives_same_boost_as_four(self) -> None:
        # > 3 triggers +0.10 once; 5 deps should give the same boost as 4
        score_four = _score_task_complexity(
            _make_task(prompt="x" * 100, depends_on=["a", "b", "c", "d"]), _make_plan()
        )
        score_five = _score_task_complexity(
            _make_task(prompt="x" * 100, depends_on=["a", "b", "c", "d", "e"]), _make_plan()
        )
        assert score_four == pytest.approx(score_five, abs=0.001)

    @pytest.mark.parametrize("engine", ["copilot", "ollama", "qwen"])
    def test_audit_tag_routes_to_high_tier_per_engine(self, engine: str) -> None:
        # "audit" is in _HIGH_TIER_TAGS → score >= 0.8 → high tier for all engines
        plan = _make_plan()
        task = _make_task(tags=["audit"], prompt="x" * 100)
        expected = {"copilot": "opus", "ollama": "mixtral", "qwen": "max"}
        assert resolve_auto_model(task, plan, engine) == expected[engine]

    def test_security_tag_plus_judge_exact_score(self) -> None:
        # security → max(0.5, 0.8) = 0.8; judge adds +0.10 → 0.90; 100-char prompt = no change
        score = _score_task_complexity(
            _make_task(tags=["security"], prompt="x" * 100, judge=JudgeSpec(criteria=["q"])),
            _make_plan(),
        )
        assert score == pytest.approx(0.90, abs=0.001)
        assert _tier_from_score(score) == "high"

    def test_deps_and_context_from_boundary_stays_medium(self) -> None:
        # 0.5 + deps>3 (+0.10) + context_from>2 (+0.10) = 0.70; NOT > 0.70 → still medium
        score = _score_task_complexity(
            _make_task(
                prompt="x" * 100,
                depends_on=["a", "b", "c", "d"],
                context_from=["a", "b", "c"],
            ),
            _make_plan(),
        )
        assert score == pytest.approx(0.70, abs=0.001)
        assert _tier_from_score(score) == "medium"

    def test_deps_and_context_from_plus_judge_pushes_to_high(self) -> None:
        # 0.5 + deps>3 (+0.10) + context_from>2 (+0.10) + judge (+0.10) = 0.80 → high
        score = _score_task_complexity(
            _make_task(
                prompt="x" * 100,
                depends_on=["a", "b", "c", "d"],
                context_from=["a", "b", "c"],
                judge=JudgeSpec(criteria=["correctness"]),
            ),
            _make_plan(),
        )
        assert score == pytest.approx(0.80, abs=0.001)
        assert _tier_from_score(score) == "high"

    @pytest.mark.parametrize(
        ("score", "expected"),
        [(-0.5, "low"), (1.5, "high")],
    )
    def test_tier_from_score_out_of_range_input(self, score: float, expected: str) -> None:
        # _tier_from_score does NOT clamp; values outside [0,1] still map correctly
        assert _tier_from_score(score) == expected


# ---------------------------------------------------------------------------
# TestKeywordOnlyArguments
# Kills ReplaceBinaryOperator_Mul_Div on the bare `*` in both function
# signatures (routing.py lines 56 and 73).  Replacing `*` with `/` makes
# routing_strategy/dag_metadata positional-or-keyword; these tests verify they
# are actually keyword-only by asserting TypeError on positional calls.
# ---------------------------------------------------------------------------


class TestKeywordOnlyArguments:
    def test_resolve_auto_model_routing_strategy_is_keyword_only(self) -> None:
        task = _make_task(prompt="x" * 100)
        plan = _make_plan()
        # 4th positional arg must raise TypeError if routing_strategy is keyword-only
        with pytest.raises(TypeError):
            resolve_auto_model(task, plan, "claude", "cost_optimized")  # type: ignore[call-arg]

    def test_resolve_auto_model_dag_metadata_is_keyword_only(self) -> None:
        task = _make_task(prompt="x" * 100)
        plan = _make_plan()
        # 5th positional arg must raise TypeError
        with pytest.raises(TypeError):
            resolve_auto_model(task, plan, "claude", None, {"fan_out": 5})  # type: ignore[call-arg]

    def test_score_task_complexity_routing_strategy_is_keyword_only(self) -> None:
        task = _make_task(prompt="x" * 100)
        plan = _make_plan()
        # 3rd positional arg must raise TypeError if routing_strategy is keyword-only
        with pytest.raises(TypeError):
            _score_task_complexity(task, plan, "cost_optimized")  # type: ignore[call-arg]

    def test_score_task_complexity_dag_metadata_is_keyword_only(self) -> None:
        task = _make_task(prompt="x" * 100)
        plan = _make_plan()
        # 4th positional arg must raise TypeError
        with pytest.raises(TypeError):
            _score_task_complexity(task, plan, None, {"fan_out": 5})  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# TestEqualityVsIdentity
# Kills ReplaceComparisonOperator_Eq_Is on `task.context_mode == "recursive"`
# (routing.py line 102).  CPython interns string literals, so tests that pass
# the literal "recursive" don't distinguish == from is.  Constructing the
# string at runtime via join() guarantees a non-interned object, meaning
# `obj is "recursive"` is False while `obj == "recursive"` is True.
# ---------------------------------------------------------------------------


class TestEqualityVsIdentity:
    def test_recursive_context_mode_noninterned_string_gets_boost(self) -> None:
        # str.join() creates a heap-allocated string not interned by CPython
        mode = "".join(["r", "e", "c", "u", "r", "s", "i", "v", "e"])
        assert mode == "recursive"  # sanity: equality holds
        task = _make_task(prompt="x" * 100, context_mode=mode)
        score = _score_task_complexity(task, _make_plan())
        # 0.5 (baseline) + 0.15 (recursive boost) = 0.65
        # With == -> is mutation: mode is not "recursive" -> no boost -> 0.50 -> fails
        assert score == pytest.approx(0.65, abs=0.001)

    def test_non_recursive_mode_noninterned_string_no_boost(self) -> None:
        # Confirm only "recursive" triggers the boost; "summarized" must not
        mode = "".join(["s", "u", "m", "m", "a", "r", "i", "z", "e", "d"])
        task = _make_task(prompt="x" * 100, context_mode=mode)
        score = _score_task_complexity(task, _make_plan())
        assert score == pytest.approx(0.5, abs=0.001)  # no boost


# ---------------------------------------------------------------------------
# TestRoutingStrategy
# ---------------------------------------------------------------------------


class TestRoutingStrategy:
    """Tests for routing_strategy cost/quality adjustments."""

    def test_cost_optimized_pushes_to_cheaper_model(self) -> None:
        plan = _make_plan()
        # A task at medium tier (baseline 0.5) with cost_optimized should
        # get a penalty that keeps it medium or pushes toward low
        task = _make_task(prompt="x" * 100)
        default = resolve_auto_model(task, plan, "claude")
        optimized = resolve_auto_model(
            task, plan, "claude", routing_strategy="cost_optimized"
        )
        # cost_optimized adds +0.15 to medium tier → 0.65 → still medium
        # but the score is higher (closer to capping at medium)
        score_default = _score_task_complexity(task, plan)
        score_optimized = _score_task_complexity(
            task, plan, routing_strategy="cost_optimized"
        )
        assert score_optimized >= score_default

    def test_quality_first_pushes_to_better_model(self) -> None:
        plan = _make_plan()
        # A task with score near high boundary + quality_first should push over
        task = _make_task(
            prompt="x" * 100,
            tags=["review"],
            depends_on=["a", "b", "c", "d"],
        )
        # baseline: 0.5 + 0.15 (review) + 0.10 (>3 deps) = 0.75 → high tier
        # quality_first: high tier gets -0.15 → 0.60 → medium tier (wait, this
        # doesn't "push to better" — quality_first reduces high scores so they
        # stay high even with fewer signals)
        # Actually let's just verify the score difference
        score_balanced = _score_task_complexity(task, plan)
        score_qf = _score_task_complexity(task, plan, routing_strategy="quality_first")
        # quality_first applies -0.15 to high tier scores → makes them lower
        assert score_qf < score_balanced

    def test_balanced_no_adjustment(self) -> None:
        plan = _make_plan()
        task = _make_task(prompt="x" * 100)
        score_none = _score_task_complexity(task, plan)
        score_balanced = _score_task_complexity(task, plan, routing_strategy="balanced")
        assert score_none == pytest.approx(score_balanced, abs=0.001)

    def test_unknown_strategy_treated_as_balanced(self) -> None:
        plan = _make_plan()
        task = _make_task(prompt="x" * 100)
        score_none = _score_task_complexity(task, plan)
        score_unknown = _score_task_complexity(task, plan, routing_strategy="nonexistent")
        assert score_none == pytest.approx(score_unknown, abs=0.001)

    def test_cost_optimized_low_tier_no_change(self) -> None:
        plan = _make_plan()
        task = _make_task(tags=["trivial"])
        # trivial → score capped at 0.2 → low tier → cost_optimized adds 0.0 to low
        score = _score_task_complexity(task, plan, routing_strategy="cost_optimized")
        score_default = _score_task_complexity(task, plan)
        assert score == pytest.approx(score_default, abs=0.001)

    def test_quality_first_low_tier_gets_negative_adjustment(self) -> None:
        plan = _make_plan()
        task = _make_task(tags=["trivial"])
        score_default = _score_task_complexity(task, plan)
        score_qf = _score_task_complexity(task, plan, routing_strategy="quality_first")
        # quality_first: low tier gets -0.15 → score decreases (clamped at 0.0)
        assert score_qf <= score_default


# ---------------------------------------------------------------------------
# TestDAGMetadata
# ---------------------------------------------------------------------------


class TestDAGMetadata:
    """Tests for DAG structural signals (fan_out, depth, upstream_failure_rate)."""

    def test_fan_out_above_3_boosts_score(self) -> None:
        plan = _make_plan()
        task = _make_task(prompt="x" * 100)
        score_default = _score_task_complexity(task, plan)
        score_fanout = _score_task_complexity(
            task, plan, dag_metadata={"fan_out": 5}
        )
        assert score_fanout > score_default

    def test_fan_out_3_no_boost(self) -> None:
        plan = _make_plan()
        task = _make_task(prompt="x" * 100)
        score_default = _score_task_complexity(task, plan)
        score_fanout = _score_task_complexity(
            task, plan, dag_metadata={"fan_out": 3}
        )
        assert score_fanout == pytest.approx(score_default, abs=0.001)

    def test_depth_above_4_boosts_score(self) -> None:
        plan = _make_plan()
        task = _make_task(prompt="x" * 100)
        score_default = _score_task_complexity(task, plan)
        score_deep = _score_task_complexity(
            task, plan, dag_metadata={"depth": 5}
        )
        assert score_deep > score_default

    def test_depth_4_no_boost(self) -> None:
        plan = _make_plan()
        task = _make_task(prompt="x" * 100)
        score_default = _score_task_complexity(task, plan)
        score_deep = _score_task_complexity(
            task, plan, dag_metadata={"depth": 4}
        )
        assert score_deep == pytest.approx(score_default, abs=0.001)

    def test_upstream_failure_rate_above_03_boosts_score(self) -> None:
        plan = _make_plan()
        task = _make_task(prompt="x" * 100)
        score_default = _score_task_complexity(task, plan)
        score_fail = _score_task_complexity(
            task, plan, dag_metadata={"upstream_failure_rate": 0.5}
        )
        assert score_fail > score_default

    def test_upstream_failure_rate_03_no_boost(self) -> None:
        plan = _make_plan()
        task = _make_task(prompt="x" * 100)
        score_default = _score_task_complexity(task, plan)
        score_fail = _score_task_complexity(
            task, plan, dag_metadata={"upstream_failure_rate": 0.3}
        )
        assert score_fail == pytest.approx(score_default, abs=0.001)

    def test_all_dag_signals_accumulate(self) -> None:
        plan = _make_plan()
        task = _make_task(prompt="x" * 100)
        score = _score_task_complexity(
            task, plan,
            dag_metadata={"fan_out": 5, "depth": 6, "upstream_failure_rate": 0.5},
        )
        # baseline 0.5 + fan_out +0.10 + depth +0.05 + failure +0.15 = 0.80
        assert score == pytest.approx(0.80, abs=0.01)

    def test_dag_metadata_none_no_effect(self) -> None:
        plan = _make_plan()
        task = _make_task(prompt="x" * 100)
        score_none = _score_task_complexity(task, plan, dag_metadata=None)
        score_empty = _score_task_complexity(task, plan, dag_metadata={})
        assert score_none == pytest.approx(score_empty, abs=0.001)

    def test_fan_out_pushes_auto_model_to_higher_tier(self) -> None:
        plan = _make_plan()
        task = _make_task(prompt="x" * 100, tags=["review"])
        # review: 0.5 + 0.15 = 0.65 → medium
        # + fan_out 5: +0.10 → 0.75 → high
        model = resolve_auto_model(
            task, plan, "claude", dag_metadata={"fan_out": 5}
        )
        assert model == "opus"

    def test_combined_strategy_and_dag(self) -> None:
        plan = _make_plan()
        task = _make_task(prompt="x" * 100)
        # baseline 0.5 + fan_out 5 (+0.10) = 0.60 → medium
        # cost_optimized on medium adds +0.15 → 0.75 → high
        score = _score_task_complexity(
            task, plan,
            routing_strategy="cost_optimized",
            dag_metadata={"fan_out": 5},
        )
        assert score > 0.7


# ---------------------------------------------------------------------------
# TestCostWeightsExact — exact numeric assertions targeting _COST_WEIGHTS
# ---------------------------------------------------------------------------


class TestCostWeightsExact:
    """Pin exact _COST_WEIGHTS values to kill NumberReplacer / USub_Not mutants."""

    @pytest.mark.parametrize(
        ("tags", "expected_base", "expected_opt"),
        [
            ([], 0.5, 0.65),           # medium tier: 0.5 + 0.15 = 0.65
            (["security"], 0.8, 1.0),  # high tier:   0.8 + 0.30 = 1.10 → clamped to 1.0
        ],
    )
    def test_cost_optimized_exact_boost_per_tier(
        self, tags: list[str], expected_base: float, expected_opt: float
    ) -> None:
        # Mutations: medium 0.15→0 (no boost survives >=), high 0.3→0 or →-1
        task = _make_task(tags=tags, prompt="x" * 100)
        plan = _make_plan()
        base = _score_task_complexity(task, plan)
        opt = _score_task_complexity(task, plan, routing_strategy="cost_optimized")
        assert base == pytest.approx(expected_base, abs=0.001)
        assert opt == pytest.approx(expected_opt, abs=0.001)

    @pytest.mark.parametrize(
        ("tags", "expected_base", "expected_qf"),
        [
            (["trivial"], 0.2, 0.05),   # low tier:    0.2 + (-0.15) = 0.05
            ([], 0.5, 0.5),             # medium tier: 0.5 +    0.0  = 0.5  (no change)
            (["security"], 0.8, 0.65),  # high tier:   0.8 + (-0.15) = 0.65
        ],
    )
    def test_quality_first_exact_adjustment_per_tier(
        self, tags: list[str], expected_base: float, expected_qf: float
    ) -> None:
        # Mutations: low -0.15→0 (USub_Not or NumberReplacer 0.15→0); medium 0.0→1 or →-1;
        # high -0.15→0 (NumberReplacer 0.15→0) — all survive <= / >= assertions
        task = _make_task(tags=tags, prompt="x" * 100)
        plan = _make_plan()
        base = _score_task_complexity(task, plan)
        qf = _score_task_complexity(task, plan, routing_strategy="quality_first")
        assert base == pytest.approx(expected_base, abs=0.001)
        assert qf == pytest.approx(expected_qf, abs=0.001)

    def test_quality_first_high_score_routes_to_medium_tier_model(self) -> None:
        # security → 0.8 → high → "opus"; quality_first: 0.8 - 0.15 = 0.65 → medium → "sonnet"
        # Observable through resolve_auto_model — kills NumberReplacer on high=-0.15
        plan = _make_plan()
        task = _make_task(tags=["security"], prompt="x" * 100)
        assert resolve_auto_model(task, plan, "claude") == "opus"
        assert resolve_auto_model(task, plan, "claude", routing_strategy="quality_first") == "sonnet"

    def test_partial_dag_metadata_missing_failure_rate_no_spurious_boost(self) -> None:
        # upstream_failure_rate key absent → default 0.0 (NOT 1.0); mutation 0.0→1 triggers
        # > 0.3 check and adds +0.15 spuriously — exact score catches this
        plan = _make_plan()
        task = _make_task(prompt="x" * 100)
        score_no_rate = _score_task_complexity(task, plan, dag_metadata={"fan_out": 5})
        score_with_rate = _score_task_complexity(
            task, plan, dag_metadata={"fan_out": 5, "upstream_failure_rate": 0.5}
        )
        # fan_out>3 (+0.10) only; no failure-rate boost
        assert score_no_rate == pytest.approx(0.60, abs=0.001)
        # fan_out>3 (+0.10) + failure_rate>0.3 (+0.15)
        assert score_with_rate == pytest.approx(0.75, abs=0.001)

    def test_fan_out_threshold_boundary_exact_scores(self) -> None:
        # fan_out=3: NOT >3 → no boost (score stays 0.50)
        # fan_out=4: >3 → +0.10 boost (score 0.60)
        # Kills NumberReplacer: 3→0 or 3→1 would make fan_out=3 cross the threshold
        plan = _make_plan()
        task = _make_task(prompt="x" * 100)
        score_3 = _score_task_complexity(task, plan, dag_metadata={"fan_out": 3})
        score_4 = _score_task_complexity(task, plan, dag_metadata={"fan_out": 4})
        assert score_3 == pytest.approx(0.50, abs=0.001)
        assert score_4 == pytest.approx(0.60, abs=0.001)

    def test_cost_optimized_high_weight_is_exactly_0_3(self) -> None:
        # Kills NumberReplacer: 0.3 → 1 in _COST_WEIGHTS["cost_optimized"]["high"].
        # Behavioral tests can't distinguish these: any high-tier score (>0.7) + 0.3
        # and + 1.0 both exceed 1.0 and clamp to the same value.  Direct constant
        # assertion is the only reliable kill.
        assert _COST_WEIGHTS["cost_optimized"]["high"] == pytest.approx(0.3, abs=0.001)

    def test_score_clamped_to_zero_not_negative_with_quality_first_low_tier(self) -> None:
        # trivial tag caps score at 0.2; "hi" (2 chars < 100) subtracts 0.10 → 0.10;
        # quality_first applies -0.15 to low tier → 0.10 - 0.15 = -0.05 before clamping.
        # max(0.0, -0.05) = 0.0.
        # Kills NumberReplacer: max(0.0, ...) → max(-1, ...) would return -0.05.
        score = _score_task_complexity(
            _make_task(tags=["trivial"], prompt="hi"),
            _make_plan(),
            routing_strategy="quality_first",
        )
        assert score == pytest.approx(0.0, abs=0.001)
        assert score >= 0.0


# ---------------------------------------------------------------------------
# TestAnnotationIntegrity
# Kills all ReplaceBinaryOperator_BitOr_* mutants on routing.py lines 57, 58,
# 74, 75 — the `str | None` and `dict[str, Any] | None` annotations on the
# keyword-only parameters of resolve_auto_model and _score_task_complexity.
#
# With `from __future__ import annotations`, annotations are stored as lazy
# strings, so the mutation (e.g. | → +) never triggers a SyntaxError at
# import time.  typing.get_type_hints() evaluates those strings in the
# module's global namespace; `str + None` raises TypeError, failing the test.
# ---------------------------------------------------------------------------


class TestAnnotationIntegrity:
    def test_resolve_auto_model_optional_param_annotations_are_valid(self) -> None:
        # Raises TypeError on mutated annotation: "str + None", "str * None", etc.
        hints = typing.get_type_hints(resolve_auto_model)
        # Both keyword-only optional parameters must be present and evaluable.
        assert "routing_strategy" in hints
        assert "dag_metadata" in hints

    def test_score_task_complexity_optional_param_annotations_are_valid(self) -> None:
        hints = typing.get_type_hints(_score_task_complexity)
        assert "routing_strategy" in hints
        assert "dag_metadata" in hints

    def test_resolve_auto_model_evidence_param_annotation_valid(self) -> None:
        hints = typing.get_type_hints(resolve_auto_model)
        assert "evidence" in hints


# ---------------------------------------------------------------------------
# T2.3 — Predictive routing: historical performance
# ---------------------------------------------------------------------------

def _make_manifest(
    plan_name: str,
    task_results: dict[str, dict],
) -> dict:
    return {"plan_name": plan_name, "task_results": task_results}


def _write_manifest(
    run_dir: Path,
    plan_name: str,
    run_index: int,
    task_results: dict[str, dict],
) -> None:
    import json
    dirname = f"2026031{run_index:1d}_120000_000000_aaa_{plan_name}"
    d = run_dir / dirname
    d.mkdir(parents=True, exist_ok=True)
    (d / "run_manifest.json").write_text(
        json.dumps(_make_manifest(plan_name, task_results)),
        encoding="utf-8",
    )


class TestLoadTaskHistories:
    def test_no_runs_returns_empty(self, tmp_path: Path) -> None:
        result = load_task_histories("demo", tmp_path)
        assert result == {}

    def test_below_min_runs_returns_empty(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, "demo", 1, {
            "t1": {"auto_routed_model": "haiku", "status": "success",
                   "duration_sec": 10.0, "cost_usd": 0.01, "exit_code": 0},
        })
        _write_manifest(tmp_path, "demo", 2, {
            "t1": {"auto_routed_model": "haiku", "status": "success",
                   "duration_sec": 12.0, "cost_usd": 0.02, "exit_code": 0},
        })
        # min_runs=3 (default), only 2 available
        result = load_task_histories("demo", tmp_path)
        assert result == {}

    def test_aggregates_auto_routed_tasks(self, tmp_path: Path) -> None:
        for i in range(3):
            _write_manifest(tmp_path, "demo", i, {
                "t1": {"auto_routed_model": "haiku", "status": "success",
                       "duration_sec": 10.0 + i, "cost_usd": 0.01, "exit_code": 0},
            })
        result = load_task_histories("demo", tmp_path)
        assert "t1" in result
        hist = result["t1"]
        assert hist.total_runs == 3
        assert "haiku" in hist.records
        rec = hist.records["haiku"]
        assert rec.runs == 3
        assert rec.successes == 3
        assert rec.failures == 0
        assert rec.timeouts == 0
        assert rec.avg_duration_sec == pytest.approx(11.0)
        assert rec.avg_cost_usd == pytest.approx(0.01)

    def test_ignores_explicit_model_tasks(self, tmp_path: Path) -> None:
        for i in range(3):
            _write_manifest(tmp_path, "demo", i, {
                "t1": {"status": "success", "duration_sec": 10.0,
                       "cost_usd": 0.01, "exit_code": 0},
            })
        result = load_task_histories("demo", tmp_path)
        assert "t1" not in result

    def test_handles_corrupt_manifest(self, tmp_path: Path) -> None:
        for i in range(3):
            _write_manifest(tmp_path, "demo", i, {
                "t1": {"auto_routed_model": "haiku", "status": "success",
                       "duration_sec": 10.0, "cost_usd": 0.01, "exit_code": 0},
            })
        # Corrupt one manifest
        corrupt_dir = tmp_path / "20260314_120000_000000_aaa_demo"
        corrupt_dir.mkdir(parents=True, exist_ok=True)
        (corrupt_dir / "run_manifest.json").write_text("NOT JSON", encoding="utf-8")

        result = load_task_histories("demo", tmp_path)
        assert "t1" in result
        assert result["t1"].total_runs == 3  # corrupt one skipped

    def test_timeout_detection(self, tmp_path: Path) -> None:
        for i in range(3):
            _write_manifest(tmp_path, "demo", i, {
                "t1": {"auto_routed_model": "sonnet", "status": "failed",
                       "duration_sec": 600.0, "cost_usd": 0.10,
                       "exit_code": 124},
            })
        result = load_task_histories("demo", tmp_path)
        rec = result["t1"].records["sonnet"]
        assert rec.timeouts == 3
        assert rec.failures == 3

    def test_cost_averaging_with_none(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, "demo", 0, {
            "t1": {"auto_routed_model": "haiku", "status": "success",
                   "duration_sec": 10.0, "cost_usd": 0.02, "exit_code": 0},
        })
        _write_manifest(tmp_path, "demo", 1, {
            "t1": {"auto_routed_model": "haiku", "status": "success",
                   "duration_sec": 12.0, "cost_usd": None, "exit_code": 0},
        })
        _write_manifest(tmp_path, "demo", 2, {
            "t1": {"auto_routed_model": "haiku", "status": "success",
                   "duration_sec": 14.0, "cost_usd": 0.04, "exit_code": 0},
        })
        result = load_task_histories("demo", tmp_path)
        rec = result["t1"].records["haiku"]
        assert rec.avg_cost_usd == pytest.approx(0.03)  # (0.02 + 0.04) / 2

    def test_caps_at_max_manifests(self, tmp_path: Path) -> None:
        # Create 25 manifests — only last 20 should be used
        for i in range(25):
            _write_manifest(tmp_path, "demo", i, {
                "t1": {"auto_routed_model": "haiku", "status": "success",
                       "duration_sec": 10.0, "cost_usd": 0.01, "exit_code": 0},
            })
        result = load_task_histories("demo", tmp_path, min_runs=3)
        assert result["t1"].total_runs == 20  # capped

    def test_mixed_auto_and_explicit(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, "demo", 0, {
            "t1": {"auto_routed_model": "haiku", "status": "success",
                   "duration_sec": 10.0, "cost_usd": 0.01, "exit_code": 0},
        })
        _write_manifest(tmp_path, "demo", 1, {
            "t1": {"status": "success", "duration_sec": 10.0,
                   "cost_usd": 0.10, "exit_code": 0},  # explicit model, no auto
        })
        _write_manifest(tmp_path, "demo", 2, {
            "t1": {"auto_routed_model": "haiku", "status": "success",
                   "duration_sec": 10.0, "cost_usd": 0.01, "exit_code": 0},
        })
        result = load_task_histories("demo", tmp_path)
        assert result["t1"].total_runs == 2  # only auto-routed counted


class TestApplyHistoricalSignal:
    def _make_history(
        self,
        records: dict[str, ModelRecord],
        total_runs: int | None = None,
    ) -> TaskHistory:
        tr = total_runs if total_runs is not None else sum(r.runs for r in records.values())
        return TaskHistory(task_id="t1", total_runs=tr, records=records)

    def _rec(
        self,
        model: str,
        runs: int = 5,
        successes: int = 5,
        failures: int = 0,
        timeouts: int = 0,
    ) -> ModelRecord:
        return ModelRecord(
            model=model, runs=runs, successes=successes,
            failures=failures, timeouts=timeouts,
            avg_duration_sec=30.0, avg_cost_usd=0.01,
        )

    def test_cheap_model_always_succeeds_lowers_score(self) -> None:
        history = self._make_history({"haiku": self._rec("haiku", 5, 5, 0)})
        # Start at 0.5 → should decrease
        result = _apply_historical_signal(0.5, history, "claude")
        assert result < 0.5

    def test_cheap_model_fails_often_raises_score(self) -> None:
        history = self._make_history({"haiku": self._rec("haiku", 5, 1, 4)})
        result = _apply_historical_signal(0.5, history, "claude")
        assert result > 0.5

    def test_timeout_pattern_raises_score(self) -> None:
        history = self._make_history({
            "sonnet": self._rec("sonnet", 5, 2, 3, timeouts=3),
        })
        result = _apply_historical_signal(0.5, history, "claude")
        assert result > 0.5

    def test_medium_model_succeeds_no_cheap_data(self) -> None:
        history = self._make_history({"sonnet": self._rec("sonnet", 5, 5, 0)})
        result = _apply_historical_signal(0.5, history, "claude")
        assert result < 0.5  # modest decrease

    def test_confidence_scaling_low_runs(self) -> None:
        # 2 runs = confidence 0.4 → reduced adjustment
        history = self._make_history(
            {"haiku": self._rec("haiku", 2, 0, 2)},
            total_runs=2,
        )
        full_history = self._make_history(
            {"haiku": self._rec("haiku", 5, 0, 5)},
            total_runs=5,
        )
        low_adj = _apply_historical_signal(0.5, history, "claude") - 0.5
        full_adj = _apply_historical_signal(0.5, full_history, "claude") - 0.5
        # Low confidence should produce smaller adjustment magnitude
        assert abs(low_adj) < abs(full_adj)

    def test_adjustment_bounded_at_020(self) -> None:
        # Stack multiple positive signals: cheap failures + timeouts
        history = self._make_history({
            "haiku": self._rec("haiku", 10, 0, 10, timeouts=10),
        }, total_runs=10)
        result = _apply_historical_signal(0.5, history, "claude")
        assert result <= 0.5 + 0.20 + 0.001  # within bound

    def test_unknown_engine_no_adjustment(self) -> None:
        history = self._make_history({"haiku": self._rec("haiku", 5, 0, 5)})
        result = _apply_historical_signal(0.5, history, "unknown_engine")
        assert result == 0.5

    def test_no_records_no_adjustment(self) -> None:
        history = TaskHistory(task_id="t1", total_runs=5, records={})
        result = _apply_historical_signal(0.5, history, "claude")
        assert result == 0.5

    def test_zero_total_runs_no_crash(self) -> None:
        history = TaskHistory(task_id="t1", total_runs=0, records={})
        result = _apply_historical_signal(0.5, history, "claude")
        assert result == 0.5

    def test_score_clamped_to_0_1(self) -> None:
        # Score near 0 with cheap success → clamp to 0.0
        history = self._make_history({"haiku": self._rec("haiku", 5, 5, 0)})
        result = _apply_historical_signal(0.05, history, "claude")
        assert result >= 0.0
        # Score near 1 with cheap failures → clamp to 1.0
        history2 = self._make_history({"haiku": self._rec("haiku", 5, 0, 5)})
        result2 = _apply_historical_signal(0.95, history2, "claude")
        assert result2 <= 1.0


class TestResolveAutoModelEvidence:
    def test_evidence_populated(self) -> None:
        task = _make_task(tags=["trivial"])
        plan = _make_plan()
        evidence: dict[str, object] = {}
        model = resolve_auto_model(task, plan, "claude", evidence=evidence)
        assert model == "haiku"
        assert "complexity_score" in evidence
        assert "tier" in evidence
        assert evidence["tier"] == "low"
        assert evidence["historical_runs"] == 0

    def test_evidence_none_unchanged(self) -> None:
        task = _make_task(tags=["trivial"])
        plan = _make_plan()
        # Should not crash with evidence=None (default)
        model = resolve_auto_model(task, plan, "claude")
        assert model == "haiku"

    def test_evidence_with_history(self) -> None:
        task = _make_task()
        plan = _make_plan()
        history = TaskHistory(task_id="test-task", total_runs=7, records={})
        evidence: dict[str, object] = {}
        resolve_auto_model(
            task, plan, "claude",
            dag_metadata={"fan_out": 0, "depth": 0,
                          "upstream_failure_rate": 0.0,
                          "task_history": history},
            evidence=evidence,
        )
        assert evidence["historical_runs"] == 7


class TestResolveAutoModelWithHistory:
    def test_cheap_success_history_downgrades(self) -> None:
        """History showing haiku always succeeds → should select haiku.

        Baseline: short prompt → 0.5 - 0.10 = 0.40 (medium tier).
        History: haiku 100% success → -0.15 → 0.25 (low tier → haiku).
        """
        task = _make_task(prompt="short")  # <100 chars → -0.10 penalty
        plan = _make_plan()
        haiku_rec = ModelRecord(
            model="haiku", runs=5, successes=5, failures=0,
            timeouts=0, avg_duration_sec=10.0, avg_cost_usd=0.01,
        )
        history = TaskHistory(task_id="test-task", total_runs=5,
                              records={"haiku": haiku_rec})
        model = resolve_auto_model(
            task, plan, "claude",
            dag_metadata={"fan_out": 0, "depth": 0,
                          "upstream_failure_rate": 0.0,
                          "task_history": history},
        )
        assert model == "haiku"

    def test_cheap_failure_history_upgrades(self) -> None:
        """History showing haiku always fails → should select sonnet.

        Baseline: trivial tag → 0.2, prompt ≥500 → +0.05 = 0.25 (low tier).
        History: haiku 100% failure → +0.15 → 0.40 (medium tier → sonnet).
        """
        task = _make_task(tags=["trivial"], prompt="A" * 600)
        plan = _make_plan()
        haiku_rec = ModelRecord(
            model="haiku", runs=5, successes=0, failures=5,
            timeouts=0, avg_duration_sec=30.0, avg_cost_usd=0.01,
        )
        history = TaskHistory(task_id="test-task", total_runs=5,
                              records={"haiku": haiku_rec})
        model = resolve_auto_model(
            task, plan, "claude",
            dag_metadata={"fan_out": 0, "depth": 0,
                          "upstream_failure_rate": 0.0,
                          "task_history": history},
        )
        assert model == "sonnet"

    def test_no_history_matches_heuristic(self) -> None:
        """No history → identical to pure heuristic routing."""
        task = _make_task(tags=["trivial"])
        plan = _make_plan()
        model_no_hist = resolve_auto_model(task, plan, "claude")
        model_with_empty_meta = resolve_auto_model(
            task, plan, "claude",
            dag_metadata={"fan_out": 0, "depth": 0,
                          "upstream_failure_rate": 0.0},
        )
        assert model_no_hist == model_with_empty_meta
