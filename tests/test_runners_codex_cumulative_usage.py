from __future__ import annotations

import json

from maestro_cli.runners import _extract_codex_cumulative_usage


# This module targets _extract_codex_cumulative_usage (runners.py L4128-4225),
# specifically the currently-uncovered lines:
#   - 4169: Strategy 1 (response.completed) where ``response`` is not a dict -> continue
#   - 4172: Strategy 1 where ``response`` is a dict but ``usage`` is not a dict -> continue
#   - 4199, 4202-4210: Strategy 3 (item.completed) read paths (type filter + usage lookup
#     on payload / payload["item"] with non-coercible or missing usage)
#   - 4214: ``return last_item_usage`` for Strategy 3.
#
# NOTE on line 4212 (set last_item_usage) and 4214 (return last_item_usage):
# These are effectively shadowed/unreachable given the current implementation.
# Strategy 2 (_extract_usage_from_line / _extract_usage_from_json_payload) iterates
# ALL lines and recursively searches every dict for a ``usage`` block requiring both
# ``input_tokens`` and ``output_tokens``. Strategy 3 reads the SAME keys via the SAME
# candidate parser. Any item.completed usage that Strategy 3 could parse is therefore
# always discovered by Strategy 2 first, which returns at line 4193 before the
# Strategy 3 loop ever runs. There is no asymmetric input that satisfies Strategy 3
# (lines 4208-4211) while Strategy 2 finds nothing, so ``last_item_usage`` can never be
# assigned in practice. The tests below exercise every reachable read path of
# Strategy 3 (4199, 4202-4210) but the function then correctly falls through to
# Strategy 4 (byte-length estimation), which is the only observable outcome.


def _dump(payload: dict[str, object]) -> str:
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# Strategy 1: response.completed branches (lines 4168-4172)
# ---------------------------------------------------------------------------


def test_strategy1_response_not_dict_continues() -> None:
    # Line 4169: response.completed event but "response" is a string, not a dict.
    line = _dump({"type": "response.completed", "response": "not-a-dict"})
    result = _extract_codex_cumulative_usage([line])
    # No usage found by S1/S2/S3 -> Strategy 4 byte-length estimation.
    assert result is not None
    input_tokens, cached_tokens, output_tokens = result
    assert input_tokens == 0
    assert cached_tokens == 0
    assert output_tokens == len(line.encode("utf-8")) // 4


def test_strategy1_usage_not_dict_continues() -> None:
    # Line 4172: response is a dict but "usage" is not a dict -> continue.
    line = _dump({"type": "response.completed", "response": {"usage": "nope"}})
    result = _extract_codex_cumulative_usage([line])
    assert result is not None
    assert result[0] == 0
    assert result[1] == 0
    assert result[2] == len(line.encode("utf-8")) // 4


def test_strategy1_usage_missing_continues() -> None:
    # Line 4170-4172: response is a dict with no "usage" key (.get -> None) -> continue.
    line = _dump({"type": "response.completed", "response": {}})
    result = _extract_codex_cumulative_usage([line])
    assert result is not None
    assert result[0] == 0
    assert result[2] == len(line.encode("utf-8")) // 4


def test_strategy1_valid_response_completed_returns_usage() -> None:
    # Happy path through Strategy 1 (lines 4173-4177): exercises the lines AROUND the
    # uncovered continues so the early-return is also confirmed not to fire spuriously.
    line = _dump(
        {
            "type": "response.completed",
            "response": {
                "usage": {
                    "input_tokens": 120,
                    "output_tokens": 45,
                    "cached_input_tokens": 30,
                }
            },
        }
    )
    result = _extract_codex_cumulative_usage([line])
    assert result == (120, 30, 45)


def test_strategy1_input_present_output_missing_falls_through() -> None:
    # response.completed usage dict has input but no output -> S1 condition (4176) false.
    # Falls to Strategy 2 (also requires both) -> Strategy 4.
    line = _dump(
        {
            "type": "response.completed",
            "response": {"usage": {"input_tokens": 99}},
        }
    )
    result = _extract_codex_cumulative_usage([line])
    assert result is not None
    assert result[0] == 0
    assert result[2] == len(line.encode("utf-8")) // 4


# ---------------------------------------------------------------------------
# Strategy 3: item.completed read paths (lines 4199, 4202-4210)
# ---------------------------------------------------------------------------


def test_strategy3_non_item_type_continues() -> None:
    # Line 4199: a line whose type is NOT item.completed -> continue inside Strategy 3.
    # Use lines that produce no usage so S1/S2 fail and the S3 loop actually runs.
    non_item = _dump({"type": "thread.started"})
    result = _extract_codex_cumulative_usage([non_item])
    assert result is not None
    # Strategy 4 estimation over the single line.
    assert result[0] == 0
    assert result[2] == len(non_item.encode("utf-8")) // 4


def test_strategy3_item_completed_no_usage() -> None:
    # Lines 4202-4206: item.completed with neither payload["usage"] nor item["usage"]
    # -> usage_dict stays None -> 4206 continue.
    line = _dump({"type": "item.completed", "item": {"id": "x"}})
    result = _extract_codex_cumulative_usage([line])
    assert result is not None
    assert result[0] == 0
    assert result[2] == len(line.encode("utf-8")) // 4


def test_strategy3_item_completed_usage_not_dict() -> None:
    # Line 4202-4206: payload["usage"] present but not a dict, item missing -> continue.
    line = _dump({"type": "item.completed", "usage": "not-a-dict"})
    result = _extract_codex_cumulative_usage([line])
    assert result is not None
    assert result[0] == 0
    assert result[2] == len(line.encode("utf-8")) // 4


def test_strategy3_item_usage_under_item_key() -> None:
    # Lines 4203-4205: usage_dict resolved from item["usage"] when payload["usage"]
    # is absent. Use uncoercible token values so Strategy 2 ALSO fails (so the S3
    # loop genuinely executes) and the 4211 condition is False.
    line = _dump(
        {
            "type": "item.completed",
            "item": {"usage": {"input_tokens": "abc", "output_tokens": "def"}},
        }
    )
    result = _extract_codex_cumulative_usage([line])
    assert result is not None
    assert result[0] == 0
    assert result[2] == len(line.encode("utf-8")) // 4


def test_strategy3_item_usage_partial_tokens() -> None:
    # Lines 4208-4211: usage dict on payload directly with input but no output ->
    # output_tokens is None -> condition False, last_item_usage stays None.
    # input_tokens-only also makes Strategy 2 reject it (it needs both) so the
    # S3 loop runs.
    line = _dump({"type": "item.completed", "usage": {"input_tokens": "qq"}})
    result = _extract_codex_cumulative_usage([line])
    assert result is not None
    assert result[0] == 0
    assert result[2] == len(line.encode("utf-8")) // 4


def test_strategy3_multiple_mixed_lines() -> None:
    # Drive several Strategy 3 read branches in one call:
    #   - a non-item line (4199 continue)
    #   - an item.completed with no usage (4206 continue)
    #   - an item.completed with item-nested uncoercible usage (4204-4211 False)
    #   - an item.completed with non-dict usage (4206 continue)
    lines = [
        _dump({"type": "turn.started"}),
        _dump({"type": "item.completed"}),
        _dump({"type": "item.completed", "item": {"usage": {"input_tokens": "x"}}}),
        _dump({"type": "item.completed", "usage": ["not", "a", "dict"]}),
    ]
    result = _extract_codex_cumulative_usage(lines)
    assert result is not None
    total_bytes = sum(len(line.encode("utf-8")) for line in lines)
    assert result == (0, 0, total_bytes // 4)


def test_strategy3_item_field_is_list_not_dict() -> None:
    # Line 4204: ``isinstance(item, dict)`` is False when item is a list, so usage_dict
    # remains None and 4206 continues.
    line = _dump({"type": "item.completed", "item": [1, 2, 3]})
    result = _extract_codex_cumulative_usage([line])
    assert result is not None
    assert result[0] == 0
    assert result[2] == len(line.encode("utf-8")) // 4


# ---------------------------------------------------------------------------
# Boundary / fall-through behavior
# ---------------------------------------------------------------------------


def test_empty_input_returns_none() -> None:
    # No lines at all -> total_bytes == 0 -> Strategy 4 skipped -> None.
    assert _extract_codex_cumulative_usage([]) is None


def test_blank_lines_only_estimate_zero_bytes() -> None:
    # Whitespace-only lines have non-zero encoded byte length, so Strategy 4 fires.
    lines = ["", "   ", "\t"]
    result = _extract_codex_cumulative_usage(lines)
    assert result is not None
    total_bytes = sum(len(line.encode("utf-8")) for line in lines)
    assert result == (0, 0, total_bytes // 4)


def test_stderr_prefixed_response_completed_parsed() -> None:
    # _parse_json_candidates strips a leading "[stderr] " prefix. The response is
    # not a dict here, exercising line 4169 even through the stderr candidate path.
    line = "[stderr] " + _dump({"type": "response.completed", "response": 42})
    result = _extract_codex_cumulative_usage([line])
    assert result is not None
    assert result[0] == 0
    assert result[2] == len(line.encode("utf-8")) // 4
