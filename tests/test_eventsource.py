from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from maestro_cli.eventsource import (
    ChainState,
    compute_artefact_hash,
    compute_event_hash,
    emit_hashed_event,
    replay_events,
    replay_run_state,
    verify_artefact_hashes,
    verify_chain,
)
from maestro_cli.models import EventRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(event: str = "test_event", **kwargs: Any) -> dict[str, Any]:
    return {"event": event, "ts": "2026-01-01T00:00:00", **kwargs}


def _write_jsonl(path: Path, lines: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line) + "\n")


# ---------------------------------------------------------------------------
# TestComputeEventHash
# ---------------------------------------------------------------------------

class TestComputeEventHash:
    def test_deterministic(self) -> None:
        ev = {"event": "run_start", "ts": "t1"}
        prev = "0" * 16
        assert compute_event_hash(ev, prev) == compute_event_hash(ev, prev)

    def test_different_payload_different_hash(self) -> None:
        prev = "0" * 16
        h1 = compute_event_hash({"event": "a"}, prev)
        h2 = compute_event_hash({"event": "b"}, prev)
        assert h1 != h2

    def test_different_prev_hash_different_hash(self) -> None:
        ev = {"event": "run_start"}
        h1 = compute_event_hash(ev, "0" * 16)
        h2 = compute_event_hash(ev, "1" * 16)
        assert h1 != h2


# ---------------------------------------------------------------------------
# TestEmitHashedEvent
# ---------------------------------------------------------------------------

class TestEmitHashedEvent:
    def test_first_event_has_zero_prev_hash(self) -> None:
        state = ChainState()
        ev = _make_event("run_start")
        record = emit_hashed_event(ev, state)
        assert record.prev_hash == "0" * 16

    def test_chain_links_correctly(self) -> None:
        state = ChainState()
        r1 = emit_hashed_event(_make_event("task_start"), state)
        r2 = emit_hashed_event(_make_event("task_complete"), state)
        assert r2.prev_hash == r1.event_hash

    def test_sequence_increments(self) -> None:
        state = ChainState()
        r1 = emit_hashed_event(_make_event("a"), state)
        r2 = emit_hashed_event(_make_event("b"), state)
        r3 = emit_hashed_event(_make_event("c"), state)
        assert r1.sequence == 0
        assert r2.sequence == 1
        assert r3.sequence == 2

    def test_state_mutates(self) -> None:
        state = ChainState()
        original_prev = state.prev_hash
        r1 = emit_hashed_event(_make_event("a"), state)
        assert state.prev_hash != original_prev
        assert state.prev_hash == r1.event_hash
        assert state.sequence == 1


# ---------------------------------------------------------------------------
# TestReplayEvents
# ---------------------------------------------------------------------------

class TestReplayEvents:
    def test_replay_valid_chain(self, tmp_path: Path) -> None:
        """Write events produced by emit_hashed_event and replay them."""
        state = ChainState()
        records: list[EventRecord] = []
        for name in ("run_start", "task_start", "task_complete"):
            records.append(emit_hashed_event(_make_event(name), state))

        events_file = tmp_path / "events.jsonl"
        _write_jsonl(events_file, [r.to_dict() for r in records])

        replayed = replay_events(events_file)
        assert len(replayed) == 3
        # Chain integrity
        for original, replayed_rec in zip(records, replayed):
            assert replayed_rec.event_hash == original.event_hash
            assert replayed_rec.prev_hash == original.prev_hash

    def test_replay_legacy_events(self, tmp_path: Path) -> None:
        """Legacy flat-format lines (no hash/prev_hash) must still parse."""
        events_file = tmp_path / "events.jsonl"
        legacy = [
            {"event": "run_start", "ts": "t1", "plan_name": "demo"},
            {"event": "task_complete", "ts": "t2", "task_id": "build"},
        ]
        _write_jsonl(events_file, legacy)

        replayed = replay_events(events_file)
        assert len(replayed) == 2
        # No hash stored for legacy events
        for rec in replayed:
            assert rec.event_hash == ""

    def test_replay_empty_file(self, tmp_path: Path) -> None:
        events_file = tmp_path / "events.jsonl"
        events_file.write_text("", encoding="utf-8")
        result = replay_events(events_file)
        assert result == []

    def test_replay_missing_file(self, tmp_path: Path) -> None:
        events_file = tmp_path / "nonexistent.jsonl"
        # replay_events opens the file directly — expect FileNotFoundError
        with pytest.raises(FileNotFoundError):
            replay_events(events_file)


# ---------------------------------------------------------------------------
# TestVerifyChain
# ---------------------------------------------------------------------------

class TestVerifyChain:
    def _build_chain(self, count: int = 3) -> list[EventRecord]:
        state = ChainState()
        return [emit_hashed_event(_make_event(f"evt_{i}"), state) for i in range(count)]

    def test_valid_chain(self) -> None:
        records = self._build_chain(4)
        assert verify_chain(records) == "valid"

    def test_tampered_payload(self) -> None:
        records = self._build_chain(3)
        # Mutate the payload of the second record — hash no longer matches
        tampered = records[1]
        tampered.payload["extra_field"] = "injected"
        # Replace in list (dataclass is mutable)
        records[1] = tampered
        assert verify_chain(records) == "tampered"

    def test_tampered_hash(self) -> None:
        records = self._build_chain(3)
        # Overwrite the stored hash of record 0
        r0 = records[0]
        object.__setattr__(r0, "event_hash", "deadbeefdeadbeef") if hasattr(r0, "__dataclass_fields__") else None
        # Direct attribute assignment (dataclass is not frozen)
        records[0] = EventRecord(
            sequence=r0.sequence,
            event_type=r0.event_type,
            timestamp=r0.timestamp,
            payload=r0.payload,
            prev_hash=r0.prev_hash,
            event_hash="deadbeefdeadbeef",  # wrong hash
        )
        assert verify_chain(records) == "tampered"

    def test_broken_prev_hash(self) -> None:
        records = self._build_chain(3)
        r1 = records[1]
        records[1] = EventRecord(
            sequence=r1.sequence,
            event_type=r1.event_type,
            timestamp=r1.timestamp,
            payload=r1.payload,
            prev_hash="ffffffffffffffff",  # broken link
            event_hash=r1.event_hash,
        )
        assert verify_chain(records) == "tampered"

    def test_empty_chain(self) -> None:
        # An empty list has nothing to verify — returns "incomplete"
        assert verify_chain([]) == "incomplete"

    def test_all_legacy_events(self) -> None:
        """Chain of events with no hash fields → incomplete (nothing verified)."""
        records = [
            EventRecord(
                sequence=i,
                event_type="task_start",
                timestamp="t",
                payload={"task_id": f"t{i}"},
                prev_hash="",
                event_hash="",  # legacy — no hash
            )
            for i in range(3)
        ]
        assert verify_chain(records) == "incomplete"


# ---------------------------------------------------------------------------
# TestReplayRunState
# ---------------------------------------------------------------------------

class TestReplayRunState:
    def _record(self, event_type: str, **payload: Any) -> EventRecord:
        return EventRecord(
            sequence=0,
            event_type=event_type,
            timestamp="t",
            payload=payload,
            prev_hash="",
            event_hash="",
        )

    def test_reconstructs_task_statuses(self) -> None:
        events = [
            self._record("task_start", task_id="build"),
            self._record("task_complete", task_id="build", status="success", cost_usd=0.01),
            self._record("task_start", task_id="test"),
            self._record("task_complete", task_id="test", status="failed", cost_usd=0.0),
        ]
        state = replay_run_state(events)
        assert state["tasks"]["build"] == "success"
        assert state["tasks"]["test"] == "failed"
        assert "build" in state["completed_tasks"]
        assert "test" in state["completed_tasks"]

    def test_sums_cost(self) -> None:
        events = [
            self._record("task_complete", task_id="a", status="success", cost_usd=0.10),
            self._record("task_complete", task_id="b", status="success", cost_usd=0.25),
            self._record("task_complete", task_id="c", status="success", cost_usd=0.05),
        ]
        state = replay_run_state(events)
        assert abs(state["total_cost_usd"] - 0.40) < 1e-9

    def test_handles_unknown_events(self) -> None:
        events = [
            self._record("some_future_event", data="irrelevant"),
            self._record("task_complete", task_id="x", status="success", cost_usd=0.0),
        ]
        state = replay_run_state(events)
        assert state["tasks"]["x"] == "success"

    def test_task_skip_marks_skipped(self) -> None:
        events = [
            self._record("task_skip", task_id="deploy", reason="when condition false"),
        ]
        state = replay_run_state(events)
        assert state["tasks"]["deploy"] == "skipped"
        assert "deploy" in state["completed_tasks"]

    def test_empty_events(self) -> None:
        state = replay_run_state([])
        assert state["tasks"] == {}
        assert state["completed_tasks"] == set()
        assert state["total_cost_usd"] == 0.0

    def test_null_cost_ignored(self) -> None:
        events = [
            self._record("task_complete", task_id="a", status="success", cost_usd=None),
        ]
        state = replay_run_state(events)
        assert state["total_cost_usd"] == 0.0


# ---------------------------------------------------------------------------
# TestArtefactIntegrity
# ---------------------------------------------------------------------------

class TestComputeArtefactHash:
    def test_returns_hash_for_existing_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.log"
        f.write_text("hello world", encoding="utf-8")
        h = compute_artefact_hash(f)
        assert h is not None
        assert len(h) == 16  # first 16 hex chars of sha256

    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        assert compute_artefact_hash(tmp_path / "nonexistent.log") is None

    def test_deterministic(self, tmp_path: Path) -> None:
        f = tmp_path / "test.log"
        f.write_text("same content", encoding="utf-8")
        assert compute_artefact_hash(f) == compute_artefact_hash(f)

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.log"
        f2 = tmp_path / "b.log"
        f1.write_text("content A", encoding="utf-8")
        f2.write_text("content B", encoding="utf-8")
        assert compute_artefact_hash(f1) != compute_artefact_hash(f2)


class TestVerifyArtefactHashes:
    def _make_task_complete(self, task_id: str, log_hash: str | None = None,
                            result_hash: str | None = None) -> EventRecord:
        return EventRecord(
            sequence=0,
            event_type="task_complete",
            timestamp="2026-01-01T00:00:00",
            payload={
                "task_id": task_id,
                "status": "success",
                "log_hash": log_hash,
                "result_hash": result_hash,
            },
            prev_hash="0" * 16,
            event_hash="a" * 16,
        )

    def test_valid_artefacts(self, tmp_path: Path) -> None:
        log = tmp_path / "my-task.log"
        result = tmp_path / "my-task.result.json"
        log.write_text("task output", encoding="utf-8")
        result.write_text('{"status": "success"}', encoding="utf-8")

        log_h = compute_artefact_hash(log)
        result_h = compute_artefact_hash(result)
        events = [self._make_task_complete("my-task", log_h, result_h)]

        mismatches = verify_artefact_hashes(tmp_path, events)
        assert mismatches == []

    def test_tampered_log(self, tmp_path: Path) -> None:
        log = tmp_path / "my-task.log"
        log.write_text("original", encoding="utf-8")
        original_hash = compute_artefact_hash(log)

        # Tamper the file after recording hash
        log.write_text("tampered content", encoding="utf-8")

        events = [self._make_task_complete("my-task", log_hash=original_hash)]
        mismatches = verify_artefact_hashes(tmp_path, events)
        assert len(mismatches) == 1
        assert "my-task.log" in mismatches[0]

    def test_tampered_result(self, tmp_path: Path) -> None:
        result = tmp_path / "my-task.result.json"
        result.write_text('{"status":"success"}', encoding="utf-8")
        original_hash = compute_artefact_hash(result)

        result.write_text('{"status":"failed"}', encoding="utf-8")

        events = [self._make_task_complete("my-task", result_hash=original_hash)]
        mismatches = verify_artefact_hashes(tmp_path, events)
        assert len(mismatches) == 1
        assert "my-task.result.json" in mismatches[0]

    def test_missing_artefact_detected(self, tmp_path: Path) -> None:
        events = [self._make_task_complete("gone", log_hash="abcdef1234567890")]
        mismatches = verify_artefact_hashes(tmp_path, events)
        assert len(mismatches) == 1
        assert "gone.log" in mismatches[0]

    def test_no_hashes_in_event_skips_check(self, tmp_path: Path) -> None:
        """Events without hash fields (legacy) are silently skipped."""
        events = [self._make_task_complete("old-task")]
        mismatches = verify_artefact_hashes(tmp_path, events)
        assert mismatches == []

    def test_ignores_non_task_complete_events(self, tmp_path: Path) -> None:
        events = [EventRecord(
            sequence=0, event_type="task_start", timestamp="",
            payload={"task_id": "x"}, prev_hash="0" * 16, event_hash="a" * 16,
        )]
        assert verify_artefact_hashes(tmp_path, events) == []


# ---------------------------------------------------------------------------
# TestComputeEventHash (extended edge cases)
# ---------------------------------------------------------------------------

class TestComputeEventHashExtended:
    def test_hash_length_is_16(self) -> None:
        h = compute_event_hash({"event": "x"}, "0" * 16)
        assert len(h) == 16

    def test_empty_payload_produces_valid_hash(self) -> None:
        h = compute_event_hash({}, "0" * 16)
        assert isinstance(h, str)
        assert len(h) == 16

    def test_sort_keys_makes_order_irrelevant(self) -> None:
        prev = "0" * 16
        h1 = compute_event_hash({"a": 1, "b": 2}, prev)
        h2 = compute_event_hash({"b": 2, "a": 1}, prev)
        assert h1 == h2

    def test_unicode_payload(self) -> None:
        prev = "0" * 16
        h = compute_event_hash({"msg": "olá mundo 日本語 🎉"}, prev)
        assert len(h) == 16
        # Deterministic
        assert h == compute_event_hash({"msg": "olá mundo 日本語 🎉"}, prev)

    def test_large_payload(self) -> None:
        prev = "0" * 16
        big = {"data": "x" * 100_000}
        h = compute_event_hash(big, prev)
        assert len(h) == 16

    def test_nested_dict_payload(self) -> None:
        prev = "0" * 16
        h1 = compute_event_hash({"outer": {"inner": "value"}}, prev)
        h2 = compute_event_hash({"outer": {"inner": "other"}}, prev)
        assert h1 != h2

    def test_numeric_values(self) -> None:
        prev = "0" * 16
        h1 = compute_event_hash({"val": 1}, prev)
        h2 = compute_event_hash({"val": 1.0}, prev)
        # JSON serialises int 1 as "1" and float 1.0 as "1.0" — different hashes
        assert h1 != h2


# ---------------------------------------------------------------------------
# TestChainState (extended)
# ---------------------------------------------------------------------------

class TestChainStateExtended:
    def test_initial_values(self) -> None:
        state = ChainState()
        assert state.sequence == 0
        assert state.prev_hash == "0" * 16

    def test_custom_initial_values(self) -> None:
        state = ChainState(sequence=5, prev_hash="a" * 16)
        assert state.sequence == 5
        assert state.prev_hash == "a" * 16

    def test_advance_multiple_times(self) -> None:
        state = ChainState()
        hashes: list[str] = []
        for i in range(10):
            record = emit_hashed_event(_make_event(f"e{i}"), state)
            hashes.append(record.event_hash)
        assert state.sequence == 10
        # All hashes unique
        assert len(set(hashes)) == 10


# ---------------------------------------------------------------------------
# TestEmitHashedEvent (extended)
# ---------------------------------------------------------------------------

class TestEmitHashedEventExtended:
    def test_event_type_from_type_key(self) -> None:
        state = ChainState()
        record = emit_hashed_event({"type": "custom_type", "ts": "t1"}, state)
        assert record.event_type == "custom_type"

    def test_event_key_takes_priority_over_type(self) -> None:
        state = ChainState()
        record = emit_hashed_event({"event": "preferred", "type": "fallback", "ts": "t1"}, state)
        assert record.event_type == "preferred"

    def test_missing_event_and_type_gives_empty_string(self) -> None:
        state = ChainState()
        record = emit_hashed_event({"ts": "t1", "data": "x"}, state)
        assert record.event_type == ""

    def test_missing_ts_gives_empty_string(self) -> None:
        state = ChainState()
        record = emit_hashed_event({"event": "test"}, state)
        assert record.timestamp == ""

    def test_payload_stored_verbatim(self) -> None:
        state = ChainState()
        original = {"event": "run_start", "ts": "t1", "extra": [1, 2, 3]}
        record = emit_hashed_event(original, state)
        assert record.payload is original

    def test_hash_is_verifiable(self) -> None:
        state = ChainState()
        record = emit_hashed_event(_make_event("x"), state)
        expected = compute_event_hash(record.payload, record.prev_hash)
        assert record.event_hash == expected

    def test_write_to_file_and_read_back(self, tmp_path: Path) -> None:
        state = ChainState()
        record = emit_hashed_event(_make_event("task_start", task_id="build"), state)
        events_file = tmp_path / "events.jsonl"
        _write_jsonl(events_file, [record.to_dict()])
        replayed = replay_events(events_file)
        assert len(replayed) == 1
        assert replayed[0].event_hash == record.event_hash
        assert replayed[0].payload == record.payload


# ---------------------------------------------------------------------------
# TestReplayEvents (extended)
# ---------------------------------------------------------------------------

class TestReplayEventsExtended:
    def test_corrupt_line_skipped(self, tmp_path: Path) -> None:
        events_file = tmp_path / "events.jsonl"
        state = ChainState()
        valid = emit_hashed_event(_make_event("a"), state)
        with events_file.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(valid.to_dict()) + "\n")
            fh.write("THIS IS NOT JSON\n")
            valid2 = emit_hashed_event(_make_event("b"), state)
            fh.write(json.dumps(valid2.to_dict()) + "\n")
        replayed = replay_events(events_file)
        assert len(replayed) == 2

    def test_blank_lines_skipped(self, tmp_path: Path) -> None:
        events_file = tmp_path / "events.jsonl"
        state = ChainState()
        record = emit_hashed_event(_make_event("x"), state)
        with events_file.open("w", encoding="utf-8") as fh:
            fh.write("\n\n")
            fh.write(json.dumps(record.to_dict()) + "\n")
            fh.write("\n")
        replayed = replay_events(events_file)
        assert len(replayed) == 1

    def test_mixed_legacy_and_hashed(self, tmp_path: Path) -> None:
        events_file = tmp_path / "events.jsonl"
        state = ChainState()
        hashed = emit_hashed_event(_make_event("hashed_evt"), state)
        legacy = {"event": "legacy_evt", "ts": "t2"}
        with events_file.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(legacy) + "\n")
            fh.write(json.dumps(hashed.to_dict()) + "\n")
        replayed = replay_events(events_file)
        assert len(replayed) == 2
        assert replayed[0].event_hash == ""
        assert replayed[1].event_hash == hashed.event_hash

    def test_unicode_in_events(self, tmp_path: Path) -> None:
        events_file = tmp_path / "events.jsonl"
        state = ChainState()
        record = emit_hashed_event(
            {"event": "unicode_test", "ts": "t1", "msg": "café résumé naïve"},
            state,
        )
        _write_jsonl(events_file, [record.to_dict()])
        replayed = replay_events(events_file)
        assert replayed[0].payload["msg"] == "café résumé naïve"


# ---------------------------------------------------------------------------
# TestVerifyChain (extended)
# ---------------------------------------------------------------------------

class TestVerifyChainExtended:
    def _build_chain(self, count: int = 3) -> list[EventRecord]:
        state = ChainState()
        return [emit_hashed_event(_make_event(f"evt_{i}"), state) for i in range(count)]

    def test_single_valid_event(self) -> None:
        records = self._build_chain(1)
        assert verify_chain(records) == "valid"

    def test_large_chain(self) -> None:
        records = self._build_chain(50)
        assert verify_chain(records) == "valid"

    def test_swapped_events_detected(self) -> None:
        records = self._build_chain(4)
        records[1], records[2] = records[2], records[1]
        assert verify_chain(records) == "tampered"

    def test_removed_event_detected(self) -> None:
        records = self._build_chain(4)
        del records[1]
        assert verify_chain(records) == "tampered"

    def test_mixed_hashed_and_legacy_still_valid(self) -> None:
        state = ChainState()
        r0 = emit_hashed_event(_make_event("a"), state)
        legacy = EventRecord(
            sequence=1, event_type="legacy", timestamp="t",
            payload={}, prev_hash="", event_hash="",
        )
        r2 = emit_hashed_event(_make_event("b"), state)
        # Legacy events are skipped — chain continues from r0 to r2
        assert verify_chain([r0, legacy, r2]) == "valid"

    def test_duplicated_event_detected(self) -> None:
        records = self._build_chain(3)
        # Duplicate the first event at the end — prev_hash won't match
        records.append(records[0])
        assert verify_chain(records) == "tampered"


# ---------------------------------------------------------------------------
# TestEventRecord (to_dict round-trip)
# ---------------------------------------------------------------------------

class TestEventRecordSerialization:
    def test_to_dict_fields(self) -> None:
        record = EventRecord(
            sequence=7,
            event_type="task_complete",
            timestamp="2026-03-20T12:00:00",
            payload={"task_id": "build", "status": "success"},
            prev_hash="abcdef0123456789",
            event_hash="9876543210fedcba",
        )
        d = record.to_dict()
        assert d["seq"] == 7
        assert d["type"] == "task_complete"
        assert d["ts"] == "2026-03-20T12:00:00"
        assert d["payload"]["task_id"] == "build"
        assert d["prev_hash"] == "abcdef0123456789"
        assert d["hash"] == "9876543210fedcba"

    def test_to_dict_keys_match_replay_format(self) -> None:
        record = EventRecord(
            sequence=0, event_type="run_start", timestamp="t",
            payload={"plan_name": "demo"}, prev_hash="0" * 16, event_hash="a" * 16,
        )
        d = record.to_dict()
        assert set(d.keys()) == {"seq", "type", "ts", "payload", "prev_hash", "hash"}

    def test_round_trip_via_file(self, tmp_path: Path) -> None:
        state = ChainState()
        original = emit_hashed_event(
            {"event": "task_start", "ts": "t1", "task_id": "deploy"},
            state,
        )
        events_file = tmp_path / "events.jsonl"
        _write_jsonl(events_file, [original.to_dict()])
        replayed = replay_events(events_file)
        assert len(replayed) == 1
        r = replayed[0]
        assert r.sequence == original.sequence
        assert r.event_type == original.event_type
        assert r.timestamp == original.timestamp
        assert r.event_hash == original.event_hash
        assert r.prev_hash == original.prev_hash

    def test_empty_payload_round_trip(self, tmp_path: Path) -> None:
        state = ChainState()
        original = emit_hashed_event({"event": "ping", "ts": "t"}, state)
        events_file = tmp_path / "events.jsonl"
        _write_jsonl(events_file, [original.to_dict()])
        replayed = replay_events(events_file)
        assert verify_chain(replayed) == "valid"


# ---------------------------------------------------------------------------
# TestVerifyArtefactHashes (extended)
# ---------------------------------------------------------------------------

class TestVerifyArtefactHashesExtended:
    def _make_task_complete(self, task_id: str, log_hash: str | None = None,
                            result_hash: str | None = None) -> EventRecord:
        return EventRecord(
            sequence=0, event_type="task_complete",
            timestamp="2026-01-01T00:00:00",
            payload={
                "task_id": task_id, "status": "success",
                "log_hash": log_hash, "result_hash": result_hash,
            },
            prev_hash="0" * 16, event_hash="a" * 16,
        )

    def test_multiple_tasks_all_valid(self, tmp_path: Path) -> None:
        for tid in ("task-a", "task-b"):
            (tmp_path / f"{tid}.log").write_text(f"output for {tid}", encoding="utf-8")
        events = [
            self._make_task_complete(
                "task-a",
                log_hash=compute_artefact_hash(tmp_path / "task-a.log"),
            ),
            self._make_task_complete(
                "task-b",
                log_hash=compute_artefact_hash(tmp_path / "task-b.log"),
            ),
        ]
        assert verify_artefact_hashes(tmp_path, events) == []

    def test_multiple_tasks_one_tampered(self, tmp_path: Path) -> None:
        for tid in ("ok", "bad"):
            (tmp_path / f"{tid}.log").write_text(f"output-{tid}", encoding="utf-8")
        ok_hash = compute_artefact_hash(tmp_path / "ok.log")
        bad_hash = compute_artefact_hash(tmp_path / "bad.log")
        # Tamper "bad"
        (tmp_path / "bad.log").write_text("modified", encoding="utf-8")
        events = [
            self._make_task_complete("ok", log_hash=ok_hash),
            self._make_task_complete("bad", log_hash=bad_hash),
        ]
        mismatches = verify_artefact_hashes(tmp_path, events)
        assert len(mismatches) == 1
        assert "bad.log" in mismatches[0]

    def test_both_log_and_result_tampered(self, tmp_path: Path) -> None:
        log = tmp_path / "t.log"
        result = tmp_path / "t.result.json"
        log.write_text("log", encoding="utf-8")
        result.write_text("{}", encoding="utf-8")
        lh = compute_artefact_hash(log)
        rh = compute_artefact_hash(result)
        log.write_text("tampered log", encoding="utf-8")
        result.write_text('{"x":1}', encoding="utf-8")
        events = [self._make_task_complete("t", log_hash=lh, result_hash=rh)]
        mismatches = verify_artefact_hashes(tmp_path, events)
        assert len(mismatches) == 2


# ---------------------------------------------------------------------------
# TestReplayRunState (extended)
# ---------------------------------------------------------------------------

class TestReplayRunStateExtended:
    def _record(self, event_type: str, **payload: Any) -> EventRecord:
        return EventRecord(
            sequence=0, event_type=event_type, timestamp="t",
            payload=payload, prev_hash="", event_hash="",
        )

    def test_invalid_cost_ignored(self) -> None:
        events = [
            self._record("task_complete", task_id="a", status="success", cost_usd="not-a-number"),
        ]
        state = replay_run_state(events)
        assert state["total_cost_usd"] == 0.0

    def test_task_start_sets_running(self) -> None:
        events = [self._record("task_start", task_id="x")]
        state = replay_run_state(events)
        assert state["tasks"]["x"] == "running"
        assert "x" not in state["completed_tasks"]

    def test_task_complete_overwrites_running(self) -> None:
        events = [
            self._record("task_start", task_id="x"),
            self._record("task_complete", task_id="x", status="success", cost_usd=0.5),
        ]
        state = replay_run_state(events)
        assert state["tasks"]["x"] == "success"
        assert "x" in state["completed_tasks"]

    def test_all_terminal_statuses(self) -> None:
        statuses = ["success", "failed", "soft_failed", "skipped", "dry_run"]
        events = [
            self._record("task_complete", task_id=f"t{i}", status=s, cost_usd=0.0)
            for i, s in enumerate(statuses)
        ]
        state = replay_run_state(events)
        for i, s in enumerate(statuses):
            assert state["tasks"][f"t{i}"] == s
            assert f"t{i}" in state["completed_tasks"]

    def test_empty_task_id_ignored(self) -> None:
        events = [
            self._record("task_start", task_id=""),
            self._record("task_complete", task_id="", status="success"),
        ]
        state = replay_run_state(events)
        # Empty task_id is skipped (if task_id: check fails)
        assert state["tasks"] == {}


# ---------------------------------------------------------------------------
# Output Envelope — compute_output_hash / check_scope_violations / build_output_envelope
# ---------------------------------------------------------------------------


class TestOutputEnvelope:
    """Tests for output envelope functions in eventsource.py."""

    def test_compute_output_hash_deterministic(self) -> None:
        from maestro_cli.eventsource import compute_output_hash
        h1 = compute_output_hash("hello world")
        h2 = compute_output_hash("hello world")
        assert h1 == h2

    def test_compute_output_hash_empty_string(self) -> None:
        from maestro_cli.eventsource import compute_output_hash
        h = compute_output_hash("")
        assert isinstance(h, str)
        assert len(h) == 16

    def test_compute_output_hash_same_input_same_hash(self) -> None:
        from maestro_cli.eventsource import compute_output_hash
        text = "some output from the task\nwith multiple lines"
        assert compute_output_hash(text) == compute_output_hash(text)

    def test_compute_output_hash_different_input_different_hash(self) -> None:
        from maestro_cli.eventsource import compute_output_hash
        h1 = compute_output_hash("output A")
        h2 = compute_output_hash("output B")
        assert h1 != h2

    def test_compute_output_hash_returns_16_hex_chars(self) -> None:
        from maestro_cli.eventsource import compute_output_hash
        h = compute_output_hash("test output")
        assert len(h) == 16
        # Must be valid hex characters
        int(h, 16)

    def test_check_scope_violations_empty_scope(self) -> None:
        from maestro_cli.eventsource import check_scope_violations
        violations = check_scope_violations(["src/a.py", "lib/b.js"], [])
        assert violations == []

    def test_check_scope_violations_file_within_scope(self) -> None:
        from maestro_cli.eventsource import check_scope_violations
        violations = check_scope_violations(["src/a.py"], ["src/*.py"])
        assert violations == []

    def test_check_scope_violations_file_outside_scope(self) -> None:
        from maestro_cli.eventsource import check_scope_violations
        violations = check_scope_violations(["lib/b.js"], ["src/*.py"])
        assert violations == ["lib/b.js"]

    def test_check_scope_violations_glob_star_py(self) -> None:
        from maestro_cli.eventsource import check_scope_violations
        files = ["src/main.py", "src/utils.py", "docs/readme.md"]
        violations = check_scope_violations(files, ["src/*.py"])
        assert violations == ["docs/readme.md"]

    def test_check_scope_violations_nested_glob(self) -> None:
        from maestro_cli.eventsource import check_scope_violations
        files = ["src/cli/main.py", "src/models/spec.py", "tests/test_a.py"]
        violations = check_scope_violations(files, ["src/**/*.py"])
        assert violations == ["tests/test_a.py"]

    def test_check_scope_violations_multiple_scope_patterns(self) -> None:
        from maestro_cli.eventsource import check_scope_violations
        files = ["src/a.py", "tests/test_a.py", "docs/guide.md"]
        violations = check_scope_violations(files, ["src/*.py", "tests/*.py"])
        assert violations == ["docs/guide.md"]

    def test_check_scope_violations_backslash_normalization(self) -> None:
        from maestro_cli.eventsource import check_scope_violations
        # Windows-style paths should be normalized to forward slashes
        violations = check_scope_violations(["src\\utils\\a.py"], ["src/**/*.py"])
        assert violations == []

    def test_build_output_envelope_no_scope(self) -> None:
        from maestro_cli.eventsource import build_output_envelope
        envelope = build_output_envelope("some output", [], ["file.py"])
        assert envelope.scope_declared == []
        assert envelope.scope_violations == []
        assert envelope.scope_verified is True
        assert len(envelope.output_hash) == 16

    def test_build_output_envelope_with_scope_no_violations(self) -> None:
        from maestro_cli.eventsource import build_output_envelope
        envelope = build_output_envelope(
            "task output here",
            ["src/*.py"],
            ["src/main.py"],
        )
        assert envelope.scope_declared == ["src/*.py"]
        assert envelope.scope_violations == []
        assert envelope.scope_verified is True

    def test_build_output_envelope_with_scope_violations(self) -> None:
        from maestro_cli.eventsource import build_output_envelope
        envelope = build_output_envelope(
            "task output",
            ["src/*.py"],
            ["src/main.py", "lib/extra.js"],
        )
        assert envelope.scope_declared == ["src/*.py"]
        assert envelope.scope_violations == ["lib/extra.js"]
        assert envelope.scope_verified is False
