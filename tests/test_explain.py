from __future__ import annotations

import json
from pathlib import Path

from maestro_cli.explain import (
    explain_context_trajectory,
    format_context_trajectory,
    format_context_trajectory_json,
)


def _write_events(tmp_path: Path, events: list[dict[str, object]]) -> None:
    (tmp_path / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events),
        encoding="utf-8",
    )


class TestExplainContextTrajectory:
    def test_reconstructs_enriched_context_compression_entries(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {
                "event": "context_compression",
                "task_id": "review",
                "context_raw_tokens": 1000,
                "context_final_tokens": 450,
                "budget_tokens": 500,
                "entries": [
                    {
                        "upstream_id": "build",
                        "score": 1.25,
                        "keywords_matched": ["api", "schema"],
                        "hop_distance": 1,
                        "hop_decay_factor": 1.0,
                        "tokens_raw": 600,
                        "tokens_final": 300,
                        "trimmed": True,
                        "trim_reason": "budget_trim",
                    },
                    {
                        "upstream_id": "lint",
                        "score": 0.75,
                        "keywords_matched": ["style"],
                        "hop_distance": 2,
                        "hop_decay_factor": 0.8,
                        "tokens_raw": 400,
                        "tokens_final": 150,
                        "trimmed": False,
                        "trim_reason": "",
                    },
                ],
            }
        ])

        reports = explain_context_trajectory(tmp_path)

        assert len(reports) == 1
        report = reports[0]
        assert report.task_id == "review"
        assert report.total_tokens_raw == 1000
        assert report.total_tokens_final == 450
        assert report.budget_tokens == 500
        assert report.upstreams_evicted == 0
        assert [entry.upstream_id for entry in report.entries] == ["build", "lint"]
        assert report.entries[0].keywords_matched == ["api", "schema"]
        assert report.entries[1].hop_decay_factor == 0.8

    def test_reconstructs_fallback_trim_events(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {
                "event": "context_budget_trim",
                "task_id": "synthesize",
                "upstream_id": "scan",
                "original_tokens": 500,
                "trimmed_tokens": 200,
                "budget": 300,
            },
            {
                "event": "context_compression",
                "task_id": "synthesize",
                "context_raw_tokens": 800,
                "context_final_tokens": 300,
            },
        ])

        reports = explain_context_trajectory(tmp_path)

        assert len(reports) == 1
        report = reports[0]
        assert report.task_id == "synthesize"
        assert report.total_tokens_raw == 800
        assert report.total_tokens_final == 300
        assert report.budget_tokens == 300
        assert len(report.entries) == 1
        entry = report.entries[0]
        assert entry.upstream_id == "scan"
        assert entry.tokens_raw == 500
        assert entry.tokens_final == 200
        assert entry.trimmed is True
        assert entry.trim_reason == "budget_trim"

    def test_formatters_render_text_and_json(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {
                "event": "context_compression",
                "task_id": "review",
                "context_raw_tokens": 100,
                "context_final_tokens": 40,
                "entries": [
                    {
                        "upstream_id": "build",
                        "score": 2.0,
                        "keywords_matched": ["api"],
                        "hop_distance": 1,
                        "hop_decay_factor": 1.0,
                        "tokens_raw": 100,
                        "tokens_final": 40,
                        "trimmed": True,
                        "trim_reason": "budget_trim",
                    }
                ],
            }
        ])

        reports = explain_context_trajectory(tmp_path)
        text = format_context_trajectory(reports)
        payload = json.loads(format_context_trajectory_json(reports))

        assert "Task: review" in text
        assert "build" in text
        assert "budget_trim" in text
        assert payload[0]["task_id"] == "review"
        assert payload[0]["entries"][0]["keywords_matched"] == ["api"]
