from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from fnmatch import fnmatchcase
import json
from pathlib import Path
from typing import Any

import yaml

from .diff import load_run_manifest
from .errors import E001, E012, E018, E020, PlanValidationError
from .loader import _to_judge_spec, _to_str_list
from .models import CLAUDE_MODELS, JUDGE_PRESETS, JudgeResult, JudgeSpec

_DEFAULT_TASK_PATTERNS = ["*"]
_DEFAULT_TIMEOUT_SEC = 45
_GLOB_CHARS = set("*?[]")


@dataclass
class EvalResult:
    task_id: str
    judge_result: JudgeResult
    passed: bool


@dataclass
class DimensionResult:
    """Result for a single evaluation dimension."""
    name: str
    results: list[EvalResult]
    skipped: list[str]

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(
            1 for r in self.results
            if not r.passed and r.judge_result.verdict != "error"
        )

    @property
    def errors(self) -> int:
        return sum(1 for r in self.results if r.judge_result.verdict == "error")

    @property
    def overall_pass(self) -> bool:
        return self.failed == 0 and self.errors == 0

    @property
    def avg_score(self) -> float:
        scores = [r.judge_result.overall_score for r in self.results
                  if r.judge_result.overall_score is not None]
        return sum(scores) / len(scores) if scores else 0.0


@dataclass
class EvalSuiteResult:
    name: str
    run_path: Path
    results: list[EvalResult]
    skipped: list[str]
    dimensions: list[DimensionResult] | None = None

    @property
    def passed(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def failed(self) -> int:
        return sum(
            1
            for result in self.results
            if not result.passed and result.judge_result.verdict != "error"
        )

    @property
    def errors(self) -> int:
        return sum(1 for result in self.results if result.judge_result.verdict == "error")

    @property
    def overall_pass(self) -> bool:
        if self.dimensions:
            return all(d.overall_pass for d in self.dimensions)
        return self.failed == 0 and self.errors == 0


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _parse_iso(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _extract_duration_sec(task_payload: dict[str, Any]) -> float:
    explicit = _coerce_float(task_payload.get("duration_sec"))
    if explicit is not None:
        return max(0.0, explicit)

    started_at = _parse_iso(task_payload.get("started_at"))
    finished_at = _parse_iso(task_payload.get("finished_at"))
    if started_at is None or finished_at is None:
        return 0.0
    return max(0.0, (finished_at - started_at).total_seconds())


def _build_judge_spec(judge_data: dict[str, Any], field_name: str = "judge") -> JudgeSpec:
    if not isinstance(judge_data, dict):
        raise PlanValidationError(f"{field_name} must be an object", code=E020)
    judge = _to_judge_spec(judge_data, field_name)
    if judge is None:
        raise PlanValidationError(f"{field_name} must be provided", code=E020)
    return judge


def _collect_judge_warnings(judge: JudgeSpec, field_name: str, warnings: list[str]) -> None:
    if judge.model not in CLAUDE_MODELS:
        warnings.append(
            f"{field_name}.model '{judge.model}' is not in known Claude models: "
            f"{sorted(CLAUDE_MODELS)}"
        )
    if judge.method == "g_eval" and judge.model == "haiku":
        warnings.append(
            f"{field_name}: judge.method 'g_eval' works best with a more capable model "
            "than 'haiku' (sonnet recommended)."
        )
    if judge.preset is not None and judge.preset not in JUDGE_PRESETS:
        warnings.append(
            f"{field_name}.preset '{judge.preset}' is not recognized. "
            f"Known presets: {sorted(JUDGE_PRESETS)}"
        )


def _normalize_patterns(value: Any, field_name: str) -> list[str]:
    patterns = [item for item in _to_str_list(value, field_name) if item.strip()]
    return patterns


def _is_glob_pattern(value: str) -> bool:
    return any(ch in value for ch in _GLOB_CHARS)


def _matches_patterns(task_id: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        if fnmatchcase(task_id, pattern):
            return True
    return False


def load_eval_spec(eval_path: Path) -> dict[str, Any]:
    eval_path = Path(eval_path)
    if not eval_path.exists() or not eval_path.is_file():
        raise PlanValidationError(f"Eval file not found: {eval_path}", code=E001)

    try:
        payload = yaml.safe_load(eval_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PlanValidationError(f"Invalid YAML: {exc}") from exc

    if not isinstance(payload, dict):
        raise PlanValidationError("Eval root must be an object", code=E018)

    name = str(payload.get("name", "")).strip()
    if not name:
        raise PlanValidationError("Eval name must be a non-empty string", code=E001)

    tasks = _normalize_patterns(payload.get("tasks", _DEFAULT_TASK_PATTERNS), "tasks")
    if not tasks:
        tasks = list(_DEFAULT_TASK_PATTERNS)

    exclude = _normalize_patterns(payload.get("exclude"), "exclude")

    judge_raw = payload.get("judge")
    if not isinstance(judge_raw, dict):
        raise PlanValidationError("judge must be an object", code=E020)
    judge = _build_judge_spec(judge_raw)

    overrides_raw = payload.get("overrides", {})
    if overrides_raw is None:
        overrides_raw = {}
    if not isinstance(overrides_raw, dict):
        raise PlanValidationError("overrides must be an object", code=E018)

    override_specs: dict[str, JudgeSpec] = {}
    for raw_task_id, override_data in overrides_raw.items():
        task_id = str(raw_task_id).strip()
        if not task_id:
            raise PlanValidationError("overrides keys must be non-empty task IDs", code=E018)
        if not isinstance(override_data, dict):
            raise PlanValidationError(f"overrides.{task_id} must be an object", code=E018)
        merged_data = dict(judge_raw)
        merged_data.update(override_data)
        override_specs[task_id] = _build_judge_spec(merged_data, f"overrides.{task_id}")

    timeout_raw = payload.get("timeout_sec")
    if timeout_raw is None:
        timeout_sec = _DEFAULT_TIMEOUT_SEC
    else:
        try:
            timeout_sec = int(timeout_raw)
        except (TypeError, ValueError) as exc:
            raise PlanValidationError("timeout_sec must be an integer", code=E018) from exc
        if timeout_sec < 1:
            raise PlanValidationError("timeout_sec must be >= 1", code=E012)

    validation_warnings: list[str] = []
    if timeout_raw is None:
        validation_warnings.append(
            f"timeout_sec was not provided; using default {_DEFAULT_TIMEOUT_SEC}s."
        )
    _collect_judge_warnings(judge, "judge", validation_warnings)
    for task_id, override_spec in override_specs.items():
        _collect_judge_warnings(override_spec, f"overrides.{task_id}", validation_warnings)

    # Multi-dimensional eval: optional dimensions block
    dimensions_raw = payload.get("dimensions")
    dimension_specs: list[dict[str, Any]] = []
    if dimensions_raw is not None:
        if not isinstance(dimensions_raw, list):
            raise PlanValidationError("dimensions must be a list", code=E018)
        for idx, dim_data in enumerate(dimensions_raw):
            if not isinstance(dim_data, dict):
                raise PlanValidationError(f"dimensions[{idx}] must be an object", code=E018)
            dim_name = str(dim_data.get("name", "")).strip()
            if not dim_name:
                raise PlanValidationError(f"dimensions[{idx}].name is required", code=E001)
            dim_judge_raw = dim_data.get("judge")
            if dim_judge_raw is not None:
                if not isinstance(dim_judge_raw, dict):
                    raise PlanValidationError(
                        f"dimensions[{idx}].judge must be an object", code=E020,
                    )
                dim_judge = _build_judge_spec(dim_judge_raw, f"dimensions[{idx}].judge")
            else:
                dim_judge = judge  # inherit from top-level
            dim_tasks = _normalize_patterns(
                dim_data.get("tasks", _DEFAULT_TASK_PATTERNS),
                f"dimensions[{idx}].tasks",
            )
            dim_exclude = _normalize_patterns(
                dim_data.get("exclude"),
                f"dimensions[{idx}].exclude",
            )
            _collect_judge_warnings(dim_judge, f"dimensions[{idx}].judge", validation_warnings)
            dimension_specs.append({
                "name": dim_name,
                "tasks": dim_tasks or list(_DEFAULT_TASK_PATTERNS),
                "exclude": dim_exclude,
                "judge": dim_judge,
            })

    return {
        "name": name,
        "tasks": tasks,
        "exclude": exclude,
        "judge": judge,
        "overrides": override_specs,
        "timeout_sec": timeout_sec,
        "validation_warnings": validation_warnings,
        "dimensions": dimension_specs,
    }


def _resolve_tasks(spec: dict[str, Any], available_tasks: list[str]) -> list[str]:
    include_patterns = _normalize_patterns(spec.get("tasks", _DEFAULT_TASK_PATTERNS), "tasks")
    if not include_patterns:
        include_patterns = list(_DEFAULT_TASK_PATTERNS)
    exclude_patterns = _normalize_patterns(spec.get("exclude"), "exclude")

    selected: list[str] = []
    for task_id in available_tasks:
        if not _matches_patterns(task_id, include_patterns):
            continue
        if _matches_patterns(task_id, exclude_patterns):
            continue
        selected.append(task_id)
    return selected


def _missing_requested_task_ids(spec: dict[str, Any], available_tasks: list[str]) -> list[str]:
    available = set(available_tasks)
    missing: list[str] = []
    for pattern in _normalize_patterns(spec.get("tasks", _DEFAULT_TASK_PATTERNS), "tasks"):
        if _is_glob_pattern(pattern):
            continue
        if pattern not in available and pattern not in missing:
            missing.append(pattern)
    return missing


def run_eval(eval_path: Path, run_path: Path) -> EvalSuiteResult:
    spec = load_eval_spec(eval_path)
    run_path = Path(run_path)
    manifest = load_run_manifest(run_path)

    raw_task_results = manifest.get("task_results")
    if not isinstance(raw_task_results, dict):
        raw_task_results = {}

    available_tasks = [task_id for task_id in raw_task_results if isinstance(task_id, str)]
    selected_tasks = _resolve_tasks(spec, available_tasks)
    skipped = _missing_requested_task_ids(spec, available_tasks)

    # Lazy import to avoid module cycles.
    from .runners import _run_judge_quorum

    base_judge: JudgeSpec = spec["judge"]
    overrides: dict[str, JudgeSpec] = spec["overrides"]
    timeout_sec = int(spec["timeout_sec"])

    results: list[EvalResult] = []
    for task_id in selected_tasks:
        task_payload = raw_task_results.get(task_id)
        if not isinstance(task_payload, dict):
            skipped.append(task_id)
            continue

        stdout_tail_raw = task_payload.get("stdout_tail", "")
        stdout_tail = (
            stdout_tail_raw if isinstance(stdout_tail_raw, str) else str(stdout_tail_raw or "")
        )
        cost_usd = _coerce_float(task_payload.get("cost_usd"))
        duration_sec = _extract_duration_sec(task_payload)

        judge_spec = overrides.get(task_id, base_judge)
        try:
            judge_result = _run_judge_quorum(
                task_id=task_id,
                judge=judge_spec,
                stdout_tail=stdout_tail,
                workdir=run_path,
                cost_usd=cost_usd,
                duration_sec=duration_sec,
                timeout_sec=timeout_sec,
            )
        except Exception as exc:
            judge_result = JudgeResult(
                verdict="error",
                overall_score=0.0,
                reasoning=f"Judge evaluation error for task '{task_id}': {exc}",
            )

        results.append(
            EvalResult(
                task_id=task_id,
                judge_result=judge_result,
                passed=judge_result.verdict == "pass",
            )
        )

    # Multi-dimensional eval
    dimension_results: list[DimensionResult] | None = None
    dimension_specs = spec.get("dimensions", [])
    if dimension_specs:
        dimension_results = []
        for dim_spec in dimension_specs:
            dim_tasks_selected = _resolve_tasks(dim_spec, available_tasks)
            dim_skipped = _missing_requested_task_ids(dim_spec, available_tasks)
            dim_judge: JudgeSpec = dim_spec["judge"]
            dim_results: list[EvalResult] = []
            for task_id in dim_tasks_selected:
                task_payload = raw_task_results.get(task_id)
                if not isinstance(task_payload, dict):
                    dim_skipped.append(task_id)
                    continue

                stdout_tail_raw = task_payload.get("stdout_tail", "")
                stdout_tail = (
                    stdout_tail_raw if isinstance(stdout_tail_raw, str)
                    else str(stdout_tail_raw or "")
                )
                cost_usd = _coerce_float(task_payload.get("cost_usd"))
                duration_sec = _extract_duration_sec(task_payload)

                try:
                    judge_result = _run_judge_quorum(
                        task_id=task_id,
                        judge=dim_judge,
                        stdout_tail=stdout_tail,
                        workdir=run_path,
                        cost_usd=cost_usd,
                        duration_sec=duration_sec,
                        timeout_sec=timeout_sec,
                    )
                except Exception as exc:
                    judge_result = JudgeResult(
                        verdict="error",
                        overall_score=0.0,
                        reasoning=f"Judge error for '{task_id}' in dimension "
                                  f"'{dim_spec['name']}': {exc}",
                    )

                dim_results.append(EvalResult(
                    task_id=task_id,
                    judge_result=judge_result,
                    passed=judge_result.verdict == "pass",
                ))

            dimension_results.append(DimensionResult(
                name=str(dim_spec["name"]),
                results=dim_results,
                skipped=dim_skipped,
            ))

    return EvalSuiteResult(
        name=str(spec["name"]),
        run_path=run_path,
        results=results,
        skipped=skipped,
        dimensions=dimension_results,
    )


def format_eval(suite: EvalSuiteResult) -> str:
    lines = [
        f"Suite: {suite.name}",
        f"Run: {suite.run_path}",
        (
            "Summary: "
            f"passed={suite.passed}, failed={suite.failed}, errors={suite.errors}, "
            f"skipped={len(suite.skipped)}, overall={'PASS' if suite.overall_pass else 'FAIL'}"
        ),
    ]

    if not suite.results:
        lines.append("No evaluated tasks.")
        if suite.skipped:
            lines.append(f"Skipped: {', '.join(sorted(set(suite.skipped)))}")
        return "\n".join(lines)

    header = ("TASK", "VERDICT", "SCORE", "PASSED")
    rows = [
        (
            result.task_id,
            result.judge_result.verdict,
            f"{result.judge_result.overall_score:.2f}",
            "yes" if result.passed else "no",
        )
        for result in suite.results
    ]

    task_w = max(len(header[0]), *(len(row[0]) for row in rows))
    verdict_w = max(len(header[1]), *(len(row[1]) for row in rows))
    score_w = max(len(header[2]), *(len(row[2]) for row in rows))
    passed_w = max(len(header[3]), *(len(row[3]) for row in rows))

    lines.append("")
    lines.append(
        f"{header[0]:<{task_w}}  {header[1]:<{verdict_w}}  "
        f"{header[2]:>{score_w}}  {header[3]:<{passed_w}}"
    )
    lines.append(
        f"{'-' * task_w}  {'-' * verdict_w}  {'-' * score_w}  {'-' * passed_w}"
    )
    for row in rows:
        lines.append(
            f"{row[0]:<{task_w}}  {row[1]:<{verdict_w}}  {row[2]:>{score_w}}  {row[3]:<{passed_w}}"
        )

    if suite.skipped:
        lines.append("")
        lines.append(f"Skipped: {', '.join(sorted(set(suite.skipped)))}")

    # Multi-dimensional breakdown
    if suite.dimensions:
        lines.append("")
        lines.append("Dimensions:")
        for dim in suite.dimensions:
            status = "PASS" if dim.overall_pass else "FAIL"
            lines.append(
                f"  {dim.name}: {status} "
                f"(passed={dim.passed}, failed={dim.failed}, "
                f"errors={dim.errors}, avg_score={dim.avg_score:.2f})"
            )

    return "\n".join(lines)


def format_eval_json(suite: EvalSuiteResult) -> str:
    payload = {
        "name": suite.name,
        "run_path": str(suite.run_path),
        "results": [
            {
                "task_id": result.task_id,
                "judge_result": result.judge_result.to_dict(),
                "passed": result.passed,
            }
            for result in suite.results
        ],
        "skipped": list(suite.skipped),
        "passed": suite.passed,
        "failed": suite.failed,
        "errors": suite.errors,
        "overall_pass": suite.overall_pass,
    }
    if suite.dimensions:
        payload["dimensions"] = [
            {
                "name": dim.name,
                "results": [
                    {
                        "task_id": r.task_id,
                        "judge_result": r.judge_result.to_dict(),
                        "passed": r.passed,
                    }
                    for r in dim.results
                ],
                "skipped": list(dim.skipped),
                "passed": dim.passed,
                "failed": dim.failed,
                "errors": dim.errors,
                "avg_score": round(dim.avg_score, 4),
                "overall_pass": dim.overall_pass,
            }
            for dim in suite.dimensions
        ]
    return json.dumps(payload, ensure_ascii=True, indent=2)


__all__ = [
    "DimensionResult",
    "EvalResult",
    "EvalSuiteResult",
    "load_eval_spec",
    "_build_judge_spec",
    "_resolve_tasks",
    "run_eval",
    "format_eval",
    "format_eval_json",
]
