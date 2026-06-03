from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .models import TaskContract, TaskResult, TaskSpec

_SECTION_HEADERS = {
    "[pre_command]",
    "[verify_command]",
    "[guard_command]",
    "[assert]",
    "[handoff_report]",
    "[judge]",
    "[judge_result]",
}
_SQL_TABLE_RE = re.compile(
    r"\b(?:CREATE|ALTER)\s+TABLE\s+(?:IF\s+(?:NOT\s+)?EXISTS\s+)?[`\"]?([A-Za-z0-9_.-]+)",
    re.IGNORECASE,
)
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$")


def normalize_task_contract(
    task: TaskSpec,
    log_path: Path,
    stdout_tail: str,
) -> TaskContract | None:
    contract_type = (task.contract_type or "").strip()
    if not contract_type:
        return None

    body = _extract_primary_output(log_path, stdout_tail)
    if contract_type == "sql-schema":
        return _normalize_sql_schema(task.id, body)
    if contract_type == "dependency-manifest":
        return _normalize_dependency_manifest(task.id, body)
    if contract_type == "conventions-doc":
        return _normalize_conventions_doc(task.id, body)
    if contract_type == "file-inventory":
        return _normalize_file_inventory(task.id, body)
    if contract_type == "api-schema":
        return _normalize_api_schema(task.id, body)
    if contract_type == "test-manifest":
        return _normalize_test_manifest(task.id, body)
    return _make_contract(task.id, contract_type, body, {}, "Generic contract output")


def build_contract_template_vars(
    task: TaskSpec,
    results: dict[str, TaskResult],
) -> dict[str, str]:
    variables: dict[str, str] = {}
    summary_lines: list[str] = []
    for producer_id in task.consumes_contracts:
        result = results.get(producer_id)
        contract = result.produced_contract if result is not None else None
        if contract is None:
            continue
        prefix = f"contract.{producer_id}"
        variables[f"{prefix}.producer"] = contract.producer_task_id
        variables[f"{prefix}.type"] = contract.contract_type
        variables[f"{prefix}.summary"] = contract.summary
        variables[f"{prefix}.body"] = contract.body
        variables[f"{prefix}.hash"] = contract.content_hash
        variables[f"{prefix}.metadata_json"] = json.dumps(
            contract.metadata,
            ensure_ascii=True,
            sort_keys=True,
        )
        summary_lines.append(
            f"- {producer_id} ({contract.contract_type}): {contract.summary}"
        )

    if summary_lines:
        variables["contracts_summary"] = "\n".join(summary_lines)
    return variables


def build_consistency_template_vars(
    task: TaskSpec,
    results: dict[str, TaskResult],
    group_members: dict[str, list[str]],
) -> dict[str, str]:
    variables: dict[str, str] = {}
    summary_lines: list[str] = []

    for group in task.reconcile_after:
        members = group_members.get(group, [])
        if not members:
            continue

        status_lines: list[str] = []
        member_summaries: list[str] = []
        contract_lines: list[str] = []
        for member_id in members:
            result = results.get(member_id)
            if result is None:
                status_lines.append(f"{member_id}: missing")
                continue
            status_lines.append(f"{member_id}: {result.status}")
            member_summary = _summarize_result(result)
            if member_summary:
                member_summaries.append(f"{member_id}: {member_summary}")
            if result.produced_contract is not None:
                contract_lines.append(
                    f"{member_id}: {result.produced_contract.summary}"
                )

        prefix = f"consistency.{group}"
        variables[f"{prefix}.tasks"] = "\n".join(members)
        variables[f"{prefix}.statuses"] = "\n".join(status_lines)
        variables[f"{prefix}.summaries"] = "\n\n".join(member_summaries) or "(none)"
        variables[f"{prefix}.contracts"] = "\n".join(contract_lines) or "(none)"
        summary_lines.append(f"- {group}: {'; '.join(status_lines)}")

    if summary_lines:
        variables["consistency_summary"] = "\n".join(summary_lines)
    return variables


def _extract_primary_output(log_path: Path, stdout_tail: str) -> str:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return stdout_tail.strip()

    lines = text.splitlines()
    body_started = False
    body: list[str] = []
    for line in lines:
        if not body_started:
            if line == "":
                body_started = True
            continue
        if line in _SECTION_HEADERS or line.startswith("[stderr] "):
            break
        if line.startswith("status=") or line.startswith("message="):
            break
        body.append(line)

    extracted = "\n".join(body).strip()
    return extracted or stdout_tail.strip()


def _normalize_sql_schema(producer_task_id: str, body: str) -> TaskContract:
    cleaned_lines = [
        line.strip()
        for line in body.splitlines()
        if line.strip() and not line.strip().startswith("--")
    ]
    normalized_body = "\n".join(cleaned_lines)
    statements = [
        statement.strip()
        for statement in normalized_body.split(";")
        if statement.strip()
    ]
    tables = sorted({match.group(1) for match in _SQL_TABLE_RE.finditer(normalized_body)})
    summary = f"SQL schema with {len(statements)} statement(s)"
    if tables:
        preview = ", ".join(tables[:5])
        if len(tables) > 5:
            preview += ", ..."
        summary += f"; tables: {preview}"
    return _make_contract(
        producer_task_id,
        "sql-schema",
        normalized_body,
        {
            "statement_count": len(statements),
            "tables": tables,
        },
        summary,
    )


def _normalize_dependency_manifest(producer_task_id: str, body: str) -> TaskContract:
    packages: list[str] = []
    metadata: dict[str, object] = {}
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        sections = (
            "require",
            "require-dev",
            "dependencies",
            "devDependencies",
            "peerDependencies",
            "optionalDependencies",
        )
        found: set[str] = set()
        for section_name in sections:
            section = payload.get(section_name)
            if isinstance(section, dict):
                found.update(str(key) for key in section)
        packages = sorted(found)
        metadata = {"package_count": len(packages), "packages": packages}
        normalized_body = json.dumps({"packages": packages}, ensure_ascii=True, sort_keys=True)
        summary = f"Dependency manifest with {len(packages)} package(s)"
        return _make_contract(
            producer_task_id,
            "dependency-manifest",
            normalized_body,
            metadata,
            summary,
        )

    return _make_generic_contract(
        producer_task_id,
        "dependency-manifest",
        body,
    )


def _normalize_conventions_doc(producer_task_id: str, body: str) -> TaskContract:
    headings = [
        match.group(1).strip()
        for line in body.splitlines()
        for match in [_HEADING_RE.match(line)]
        if match is not None
    ]
    metadata = {
        "heading_count": len(headings),
        "headings": headings,
    }
    summary = f"Conventions document with {len(headings)} heading(s)"
    if headings:
        summary += f"; first heading: {headings[0]}"
    return _make_contract(
        producer_task_id,
        "conventions-doc",
        body.strip(),
        metadata,
        summary,
    )


def _normalize_file_inventory(producer_task_id: str, body: str) -> TaskContract:
    files: list[str] = []
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, list):
        files = [str(item).replace("\\", "/") for item in payload if str(item).strip()]
    elif isinstance(payload, dict) and isinstance(payload.get("files"), list):
        files = [
            str(item).replace("\\", "/")
            for item in payload["files"]
            if str(item).strip()
        ]
    else:
        files = [
            line.strip().replace("\\", "/")
            for line in body.splitlines()
            if line.strip()
        ]

    normalized = sorted(dict.fromkeys(files))
    return _make_contract(
        producer_task_id,
        "file-inventory",
        "\n".join(normalized),
        {
            "file_count": len(normalized),
            "files": normalized,
        },
        f"File inventory with {len(normalized)} file(s)",
    )


def _normalize_api_schema(producer_task_id: str, body: str) -> TaskContract:
    """Parse OpenAPI/Swagger-style JSON body.

    Extracts: schema version (openapi/swagger), path count, schema/component names.
    Falls back to generic contract for non-parseable input.
    """
    payload: dict[str, object] | None = None
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            payload = parsed
    except json.JSONDecodeError:
        pass

    if payload is None:
        return _make_generic_contract(producer_task_id, "api-schema", body)

    paths_obj = payload.get("paths")
    path_keys: list[str] = sorted(paths_obj.keys()) if isinstance(paths_obj, dict) else []

    # OpenAPI 3.0: schemas in components.schemas; Swagger 2.0: top-level definitions
    components_raw = payload.get("components")
    if isinstance(components_raw, dict):
        schemas_dict = components_raw.get("schemas") or {}
        schema_names: list[str] = sorted(schemas_dict.keys()) if isinstance(schemas_dict, dict) else []
    else:
        definitions = payload.get("definitions") or {}
        schema_names = sorted(definitions.keys()) if isinstance(definitions, dict) else []

    openapi_version: str = (
        str(payload.get("openapi") or payload.get("swagger") or "unknown")
    )

    summary = f"API schema (OpenAPI {openapi_version}) with {len(path_keys)} path(s)"
    if schema_names:
        preview = ", ".join(schema_names[:5])
        if len(schema_names) > 5:
            preview += ", ..."
        summary += f"; schemas: {preview}"

    return _make_contract(
        producer_task_id,
        "api-schema",
        json.dumps({"openapi": openapi_version, "paths": path_keys, "schemas": schema_names},
                   ensure_ascii=True, sort_keys=True),
        {
            "openapi_version": openapi_version,
            "path_count": len(path_keys),
            "paths": path_keys,
            "schema_count": len(schema_names),
            "schemas": schema_names,
        },
        summary,
    )


def _normalize_test_manifest(producer_task_id: str, body: str) -> TaskContract:
    """Parse a test run report.

    Accepts pytest/jest JSON output or plain text summary lines.
    Extracts: passed, failed, skipped, total counts.
    Falls back to generic contract for unrecognized formats.
    """
    payload: dict[str, object] | None = None
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            payload = parsed
    except json.JSONDecodeError:
        pass

    passed = 0
    failed = 0
    skipped = 0
    total = 0

    if isinstance(payload, dict):
        # pytest --json / pytest-json-report format
        passed = int(str(payload.get("passed", payload.get("numPassedTests", 0)) or 0))
        failed = int(str(payload.get("failed", payload.get("numFailedTests", 0)) or 0))
        skipped = int(str(payload.get("skipped", payload.get("numPendingTests", 0)) or 0))
        total = int(str(
            payload.get("total", payload.get("numTotalTests", passed + failed + skipped)) or 0
        ))
        if total == 0:
            total = passed + failed + skipped
    else:
        # Parse plain text like "10 passed, 2 failed" or "passed: 10, failed: 2"
        passed_m = re.search(r"(\d+)\s+passed", body, re.IGNORECASE)
        failed_m = re.search(r"(\d+)\s+(?:failed|error)", body, re.IGNORECASE)
        skipped_m = re.search(r"(\d+)\s+(?:skipped|pending|deselected)", body, re.IGNORECASE)
        passed = int(passed_m.group(1)) if passed_m else 0
        failed = int(failed_m.group(1)) if failed_m else 0
        skipped = int(skipped_m.group(1)) if skipped_m else 0
        total = passed + failed + skipped

    if total == 0 and not passed and not failed:
        return _make_generic_contract(producer_task_id, "test-manifest", body)

    summary = f"Test manifest: {passed} passed / {failed} failed"
    if skipped:
        summary += f" / {skipped} skipped"
    summary += f" (total: {total})"

    return _make_contract(
        producer_task_id,
        "test-manifest",
        json.dumps({"passed": passed, "failed": failed, "skipped": skipped, "total": total},
                   ensure_ascii=True, sort_keys=True),
        {
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "total": total,
        },
        summary,
    )


def _make_generic_contract(
    producer_task_id: str,
    contract_type: str,
    body: str,
) -> TaskContract:
    lines = [line for line in body.splitlines() if line.strip()]
    return _make_contract(
        producer_task_id,
        contract_type,
        body.strip(),
        {
            "line_count": len(lines),
            "char_count": len(body.strip()),
        },
        f"{contract_type} contract with {len(lines)} non-empty line(s)",
    )


def _make_contract(
    producer_task_id: str,
    contract_type: str,
    body: str,
    metadata: dict[str, object],
    summary: str,
) -> TaskContract:
    normalized_body = body.strip()
    content_hash = hashlib.sha256(
        normalized_body.encode("utf-8", errors="replace")
    ).hexdigest()
    return TaskContract(
        producer_task_id=producer_task_id,
        contract_type=contract_type,
        summary=summary,
        body=normalized_body,
        content_hash=content_hash,
        metadata=dict(metadata),
    )


def _summarize_result(result: TaskResult) -> str:
    if result.produced_contract is not None:
        return result.produced_contract.summary
    structured = result.structured_context
    if structured is not None and structured.summary:
        return structured.summary
    tail = result.stdout_tail.strip()
    if not tail:
        return ""
    single_line = " ".join(tail.splitlines())
    return single_line[:240]
