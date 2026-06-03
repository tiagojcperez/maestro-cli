from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from maestro_cli.contracts import (
    build_consistency_template_vars,
    build_contract_template_vars,
    normalize_task_contract,
)
from maestro_cli.models import TaskContract, TaskResult, TaskSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task(task_id: str, **kwargs) -> TaskSpec:  # type: ignore[type-arg]
    return TaskSpec(id=task_id, **kwargs)


def _make_result(
    task_id: str,
    contract: TaskContract | None = None,
    stdout_tail: str = "",
) -> TaskResult:
    return TaskResult(
        task_id=task_id,
        status="success",
        stdout_tail=stdout_tail,
        produced_contract=contract,
    )


def _make_contract(
    producer_id: str,
    contract_type: str = "conventions-doc",
    body: str = "# Heading",
    summary: str = "test summary",
) -> TaskContract:
    h = hashlib.sha256(body.encode()).hexdigest()
    return TaskContract(
        producer_task_id=producer_id,
        contract_type=contract_type,
        summary=summary,
        body=body,
        content_hash=h,
    )


def _write_log(path: Path, body: str) -> None:
    """Write a log file with header + blank line + body."""
    path.write_text(f"header\n\n{body}\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# normalize_task_contract — contract type dispatch
# ---------------------------------------------------------------------------


class TestNormalizeTaskContract:
    def test_no_contract_type_returns_none(self) -> None:
        task = _make_task("t1")
        result = normalize_task_contract(task, Path("/nonexistent"), "")
        assert result is None

    def test_empty_contract_type_returns_none(self) -> None:
        task = _make_task("t1", contract_type="  ")
        result = normalize_task_contract(task, Path("/nonexistent"), "")
        assert result is None

    def test_sql_schema_contract_type(self, tmp_path: Path) -> None:
        log = tmp_path / "t.log"
        _write_log(log, "CREATE TABLE users (id INT);")
        task = _make_task("t1", contract_type="sql-schema")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert contract.contract_type == "sql-schema"
        assert "users" in contract.metadata["tables"]

    def test_dependency_manifest_contract_type(self, tmp_path: Path) -> None:
        payload = json.dumps({"require": {"vendor/pkg": "^1.0"}})
        log = tmp_path / "t.log"
        _write_log(log, payload)
        task = _make_task("t1", contract_type="dependency-manifest")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert contract.contract_type == "dependency-manifest"
        assert "vendor/pkg" in contract.metadata["packages"]

    def test_conventions_doc_contract_type(self, tmp_path: Path) -> None:
        log = tmp_path / "t.log"
        _write_log(log, "# Naming Conventions\n\nSome text.")
        task = _make_task("t1", contract_type="conventions-doc")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert contract.contract_type == "conventions-doc"
        assert "Naming Conventions" in contract.metadata["headings"]

    def test_file_inventory_contract_type(self, tmp_path: Path) -> None:
        payload = json.dumps(["src/a.py", "src/b.py"])
        log = tmp_path / "t.log"
        _write_log(log, payload)
        task = _make_task("t1", contract_type="file-inventory")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert contract.contract_type == "file-inventory"
        assert "src/a.py" in contract.metadata["files"]

    def test_unknown_contract_type_returns_generic(self) -> None:
        task = _make_task("t1", contract_type="custom-type")
        contract = normalize_task_contract(task, Path("/nonexistent"), "some output")
        assert contract is not None
        assert contract.contract_type == "custom-type"

    def test_log_not_found_falls_back_to_stdout_tail(self) -> None:
        task = _make_task("t1", contract_type="conventions-doc")
        contract = normalize_task_contract(task, Path("/nonexistent/file.log"), "# My Heading")
        assert contract is not None
        assert "My Heading" in contract.metadata["headings"]

    def test_producer_task_id_set_correctly(self, tmp_path: Path) -> None:
        log = tmp_path / "t.log"
        _write_log(log, "# Doc")
        task = _make_task("my-producer", contract_type="conventions-doc")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert contract.producer_task_id == "my-producer"

    def test_content_hash_is_sha256(self, tmp_path: Path) -> None:
        log = tmp_path / "t.log"
        _write_log(log, "# Doc")
        task = _make_task("t1", contract_type="conventions-doc")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert len(contract.content_hash) == 64  # SHA-256 hex digest


# ---------------------------------------------------------------------------
# _extract_primary_output (tested via normalize_task_contract)
# ---------------------------------------------------------------------------


class TestExtractPrimaryOutput:
    def test_body_starts_after_blank_line(self, tmp_path: Path) -> None:
        log = tmp_path / "t.log"
        log.write_text("## pre-header\n\n# Body Heading\nbody text\n", encoding="utf-8")
        task = _make_task("t1", contract_type="conventions-doc")
        contract = normalize_task_contract(task, log, "fallback")
        assert contract is not None
        assert "Body Heading" in contract.metadata["headings"]

    def test_no_blank_line_falls_back_to_stdout_tail(self, tmp_path: Path) -> None:
        log = tmp_path / "t.log"
        log.write_text("no-blank-line\ncontinued\n", encoding="utf-8")
        task = _make_task("t1", contract_type="conventions-doc")
        contract = normalize_task_contract(task, log, "# Fallback Heading")
        assert contract is not None
        assert "Fallback Heading" in contract.metadata["headings"]

    def test_section_header_stops_body_collection(self, tmp_path: Path) -> None:
        log = tmp_path / "t.log"
        log.write_text("header\n\n# Good Heading\n[judge]\njudge stuff\n", encoding="utf-8")
        task = _make_task("t1", contract_type="conventions-doc")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "Good Heading" in contract.metadata["headings"]
        assert "judge stuff" not in contract.body

    def test_status_line_stops_body_collection(self, tmp_path: Path) -> None:
        log = tmp_path / "t.log"
        log.write_text("header\n\n# Heading\nstatus=success\nextra\n", encoding="utf-8")
        task = _make_task("t1", contract_type="conventions-doc")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "Heading" in contract.metadata["headings"]
        assert "status=success" not in contract.body

    def test_empty_body_falls_back_to_stdout_tail(self, tmp_path: Path) -> None:
        log = tmp_path / "t.log"
        # Blank line then immediately a section header → empty body
        log.write_text("header\n\n[judge]\njudge stuff\n", encoding="utf-8")
        task = _make_task("t1", contract_type="conventions-doc")
        contract = normalize_task_contract(task, log, "# Tail Heading")
        assert contract is not None
        assert "Tail Heading" in contract.metadata["headings"]


# ---------------------------------------------------------------------------
# _normalize_sql_schema (tested via normalize_task_contract)
# ---------------------------------------------------------------------------


class TestNormalizeSqlSchema:
    def test_single_table_extracted(self, tmp_path: Path) -> None:
        log = tmp_path / "t.log"
        _write_log(log, "CREATE TABLE users (id INT);")
        task = _make_task("t1", contract_type="sql-schema")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "users" in contract.metadata["tables"]
        assert contract.metadata["statement_count"] == 1

    def test_multiple_tables(self, tmp_path: Path) -> None:
        sql = "CREATE TABLE orders (id INT);\nCREATE TABLE items (id INT);"
        log = tmp_path / "t.log"
        _write_log(log, sql)
        task = _make_task("t1", contract_type="sql-schema")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "orders" in contract.metadata["tables"]
        assert "items" in contract.metadata["tables"]
        assert contract.metadata["statement_count"] == 2

    def test_strips_comment_lines(self, tmp_path: Path) -> None:
        sql = "-- This is a comment\nCREATE TABLE users (id INT);"
        log = tmp_path / "t.log"
        _write_log(log, sql)
        task = _make_task("t1", contract_type="sql-schema")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "-- " not in contract.body

    def test_no_sql_returns_contract_with_empty_tables(self, tmp_path: Path) -> None:
        log = tmp_path / "t.log"
        _write_log(log, "just some prose text")
        task = _make_task("t1", contract_type="sql-schema")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert contract.metadata["tables"] == []

    def test_more_than_5_tables_summary_truncated(self, tmp_path: Path) -> None:
        tables = [f"CREATE TABLE tbl_{i} (id INT);" for i in range(7)]
        log = tmp_path / "t.log"
        _write_log(log, "\n".join(tables))
        task = _make_task("t1", contract_type="sql-schema")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "..." in contract.summary
        assert len(contract.metadata["tables"]) == 7

    def test_alter_table_detected(self, tmp_path: Path) -> None:
        log = tmp_path / "t.log"
        _write_log(log, "ALTER TABLE users ADD COLUMN email TEXT;")
        task = _make_task("t1", contract_type="sql-schema")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "users" in contract.metadata["tables"]

    def test_backtick_quoted_table_name_extracted(self, tmp_path: Path) -> None:
        log = tmp_path / "t.log"
        _write_log(log, "CREATE TABLE `my_table` (id INT);")
        task = _make_task("t1", contract_type="sql-schema")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "my_table" in contract.metadata["tables"]

    def test_create_table_if_not_exists(self, tmp_path: Path) -> None:
        log = tmp_path / "t.log"
        _write_log(log, "CREATE TABLE IF NOT EXISTS sessions (token TEXT);")
        task = _make_task("t1", contract_type="sql-schema")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "sessions" in contract.metadata["tables"]


# ---------------------------------------------------------------------------
# _normalize_dependency_manifest (tested via normalize_task_contract)
# ---------------------------------------------------------------------------


class TestNormalizeDependencyManifest:
    def test_require_section(self, tmp_path: Path) -> None:
        payload = json.dumps({"require": {"vendor/a": "^1.0", "vendor/b": "^2.0"}})
        log = tmp_path / "t.log"
        _write_log(log, payload)
        task = _make_task("t1", contract_type="dependency-manifest")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "vendor/a" in contract.metadata["packages"]
        assert "vendor/b" in contract.metadata["packages"]

    def test_dependencies_section(self, tmp_path: Path) -> None:
        payload = json.dumps({"dependencies": {"express": "^4.0"}})
        log = tmp_path / "t.log"
        _write_log(log, payload)
        task = _make_task("t1", contract_type="dependency-manifest")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "express" in contract.metadata["packages"]

    def test_invalid_json_falls_back_to_generic(self, tmp_path: Path) -> None:
        log = tmp_path / "t.log"
        _write_log(log, "not json at all")
        task = _make_task("t1", contract_type="dependency-manifest")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        # Falls back to _make_generic_contract which has line_count
        assert "line_count" in contract.metadata

    def test_dev_dependencies_included(self, tmp_path: Path) -> None:
        payload = json.dumps({"devDependencies": {"jest": "^29.0"}})
        log = tmp_path / "t.log"
        _write_log(log, payload)
        task = _make_task("t1", contract_type="dependency-manifest")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "jest" in contract.metadata["packages"]

    def test_merged_require_and_dependencies_sections(self, tmp_path: Path) -> None:
        payload = json.dumps({
            "require": {"vendor/a": "^1.0"},
            "dependencies": {"express": "^4.0"},
            "devDependencies": {"jest": "^29.0"},
        })
        log = tmp_path / "t.log"
        _write_log(log, payload)
        task = _make_task("t1", contract_type="dependency-manifest")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        packages = contract.metadata["packages"]
        assert "vendor/a" in packages
        assert "express" in packages
        assert "jest" in packages
        assert contract.metadata["package_count"] == 3

    def test_peer_dependencies_included(self, tmp_path: Path) -> None:
        payload = json.dumps({"peerDependencies": {"react": "^18.0"}})
        log = tmp_path / "t.log"
        _write_log(log, payload)
        task = _make_task("t1", contract_type="dependency-manifest")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "react" in contract.metadata["packages"]

    def test_optional_dependencies_included(self, tmp_path: Path) -> None:
        payload = json.dumps({"optionalDependencies": {"fsevents": "^2.3"}})
        log = tmp_path / "t.log"
        _write_log(log, payload)
        task = _make_task("t1", contract_type="dependency-manifest")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "fsevents" in contract.metadata["packages"]


# ---------------------------------------------------------------------------
# _normalize_conventions_doc (tested via normalize_task_contract)
# ---------------------------------------------------------------------------


class TestNormalizeConventionsDoc:
    def test_headings_extracted(self, tmp_path: Path) -> None:
        log = tmp_path / "t.log"
        _write_log(log, "# Title\n## Section One\n### Subsection")
        task = _make_task("t1", contract_type="conventions-doc")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        headings = contract.metadata["headings"]
        assert "Title" in headings
        assert "Section One" in headings
        assert "Subsection" in headings
        assert contract.metadata["heading_count"] == 3

    def test_no_headings(self, tmp_path: Path) -> None:
        log = tmp_path / "t.log"
        _write_log(log, "Just plain text, no headings here.")
        task = _make_task("t1", contract_type="conventions-doc")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert contract.metadata["headings"] == []
        assert contract.metadata["heading_count"] == 0

    def test_summary_includes_first_heading(self, tmp_path: Path) -> None:
        log = tmp_path / "t.log"
        _write_log(log, "# My Convention Doc\nsome content")
        task = _make_task("t1", contract_type="conventions-doc")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "My Convention Doc" in contract.summary


# ---------------------------------------------------------------------------
# _normalize_file_inventory (tested via normalize_task_contract)
# ---------------------------------------------------------------------------


class TestNormalizeFileInventory:
    def test_json_array(self, tmp_path: Path) -> None:
        payload = json.dumps(["src/a.py", "src/b.py", "tests/test_a.py"])
        log = tmp_path / "t.log"
        _write_log(log, payload)
        task = _make_task("t1", contract_type="file-inventory")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "src/a.py" in contract.metadata["files"]

    def test_json_dict_with_files_key(self, tmp_path: Path) -> None:
        payload = json.dumps({"files": ["a.py", "b.py"]})
        log = tmp_path / "t.log"
        _write_log(log, payload)
        task = _make_task("t1", contract_type="file-inventory")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "a.py" in contract.metadata["files"]
        assert "b.py" in contract.metadata["files"]

    def test_line_per_file_fallback(self, tmp_path: Path) -> None:
        log = tmp_path / "t.log"
        _write_log(log, "src/a.py\nsrc/b.py")
        task = _make_task("t1", contract_type="file-inventory")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "src/a.py" in contract.metadata["files"]
        assert "src/b.py" in contract.metadata["files"]

    def test_deduplicated_and_sorted(self, tmp_path: Path) -> None:
        payload = json.dumps(["b.py", "a.py", "b.py"])
        log = tmp_path / "t.log"
        _write_log(log, payload)
        task = _make_task("t1", contract_type="file-inventory")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        files = contract.metadata["files"]
        assert files == sorted(set(files))
        assert files.count("b.py") == 1

    def test_file_count_in_metadata(self, tmp_path: Path) -> None:
        payload = json.dumps(["x.py", "y.py"])
        log = tmp_path / "t.log"
        _write_log(log, payload)
        task = _make_task("t1", contract_type="file-inventory")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert contract.metadata["file_count"] == 2

    def test_backslash_paths_normalized_to_forward_slash(self, tmp_path: Path) -> None:
        payload = json.dumps(["src\\models\\user.py", "tests\\test_user.py"])
        log = tmp_path / "t.log"
        _write_log(log, payload)
        task = _make_task("t1", contract_type="file-inventory")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "src/models/user.py" in contract.metadata["files"]
        assert "tests/test_user.py" in contract.metadata["files"]

    def test_json_dict_without_files_key_falls_back_to_line_per_file(self, tmp_path: Path) -> None:
        payload = json.dumps({"other_key": ["ignored"]})
        log = tmp_path / "t.log"
        # The dict has no "files" key so the line-per-file branch fires.
        # json.dumps produces a single line, so only that one "file" is recorded.
        _write_log(log, payload)
        task = _make_task("t1", contract_type="file-inventory")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        # line-per-file: the raw JSON string is treated as one file path
        assert contract.metadata["file_count"] >= 1

    def test_empty_json_array_gives_zero_file_count(self, tmp_path: Path) -> None:
        payload = json.dumps([])
        log = tmp_path / "t.log"
        _write_log(log, payload)
        task = _make_task("t1", contract_type="file-inventory")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert contract.metadata["file_count"] == 0
        assert contract.metadata["files"] == []


# ---------------------------------------------------------------------------
# build_contract_template_vars
# ---------------------------------------------------------------------------


class TestBuildContractTemplateVars:
    def test_upstream_with_contract_populates_vars(self) -> None:
        producer_id = "producer"
        contract = _make_contract(producer_id)
        consumer = _make_task("consumer", consumes_contracts=[producer_id])
        results = {producer_id: _make_result(producer_id, contract=contract)}

        vars_ = build_contract_template_vars(consumer, results)

        assert f"contract.{producer_id}.type" in vars_
        assert f"contract.{producer_id}.body" in vars_
        assert f"contract.{producer_id}.hash" in vars_
        assert f"contract.{producer_id}.producer" in vars_
        assert f"contract.{producer_id}.summary" in vars_
        assert f"contract.{producer_id}.metadata_json" in vars_

    def test_upstream_without_contract_not_in_vars(self) -> None:
        producer_id = "producer"
        consumer = _make_task("consumer", consumes_contracts=[producer_id])
        results = {producer_id: _make_result(producer_id, contract=None)}

        vars_ = build_contract_template_vars(consumer, results)

        assert f"contract.{producer_id}.type" not in vars_

    def test_missing_upstream_result_not_in_vars(self) -> None:
        consumer = _make_task("consumer", consumes_contracts=["missing"])
        vars_ = build_contract_template_vars(consumer, {})
        assert "contract.missing.type" not in vars_

    def test_multiple_upstreams_all_populated(self) -> None:
        c1 = _make_contract("p1")
        c2 = _make_contract("p2")
        consumer = _make_task("consumer", consumes_contracts=["p1", "p2"])
        results = {
            "p1": _make_result("p1", contract=c1),
            "p2": _make_result("p2", contract=c2),
        }
        vars_ = build_contract_template_vars(consumer, results)
        assert "contract.p1.type" in vars_
        assert "contract.p2.type" in vars_
        assert "contracts_summary" in vars_

    def test_no_consumes_contracts_returns_empty(self) -> None:
        consumer = _make_task("consumer")
        vars_ = build_contract_template_vars(consumer, {})
        assert vars_ == {}

    def test_contracts_summary_not_set_with_no_contracts(self) -> None:
        producer_id = "producer"
        consumer = _make_task("consumer", consumes_contracts=[producer_id])
        results = {producer_id: _make_result(producer_id, contract=None)}

        vars_ = build_contract_template_vars(consumer, results)
        assert "contracts_summary" not in vars_

    def test_contracts_summary_format_includes_id_type_and_summary(self) -> None:
        contract = _make_contract("p1", contract_type="sql-schema", summary="SQL with 2 tables")
        consumer = _make_task("consumer", consumes_contracts=["p1"])
        results = {"p1": _make_result("p1", contract=contract)}

        vars_ = build_contract_template_vars(consumer, results)
        summary = vars_["contracts_summary"]
        assert "p1" in summary
        assert "sql-schema" in summary
        assert "SQL with 2 tables" in summary

    def test_metadata_json_field_is_parseable_json(self) -> None:
        contract = _make_contract("p1")
        consumer = _make_task("consumer", consumes_contracts=["p1"])
        results = {"p1": _make_result("p1", contract=contract)}

        vars_ = build_contract_template_vars(consumer, results)
        parsed = json.loads(vars_["contract.p1.metadata_json"])
        assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# build_consistency_template_vars
# ---------------------------------------------------------------------------


class TestBuildConsistencyTemplateVars:
    def test_group_with_completed_peers(self) -> None:
        r1 = _make_result("impl-a", stdout_tail="done")
        r2 = _make_result("impl-b", stdout_tail="done")
        reconciler = _make_task("reconciler", reconcile_after=["my-group"])
        group_members = {"my-group": ["impl-a", "impl-b"]}
        results = {"impl-a": r1, "impl-b": r2}

        vars_ = build_consistency_template_vars(reconciler, results, group_members)

        assert "consistency.my-group.tasks" in vars_
        assert "consistency.my-group.statuses" in vars_
        assert "impl-a" in vars_["consistency.my-group.tasks"]
        assert "consistency_summary" in vars_

    def test_statuses_reflect_result_status(self) -> None:
        r1 = _make_result("impl-a")
        reconciler = _make_task("reconciler", reconcile_after=["grp"])
        group_members = {"grp": ["impl-a"]}
        results = {"impl-a": r1}

        vars_ = build_consistency_template_vars(reconciler, results, group_members)

        assert "impl-a: success" in vars_["consistency.grp.statuses"]

    def test_missing_member_result_shows_missing(self) -> None:
        reconciler = _make_task("reconciler", reconcile_after=["grp"])
        group_members = {"grp": ["impl-a"]}

        vars_ = build_consistency_template_vars(reconciler, {}, group_members)

        assert "impl-a: missing" in vars_["consistency.grp.statuses"]

    def test_no_group_members_no_vars(self) -> None:
        reconciler = _make_task("reconciler", reconcile_after=["empty-group"])
        group_members: dict[str, list[str]] = {"empty-group": []}

        vars_ = build_consistency_template_vars(reconciler, {}, group_members)

        assert "consistency.empty-group.tasks" not in vars_

    def test_missing_group_in_group_members_no_vars(self) -> None:
        reconciler = _make_task("reconciler", reconcile_after=["grp"])
        vars_ = build_consistency_template_vars(reconciler, {}, {})
        assert "consistency.grp.tasks" not in vars_

    def test_no_reconcile_after_returns_empty(self) -> None:
        task = _make_task("t1")
        vars_ = build_consistency_template_vars(task, {}, {})
        assert vars_ == {}

    def test_contract_on_member_populates_contracts_field(self) -> None:
        contract = _make_contract("impl-a", summary="Schema v1")
        r = _make_result("impl-a", contract=contract)
        reconciler = _make_task("reconciler", reconcile_after=["grp"])
        group_members = {"grp": ["impl-a"]}
        results = {"impl-a": r}

        vars_ = build_consistency_template_vars(reconciler, results, group_members)

        assert "Schema v1" in vars_["consistency.grp.contracts"]

    def test_summaries_uses_stdout_tail_when_no_contract(self) -> None:
        r = _make_result("impl-a", stdout_tail="All checks passed OK")
        reconciler = _make_task("reconciler", reconcile_after=["grp"])
        group_members = {"grp": ["impl-a"]}
        results = {"impl-a": r}

        vars_ = build_consistency_template_vars(reconciler, results, group_members)

        assert "All checks passed OK" in vars_["consistency.grp.summaries"]

    def test_multiple_groups_all_populated(self) -> None:
        r1 = _make_result("a", stdout_tail="done a")
        r2 = _make_result("b", stdout_tail="done b")
        reconciler = _make_task("reconciler", reconcile_after=["g1", "g2"])
        group_members = {"g1": ["a"], "g2": ["b"]}
        results = {"a": r1, "b": r2}

        vars_ = build_consistency_template_vars(reconciler, results, group_members)

        assert "consistency.g1.tasks" in vars_
        assert "consistency.g2.tasks" in vars_
        assert "consistency_summary" in vars_


# ---------------------------------------------------------------------------
# Additional edge-case tests — _extract_primary_output ([stderr] prefix)
# ---------------------------------------------------------------------------


class TestExtractPrimaryOutputSterrPrefix:
    def test_stderr_prefix_line_stops_body_collection(self, tmp_path: Path) -> None:
        log = tmp_path / "t.log"
        log.write_text(
            "header\n\n# Good Heading\n[stderr] something went wrong\nextra line\n",
            encoding="utf-8",
        )
        task = _make_task("t1", contract_type="conventions-doc")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "Good Heading" in contract.metadata["headings"]
        assert "extra line" not in contract.body


# ---------------------------------------------------------------------------
# Additional edge-case tests — generic/unknown contract metadata
# ---------------------------------------------------------------------------


class TestNormalizeGenericContract:
    def test_unknown_type_metadata_is_empty_dict(self) -> None:
        task = _make_task("t1", contract_type="custom-type")
        contract = normalize_task_contract(task, Path("/nonexistent"), "line one\nline two")
        assert contract is not None
        assert contract.metadata == {}

    def test_unknown_type_summary_is_generic_label(self) -> None:
        task = _make_task("t1", contract_type="my-contract")
        contract = normalize_task_contract(task, Path("/nonexistent"), "only one line")
        assert contract is not None
        assert contract.summary == "Generic contract output"


# ---------------------------------------------------------------------------
# Additional edge-case tests — consistency summaries and contracts fields
# ---------------------------------------------------------------------------


class TestBuildConsistencyTemplateVarsNoneFields:
    def test_summaries_shows_none_when_members_have_no_output(self) -> None:
        r = _make_result("impl-a")  # empty stdout_tail, no contract
        reconciler = _make_task("reconciler", reconcile_after=["grp"])
        group_members = {"grp": ["impl-a"]}
        results = {"impl-a": r}

        vars_ = build_consistency_template_vars(reconciler, results, group_members)

        assert vars_["consistency.grp.summaries"] == "(none)"

    def test_contracts_shows_none_when_members_have_no_contracts(self) -> None:
        r = _make_result("impl-a", stdout_tail="some output")  # no produced_contract
        reconciler = _make_task("reconciler", reconcile_after=["grp"])
        group_members = {"grp": ["impl-a"]}
        results = {"impl-a": r}

        vars_ = build_consistency_template_vars(reconciler, results, group_members)

        assert vars_["consistency.grp.contracts"] == "(none)"


# ---------------------------------------------------------------------------
# Additional edge-case tests — generic contract fallback char_count field
# ---------------------------------------------------------------------------


class TestNormalizeGenericContractCharCount:
    def test_dependency_manifest_non_dict_json_fallback_has_char_count(
        self, tmp_path: Path
    ) -> None:
        """A JSON array (not a dict) body falls back to _make_generic_contract,
        which includes both line_count AND char_count in the metadata."""
        payload = json.dumps(["not", "a", "dict"])
        log = tmp_path / "t.log"
        _write_log(log, payload)
        task = _make_task("t1", contract_type="dependency-manifest")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "char_count" in contract.metadata
        assert isinstance(contract.metadata["char_count"], int)
        assert contract.metadata["char_count"] > 0

    def test_dependency_manifest_non_dict_json_fallback_line_count_correct(
        self, tmp_path: Path
    ) -> None:
        """Non-dict JSON body fallback: line_count reflects non-empty lines."""
        payload = "line one\nline two\nline three"
        log = tmp_path / "t.log"
        _write_log(log, payload)
        task = _make_task("t1", contract_type="dependency-manifest")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert contract.metadata["line_count"] == 3


# ---------------------------------------------------------------------------
# Additional edge-case tests — file-inventory and dependency-manifest
# ---------------------------------------------------------------------------


class TestNormalizeFileInventoryFilesNotList:
    def test_files_key_not_a_list_falls_back_to_line_per_file(
        self, tmp_path: Path
    ) -> None:
        """If the JSON has a 'files' key whose value is not a list, the
        line-per-file branch fires and treats each line of the raw body
        as a file path."""
        payload = json.dumps({"files": "not_a_list"})
        log = tmp_path / "t.log"
        _write_log(log, payload)
        task = _make_task("t1", contract_type="file-inventory")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        # The raw JSON line is treated as one "file" path
        assert contract.metadata["file_count"] >= 1
        # None of the metadata files should be just a comma-separated list;
        # verify the contract type is correct
        assert contract.contract_type == "file-inventory"


class TestNormalizeDependencyManifestNullJson:
    def test_json_null_body_falls_back_to_generic_contract(
        self, tmp_path: Path
    ) -> None:
        """JSON literal 'null' deserializes to None.  isinstance(None, dict) is
        False so _make_generic_contract is called, which adds line_count /
        char_count instead of package_count."""
        log = tmp_path / "t.log"
        _write_log(log, "null")
        task = _make_task("t1", contract_type="dependency-manifest")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "line_count" in contract.metadata
        assert "package_count" not in contract.metadata


# ---------------------------------------------------------------------------
# Additional edge-case tests — _extract_primary_output (message= prefix)
# ---------------------------------------------------------------------------


class TestExtractPrimaryOutputMessagePrefix:
    def test_message_prefix_stops_body_collection(self, tmp_path: Path) -> None:
        """Lines starting with 'message=' must stop body extraction, just like
        'status=' does.  Verify the body does NOT include the message= line or
        anything after it."""
        log = tmp_path / "t.log"
        log.write_text(
            "header\n\n# Good Heading\nmessage=some diagnostic\nextra ignored\n",
            encoding="utf-8",
        )
        task = _make_task("t1", contract_type="conventions-doc")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "Good Heading" in contract.metadata["headings"]
        assert "message=some diagnostic" not in contract.body
        assert "extra ignored" not in contract.body

    def test_message_prefix_alone_falls_back_to_stdout_tail(self, tmp_path: Path) -> None:
        """If the FIRST body line is message=..., extracted body is empty and
        the function must fall back to stdout_tail."""
        log = tmp_path / "t.log"
        log.write_text("header\n\nmessage=failed early\n", encoding="utf-8")
        task = _make_task("t1", contract_type="conventions-doc")
        contract = normalize_task_contract(task, log, "# Fallback Heading")
        assert contract is not None
        assert "Fallback Heading" in contract.metadata["headings"]


# ---------------------------------------------------------------------------
# Additional tests — _normalize_dependency_manifest (require-dev key)
# ---------------------------------------------------------------------------


class TestNormalizeDependencyManifestRequireDev:
    def test_require_dev_section_included(self, tmp_path: Path) -> None:
        """The 'require-dev' key (Composer-style) must be included alongside
        'require'.  This is distinct from 'devDependencies' (npm-style)."""
        payload = json.dumps({"require-dev": {"phpunit/phpunit": "^10.0", "mockery/mockery": "^1.6"}})
        log = tmp_path / "t.log"
        _write_log(log, payload)
        task = _make_task("t1", contract_type="dependency-manifest")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "phpunit/phpunit" in contract.metadata["packages"]
        assert "mockery/mockery" in contract.metadata["packages"]
        assert contract.metadata["package_count"] == 2


# ---------------------------------------------------------------------------
# Additional tests — conventions doc summary format
# ---------------------------------------------------------------------------


class TestNormalizeConventionsDocSummary:
    def test_summary_with_zero_headings_has_no_first_heading_part(
        self, tmp_path: Path
    ) -> None:
        """When there are no headings, summary is just 'Conventions document
        with 0 heading(s)' — no '; first heading:' suffix is appended."""
        log = tmp_path / "t.log"
        _write_log(log, "Plain text, no markdown headings.")
        task = _make_task("t1", contract_type="conventions-doc")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert contract.summary == "Conventions document with 0 heading(s)"
        assert "first heading" not in contract.summary

    def test_summary_format_includes_heading_count_and_first_heading(
        self, tmp_path: Path
    ) -> None:
        """With headings present, summary must contain the count and the
        '; first heading: X' suffix."""
        log = tmp_path / "t.log"
        _write_log(log, "# Architecture Patterns\n## Naming")
        task = _make_task("t1", contract_type="conventions-doc")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "Conventions document with 2 heading(s)" in contract.summary
        assert "first heading: Architecture Patterns" in contract.summary


# ---------------------------------------------------------------------------
# Additional tests — generic contract summary string format
# ---------------------------------------------------------------------------


class TestNormalizeGenericContractSummaryFormat:
    def test_generic_summary_includes_contract_type_and_line_count(self) -> None:
        """_make_generic_contract uses '{contract_type} contract with N
        non-empty line(s)' as the summary.  Verify for a multi-line body."""
        task = _make_task("t1", contract_type="my-custom-type")
        # 3 non-empty lines
        contract = normalize_task_contract(
            task, Path("/nonexistent"), "line one\nline two\nline three"
        )
        assert contract is not None
        assert contract.summary == "Generic contract output"

    def test_unknown_type_body_stored_as_stripped_body(self) -> None:
        """For an unknown contract type, body is stored stripped."""
        body = "  some output with whitespace   "
        task = _make_task("t1", contract_type="custom-type")
        contract = normalize_task_contract(task, Path("/nonexistent"), body)
        assert contract is not None
        assert contract.body == body.strip()


class TestConsistencyVarsLongStdoutTruncated:
    def test_long_stdout_tail_truncated_to_240_chars_in_summaries(self) -> None:
        """_summarize_result joins lines and slices at 240 chars.  A 300-char
        single-line stdout_tail must appear truncated in the summaries field."""
        long_tail = "x" * 300
        r = _make_result("impl-a", stdout_tail=long_tail)
        reconciler = _make_task("reconciler", reconcile_after=["grp"])
        group_members = {"grp": ["impl-a"]}
        results = {"impl-a": r}

        vars_ = build_consistency_template_vars(reconciler, results, group_members)

        # The summaries field is "impl-a: <truncated>"; the total length of
        # the member summary portion must not exceed 240 characters.
        summary_entry = vars_["consistency.grp.summaries"]
        # Strip the "impl-a: " prefix to get the summary text itself
        prefix = "impl-a: "
        assert summary_entry.startswith(prefix)
        member_text = summary_entry[len(prefix):]
        assert len(member_text) <= 240
        assert len(member_text) == 240  # exactly sliced at 240


# ---------------------------------------------------------------------------
# Additional tests — SQL schema double-quoted table name
# ---------------------------------------------------------------------------


class TestNormalizeSqlSchemaDoubleQuoted:
    def test_double_quoted_table_name_extracted(self, tmp_path: Path) -> None:
        """The SQL regex accepts double-quoted identifiers like \"my_table\"."""
        log = tmp_path / "t.log"
        _write_log(log, 'CREATE TABLE "my_schema_table" (id INT, name TEXT);')
        task = _make_task("t1", contract_type="sql-schema")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "my_schema_table" in contract.metadata["tables"]


# ---------------------------------------------------------------------------
# Additional tests — file-inventory blank-line handling in line-per-file mode
# ---------------------------------------------------------------------------


class TestNormalizeFileInventoryBlankLinesSkipped:
    def test_blank_lines_in_line_per_file_are_excluded(self, tmp_path: Path) -> None:
        """In the line-per-file fallback, empty lines must not appear as file
        paths.  The implementation filters via ``if line.strip()``."""
        log = tmp_path / "t.log"
        # Three non-empty paths separated by blank lines
        _write_log(log, "src/a.py\n\nsrc/b.py\n\ntests/test_a.py\n")
        task = _make_task("t1", contract_type="file-inventory")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        # Exactly 3 real paths, no empty string
        assert contract.metadata["file_count"] == 3
        assert "" not in contract.metadata["files"]


# ---------------------------------------------------------------------------
# Additional tests — consistency template vars: multiple member contracts
# ---------------------------------------------------------------------------


class TestBuildConsistencyMultipleMemberContracts:
    def test_two_member_contracts_both_in_contracts_field(self) -> None:
        """When two group members each have a produced_contract, both summaries
        must appear in ``consistency.<group>.contracts``."""
        c1 = _make_contract("impl-a", summary="Schema v1")
        c2 = _make_contract("impl-b", summary="Schema v2")
        r1 = _make_result("impl-a", contract=c1)
        r2 = _make_result("impl-b", contract=c2)
        reconciler = _make_task("reconciler", reconcile_after=["grp"])
        group_members = {"grp": ["impl-a", "impl-b"]}
        results = {"impl-a": r1, "impl-b": r2}

        vars_ = build_consistency_template_vars(reconciler, results, group_members)

        contracts_field = vars_["consistency.grp.contracts"]
        assert "Schema v1" in contracts_field
        assert "Schema v2" in contracts_field


# ---------------------------------------------------------------------------
# Additional tests — _summarize_result uses structured_context.summary
# ---------------------------------------------------------------------------


class TestBuildConsistencyStructuredContextSummary:
    def test_structured_context_summary_used_when_no_contract(self) -> None:
        """When ``produced_contract`` is None but ``structured_context.summary``
        is set, ``_summarize_result`` must return that summary."""
        from maestro_cli.models import StructuredContext

        sc = StructuredContext(
            task_id="impl-a",
            status="success",
            exit_code=0,
            duration_sec=1.0,
            summary="Structured summary from extraction",
        )
        r = TaskResult(
            task_id="impl-a",
            status="success",
            stdout_tail="raw tail",
            structured_context=sc,
        )
        reconciler = _make_task("reconciler", reconcile_after=["grp"])
        group_members = {"grp": ["impl-a"]}
        results = {"impl-a": r}

        vars_ = build_consistency_template_vars(reconciler, results, group_members)

        assert "Structured summary from extraction" in vars_["consistency.grp.summaries"]


# ---------------------------------------------------------------------------
# Additional tests — _normalize_sql_schema backtick and IF NOT EXISTS
# ---------------------------------------------------------------------------


class TestNormalizeSqlSchemaBacktickAndIfNotExists:
    def test_backtick_quoted_table_name_extracted(self, tmp_path: Path) -> None:
        """CREATE TABLE with a backtick-quoted identifier; the regex
        must strip the backtick and return the bare name."""
        log = tmp_path / "t.log"
        _write_log(log, "CREATE TABLE `schema_events` (id INT, ts DATETIME);")
        task = _make_task("t1", contract_type="sql-schema")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "schema_events" in contract.metadata["tables"]

    def test_if_not_exists_table_name_extracted(self, tmp_path: Path) -> None:
        """CREATE TABLE IF NOT EXISTS should still extract the table name."""
        log = tmp_path / "t.log"
        _write_log(log, "CREATE TABLE IF NOT EXISTS audit_log (id INT);")
        task = _make_task("t1", contract_type="sql-schema")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "audit_log" in contract.metadata["tables"]

    def test_if_exists_alter_table(self, tmp_path: Path) -> None:
        """ALTER TABLE with a plain name after it should also be extracted."""
        log = tmp_path / "t.log"
        _write_log(log, "ALTER TABLE products ADD COLUMN price DECIMAL(10,2);")
        task = _make_task("t1", contract_type="sql-schema")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "products" in contract.metadata["tables"]


# ---------------------------------------------------------------------------
# Additional tests — _normalize_dependency_manifest npm-style sections
# ---------------------------------------------------------------------------


class TestNormalizeDependencyManifestNpmSections:
    def test_npm_dependencies_extracted(self, tmp_path: Path) -> None:
        """The 'dependencies' key (npm-style) must be parsed into packages."""
        payload = json.dumps({"dependencies": {"express": "^4.18", "lodash": "^4.17"}})
        log = tmp_path / "t.log"
        _write_log(log, payload)
        task = _make_task("t1", contract_type="dependency-manifest")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "express" in contract.metadata["packages"]
        assert "lodash" in contract.metadata["packages"]

    def test_peer_dependencies_extracted(self, tmp_path: Path) -> None:
        """The 'peerDependencies' key must be included in the packages list."""
        payload = json.dumps({"peerDependencies": {"react": "^18.0", "react-dom": "^18.0"}})
        log = tmp_path / "t.log"
        _write_log(log, payload)
        task = _make_task("t1", contract_type="dependency-manifest")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "react" in contract.metadata["packages"]
        assert "react-dom" in contract.metadata["packages"]

    def test_optional_dependencies_extracted(self, tmp_path: Path) -> None:
        """The 'optionalDependencies' key must be included in the packages list."""
        payload = json.dumps({"optionalDependencies": {"fsevents": "^2.3"}})
        log = tmp_path / "t.log"
        _write_log(log, payload)
        task = _make_task("t1", contract_type="dependency-manifest")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "fsevents" in contract.metadata["packages"]

    def test_mixed_npm_sections_merged_and_deduplicated(self, tmp_path: Path) -> None:
        """Packages appearing in multiple sections must be deduplicated."""
        payload = json.dumps({
            "dependencies": {"react": "^18.0"},
            "devDependencies": {"jest": "^29.0", "react": "^18.0"},
        })
        log = tmp_path / "t.log"
        _write_log(log, payload)
        task = _make_task("t1", contract_type="dependency-manifest")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        pkgs = contract.metadata["packages"]
        assert "react" in pkgs
        assert "jest" in pkgs
        assert pkgs.count("react") == 1


# ---------------------------------------------------------------------------
# Additional tests — consistency.tasks field newline-separated format
# ---------------------------------------------------------------------------


class TestConsistencyTasksFieldFormat:
    def test_two_members_listed_newline_separated(self) -> None:
        """consistency.X.tasks must contain all member IDs separated by newlines
        when there are multiple members in the group."""
        r1 = _make_result("worker-a")
        r2 = _make_result("worker-b")
        reconciler = _make_task("reconciler", reconcile_after=["workers"])
        group_members = {"workers": ["worker-a", "worker-b"]}
        results = {"worker-a": r1, "worker-b": r2}

        vars_ = build_consistency_template_vars(reconciler, results, group_members)

        tasks_field = vars_["consistency.workers.tasks"]
        assert "worker-a" in tasks_field
        assert "worker-b" in tasks_field
        assert "\n" in tasks_field


# ---------------------------------------------------------------------------
# Additional tests — contract.X.producer field value
# ---------------------------------------------------------------------------


class TestContractProducerFieldValue:
    def test_producer_field_value_equals_producer_task_id(self) -> None:
        """The contract.X.producer variable must equal the producer task ID,
        not just be present — verify the actual string value."""
        contract = _make_contract("my-producer")
        consumer = _make_task("consumer", consumes_contracts=["my-producer"])
        results = {"my-producer": _make_result("my-producer", contract=contract)}

        vars_ = build_contract_template_vars(consumer, results)

        assert vars_["contract.my-producer.producer"] == "my-producer"


# ---------------------------------------------------------------------------
# Additional tests — consistency_summary field content
# ---------------------------------------------------------------------------


class TestConsistencySummaryContent:
    def test_consistency_summary_contains_group_name(self) -> None:
        """consistency_summary must include the group name so downstream tasks
        can identify which groups are summarised."""
        r = _make_result("impl-x")
        reconciler = _make_task("reconciler", reconcile_after=["my-special-group"])
        group_members = {"my-special-group": ["impl-x"]}
        results = {"impl-x": r}

        vars_ = build_consistency_template_vars(reconciler, results, group_members)

        assert "my-special-group" in vars_["consistency_summary"]


# ---------------------------------------------------------------------------
# Additional tests — ALTER TABLE IF EXISTS (without NOT)
# ---------------------------------------------------------------------------


class TestNormalizeSqlAlterTableIfExists:
    def test_alter_table_if_exists_without_not_extracted(self, tmp_path: Path) -> None:
        """The SQL regex accepts ALTER TABLE IF EXISTS (without NOT).
        The (?:NOT\\s+)? part of the pattern is optional."""
        log = tmp_path / "t.log"
        _write_log(log, "ALTER TABLE IF EXISTS legacy_table ADD COLUMN new_col INT;")
        task = _make_task("t1", contract_type="sql-schema")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "legacy_table" in contract.metadata["tables"]


# ---------------------------------------------------------------------------
# Additional tests — _normalize_sql_schema: exactly 5 tables (no ellipsis)
# ---------------------------------------------------------------------------


class TestNormalizeSqlSchemaExactly5Tables:
    def test_exactly_5_tables_no_ellipsis_in_summary(self, tmp_path: Path) -> None:
        """When there are exactly 5 tables the code uses ``if len(tables) > 5``
        so the preview does NOT get the trailing ', ...' suffix."""
        tables_sql = "\n".join(
            f"CREATE TABLE tbl_{i} (id INT);" for i in range(5)
        )
        log = tmp_path / "t.log"
        _write_log(log, tables_sql)
        task = _make_task("t1", contract_type="sql-schema")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "..." not in contract.summary
        assert contract.metadata["statement_count"] == 5
        assert len(contract.metadata["tables"]) == 5


# ---------------------------------------------------------------------------
# Additional tests — _normalize_dependency_manifest: no matching sections
# ---------------------------------------------------------------------------


class TestNormalizeDependencyManifestNoMatchingSections:
    def test_json_dict_no_matching_sections_returns_empty_packages(
        self, tmp_path: Path
    ) -> None:
        """A JSON dict with no recognized package-section keys produces a
        contract with packages=[] and package_count=0."""
        payload = json.dumps({"name": "my-project", "version": "1.0.0"})
        log = tmp_path / "t.log"
        _write_log(log, payload)
        task = _make_task("t1", contract_type="dependency-manifest")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert contract.metadata["packages"] == []
        assert contract.metadata["package_count"] == 0

    def test_json_dict_with_non_dict_require_section_skipped(
        self, tmp_path: Path
    ) -> None:
        """If the 'require' value is a list (not a dict), the
        ``isinstance(section, dict)`` guard skips it — no packages extracted."""
        payload = json.dumps({"require": ["not-a-dict-value"]})
        log = tmp_path / "t.log"
        _write_log(log, payload)
        task = _make_task("t1", contract_type="dependency-manifest")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert contract.metadata["packages"] == []
        assert contract.metadata["package_count"] == 0


# ---------------------------------------------------------------------------
# Additional tests — _normalize_file_inventory: whitespace-only items filtered
# ---------------------------------------------------------------------------


class TestNormalizeFileInventoryWhitespaceItems:
    def test_whitespace_only_items_in_json_array_excluded(self, tmp_path: Path) -> None:
        """Items in a JSON array whose .strip() is empty must be excluded from
        the file list; the implementation filters via ``if str(item).strip()``."""
        payload = json.dumps(["src/a.py", "  ", "", "src/b.py"])
        log = tmp_path / "t.log"
        _write_log(log, payload)
        task = _make_task("t1", contract_type="file-inventory")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "" not in contract.metadata["files"]
        assert "  " not in contract.metadata["files"]
        assert "src/a.py" in contract.metadata["files"]
        assert "src/b.py" in contract.metadata["files"]
        assert contract.metadata["file_count"] == 2


# ---------------------------------------------------------------------------
# Additional tests — _normalize_dependency_manifest: JSON array body
# ---------------------------------------------------------------------------


class TestNormalizeDependencyManifestJsonArray:
    def test_json_array_body_falls_back_to_generic_contract(
        self, tmp_path: Path
    ) -> None:
        """A JSON array is not a dict, so isinstance(payload, dict) is False
        and _make_generic_contract is called, producing line_count/char_count
        instead of package_count."""
        payload = json.dumps(["vendor/a", "vendor/b"])
        log = tmp_path / "t.log"
        _write_log(log, payload)
        task = _make_task("t1", contract_type="dependency-manifest")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "line_count" in contract.metadata
        assert "package_count" not in contract.metadata
        assert contract.contract_type == "dependency-manifest"


# ---------------------------------------------------------------------------
# Additional tests — build_contract_template_vars: mixed upstreams
# ---------------------------------------------------------------------------


class TestBuildContractTemplateVarsMixedUpstreams:
    def test_mixed_upstreams_only_contract_bearing_producer_in_summary(
        self,
    ) -> None:
        """When one upstream has a contract and another does not, only the
        contract-bearing producer appears in contracts_summary."""
        contract = _make_contract("has-contract")
        consumer = _make_task(
            "consumer", consumes_contracts=["has-contract", "no-contract"]
        )
        results = {
            "has-contract": _make_result("has-contract", contract=contract),
            "no-contract": _make_result("no-contract", contract=None),
        }
        vars_ = build_contract_template_vars(consumer, results)
        assert "contract.has-contract.type" in vars_
        assert "contract.no-contract.type" not in vars_
        assert "contracts_summary" in vars_
        assert "has-contract" in vars_["contracts_summary"]
        assert "no-contract" not in vars_["contracts_summary"]


# ---------------------------------------------------------------------------
# Additional tests — section headers that stop body extraction
# ---------------------------------------------------------------------------


class TestExtractPrimaryOutputSectionHeaders:
    def test_pre_command_header_stops_body_collection(self, tmp_path: Path) -> None:
        """Lines matching '[pre_command]' must stop body extraction — verifies
        that the full _SECTION_HEADERS set is honoured, not just [judge]."""
        log = tmp_path / "t.log"
        log.write_text(
            "header\n\n# Good Heading\n[pre_command]\npre stuff\n",
            encoding="utf-8",
        )
        task = _make_task("t1", contract_type="conventions-doc")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "Good Heading" in contract.metadata["headings"]
        assert "pre stuff" not in contract.body

    def test_assert_header_stops_body_collection(self, tmp_path: Path) -> None:
        """Lines matching '[assert]' must stop body extraction."""
        log = tmp_path / "t.log"
        log.write_text(
            "header\n\n# Valid Heading\n[assert]\nassert stuff\n",
            encoding="utf-8",
        )
        task = _make_task("t1", contract_type="conventions-doc")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "Valid Heading" in contract.metadata["headings"]
        assert "assert stuff" not in contract.body


# ---------------------------------------------------------------------------
# Additional tests — dependency-manifest and file-inventory summary formats
# ---------------------------------------------------------------------------


class TestNormalizeDependencyManifestSummaryFormat:
    def test_summary_says_dependency_manifest_with_n_packages(
        self, tmp_path: Path
    ) -> None:
        """The summary for a dependency-manifest contract must be exactly
        'Dependency manifest with N package(s)'."""
        payload = json.dumps({"require": {"vendor/a": "^1.0", "vendor/b": "^2.0"}})
        log = tmp_path / "t.log"
        _write_log(log, payload)
        task = _make_task("t1", contract_type="dependency-manifest")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert contract.summary == "Dependency manifest with 2 package(s)"

    def test_summary_zero_packages_format(self, tmp_path: Path) -> None:
        """Empty manifest dict produces 'Dependency manifest with 0 package(s)'."""
        payload = json.dumps({"name": "empty-project"})
        log = tmp_path / "t.log"
        _write_log(log, payload)
        task = _make_task("t1", contract_type="dependency-manifest")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert contract.summary == "Dependency manifest with 0 package(s)"


class TestNormalizeFileInventorySummaryFormat:
    def test_summary_says_file_inventory_with_n_files(self, tmp_path: Path) -> None:
        """The summary for a file-inventory contract must be exactly
        'File inventory with N file(s)'."""
        payload = json.dumps(["src/a.py", "src/b.py", "src/c.py"])
        log = tmp_path / "t.log"
        _write_log(log, payload)
        task = _make_task("t1", contract_type="file-inventory")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert contract.summary == "File inventory with 3 file(s)"


# ---------------------------------------------------------------------------
# Additional tests — remaining _SECTION_HEADERS not yet covered
# ---------------------------------------------------------------------------


class TestExtractPrimaryOutputRemainingHeaders:
    def test_verify_command_header_stops_body_collection(self, tmp_path: Path) -> None:
        """Lines matching '[verify_command]' (an entry in _SECTION_HEADERS) must
        stop body extraction before the content that follows it."""
        log = tmp_path / "t.log"
        log.write_text(
            "header\n\n# Good Heading\n[verify_command]\nverify stuff\n",
            encoding="utf-8",
        )
        task = _make_task("t1", contract_type="conventions-doc")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "Good Heading" in contract.metadata["headings"]
        assert "verify stuff" not in contract.body

    def test_guard_command_header_stops_body_collection(self, tmp_path: Path) -> None:
        """Lines matching '[guard_command]' must stop body extraction."""
        log = tmp_path / "t.log"
        log.write_text(
            "header\n\n# Valid Heading\n[guard_command]\nguard stuff\n",
            encoding="utf-8",
        )
        task = _make_task("t1", contract_type="conventions-doc")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "Valid Heading" in contract.metadata["headings"]
        assert "guard stuff" not in contract.body

    def test_handoff_report_header_stops_body_collection(self, tmp_path: Path) -> None:
        """Lines matching '[handoff_report]' must stop body extraction."""
        log = tmp_path / "t.log"
        log.write_text(
            "header\n\n# Valid Heading\n[handoff_report]\nhandoff stuff\n",
            encoding="utf-8",
        )
        task = _make_task("t1", contract_type="conventions-doc")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "Valid Heading" in contract.metadata["headings"]
        assert "handoff stuff" not in contract.body

    def test_judge_result_header_stops_body_collection(self, tmp_path: Path) -> None:
        """Lines matching '[judge_result]' must stop body extraction."""
        log = tmp_path / "t.log"
        log.write_text(
            "header\n\n# Valid Heading\n[judge_result]\njudge result stuff\n",
            encoding="utf-8",
        )
        task = _make_task("t1", contract_type="conventions-doc")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "Valid Heading" in contract.metadata["headings"]
        assert "judge result stuff" not in contract.body


# ---------------------------------------------------------------------------
# Additional tests — SQL body consisting of only empty/whitespace lines
# ---------------------------------------------------------------------------


class TestNormalizeSqlSchemaEmptyLinesOnly:
    def test_body_with_only_whitespace_lines_gives_empty_tables_and_zero_statements(
        self, tmp_path: Path
    ) -> None:
        """When the SQL body contains only whitespace lines they are all
        filtered by the ``if line.strip()`` guard in cleaned_lines, producing
        an empty normalized_body, zero statements, and no tables."""
        log = tmp_path / "t.log"
        _write_log(log, "   \n  \n\t\n")
        task = _make_task("t1", contract_type="sql-schema")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert contract.metadata["tables"] == []
        assert contract.metadata["statement_count"] == 0


# ---------------------------------------------------------------------------
# Additional tests — SQL schema summary format (tables label)
# ---------------------------------------------------------------------------


class TestNormalizeSqlSchemaSummaryFormat:
    def test_single_table_summary_includes_tables_label(self, tmp_path: Path) -> None:
        """SQL schema summary for a single table must include '; tables: <name>'."""
        log = tmp_path / "t.log"
        _write_log(log, "CREATE TABLE users (id INT);")
        task = _make_task("t1", contract_type="sql-schema")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "tables: users" in contract.summary

    def test_no_tables_summary_omits_tables_label(self, tmp_path: Path) -> None:
        """When no tables are matched, the summary must NOT contain '; tables:'."""
        log = tmp_path / "t.log"
        _write_log(log, "just prose text, no SQL at all")
        task = _make_task("t1", contract_type="sql-schema")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "tables:" not in contract.summary


# ---------------------------------------------------------------------------
# Additional tests — _make_generic_contract summary from dependency-manifest
# ---------------------------------------------------------------------------


class TestNormalizeDependencyManifestGenericFallbackSummary:
    def test_null_json_fallback_summary_format(self, tmp_path: Path) -> None:
        """json.loads('null') == None (not a dict) → _make_generic_contract is
        called; its summary is 'dependency-manifest contract with N non-empty
        line(s)', NOT 'Dependency manifest with ...'."""
        log = tmp_path / "t.log"
        _write_log(log, "null")
        task = _make_task("t1", contract_type="dependency-manifest")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert contract.summary.startswith("dependency-manifest contract with")
        assert "non-empty line" in contract.summary

    def test_json_array_fallback_summary_format(self, tmp_path: Path) -> None:
        """A JSON array body (not a dict) falls back to _make_generic_contract;
        summary starts with the contract type, not 'Dependency manifest with'."""
        payload = json.dumps(["vendor/a", "vendor/b"])
        log = tmp_path / "t.log"
        _write_log(log, payload)
        task = _make_task("t1", contract_type="dependency-manifest")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert contract.summary.startswith("dependency-manifest contract with")
        assert "non-empty line" in contract.summary


# ---------------------------------------------------------------------------
# Additional tests — _summarize_result multi-line stdout_tail joining
# ---------------------------------------------------------------------------


class TestSummarizeResultMultilineJoin:
    def test_multiline_stdout_tail_joined_with_spaces_in_summaries(self) -> None:
        """_summarize_result joins multi-line stdout_tail lines with a single
        space, producing a flat single-line string in consistency.X.summaries."""
        r = _make_result("impl-a", stdout_tail="line one\nline two\nline three")
        reconciler = _make_task("reconciler", reconcile_after=["grp"])
        group_members = {"grp": ["impl-a"]}
        results = {"impl-a": r}

        vars_ = build_consistency_template_vars(reconciler, results, group_members)

        summaries = vars_["consistency.grp.summaries"]
        assert "line one line two line three" in summaries


# ---------------------------------------------------------------------------
# SQL schema — dotted and hyphenated table names
# ---------------------------------------------------------------------------


class TestNormalizeSqlSchemaDottedNames:
    def test_dotted_schema_table_name_extracted(self, tmp_path: Path) -> None:
        """Table names with dotted schema prefix (e.g., dbo.users) are
        captured correctly by the regex."""
        log = tmp_path / "t.log"
        _write_log(log, "CREATE TABLE dbo.users (id INT);")
        task = _make_task("t1", contract_type="sql-schema")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "dbo.users" in contract.metadata["tables"]

    def test_hyphenated_table_name_extracted(self, tmp_path: Path) -> None:
        """Table names containing hyphens are captured."""
        log = tmp_path / "t.log"
        _write_log(log, "CREATE TABLE `my-table` (id INT);")
        task = _make_task("t1", contract_type="sql-schema")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "my-table" in contract.metadata["tables"]


# ---------------------------------------------------------------------------
# _extract_primary_output — [judge] header
# ---------------------------------------------------------------------------


class TestExtractPrimaryOutputJudgeHeader:
    def test_judge_header_stops_body_collection(self, tmp_path: Path) -> None:
        """The [judge] section header (distinct from [judge_result]) also
        stops body collection."""
        log = tmp_path / "t.log"
        log.write_text(
            "header\n\nprimary output\n[judge]\njudge content\n",
            encoding="utf-8",
        )
        task = _make_task("t1", contract_type="conventions-doc")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "primary output" in contract.body
        assert "judge content" not in contract.body


# ---------------------------------------------------------------------------
# Content hash determinism
# ---------------------------------------------------------------------------


class TestContentHashDeterminism:
    def test_same_body_same_hash(self, tmp_path: Path) -> None:
        """Two normalization calls with the same body produce the same
        content_hash."""
        body = "CREATE TABLE t1 (id INT);"
        log1 = tmp_path / "a.log"
        log2 = tmp_path / "b.log"
        _write_log(log1, body)
        _write_log(log2, body)
        task = _make_task("t1", contract_type="sql-schema")
        c1 = normalize_task_contract(task, log1, "")
        c2 = normalize_task_contract(task, log2, "")
        assert c1 is not None and c2 is not None
        assert c1.content_hash == c2.content_hash

    def test_whitespace_trimmed_before_hashing(self, tmp_path: Path) -> None:
        """Leading/trailing whitespace in body is stripped before hashing, so
        bodies that differ only in surrounding whitespace share the same hash."""
        log1 = tmp_path / "a.log"
        log2 = tmp_path / "b.log"
        log1.write_text("header\n\n  hello world  \n", encoding="utf-8")
        log2.write_text("header\n\nhello world\n", encoding="utf-8")
        task = _make_task("t1", contract_type="conventions-doc")
        c1 = normalize_task_contract(task, log1, "")
        c2 = normalize_task_contract(task, log2, "")
        assert c1 is not None and c2 is not None
        assert c1.content_hash == c2.content_hash


# ---------------------------------------------------------------------------
# File inventory — non-string items coerced via str()
# ---------------------------------------------------------------------------


class TestNormalizeFileInventoryNonStringItems:
    def test_integer_items_coerced_to_string(self, tmp_path: Path) -> None:
        """JSON array containing integers is coerced to strings."""
        log = tmp_path / "t.log"
        _write_log(log, json.dumps([1, 2, 3]))
        task = _make_task("t1", contract_type="file-inventory")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert contract.metadata["file_count"] == 3
        assert "1" in contract.metadata["files"]

    def test_mixed_types_in_json_array(self, tmp_path: Path) -> None:
        """Array with mixed types (str, int, bool) all coerced."""
        log = tmp_path / "t.log"
        _write_log(log, json.dumps(["src/main.py", 42, True]))
        task = _make_task("t1", contract_type="file-inventory")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "src/main.py" in contract.metadata["files"]
        assert "42" in contract.metadata["files"]
        assert "True" in contract.metadata["files"]


# ---------------------------------------------------------------------------
# normalize_task_contract — whitespace-padded contract type
# ---------------------------------------------------------------------------


class TestNormalizeTaskContractWhitespacePaddedType:
    def test_whitespace_padded_type_still_dispatches_correctly(self, tmp_path: Path) -> None:
        """A contract_type with surrounding whitespace is stripped before
        dispatch, so '  conventions-doc  ' behaves the same as 'conventions-doc'."""
        log = tmp_path / "t.log"
        _write_log(log, "# Padded Type Heading")
        task = _make_task("t1", contract_type="  conventions-doc  ")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert contract.contract_type == "conventions-doc"
        assert "Padded Type Heading" in contract.metadata["headings"]

    def test_whitespace_padded_sql_type_dispatches(self, tmp_path: Path) -> None:
        """'  sql-schema  ' strips to 'sql-schema' and dispatches to SQL normalizer."""
        log = tmp_path / "t.log"
        _write_log(log, "CREATE TABLE padded (id INT);")
        task = _make_task("t1", contract_type="  sql-schema  ")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert contract.contract_type == "sql-schema"
        assert "padded" in contract.metadata["tables"]


# ---------------------------------------------------------------------------
# _normalize_file_inventory — dict 'files' key with whitespace-only items
# ---------------------------------------------------------------------------


class TestNormalizeFileInventoryDictFilesWhitespaceItems:
    def test_whitespace_only_items_in_dict_files_key_excluded(self, tmp_path: Path) -> None:
        """In the dict branch (payload['files'] is a list), items that strip
        to empty must be excluded from the resulting file list."""
        payload = json.dumps({"files": ["  ", "", "src/kept.py", "\t"]})
        log = tmp_path / "t.log"
        _write_log(log, payload)
        task = _make_task("t1", contract_type="file-inventory")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "src/kept.py" in contract.metadata["files"]
        assert "" not in contract.metadata["files"]
        assert "  " not in contract.metadata["files"]
        assert contract.metadata["file_count"] == 1


# ---------------------------------------------------------------------------
# _extract_primary_output — log starts immediately with blank line
# ---------------------------------------------------------------------------


class TestExtractPrimaryOutputImmediateBlankLine:
    def test_log_starting_with_blank_line_extracts_body(self, tmp_path: Path) -> None:
        """When the first line of the log is blank (no header at all), body
        extraction begins immediately.  Content after the blank line is returned."""
        log = tmp_path / "t.log"
        log.write_text("\n# Immediate Heading\nbody line\n", encoding="utf-8")
        task = _make_task("t1", contract_type="conventions-doc")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "Immediate Heading" in contract.metadata["headings"]
        assert "body line" in contract.body


# ---------------------------------------------------------------------------
# _normalize_conventions_doc — multi-level headings (#, ##, ###)
# ---------------------------------------------------------------------------


class TestNormalizeConventionsDocMultiLevelHeadings:
    def test_all_heading_levels_extracted(self, tmp_path: Path) -> None:
        """The _HEADING_RE pattern (^#{1,6}\\s+...) captures headings at all
        levels 1-6.  Verify ##, ###, and #### are included alongside #."""
        body = "# Top Level\n## Section\n### Subsection\n#### Detail\nplain text"
        log = tmp_path / "t.log"
        _write_log(log, body)
        task = _make_task("t1", contract_type="conventions-doc")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        headings = contract.metadata["headings"]
        assert "Top Level" in headings
        assert "Section" in headings
        assert "Subsection" in headings
        assert "Detail" in headings
        assert len(headings) == 4
        assert contract.summary.startswith("Conventions document with 4 heading(s)")


# ---------------------------------------------------------------------------
# _normalize_dependency_manifest — require-dev section only
# ---------------------------------------------------------------------------


class TestNormalizeDependencyManifestRequireDevOnly:
    def test_require_dev_section_extracted(self, tmp_path: Path) -> None:
        """The 'require-dev' key (composer dev-dependencies) must be parsed
        into the packages list even when 'require' is absent."""
        payload = json.dumps({"require-dev": {"phpunit/phpunit": "^10.0"}})
        log = tmp_path / "t.log"
        _write_log(log, payload)
        task = _make_task("t1", contract_type="dependency-manifest")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "phpunit/phpunit" in contract.metadata["packages"]
        assert contract.metadata["package_count"] == 1


# ---------------------------------------------------------------------------
# _normalize_dependency_manifest — normalized body is valid sorted JSON
# ---------------------------------------------------------------------------


class TestNormalizeDependencyManifestBodyFormat:
    def test_normalized_body_is_valid_json_with_sorted_packages(self, tmp_path: Path) -> None:
        """The normalized body for a valid dependency manifest must be valid
        JSON containing a sorted 'packages' list."""
        payload = json.dumps({"require": {"z-pkg": "^1.0", "a-pkg": "^2.0"}})
        log = tmp_path / "t.log"
        _write_log(log, payload)
        task = _make_task("t1", contract_type="dependency-manifest")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        body_parsed = json.loads(contract.body)
        assert "packages" in body_parsed
        # Packages must be sorted alphabetically
        assert body_parsed["packages"] == ["a-pkg", "z-pkg"]


# ---------------------------------------------------------------------------
# _extract_primary_output — multiple consecutive blank lines
# ---------------------------------------------------------------------------


class TestExtractPrimaryOutputMultipleBlankLines:
    def test_only_first_blank_line_triggers_body_start(self, tmp_path: Path) -> None:
        """Body extraction starts after the first blank line; subsequent blank
        lines within the body are preserved as part of the content."""
        log = tmp_path / "t.log"
        log.write_text(
            "header line\n\nfirst body line\n\nsecond body line\n",
            encoding="utf-8",
        )
        task = _make_task("t1", contract_type="conventions-doc")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "first body line" in contract.body
        assert "second body line" in contract.body


# ---------------------------------------------------------------------------
# build_consistency_template_vars — consistency_summary absent for empty groups
# ---------------------------------------------------------------------------


class TestBuildConsistencyTemplateVarsSummaryAbsent:
    def test_consistency_summary_absent_when_only_group_has_no_members(self) -> None:
        """When all reconcile_after groups map to empty member lists the
        summary_lines list stays empty and consistency_summary must NOT be set
        in the returned variables."""
        reconciler = _make_task("reconciler", reconcile_after=["grp"])
        group_members: dict[str, list[str]] = {"grp": []}

        vars_ = build_consistency_template_vars(reconciler, {}, group_members)

        assert "consistency_summary" not in vars_

    def test_consistency_summary_absent_when_group_not_in_group_members(self) -> None:
        """A reconcile_after group that has no entry in group_members at all
        must also not produce a consistency_summary key."""
        reconciler = _make_task("reconciler", reconcile_after=["ghost-group"])
        vars_ = build_consistency_template_vars(reconciler, {}, {})
        assert "consistency_summary" not in vars_


# ---------------------------------------------------------------------------
# build_contract_template_vars — hash field value equals sha256 of body
# ---------------------------------------------------------------------------


class TestBuildContractTemplateVarsHashValue:
    def test_hash_field_value_matches_sha256_of_body(self) -> None:
        """contract.X.hash in the template vars must equal the SHA-256 hex
        digest of the contract body string."""
        import hashlib

        body = "CREATE TABLE orders (id INT);"
        expected_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
        contract = _make_contract("db-task", contract_type="sql-schema", body=body)
        consumer = _make_task("consumer", consumes_contracts=["db-task"])
        results = {"db-task": _make_result("db-task", contract=contract)}

        vars_ = build_contract_template_vars(consumer, results)

        assert vars_["contract.db-task.hash"] == expected_hash

    def test_metadata_json_keys_are_sorted(self) -> None:
        """metadata_json must be produced with sort_keys=True so that the
        output is deterministic regardless of insertion order."""
        import hashlib

        body = "# Conventions"
        contract = _make_contract(
            "conv-task",
            contract_type="conventions-doc",
            body=body,
        )
        consumer = _make_task("consumer", consumes_contracts=["conv-task"])
        results = {"conv-task": _make_result("conv-task", contract=contract)}

        vars_ = build_contract_template_vars(consumer, results)
        raw_json = vars_["contract.conv-task.metadata_json"]
        # Parse and re-serialise with sorted keys — must produce the same string.
        parsed = json.loads(raw_json)
        re_serialised = json.dumps(parsed, ensure_ascii=True, sort_keys=True)
        assert raw_json == re_serialised


# ---------------------------------------------------------------------------
# _normalize_sql_schema — body consisting entirely of SQL comments
# ---------------------------------------------------------------------------


class TestNormalizeSqlSchemaOnlyComments:
    def test_sql_schema_body_only_comments_zero_statements(self, tmp_path: Path) -> None:
        """A body consisting entirely of SQL comment lines should produce a
        contract with zero statements and an empty tables list."""
        body = "-- this is a comment\n-- another comment\n-- third comment"
        log = tmp_path / "t.log"
        _write_log(log, body)
        task = _make_task("t1", contract_type="sql-schema")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert contract.metadata["statement_count"] == 0
        assert contract.metadata["tables"] == []
        assert "0 statement(s)" in contract.summary


# ---------------------------------------------------------------------------
# _normalize_dependency_manifest — JSON string value (not dict/array)
# ---------------------------------------------------------------------------


class TestNormalizeDependencyManifestJsonString:
    def test_dependency_manifest_json_string_falls_to_generic(self, tmp_path: Path) -> None:
        """A JSON string value (not a dict) should fall through to the generic
        contract path since it cannot contain package sections."""
        body = json.dumps("just a string value")
        log = tmp_path / "t.log"
        _write_log(log, body)
        task = _make_task("t1", contract_type="dependency-manifest")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert contract.contract_type == "dependency-manifest"
        # Generic path sets line_count and char_count metadata
        assert "line_count" in contract.metadata
        assert "char_count" in contract.metadata


# ---------------------------------------------------------------------------
# normalize_task_contract — whitespace-only contract_type returns None
# ---------------------------------------------------------------------------


class TestNormalizeTaskContractWhitespaceType:
    def test_whitespace_only_contract_type_returns_none(self, tmp_path: Path) -> None:
        """A contract_type consisting only of whitespace should be treated as
        empty after stripping and return None."""
        log = tmp_path / "t.log"
        _write_log(log, "some output")
        task = _make_task("t1", contract_type="   ")
        result = normalize_task_contract(task, log, "")
        assert result is None


# ---------------------------------------------------------------------------
# _extract_primary_output — section header with trailing text is not boundary
# ---------------------------------------------------------------------------


class TestExtractPrimaryOutputTrailingTextNotBoundary:
    def test_section_header_with_trailing_text_not_treated_as_boundary(
        self, tmp_path: Path
    ) -> None:
        """A line that starts like a section header but has trailing text (e.g.
        '[pre_command] something') must NOT stop body collection because section
        headers are matched by exact equality, not prefix."""
        log = tmp_path / "t.log"
        log.write_text(
            "header line\n\nfirst line\n[pre_command] extra text\nlast line\n",
            encoding="utf-8",
        )
        task = _make_task("t1", contract_type="conventions-doc")
        contract = normalize_task_contract(task, log, "")
        assert contract is not None
        assert "[pre_command] extra text" in contract.body
        assert "last line" in contract.body


# ---------------------------------------------------------------------------
# Feature 1: api-schema and test-manifest contract types
# ---------------------------------------------------------------------------


class TestApiSchemaContract:
    def _make_log(self, tmp_path: Path, content: str) -> Path:
        log = tmp_path / "task.log"
        log.write_text(f"\n{content}\n", encoding="utf-8")
        return log

    def test_valid_openapi_json(self, tmp_path: Path) -> None:
        body = json.dumps({
            "openapi": "3.0.0",
            "info": {"title": "Test API"},
            "paths": {"/users": {}, "/items": {}, "/health": {}},
            "components": {"schemas": {"User": {}, "Item": {}}},
        })
        log = self._make_log(tmp_path, body)
        task = _make_task("api-task", contract_type="api-schema")
        contract = normalize_task_contract(task, log, body)
        assert contract is not None
        assert contract.contract_type == "api-schema"
        assert contract.metadata["path_count"] == 3
        assert contract.metadata["schema_count"] == 2
        assert contract.metadata["openapi_version"] == "3.0.0"
        assert "3 path(s)" in contract.summary

    def test_swagger_v2(self, tmp_path: Path) -> None:
        body = json.dumps({
            "swagger": "2.0",
            "info": {"title": "Swagger API"},
            "paths": {"/pets": {}},
            "definitions": {"Pet": {}},
        })
        log = self._make_log(tmp_path, body)
        task = _make_task("api-task", contract_type="api-schema")
        contract = normalize_task_contract(task, log, body)
        assert contract is not None
        assert contract.metadata["openapi_version"] == "2.0"
        assert "/pets" in contract.metadata["paths"]

    def test_plain_text_falls_back_to_generic(self, tmp_path: Path) -> None:
        body = "Not JSON at all"
        log = self._make_log(tmp_path, body)
        task = _make_task("api-task", contract_type="api-schema")
        contract = normalize_task_contract(task, log, body)
        assert contract is not None
        assert contract.contract_type == "api-schema"
        # generic contract: has line_count metadata
        assert "line_count" in contract.metadata

    def test_empty_paths(self, tmp_path: Path) -> None:
        body = json.dumps({"openapi": "3.1.0", "paths": {}, "components": {}})
        log = self._make_log(tmp_path, body)
        task = _make_task("api-task", contract_type="api-schema")
        contract = normalize_task_contract(task, log, body)
        assert contract is not None
        assert contract.metadata["path_count"] == 0
        assert "0 path(s)" in contract.summary


class TestTestManifestContract:
    def _make_log(self, tmp_path: Path, content: str) -> Path:
        log = tmp_path / "task.log"
        log.write_text(f"\n{content}\n", encoding="utf-8")
        return log

    def test_pytest_json_format(self, tmp_path: Path) -> None:
        body = json.dumps({"passed": 10, "failed": 2, "skipped": 1, "total": 13})
        log = self._make_log(tmp_path, body)
        task = _make_task("test-task", contract_type="test-manifest")
        contract = normalize_task_contract(task, log, body)
        assert contract is not None
        assert contract.contract_type == "test-manifest"
        assert contract.metadata["passed"] == 10
        assert contract.metadata["failed"] == 2
        assert contract.metadata["skipped"] == 1
        assert contract.metadata["total"] == 13
        assert "10 passed" in contract.summary
        assert "2 failed" in contract.summary

    def test_jest_json_format(self, tmp_path: Path) -> None:
        body = json.dumps({"numPassedTests": 20, "numFailedTests": 0, "numTotalTests": 20})
        log = self._make_log(tmp_path, body)
        task = _make_task("test-task", contract_type="test-manifest")
        contract = normalize_task_contract(task, log, body)
        assert contract is not None
        assert contract.metadata["passed"] == 20
        assert contract.metadata["failed"] == 0

    def test_plain_text_format(self, tmp_path: Path) -> None:
        body = "15 passed, 3 failed, 2 skipped in 4.2s"
        log = self._make_log(tmp_path, body)
        task = _make_task("test-task", contract_type="test-manifest")
        contract = normalize_task_contract(task, log, body)
        assert contract is not None
        assert contract.metadata["passed"] == 15
        assert contract.metadata["failed"] == 3
        assert contract.metadata["skipped"] == 2

    def test_all_pass_no_failure_key(self, tmp_path: Path) -> None:
        body = json.dumps({"passed": 5, "total": 5})
        log = self._make_log(tmp_path, body)
        task = _make_task("test-task", contract_type="test-manifest")
        contract = normalize_task_contract(task, log, body)
        assert contract is not None
        assert contract.metadata["failed"] == 0
        assert contract.metadata["passed"] == 5

    def test_unrecognized_falls_back_to_generic(self, tmp_path: Path) -> None:
        body = "this output has no test counts at all"
        log = self._make_log(tmp_path, body)
        task = _make_task("test-task", contract_type="test-manifest")
        contract = normalize_task_contract(task, log, body)
        assert contract is not None
        # Falls back to generic — has line_count metadata
        assert "line_count" in contract.metadata
