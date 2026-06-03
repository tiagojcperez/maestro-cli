from __future__ import annotations

import pytest

from maestro_cli.models import TaskSpec
from maestro_cli.relationships import (
    build_consistency_group_members,
    clone_tasks_with_resolved_dependencies,
    resolve_task_dependencies,
)


class TestBuildConsistencyGroupMembers:
    def test_no_groups_returns_empty(self) -> None:
        tasks = [TaskSpec(id="a"), TaskSpec(id="b")]
        result = build_consistency_group_members(tasks)
        assert result == {}

    def test_single_group_collects_members(self) -> None:
        tasks = [
            TaskSpec(id="a", consistency_group=["grp1"]),
            TaskSpec(id="b", consistency_group=["grp1"]),
            TaskSpec(id="c"),
        ]
        result = build_consistency_group_members(tasks)
        assert set(result["grp1"]) == {"a", "b"}

    def test_multiple_groups(self) -> None:
        tasks = [
            TaskSpec(id="a", consistency_group=["grp1"]),
            TaskSpec(id="b", consistency_group=["grp2"]),
            TaskSpec(id="c", consistency_group=["grp1", "grp2"]),
        ]
        result = build_consistency_group_members(tasks)
        assert set(result["grp1"]) == {"a", "c"}
        assert set(result["grp2"]) == {"b", "c"}

    def test_no_duplicate_members(self) -> None:
        # Same ID appearing via two TaskSpec instances with same ID
        tasks = [
            TaskSpec(id="a", consistency_group=["grp1"]),
            TaskSpec(id="b", consistency_group=["grp1"]),
        ]
        result = build_consistency_group_members(tasks)
        assert result["grp1"].count("a") == 1
        assert result["grp1"].count("b") == 1

    def test_empty_task_list(self) -> None:
        result = build_consistency_group_members([])
        assert result == {}


class TestResolveTaskDependencies:
    def test_simple_depends_on_preserved(self) -> None:
        tasks = [
            TaskSpec(id="a"),
            TaskSpec(id="b", depends_on=["a"]),
        ]
        result = resolve_task_dependencies(tasks)
        assert result["b"] == ["a"]

    def test_consumes_contracts_added_to_deps(self) -> None:
        tasks = [
            TaskSpec(id="producer"),
            TaskSpec(id="consumer", consumes_contracts=["producer"]),
        ]
        result = resolve_task_dependencies(tasks)
        assert "producer" in result["consumer"]

    def test_reconcile_after_expands_group_members(self) -> None:
        tasks = [
            TaskSpec(id="impl-a", consistency_group=["my-group"]),
            TaskSpec(id="impl-b", consistency_group=["my-group"]),
            TaskSpec(id="reconciler", reconcile_after=["my-group"]),
        ]
        result = resolve_task_dependencies(tasks)
        deps = result["reconciler"]
        assert "impl-a" in deps
        assert "impl-b" in deps

    def test_self_reference_skipped(self) -> None:
        """Self-references via consumes_contracts are filtered by _add()."""
        tasks = [
            TaskSpec(id="a", consumes_contracts=["a"]),
        ]
        result = resolve_task_dependencies(tasks)
        assert "a" not in result["a"]

    def test_unknown_consumes_contracts_silently_ignored(self) -> None:
        tasks = [
            TaskSpec(id="a", consumes_contracts=["nonexistent"]),
        ]
        result = resolve_task_dependencies(tasks)
        assert "nonexistent" not in result["a"]

    def test_no_duplicates_when_deps_overlap(self) -> None:
        tasks = [
            TaskSpec(id="upstream"),
            TaskSpec(
                id="consumer",
                depends_on=["upstream"],
                consumes_contracts=["upstream"],
            ),
        ]
        result = resolve_task_dependencies(tasks)
        assert result["consumer"].count("upstream") == 1

    def test_empty_plan(self) -> None:
        result = resolve_task_dependencies([])
        assert result == {}

    def test_task_with_no_deps_has_empty_list(self) -> None:
        tasks = [TaskSpec(id="lone")]
        result = resolve_task_dependencies(tasks)
        assert result["lone"] == []

    def test_reconcile_after_unknown_group_adds_nothing(self) -> None:
        tasks = [
            TaskSpec(id="reconciler", reconcile_after=["no-such-group"]),
        ]
        result = resolve_task_dependencies(tasks)
        assert result["reconciler"] == []


class TestCloneTasksWithResolvedDependencies:
    def test_consumes_contracts_reflected_in_depends_on(self) -> None:
        tasks = [
            TaskSpec(id="producer"),
            TaskSpec(id="consumer", consumes_contracts=["producer"]),
        ]
        cloned = clone_tasks_with_resolved_dependencies(tasks)
        consumer = next(t for t in cloned if t.id == "consumer")
        assert "producer" in consumer.depends_on

    def test_original_tasks_not_mutated(self) -> None:
        tasks = [
            TaskSpec(id="producer"),
            TaskSpec(id="consumer", consumes_contracts=["producer"]),
        ]
        original_deps = list(tasks[1].depends_on)
        clone_tasks_with_resolved_dependencies(tasks)
        assert tasks[1].depends_on == original_deps

    def test_order_preserved(self) -> None:
        tasks = [
            TaskSpec(id="a"),
            TaskSpec(id="b"),
            TaskSpec(id="c"),
        ]
        cloned = clone_tasks_with_resolved_dependencies(tasks)
        assert [t.id for t in cloned] == ["a", "b", "c"]

    def test_cloned_task_ids_unchanged(self) -> None:
        tasks = [TaskSpec(id="x"), TaskSpec(id="y", depends_on=["x"])]
        cloned = clone_tasks_with_resolved_dependencies(tasks)
        assert {t.id for t in cloned} == {"x", "y"}

    def test_reconcile_after_excludes_self_from_group(self) -> None:
        """When reconciler is itself a member of the group, it should not
        depend on itself."""
        tasks = [
            TaskSpec(id="impl-a", consistency_group=["grp"]),
            TaskSpec(
                id="reconciler",
                consistency_group=["grp"],
                reconcile_after=["grp"],
            ),
        ]
        cloned = clone_tasks_with_resolved_dependencies(tasks)
        reconciler = next(t for t in cloned if t.id == "reconciler")
        assert "impl-a" in reconciler.depends_on
        assert "reconciler" not in reconciler.depends_on

    def test_depends_on_member_also_in_group_not_duplicated_after_clone(self) -> None:
        """If a dep appears in both depends_on and via reconcile_after, cloned
        task still has it only once."""
        tasks = [
            TaskSpec(id="impl-a", consistency_group=["grp"]),
            TaskSpec(
                id="reconciler",
                depends_on=["impl-a"],
                reconcile_after=["grp"],
            ),
        ]
        cloned = clone_tasks_with_resolved_dependencies(tasks)
        reconciler = next(t for t in cloned if t.id == "reconciler")
        assert reconciler.depends_on.count("impl-a") == 1

    def test_combined_deps_from_all_three_sources(self) -> None:
        """depends_on + consumes_contracts + reconcile_after all contribute."""
        tasks = [
            TaskSpec(id="base"),
            TaskSpec(id="producer"),
            TaskSpec(id="peer", consistency_group=["team"]),
            TaskSpec(
                id="consumer",
                depends_on=["base"],
                consumes_contracts=["producer"],
                reconcile_after=["team"],
            ),
        ]
        cloned = clone_tasks_with_resolved_dependencies(tasks)
        consumer = next(t for t in cloned if t.id == "consumer")
        assert "base" in consumer.depends_on
        assert "producer" in consumer.depends_on
        assert "peer" in consumer.depends_on

    def test_clone_empty_list_returns_empty(self) -> None:
        cloned = clone_tasks_with_resolved_dependencies([])
        assert cloned == []


# ---------------------------------------------------------------------------
# resolve_task_dependencies — depends_on with unknown IDs
# ---------------------------------------------------------------------------


class TestResolveTaskDepsUnknownIds:
    def test_unknown_depends_on_id_preserved_in_resolved(self) -> None:
        """Unlike consumes_contracts (filtered via _add), depends_on entries are
        added directly to the resolved list without a task_map check.  An
        unknown ID in depends_on should therefore remain in the resolved deps."""
        tasks = [
            TaskSpec(id="a", depends_on=["ghost-task"]),
        ]
        result = resolve_task_dependencies(tasks)
        assert "ghost-task" in result["a"]

    def test_ordering_depends_on_before_consumes_contracts(self) -> None:
        """depends_on entries must appear before consumes_contracts additions in
        the resolved list (insertion order is preserved)."""
        tasks = [
            TaskSpec(id="first"),
            TaskSpec(id="second"),
            TaskSpec(
                id="consumer",
                depends_on=["first"],
                consumes_contracts=["second"],
            ),
        ]
        result = resolve_task_dependencies(tasks)
        deps = result["consumer"]
        assert deps.index("first") < deps.index("second")


class TestCloneOnlyExplicitDeps:
    def test_clone_with_only_explicit_deps_keeps_them(self) -> None:
        """A task with only depends_on and no consumes_contracts/reconcile_after
        should have exactly those deps in the cloned version."""
        tasks = [
            TaskSpec(id="a"),
            TaskSpec(id="b"),
            TaskSpec(id="c", depends_on=["a", "b"]),
        ]
        cloned = clone_tasks_with_resolved_dependencies(tasks)
        c = next(t for t in cloned if t.id == "c")
        assert c.depends_on == ["a", "b"]

    def test_reconcile_after_two_groups_merges_all_members(self) -> None:
        """reconcile_after with two groups adds members from both groups."""
        tasks = [
            TaskSpec(id="x", consistency_group=["g1"]),
            TaskSpec(id="y", consistency_group=["g2"]),
            TaskSpec(id="z", consistency_group=["g2"]),
            TaskSpec(id="reconciler", reconcile_after=["g1", "g2"]),
        ]
        result = resolve_task_dependencies(tasks)
        deps = result["reconciler"]
        assert "x" in deps
        assert "y" in deps
        assert "z" in deps


# ---------------------------------------------------------------------------
# build_consistency_group_members — empty consistency_group list
# ---------------------------------------------------------------------------


class TestConsistencyGroupEmptyList:
    def test_empty_consistency_group_list_not_registered(self) -> None:
        """A task with consistency_group=[] doesn't appear in any group."""
        tasks = [
            TaskSpec(id="a", consistency_group=[]),
            TaskSpec(id="b", consistency_group=["grp1"]),
        ]
        result = build_consistency_group_members(tasks)
        assert "a" not in result.get("grp1", [])
        assert len(result) == 1

    def test_task_in_group_with_itself_only(self) -> None:
        """A single task in a consistency group produces a group with one member."""
        tasks = [TaskSpec(id="solo", consistency_group=["lonely"])]
        result = build_consistency_group_members(tasks)
        assert result["lonely"] == ["solo"]


# ---------------------------------------------------------------------------
# resolve_task_dependencies — duplicate IDs in depends_on
# ---------------------------------------------------------------------------


class TestResolveTaskDepsDuplicateDependsOn:
    def test_duplicate_entry_in_depends_on_appears_once(self) -> None:
        """When depends_on lists the same ID twice the resolved deps list must
        contain it only once (the ``if dep_id in seen: continue`` branch)."""
        tasks = [
            TaskSpec(id="upstream"),
            TaskSpec(id="consumer", depends_on=["upstream", "upstream"]),
        ]
        result = resolve_task_dependencies(tasks)
        assert result["consumer"].count("upstream") == 1

    def test_three_duplicate_depends_on_entries_deduped(self) -> None:
        """Three identical depends_on entries collapse to a single dep."""
        tasks = [
            TaskSpec(id="a"),
            TaskSpec(id="b", depends_on=["a", "a", "a"]),
        ]
        result = resolve_task_dependencies(tasks)
        assert result["b"] == ["a"]


# ---------------------------------------------------------------------------
# resolve_task_dependencies — reconcile_after across multiple groups, order
# ---------------------------------------------------------------------------


class TestResolveTaskDepsReconcileMultipleGroups:
    def test_reconcile_after_two_groups_appends_all_unique_members(self) -> None:
        """reconcile_after=[g1, g2] adds all unique members from both groups via
        the _add() closure without duplication."""
        tasks = [
            TaskSpec(id="x", consistency_group=["g1"]),
            TaskSpec(id="y", consistency_group=["g2"]),
            TaskSpec(id="reconciler", reconcile_after=["g1", "g2"]),
        ]
        result = resolve_task_dependencies(tasks)
        deps = result["reconciler"]
        assert "x" in deps
        assert "y" in deps
        assert deps.count("x") == 1
        assert deps.count("y") == 1

    def test_reconcile_after_member_also_in_depends_on_not_duplicated(self) -> None:
        """A group member that is already present in resolved deps via
        depends_on is not added again by the reconcile_after loop."""
        tasks = [
            TaskSpec(id="impl", consistency_group=["grp"]),
            TaskSpec(
                id="reconciler",
                depends_on=["impl"],
                reconcile_after=["grp"],
            ),
        ]
        result = resolve_task_dependencies(tasks)
        assert result["reconciler"].count("impl") == 1


# ---------------------------------------------------------------------------
# resolve_task_dependencies — depends_on self-reference is NOT filtered
# ---------------------------------------------------------------------------


class TestResolveTaskDepsSelfRefDependsOn:
    def test_depends_on_self_reference_is_preserved(self) -> None:
        """Unlike consumes_contracts which are filtered by _add() (checking
        dep_id == task.id), entries in depends_on go through the direct loop
        without a self-reference check.  A self-dep is therefore preserved."""
        tasks = [
            TaskSpec(id="a", depends_on=["a"]),
        ]
        result = resolve_task_dependencies(tasks)
        assert "a" in result["a"]


# ---------------------------------------------------------------------------
# resolve_task_dependencies — consumes_contracts + reconcile_after overlap
# ---------------------------------------------------------------------------


class TestResolveTaskDepsConsumesAndReconcileOverlap:
    def test_consumes_and_reconcile_same_target_deduped(self) -> None:
        """When the same task ID appears via both consumes_contracts and
        reconcile_after (group membership), it must only appear once in the
        resolved deps because _add() uses the seen set."""
        tasks = [
            TaskSpec(id="producer", consistency_group=["grp"]),
            TaskSpec(
                id="consumer",
                consumes_contracts=["producer"],
                reconcile_after=["grp"],
            ),
        ]
        result = resolve_task_dependencies(tasks)
        assert result["consumer"].count("producer") == 1
        assert len(result["consumer"]) == 1
