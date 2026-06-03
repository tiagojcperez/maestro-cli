from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from maestro_cli.models import DualVerificationResult, WorktreeMergeResult
from maestro_cli.worktree import (
    _extract_claimed_files,
    _normalize_path,
    _sanitize_branch_name,
    cleanup_worktree,
    create_worktree,
    get_base_branch,
    merge_worktree,
    verify_worktree_output,
)


def _mock_run(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    return MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)


def _reset_git_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("GIT_DIR", raising=False)
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path))


class TestGetBaseBranch:
    def test_returns_branch_name(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _reset_git_env(monkeypatch, tmp_path)

        def mock_fn(*args: Any, **kwargs: Any) -> MagicMock:
            assert args[0] == ["git", "rev-parse", "--abbrev-ref", "HEAD"]
            assert kwargs["cwd"] == tmp_path
            return _mock_run(stdout="main\n")

        monkeypatch.setattr("maestro_cli.worktree.subprocess.run", mock_fn)

        assert get_base_branch(tmp_path) == "main"

    def test_detached_head(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "maestro_cli.worktree.subprocess.run",
            lambda *args, **kwargs: _mock_run(stdout="HEAD\n"),
        )

        assert get_base_branch(tmp_path) == "HEAD"

    def test_not_git_repo(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "maestro_cli.worktree.subprocess.run",
            lambda *args, **kwargs: _mock_run(returncode=128, stderr="fatal: not a git repository"),
        )

        with pytest.raises(ValueError, match="git repository"):
            get_base_branch(tmp_path)

    def test_empty_output_raises_value_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "maestro_cli.worktree.subprocess.run",
            lambda *args, **kwargs: _mock_run(stdout=" \n"),
        )

        with pytest.raises(ValueError, match="unable to determine current git branch"):
            get_base_branch(tmp_path)

    def test_branch_name_is_stripped(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "maestro_cli.worktree.subprocess.run",
            lambda *args, **kwargs: _mock_run(stdout="  feature/worktree  \n"),
        )

        assert get_base_branch(tmp_path) == "feature/worktree"

    def test_oserror_raises_runtime_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _reset_git_env(monkeypatch, tmp_path)

        def mock_fn(*args: Any, **kwargs: Any) -> MagicMock:
            raise OSError("git missing")

        monkeypatch.setattr("maestro_cli.worktree.subprocess.run", mock_fn)

        with pytest.raises(RuntimeError, match="failed to run git: git missing"):
            get_base_branch(tmp_path)


class TestCreateWorktree:
    def test_creates_worktree_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        calls: list[list[str]] = []

        def mock_fn(*args: Any, **kwargs: Any) -> MagicMock:
            calls.append(list(args[0]))
            assert kwargs["cwd"] == tmp_path
            return _mock_run()

        monkeypatch.setattr("maestro_cli.worktree.subprocess.run", mock_fn)

        worktree_path = create_worktree(tmp_path, "task-123", base_branch="main")

        assert worktree_path == tmp_path / ".maestro-worktrees" / "task-123"
        assert worktree_path.parent.exists()
        assert calls == [
            [
                "git",
                "worktree",
                "add",
                str(worktree_path),
                "-b",
                "maestro/task-123",
                "main",
            ]
        ]

    @pytest.mark.parametrize(
        ("task_id", "expected_branch"),
        [
            ("task@1.2", "maestro/task-1-2"),
            ("task one/two", "maestro/task-one-two"),
        ],
    )
    def test_branch_name_sanitized(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        task_id: str,
        expected_branch: str,
    ) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        calls: list[list[str]] = []

        def mock_fn(*args: Any, **kwargs: Any) -> MagicMock:
            calls.append(list(args[0]))
            return _mock_run()

        monkeypatch.setattr("maestro_cli.worktree.subprocess.run", mock_fn)

        create_worktree(tmp_path, task_id, base_branch="main")

        assert calls[0][5] == expected_branch

    def test_failure_raises_runtime_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "maestro_cli.worktree.subprocess.run",
            lambda *args, **kwargs: _mock_run(returncode=1, stderr="bad branch"),
        )

        with pytest.raises(RuntimeError, match="failed to create worktree"):
            create_worktree(tmp_path, "task-123", base_branch="main")

    def test_failure_uses_stdout_when_stderr_empty(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "maestro_cli.worktree.subprocess.run",
            lambda *args, **kwargs: _mock_run(returncode=1, stdout="branch already exists"),
        )

        with pytest.raises(RuntimeError, match="branch already exists"):
            create_worktree(tmp_path, "task-123", base_branch="main")

    def test_failure_prefers_stderr_over_stdout(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "maestro_cli.worktree.subprocess.run",
            lambda *args, **kwargs: _mock_run(
                returncode=1,
                stdout="stdout detail",
                stderr="stderr detail",
            ),
        )

        with pytest.raises(RuntimeError, match="stderr detail"):
            create_worktree(tmp_path, "task-123", base_branch="main")

    def test_failure_uses_unknown_git_error_when_no_output(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "maestro_cli.worktree.subprocess.run",
            lambda *args, **kwargs: _mock_run(returncode=1),
        )

        with pytest.raises(RuntimeError, match="unknown git error"):
            create_worktree(tmp_path, "task-123", base_branch="main")

    def test_oserror_raises_runtime_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _reset_git_env(monkeypatch, tmp_path)

        def mock_fn(*args: Any, **kwargs: Any) -> MagicMock:
            raise OSError("git not found")

        monkeypatch.setattr("maestro_cli.worktree.subprocess.run", mock_fn)

        with pytest.raises(RuntimeError, match="failed to run git: git not found"):
            create_worktree(tmp_path, "task-123", base_branch="main")

    def test_uses_auto_detected_base_branch(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        calls: list[list[str]] = []
        responses = iter([_mock_run(stdout="develop\n"), _mock_run()])

        def mock_fn(*args: Any, **kwargs: Any) -> MagicMock:
            calls.append(list(args[0]))
            assert kwargs["cwd"] == tmp_path
            return next(responses)

        monkeypatch.setattr("maestro_cli.worktree.subprocess.run", mock_fn)

        worktree_path = create_worktree(tmp_path, "task-123")

        assert worktree_path == tmp_path / ".maestro-worktrees" / "task-123"
        assert calls == [
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            [
                "git",
                "worktree",
                "add",
                str(worktree_path),
                "-b",
                "maestro/task-123",
                "develop",
            ],
        ]

    def test_auto_detected_base_branch_failure_stops_before_worktree_add(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        calls: list[list[str]] = []

        def mock_fn(*args: Any, **kwargs: Any) -> MagicMock:
            calls.append(list(args[0]))
            return _mock_run(stdout=" \n")

        monkeypatch.setattr("maestro_cli.worktree.subprocess.run", mock_fn)

        with pytest.raises(ValueError, match="unable to determine current git branch"):
            create_worktree(tmp_path, "task-123")

        assert calls == [["git", "rev-parse", "--abbrev-ref", "HEAD"]]

    def test_existing_worktree_path_still_calls_git(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        worktree_path = tmp_path / ".maestro-worktrees" / "task-123"
        worktree_path.mkdir(parents=True)
        calls: list[list[str]] = []

        def mock_fn(*args: Any, **kwargs: Any) -> MagicMock:
            calls.append(list(args[0]))
            assert kwargs["cwd"] == tmp_path
            return _mock_run()

        monkeypatch.setattr("maestro_cli.worktree.subprocess.run", mock_fn)

        result = create_worktree(tmp_path, "task-123", base_branch="main")

        assert result == worktree_path
        assert calls == [
            [
                "git",
                "worktree",
                "add",
                str(worktree_path),
                "-b",
                "maestro/task-123",
                "main",
            ]
        ]

    def test_task_id_path_is_preserved_while_branch_name_is_sanitized(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        calls: list[list[str]] = []

        def mock_fn(*args: Any, **kwargs: Any) -> MagicMock:
            calls.append(list(args[0]))
            assert kwargs["cwd"] == tmp_path
            return _mock_run()

        monkeypatch.setattr("maestro_cli.worktree.subprocess.run", mock_fn)

        worktree_path = create_worktree(tmp_path, "feature/task@1", base_branch="main")

        assert worktree_path == tmp_path / ".maestro-worktrees" / "feature" / "task@1"
        assert worktree_path.parent.exists()
        assert calls == [
            [
                "git",
                "worktree",
                "add",
                str(worktree_path),
                "-b",
                "maestro/feature-task-1",
                "main",
            ]
        ]


class TestMergeWorktree:
    def test_merge_success(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        from maestro_cli.worktree import reset_merge_ledger
        reset_merge_ledger()
        responses = iter(
            [
                _mock_run(
                    stdout=(
                        "src/app.py | 2 +-\n"
                        "README.md | 4 ++--\n"
                        " 2 files changed, 3 insertions(+), 3 deletions(-)\n"
                    )
                ),
                _mock_run(),                         # checkout
                _mock_run(),                         # merge --no-commit --no-ff
                _mock_run(),                         # commit
                _mock_run(stdout="abc123\n"),         # rev-parse HEAD
            ]
        )

        monkeypatch.setattr(
            "maestro_cli.worktree.subprocess.run",
            lambda *args, **kwargs: next(responses),
        )

        result = merge_worktree(tmp_path, "task-123", tmp_path / "wt", "main")

        assert isinstance(result, WorktreeMergeResult)
        assert result.status == "merged"
        assert result.files_changed == ["src/app.py", "README.md"]
        assert result.merge_commit == "abc123"
        assert result.review is not None
        assert result.review.verdict == "safe"

    def test_merge_success_calls_expected_git_commands(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        from maestro_cli.worktree import reset_merge_ledger
        reset_merge_ledger()
        calls: list[list[str]] = []
        responses = iter(
            [
                _mock_run(stdout="src/app.py | 2 +-\n"),
                _mock_run(),                          # checkout
                _mock_run(),                          # merge --no-commit --no-ff
                _mock_run(),                          # commit
                _mock_run(stdout="abc123 \n"),         # rev-parse HEAD
            ]
        )

        def mock_fn(*args: Any, **kwargs: Any) -> MagicMock:
            calls.append(list(args[0]))
            assert kwargs["cwd"] == tmp_path
            return next(responses)

        monkeypatch.setattr("maestro_cli.worktree.subprocess.run", mock_fn)

        result = merge_worktree(tmp_path, "task@1.2", tmp_path / "ignored", "main")

        assert isinstance(result, WorktreeMergeResult)
        assert result.status == "merged"
        assert result.merge_commit == "abc123"
        assert calls == [
            ["git", "diff", "main...maestro/task-1-2", "--stat"],
            ["git", "checkout", "main"],
            ["git", "merge", "--no-commit", "--no-ff", "maestro/task-1-2"],
            ["git", "commit", "-m", "maestro: merge task@1.2"],
            ["git", "rev-parse", "HEAD"],
        ]

    def test_merge_success_ignores_non_file_stat_lines(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        from maestro_cli.worktree import reset_merge_ledger
        reset_merge_ledger()
        responses = iter(
            [
                _mock_run(
                    stdout=(
                        " src/app.py | 2 +-\n"
                        " docs/guide.md | 5 +++--\n"
                        " 2 files changed, 4 insertions(+), 3 deletions(-)\n"
                        " create mode 100644 docs/guide.md\n"
                    )
                ),
                _mock_run(),                          # checkout
                _mock_run(),                          # merge --no-commit --no-ff
                _mock_run(),                          # commit
                _mock_run(stdout="def456\n"),          # rev-parse HEAD
            ]
        )

        monkeypatch.setattr(
            "maestro_cli.worktree.subprocess.run",
            lambda *args, **kwargs: next(responses),
        )

        result = merge_worktree(tmp_path, "task-123", tmp_path / "wt", "main")

        assert isinstance(result, WorktreeMergeResult)
        assert result.status == "merged"
        assert result.files_changed == ["src/app.py", "docs/guide.md"]
        assert result.merge_commit == "def456"

    def test_merge_empty_no_changes(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "maestro_cli.worktree.subprocess.run",
            lambda *args, **kwargs: _mock_run(stdout=""),
        )

        result = merge_worktree(tmp_path, "task-123", tmp_path / "wt", "main")

        assert isinstance(result, WorktreeMergeResult)
        assert result.status == "empty"
        assert result.files_changed == []

    def test_merge_empty_ignores_summary_only_diff(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "maestro_cli.worktree.subprocess.run",
            lambda *args, **kwargs: _mock_run(
                stdout=(
                    " | 0\n"
                    "------\n"
                    "0 files changed\n"
                )
            ),
        )

        result = merge_worktree(tmp_path, "task-123", tmp_path / "wt", "main")

        assert isinstance(result, WorktreeMergeResult)
        assert result.status == "empty"
        assert result.files_changed == []

    def test_merge_conflict(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from maestro_cli.models import MergeReview
        from maestro_cli.worktree import reset_merge_ledger
        _reset_git_env(monkeypatch, tmp_path)
        reset_merge_ledger()
        calls: list[list[str]] = []
        responses = iter(
            [
                _mock_run(stdout="src/app.py | 2 +-\n"),
                _mock_run(),                          # checkout
                _mock_run(
                    returncode=1,
                    stdout="CONFLICT (content): Merge conflict in src/app.py\n",
                    stderr="Automatic merge failed; fix conflicts and then commit the result.\n",
                ),
                _mock_run(),                          # merge --abort
            ]
        )

        def mock_fn(*args: Any, **kwargs: Any) -> MagicMock:
            calls.append(list(args[0]))
            return next(responses)

        monkeypatch.setattr("maestro_cli.worktree.subprocess.run", mock_fn)
        # Skip LLM review in test
        monkeypatch.setattr(
            "maestro_cli.worktree._build_conflict_review",
            lambda *a, **kw: MergeReview(verdict="conflict", conflict_files=["src/app.py"]),
        )

        result = merge_worktree(tmp_path, "task-123", tmp_path / "wt", "main")

        assert isinstance(result, WorktreeMergeResult)
        assert result.status == "conflict"
        assert result.conflict_files == ["src/app.py"]
        assert calls[-1] == ["git", "merge", "--abort"]

    def test_merge_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _reset_git_env(monkeypatch, tmp_path)

        def mock_fn(*args: Any, **kwargs: Any) -> MagicMock:
            raise OSError("git unavailable")

        monkeypatch.setattr("maestro_cli.worktree.subprocess.run", mock_fn)

        result = merge_worktree(tmp_path, "task-123", tmp_path / "wt", "main")

        assert isinstance(result, WorktreeMergeResult)
        assert result.status == "error"
        assert "git unavailable" in (result.error or "")

    def test_merge_checkout_failure_returns_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        calls: list[list[str]] = []
        responses = iter(
            [
                _mock_run(stdout="src/app.py | 2 +-\n"),
                _mock_run(returncode=1, stderr="pathspec 'main' did not match"),
            ]
        )

        def mock_fn(*args: Any, **kwargs: Any) -> MagicMock:
            calls.append(list(args[0]))
            return next(responses)

        monkeypatch.setattr("maestro_cli.worktree.subprocess.run", mock_fn)

        result = merge_worktree(tmp_path, "task-123", tmp_path / "wt", "main")

        assert isinstance(result, WorktreeMergeResult)
        assert result.status == "error"
        assert result.error == "pathspec 'main' did not match"
        assert calls == [
            ["git", "diff", "main...maestro/task-123", "--stat"],
            ["git", "checkout", "main"],
        ]

    def test_merge_checkout_failure_uses_stdout_when_stderr_empty(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        responses = iter(
            [
                _mock_run(stdout="src/app.py | 2 +-\n"),
                _mock_run(returncode=1, stdout="checkout failed"),
            ]
        )

        monkeypatch.setattr(
            "maestro_cli.worktree.subprocess.run",
            lambda *args, **kwargs: next(responses),
        )

        result = merge_worktree(tmp_path, "task-123", tmp_path / "wt", "main")

        assert isinstance(result, WorktreeMergeResult)
        assert result.status == "error"
        assert result.error == "checkout failed"

    def test_merge_diff_failure_returns_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        calls: list[list[str]] = []

        def mock_fn(*args: Any, **kwargs: Any) -> MagicMock:
            calls.append(list(args[0]))
            return _mock_run(returncode=1, stderr="diff failed")

        monkeypatch.setattr("maestro_cli.worktree.subprocess.run", mock_fn)

        result = merge_worktree(tmp_path, "task-123", tmp_path / "wt", "main")

        assert isinstance(result, WorktreeMergeResult)
        assert result.status == "error"
        assert result.error == "diff failed"
        assert calls == [["git", "diff", "main...maestro/task-123", "--stat"]]

    def test_merge_diff_failure_uses_stdout_when_stderr_empty(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "maestro_cli.worktree.subprocess.run",
            lambda *args, **kwargs: _mock_run(returncode=1, stdout="diff fallback"),
        )

        result = merge_worktree(tmp_path, "task-123", tmp_path / "wt", "main")

        assert isinstance(result, WorktreeMergeResult)
        assert result.status == "error"
        assert result.error == "diff fallback"

    def test_merge_diff_failure_uses_unknown_git_error_when_no_output(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "maestro_cli.worktree.subprocess.run",
            lambda *args, **kwargs: _mock_run(returncode=1),
        )

        result = merge_worktree(tmp_path, "task-123", tmp_path / "wt", "main")

        assert isinstance(result, WorktreeMergeResult)
        assert result.status == "error"
        assert result.error == "unknown git error"

    def test_merge_head_lookup_failure_returns_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        from maestro_cli.worktree import reset_merge_ledger
        reset_merge_ledger()
        responses = iter(
            [
                _mock_run(stdout="src/app.py | 2 +-\n"),
                _mock_run(),                          # checkout
                _mock_run(),                          # merge --no-commit (success)
                _mock_run(returncode=1, stderr="commit failed"),  # commit fails
                _mock_run(),                          # merge --abort
            ]
        )

        monkeypatch.setattr(
            "maestro_cli.worktree.subprocess.run",
            lambda *args, **kwargs: next(responses),
        )

        result = merge_worktree(tmp_path, "task-123", tmp_path / "wt", "main")

        assert isinstance(result, WorktreeMergeResult)
        assert result.status == "error"
        assert result.files_changed == ["src/app.py"]
        assert result.error == "commit failed"

    def test_merge_head_lookup_failure_uses_unknown_git_error_when_no_output(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        from maestro_cli.worktree import reset_merge_ledger
        reset_merge_ledger()
        responses = iter(
            [
                _mock_run(stdout="src/app.py | 2 +-\n"),
                _mock_run(),                          # checkout
                _mock_run(),                          # merge --no-commit (success)
                _mock_run(returncode=1),              # commit fails (no output)
                _mock_run(),                          # merge --abort
            ]
        )

        monkeypatch.setattr(
            "maestro_cli.worktree.subprocess.run",
            lambda *args, **kwargs: next(responses),
        )

        result = merge_worktree(tmp_path, "task-123", tmp_path / "wt", "main")

        assert isinstance(result, WorktreeMergeResult)
        assert result.status == "error"
        assert result.files_changed == ["src/app.py"]
        assert result.error == "commit failed"

    def test_merge_head_lookup_failure_uses_stdout_when_stderr_empty(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        from maestro_cli.worktree import reset_merge_ledger
        reset_merge_ledger()
        responses = iter(
            [
                _mock_run(stdout="src/app.py | 2 +-\n"),
                _mock_run(),                          # checkout
                _mock_run(),                          # merge --no-commit (success)
                _mock_run(returncode=1, stdout="commit fallback"),  # commit fails
                _mock_run(),                          # merge --abort
            ]
        )

        monkeypatch.setattr(
            "maestro_cli.worktree.subprocess.run",
            lambda *args, **kwargs: next(responses),
        )

        result = merge_worktree(tmp_path, "task-123", tmp_path / "wt", "main")

        assert isinstance(result, WorktreeMergeResult)
        assert result.status == "error"
        assert result.files_changed == ["src/app.py"]
        assert result.error == "commit fallback"

    def test_merge_conflict_collects_files_from_stdout_and_stderr(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from maestro_cli.models import MergeReview
        from maestro_cli.worktree import reset_merge_ledger
        _reset_git_env(monkeypatch, tmp_path)
        reset_merge_ledger()
        calls: list[list[str]] = []
        responses = iter(
            [
                _mock_run(stdout="src/app.py | 2 +-\nsrc/lib.py | 4 ++--\n"),
                _mock_run(),                          # checkout
                _mock_run(
                    returncode=1,
                    stdout="CONFLICT (content): Merge conflict in src/app.py\n",
                    stderr="CONFLICT (modify/delete): Merge conflict in src/lib.py\n",
                ),
                _mock_run(),                          # merge --abort
            ]
        )

        def mock_fn(*args: Any, **kwargs: Any) -> MagicMock:
            calls.append(list(args[0]))
            return next(responses)

        monkeypatch.setattr("maestro_cli.worktree.subprocess.run", mock_fn)
        monkeypatch.setattr(
            "maestro_cli.worktree._build_conflict_review",
            lambda *a, **kw: MergeReview(verdict="conflict", conflict_files=["src/app.py", "src/lib.py"]),
        )

        result = merge_worktree(tmp_path, "task-123", tmp_path / "wt", "main")

        assert isinstance(result, WorktreeMergeResult)
        assert result.status == "conflict"
        assert result.files_changed == ["src/app.py", "src/lib.py"]
        assert result.conflict_files == ["src/app.py", "src/lib.py"]
        assert calls[-1] == ["git", "merge", "--abort"]

    def test_merge_conflict_without_markers_returns_empty_conflict_files(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from maestro_cli.models import MergeReview
        from maestro_cli.worktree import reset_merge_ledger
        _reset_git_env(monkeypatch, tmp_path)
        reset_merge_ledger()
        calls: list[list[str]] = []
        responses = iter(
            [
                _mock_run(stdout="src/app.py | 2 +-\n"),
                _mock_run(),                          # checkout
                _mock_run(
                    returncode=1,
                    stdout="Automatic merge failed.\n",
                    stderr="Resolve the merge manually.\n",
                ),
                _mock_run(),                          # merge --abort
            ]
        )

        def mock_fn(*args: Any, **kwargs: Any) -> MagicMock:
            calls.append(list(args[0]))
            return next(responses)

        monkeypatch.setattr("maestro_cli.worktree.subprocess.run", mock_fn)
        monkeypatch.setattr(
            "maestro_cli.worktree._build_conflict_review",
            lambda *a, **kw: MergeReview(verdict="conflict"),
        )

        result = merge_worktree(tmp_path, "task-123", tmp_path / "wt", "main")

        assert isinstance(result, WorktreeMergeResult)
        assert result.status == "conflict"
        assert result.files_changed == ["src/app.py"]
        assert result.conflict_files == []
        assert calls[-1] == ["git", "merge", "--abort"]

    def test_merge_conflict_abort_failure_returns_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from maestro_cli.worktree import reset_merge_ledger
        _reset_git_env(monkeypatch, tmp_path)
        reset_merge_ledger()
        calls: list[list[str]] = []
        responses = iter(
            [
                _mock_run(stdout="src/app.py | 2 +-\n"),
                _mock_run(),                          # checkout
                _mock_run(
                    returncode=1,
                    stdout="CONFLICT (content): Merge conflict in src/app.py\n",
                ),
            ]
        )

        def mock_fn(*args: Any, **kwargs: Any) -> MagicMock:
            calls.append(list(args[0]))
            if args[0][1:3] == ["merge", "--abort"]:
                raise OSError("abort failed")
            return next(responses)

        monkeypatch.setattr("maestro_cli.worktree.subprocess.run", mock_fn)

        result = merge_worktree(tmp_path, "task-123", tmp_path / "wt", "main")

        assert isinstance(result, WorktreeMergeResult)
        assert result.status == "error"
        assert "abort failed" in (result.error or "")
        assert calls == [
            ["git", "diff", "main...maestro/task-123", "--stat"],
            ["git", "checkout", "main"],
            ["git", "merge", "--no-commit", "--no-ff", "maestro/task-123"],
            ["git", "merge", "--abort"],
        ]


class TestCleanupWorktree:
    def test_cleanup_removes_worktree(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        calls: list[list[str]] = []

        def mock_fn(*args: Any, **kwargs: Any) -> MagicMock:
            calls.append(list(args[0]))
            return _mock_run()

        monkeypatch.setattr("maestro_cli.worktree.subprocess.run", mock_fn)

        cleanup_worktree(tmp_path, "task@1.2", tmp_path / "wt")

        assert calls == [
            ["git", "worktree", "remove", str(tmp_path / "wt"), "--force"],
            ["git", "branch", "-D", "maestro/task-1-2"],
        ]

    def test_cleanup_ignores_failures(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        calls: list[list[str]] = []

        def mock_fn(*args: Any, **kwargs: Any) -> MagicMock:
            calls.append(list(args[0]))
            raise OSError("git unavailable")

        monkeypatch.setattr("maestro_cli.worktree.subprocess.run", mock_fn)

        cleanup_worktree(tmp_path, "task-123", tmp_path / "wt")

        assert calls == [
            ["git", "worktree", "remove", str(tmp_path / "wt"), "--force"],
            ["git", "branch", "-D", "maestro/task-123"],
        ]

    def test_cleanup_ignores_nonzero_git_returncodes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        calls: list[list[str]] = []
        responses = iter(
            [
                _mock_run(returncode=1, stderr="worktree already removed"),
                _mock_run(returncode=1, stderr="branch not found"),
            ]
        )

        def mock_fn(*args: Any, **kwargs: Any) -> MagicMock:
            calls.append(list(args[0]))
            return next(responses)

        monkeypatch.setattr("maestro_cli.worktree.subprocess.run", mock_fn)

        cleanup_worktree(tmp_path, "task-123", tmp_path / "wt")

        assert calls == [
            ["git", "worktree", "remove", str(tmp_path / "wt"), "--force"],
            ["git", "branch", "-D", "maestro/task-123"],
        ]

    def test_cleanup_deletes_branch_when_worktree_remove_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        calls: list[list[str]] = []

        def mock_fn(*args: Any, **kwargs: Any) -> MagicMock:
            calls.append(list(args[0]))
            if args[0][1:3] == ["worktree", "remove"]:
                raise OSError("worktree missing")
            return _mock_run()

        monkeypatch.setattr("maestro_cli.worktree.subprocess.run", mock_fn)

        cleanup_worktree(tmp_path, "task-123", tmp_path / "wt")

        assert calls == [
            ["git", "worktree", "remove", str(tmp_path / "wt"), "--force"],
            ["git", "branch", "-D", "maestro/task-123"],
        ]


class TestSanitizeBranchName:
    def test_alphanumeric_unchanged(self) -> None:
        assert _sanitize_branch_name("my-task-1") == "my-task-1"

    @pytest.mark.parametrize(
        ("task_id", "expected"),
        [("task@1.2", "task-1-2"), ("task one/two", "task-one-two")],
    )
    def test_special_chars_replaced(self, task_id: str, expected: str) -> None:
        assert _sanitize_branch_name(task_id) == expected

    def test_unicode_chars_replaced(self) -> None:
        assert _sanitize_branch_name("naïve-task") == "na-ve-task"

    def test_consecutive_special_chars_are_replaced_individually(self) -> None:
        assert _sanitize_branch_name("task..@@ name") == "task-----name"


# ---------------------------------------------------------------------------
# NEW COVERAGE: parsing helpers, merge ledger, overlap detection,
# dataclass serialization, conflict review, review_callback, edge cases
# ---------------------------------------------------------------------------

from maestro_cli.models import MergeOverlap, MergeReview, MergeReviewVerdict
from maestro_cli.worktree import (
    _branch_name,
    _get_overlaps,
    _parse_changed_files,
    _parse_conflict_files,
    _record_merged_files,
    reset_merge_ledger,
)


class TestBranchName:
    """Direct coverage for _branch_name (only tested indirectly before)."""

    def test_simple_task_id(self) -> None:
        assert _branch_name("my-task") == "maestro/my-task"

    def test_special_chars_sanitized(self) -> None:
        assert _branch_name("task@1.2") == "maestro/task-1-2"

    def test_underscores_preserved(self) -> None:
        assert _branch_name("task_with_underscores") == "maestro/task_with_underscores"

    def test_empty_string(self) -> None:
        assert _branch_name("") == "maestro/"


class TestParseChangedFiles:
    """Direct tests for _parse_changed_files parsing helper."""

    def test_typical_stat_output(self) -> None:
        output = (
            " src/app.py | 2 +-\n"
            " README.md  | 4 ++--\n"
            " 2 files changed, 3 insertions(+), 3 deletions(-)\n"
        )
        assert _parse_changed_files(output) == ["src/app.py", "README.md"]

    def test_empty_string_returns_empty(self) -> None:
        assert _parse_changed_files("") == []

    def test_whitespace_only_returns_empty(self) -> None:
        assert _parse_changed_files("   \n  \n") == []

    def test_summary_line_only_returns_empty(self) -> None:
        assert _parse_changed_files("0 files changed") == []

    def test_dashes_separator_line_ignored(self) -> None:
        assert _parse_changed_files("-------") == []

    def test_pipe_bar_header_lines_ignored(self) -> None:
        # Lines starting with | are filtered
        assert _parse_changed_files("| header |\n") == []

    def test_line_without_pipe_ignored(self) -> None:
        assert _parse_changed_files("create mode 100644 docs/guide.md\n") == []

    def test_binary_file_stat(self) -> None:
        output = " image.png | Bin 0 -> 1234 bytes\n"
        assert _parse_changed_files(output) == ["image.png"]

    def test_single_file(self) -> None:
        assert _parse_changed_files("setup.py | 1 +\n") == ["setup.py"]


class TestParseConflictFiles:
    """Direct tests for _parse_conflict_files parsing helper."""

    def test_content_conflict_in_stdout(self) -> None:
        stdout = "CONFLICT (content): Merge conflict in src/app.py\n"
        assert _parse_conflict_files(stdout, "") == ["src/app.py"]

    def test_modify_delete_conflict_in_stderr(self) -> None:
        stderr = "CONFLICT (modify/delete): Merge conflict in lib.py\n"
        assert _parse_conflict_files("", stderr) == ["lib.py"]

    def test_multiple_conflicts_across_stdout_and_stderr(self) -> None:
        stdout = "CONFLICT (content): Merge conflict in a.py\n"
        stderr = "CONFLICT (rename/delete): Merge conflict in b.py\n"
        assert _parse_conflict_files(stdout, stderr) == ["a.py", "b.py"]

    def test_no_conflict_markers_returns_empty(self) -> None:
        assert _parse_conflict_files("Merge failed\n", "error\n") == []

    def test_empty_input_returns_empty(self) -> None:
        assert _parse_conflict_files("", "") == []

    def test_conflict_file_with_spaces_in_path(self) -> None:
        stdout = "CONFLICT (content): Merge conflict in path/with spaces/file.py\n"
        assert _parse_conflict_files(stdout, "") == ["path/with spaces/file.py"]


class TestMergeLedger:
    """Tests for the merge ledger: reset, record, overlap detection."""

    def test_reset_clears_ledger(self) -> None:
        _record_merged_files("task-a", ["file1.py"])
        reset_merge_ledger()
        assert _get_overlaps(["file1.py"]) == []

    def test_record_and_detect_overlap(self) -> None:
        reset_merge_ledger()
        _record_merged_files("task-a", ["shared.py", "other.py"])
        overlaps = _get_overlaps(["shared.py", "new.py"])
        assert len(overlaps) == 1
        assert overlaps[0].file == "shared.py"
        assert overlaps[0].merged_by == ["task-a"]

    def test_multiple_tasks_overlap_same_file(self) -> None:
        reset_merge_ledger()
        _record_merged_files("task-a", ["shared.py"])
        _record_merged_files("task-b", ["shared.py"])
        overlaps = _get_overlaps(["shared.py"])
        assert len(overlaps) == 1
        assert overlaps[0].merged_by == ["task-a", "task-b"]

    def test_no_overlap_when_different_files(self) -> None:
        reset_merge_ledger()
        _record_merged_files("task-a", ["a.py"])
        assert _get_overlaps(["b.py"]) == []

    def test_empty_files_list(self) -> None:
        reset_merge_ledger()
        _record_merged_files("task-a", [])
        assert _get_overlaps([]) == []


class TestMergeWorktreeOverlaps:
    """Test the overlap detection path within merge_worktree."""

    def test_merge_success_with_overlaps_returns_resolvable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        reset_merge_ledger()
        # Pre-populate ledger so src/app.py overlaps
        _record_merged_files("prev-task", ["src/app.py"])

        responses = iter(
            [
                _mock_run(stdout="src/app.py | 2 +-\n"),
                _mock_run(),                         # checkout
                _mock_run(),                         # merge --no-commit --no-ff
                _mock_run(),                         # commit
                _mock_run(stdout="deadbeef\n"),       # rev-parse HEAD
            ]
        )

        monkeypatch.setattr(
            "maestro_cli.worktree.subprocess.run",
            lambda *args, **kwargs: next(responses),
        )

        result = merge_worktree(tmp_path, "task-overlap", tmp_path / "wt", "main")

        assert result.status == "merged"
        assert result.review is not None
        assert result.review.verdict == "resolvable"
        assert len(result.review.overlapping_files) == 1
        assert result.review.overlapping_files[0].file == "src/app.py"
        assert result.review.overlapping_files[0].merged_by == ["prev-task"]


class TestMergeWorktreeReviewCallback:
    """Test review_callback invocation on conflict."""

    def test_review_callback_invoked_on_conflict(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        reset_merge_ledger()
        responses = iter(
            [
                _mock_run(stdout="src/app.py | 2 +-\n"),
                _mock_run(),                          # checkout
                _mock_run(
                    returncode=1,
                    stdout="CONFLICT (content): Merge conflict in src/app.py\n",
                ),
                _mock_run(),                          # merge --abort
            ]
        )

        monkeypatch.setattr(
            "maestro_cli.worktree.subprocess.run",
            lambda *args, **kwargs: next(responses),
        )
        monkeypatch.setattr(
            "maestro_cli.worktree._build_conflict_review",
            lambda *a, **kw: MergeReview(
                verdict="conflict",
                conflict_files=["src/app.py"],
                resolution_suggestion="Manual resolve needed",
            ),
        )

        callback_calls: list[tuple[str, dict[str, object]]] = []

        def cb(event_type: str, payload: dict[str, object]) -> None:
            callback_calls.append((event_type, payload))

        result = merge_worktree(
            tmp_path, "task-cb", tmp_path / "wt", "main",
            review_callback=cb,
        )

        assert result.status == "conflict"
        assert len(callback_calls) == 1
        assert callback_calls[0][0] == "worktree_review"
        assert callback_calls[0][1]["task_id"] == "task-cb"
        assert callback_calls[0][1]["verdict"] == "conflict"

    def test_review_callback_exception_does_not_break_merge(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        reset_merge_ledger()
        responses = iter(
            [
                _mock_run(stdout="src/app.py | 2 +-\n"),
                _mock_run(),                          # checkout
                _mock_run(
                    returncode=1,
                    stdout="CONFLICT (content): Merge conflict in src/app.py\n",
                ),
                _mock_run(),                          # merge --abort
            ]
        )

        monkeypatch.setattr(
            "maestro_cli.worktree.subprocess.run",
            lambda *args, **kwargs: next(responses),
        )
        monkeypatch.setattr(
            "maestro_cli.worktree._build_conflict_review",
            lambda *a, **kw: MergeReview(verdict="conflict", conflict_files=["src/app.py"]),
        )

        def bad_callback(event_type: str, payload: dict[str, object]) -> None:
            raise ValueError("callback exploded")

        result = merge_worktree(
            tmp_path, "task-bad-cb", tmp_path / "wt", "main",
            review_callback=bad_callback,
        )

        # Should still return conflict, not error
        assert result.status == "conflict"


class TestMergeWorktreeRevParseFailure:
    """Test rev-parse HEAD failure after successful commit."""

    def test_merge_commit_is_none_when_rev_parse_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        reset_merge_ledger()
        responses = iter(
            [
                _mock_run(stdout="src/app.py | 2 +-\n"),
                _mock_run(),                          # checkout
                _mock_run(),                          # merge --no-commit --no-ff
                _mock_run(),                          # commit
                _mock_run(returncode=1, stderr="HEAD missing"),  # rev-parse fails
            ]
        )

        monkeypatch.setattr(
            "maestro_cli.worktree.subprocess.run",
            lambda *args, **kwargs: next(responses),
        )

        result = merge_worktree(tmp_path, "task-nohead", tmp_path / "wt", "main")

        assert result.status == "merged"
        assert result.merge_commit is None


class TestBuildConflictReview:
    """Test _build_conflict_review (LLM conflict analysis, previously mocked out)."""

    def test_no_diff_output_returns_basic_review(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from maestro_cli.worktree import _build_conflict_review

        _reset_git_env(monkeypatch, tmp_path)
        # All diff calls return empty
        monkeypatch.setattr(
            "maestro_cli.worktree.subprocess.run",
            lambda *args, **kwargs: _mock_run(returncode=1),
        )

        review = _build_conflict_review(
            workspace_root=tmp_path,
            task_id="task-x",
            branch_name="maestro/task-x",
            base_branch="main",
            files_changed=["a.py"],
            conflict_files=["a.py"],
            overlaps=[],
        )

        assert review.verdict == "conflict"
        assert review.conflict_files == ["a.py"]
        assert review.resolution_suggestion is None

    def test_llm_call_failure_returns_fallback_review(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from maestro_cli.worktree import _build_conflict_review

        _reset_git_env(monkeypatch, tmp_path)
        call_count = [0]

        def mock_fn(*args: Any, **kwargs: Any) -> MagicMock:
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: diff returns content
                return _mock_run(stdout="diff content here\n")
            # LLM call raises
            raise OSError("no claude")

        monkeypatch.setattr("maestro_cli.worktree.subprocess.run", mock_fn)
        # Mock the imports from runners
        monkeypatch.setattr(
            "maestro_cli.worktree._build_safe_env",
            lambda env, secrets: {},
            raising=False,
        )
        # Need to mock at import level inside the function
        import maestro_cli.worktree as wt_mod
        orig_build = wt_mod._build_conflict_review

        # Actually, let's mock the runners import
        monkeypatch.setattr(
            "maestro_cli.runners._build_safe_env",
            lambda env, secrets: {},
            raising=False,
        )
        monkeypatch.setattr(
            "maestro_cli.runners._resolve_executable",
            lambda name: ["claude"],
            raising=False,
        )

        review = _build_conflict_review(
            workspace_root=tmp_path,
            task_id="task-y",
            branch_name="maestro/task-y",
            base_branch="main",
            files_changed=["b.py"],
            conflict_files=["b.py"],
            overlaps=[],
        )

        assert review.verdict == "conflict"
        assert review.conflict_files == ["b.py"]

    def test_llm_returns_resolvable_strategy(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from maestro_cli.worktree import _build_conflict_review

        _reset_git_env(monkeypatch, tmp_path)
        call_count = [0]

        def mock_fn(*args: Any, **kwargs: Any) -> MagicMock:
            call_count[0] += 1
            if call_count[0] == 1:
                # diff call
                return _mock_run(stdout="some diff output\n")
            # LLM call
            return _mock_run(
                stdout='{"resolution_strategy": "additive", "safe_to_auto_resolve": true, "reasoning": "test", "suggestion": "just add it"}'
            )

        monkeypatch.setattr("maestro_cli.worktree.subprocess.run", mock_fn)
        monkeypatch.setattr(
            "maestro_cli.runners._build_safe_env",
            lambda env, secrets: {},
            raising=False,
        )
        monkeypatch.setattr(
            "maestro_cli.runners._resolve_executable",
            lambda name: ["claude"],
            raising=False,
        )

        review = _build_conflict_review(
            workspace_root=tmp_path,
            task_id="task-z",
            branch_name="maestro/task-z",
            base_branch="main",
            files_changed=["c.py"],
            conflict_files=["c.py"],
            overlaps=[],
            model="haiku",
        )

        assert review.verdict == "resolvable"
        assert review.resolution_suggestion == "just add it"
        assert review.review_model == "haiku"

    def test_llm_returns_non_overlapping_strategy(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from maestro_cli.worktree import _build_conflict_review

        _reset_git_env(monkeypatch, tmp_path)
        call_count = [0]

        def mock_fn(*args: Any, **kwargs: Any) -> MagicMock:
            call_count[0] += 1
            if call_count[0] == 1:
                return _mock_run(stdout="some diff\n")
            return _mock_run(
                stdout='{"resolution_strategy": "non_overlapping", "suggestion": "separate areas"}'
            )

        monkeypatch.setattr("maestro_cli.worktree.subprocess.run", mock_fn)
        monkeypatch.setattr(
            "maestro_cli.runners._build_safe_env",
            lambda env, secrets: {},
            raising=False,
        )
        monkeypatch.setattr(
            "maestro_cli.runners._resolve_executable",
            lambda name: ["claude"],
            raising=False,
        )

        review = _build_conflict_review(
            workspace_root=tmp_path,
            task_id="t",
            branch_name="maestro/t",
            base_branch="main",
            files_changed=["d.py"],
            conflict_files=["d.py"],
            overlaps=[],
        )

        assert review.verdict == "resolvable"

    def test_llm_returns_true_conflict_strategy(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from maestro_cli.worktree import _build_conflict_review

        _reset_git_env(monkeypatch, tmp_path)
        call_count = [0]

        def mock_fn(*args: Any, **kwargs: Any) -> MagicMock:
            call_count[0] += 1
            if call_count[0] == 1:
                return _mock_run(stdout="diff\n")
            return _mock_run(
                stdout='{"resolution_strategy": "true_conflict", "suggestion": "manual fix"}'
            )

        monkeypatch.setattr("maestro_cli.worktree.subprocess.run", mock_fn)
        monkeypatch.setattr(
            "maestro_cli.runners._build_safe_env",
            lambda env, secrets: {},
            raising=False,
        )
        monkeypatch.setattr(
            "maestro_cli.runners._resolve_executable",
            lambda name: ["claude"],
            raising=False,
        )

        review = _build_conflict_review(
            workspace_root=tmp_path,
            task_id="t",
            branch_name="maestro/t",
            base_branch="main",
            files_changed=["e.py"],
            conflict_files=["e.py"],
            overlaps=[],
        )

        assert review.verdict == "conflict"

    def test_llm_returns_invalid_json_falls_back(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from maestro_cli.worktree import _build_conflict_review

        _reset_git_env(monkeypatch, tmp_path)
        call_count = [0]

        def mock_fn(*args: Any, **kwargs: Any) -> MagicMock:
            call_count[0] += 1
            if call_count[0] == 1:
                return _mock_run(stdout="diff\n")
            return _mock_run(stdout="not valid json at all")

        monkeypatch.setattr("maestro_cli.worktree.subprocess.run", mock_fn)
        monkeypatch.setattr(
            "maestro_cli.runners._build_safe_env",
            lambda env, secrets: {},
            raising=False,
        )
        monkeypatch.setattr(
            "maestro_cli.runners._resolve_executable",
            lambda name: ["claude"],
            raising=False,
        )

        review = _build_conflict_review(
            workspace_root=tmp_path,
            task_id="t",
            branch_name="maestro/t",
            base_branch="main",
            files_changed=["f.py"],
            conflict_files=["f.py"],
            overlaps=[],
        )

        assert review.verdict == "conflict"

    def test_llm_returns_nonzero_exit_code_falls_back(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from maestro_cli.worktree import _build_conflict_review

        _reset_git_env(monkeypatch, tmp_path)
        call_count = [0]

        def mock_fn(*args: Any, **kwargs: Any) -> MagicMock:
            call_count[0] += 1
            if call_count[0] == 1:
                return _mock_run(stdout="diff\n")
            return _mock_run(returncode=1, stderr="claude error")

        monkeypatch.setattr("maestro_cli.worktree.subprocess.run", mock_fn)
        monkeypatch.setattr(
            "maestro_cli.runners._build_safe_env",
            lambda env, secrets: {},
            raising=False,
        )
        monkeypatch.setattr(
            "maestro_cli.runners._resolve_executable",
            lambda name: ["claude"],
            raising=False,
        )

        review = _build_conflict_review(
            workspace_root=tmp_path,
            task_id="t",
            branch_name="maestro/t",
            base_branch="main",
            files_changed=["g.py"],
            conflict_files=["g.py"],
            overlaps=[MergeOverlap(file="g.py", merged_by=["prev"])],
        )

        assert review.verdict == "conflict"
        assert review.review_model == "haiku"
        assert len(review.overlapping_files) == 1

    def test_overlap_list_formatted_when_present(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from maestro_cli.worktree import _build_conflict_review

        _reset_git_env(monkeypatch, tmp_path)
        call_count = [0]

        def mock_fn(*args: Any, **kwargs: Any) -> MagicMock:
            call_count[0] += 1
            if call_count[0] == 1:
                return _mock_run(stdout="diff\n")
            return _mock_run(returncode=1)

        monkeypatch.setattr("maestro_cli.worktree.subprocess.run", mock_fn)
        monkeypatch.setattr(
            "maestro_cli.runners._build_safe_env",
            lambda env, secrets: {},
            raising=False,
        )
        monkeypatch.setattr(
            "maestro_cli.runners._resolve_executable",
            lambda name: ["claude"],
            raising=False,
        )

        overlaps = [
            MergeOverlap(file="a.py", merged_by=["task-1", "task-2"]),
            MergeOverlap(file="b.py", merged_by=["task-3"]),
        ]

        review = _build_conflict_review(
            workspace_root=tmp_path,
            task_id="t",
            branch_name="maestro/t",
            base_branch="main",
            files_changed=["a.py", "b.py"],
            conflict_files=["a.py"],
            overlaps=overlaps,
        )

        assert review.verdict == "conflict"
        assert review.overlapping_files == overlaps

    def test_conflict_files_limited_to_five_diffs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """_build_conflict_review only diffs the first 5 conflict files."""
        from maestro_cli.worktree import _build_conflict_review

        _reset_git_env(monkeypatch, tmp_path)
        diff_calls: list[str] = []

        def mock_fn(*args: Any, **kwargs: Any) -> MagicMock:
            cmd = list(args[0])
            if cmd[1] == "diff" and "--" in cmd:
                diff_calls.append(cmd[-1])
                return _mock_run(stdout="diff content\n")
            if cmd[1] == "diff":
                # Should not reach here for --stat as we call directly
                return _mock_run(returncode=1)
            return _mock_run(returncode=1)

        monkeypatch.setattr("maestro_cli.worktree.subprocess.run", mock_fn)

        conflict_files = [f"file{i}.py" for i in range(8)]

        review = _build_conflict_review(
            workspace_root=tmp_path,
            task_id="t",
            branch_name="maestro/t",
            base_branch="main",
            files_changed=conflict_files,
            conflict_files=conflict_files,
            overlaps=[],
        )

        # Should only diff first 5
        assert len(diff_calls) == 5
        assert diff_calls == [f"file{i}.py" for i in range(5)]
        assert review.verdict == "conflict"


class TestWorktreeMergeResultToDict:
    """Coverage for WorktreeMergeResult.to_dict()."""

    def test_minimal_result(self) -> None:
        r = WorktreeMergeResult(status="empty")
        d = r.to_dict()
        assert d["status"] == "empty"
        assert d["files_changed"] == []
        assert d["conflict_files"] == []
        assert d["merge_commit"] is None
        assert d["error"] is None
        assert "review" not in d

    def test_merged_with_review(self) -> None:
        review = MergeReview(
            verdict="safe",
            overlapping_files=[],
            review_model="haiku",
            review_duration_sec=1.234,
        )
        r = WorktreeMergeResult(
            status="merged",
            files_changed=["a.py", "b.py"],
            merge_commit="abc123",
            review=review,
        )
        d = r.to_dict()
        assert d["status"] == "merged"
        assert d["files_changed"] == ["a.py", "b.py"]
        assert d["merge_commit"] == "abc123"
        assert "review" in d
        assert d["review"]["verdict"] == "safe"
        assert d["review"]["review_model"] == "haiku"
        assert d["review"]["review_duration_sec"] == 1.23

    def test_conflict_with_error(self) -> None:
        r = WorktreeMergeResult(
            status="error",
            error="something broke",
        )
        d = r.to_dict()
        assert d["status"] == "error"
        assert d["error"] == "something broke"

    def test_conflict_with_files(self) -> None:
        r = WorktreeMergeResult(
            status="conflict",
            files_changed=["a.py"],
            conflict_files=["a.py"],
        )
        d = r.to_dict()
        assert d["status"] == "conflict"
        assert d["conflict_files"] == ["a.py"]


class TestMergeReviewToDict:
    """Coverage for MergeReview.to_dict()."""

    def test_minimal_review(self) -> None:
        r = MergeReview(verdict="safe")
        d = r.to_dict()
        assert d["verdict"] == "safe"
        assert d["overlapping_files"] == []
        assert d["conflict_files"] == []
        assert d["auto_resolved"] is False
        assert "resolution_suggestion" not in d
        assert "review_model" not in d
        assert "review_duration_sec" not in d
        assert "review_cost_usd" not in d

    def test_full_review(self) -> None:
        r = MergeReview(
            verdict="resolvable",
            overlapping_files=[MergeOverlap(file="x.py", merged_by=["t1", "t2"])],
            conflict_files=["x.py", "y.py"],
            resolution_suggestion="Combine changes manually",
            auto_resolved=True,
            review_model="sonnet",
            review_duration_sec=5.678,
            review_cost_usd=0.01,
        )
        d = r.to_dict()
        assert d["verdict"] == "resolvable"
        assert len(d["overlapping_files"]) == 1
        assert d["overlapping_files"][0] == {"file": "x.py", "merged_by": ["t1", "t2"]}
        assert d["conflict_files"] == ["x.py", "y.py"]
        assert d["auto_resolved"] is True
        assert d["resolution_suggestion"] == "Combine changes manually"
        assert d["review_model"] == "sonnet"
        assert d["review_duration_sec"] == 5.68
        assert d["review_cost_usd"] == 0.01

    def test_review_duration_zero_excluded(self) -> None:
        r = MergeReview(verdict="conflict", review_duration_sec=0.0)
        d = r.to_dict()
        assert "review_duration_sec" not in d

    def test_review_cost_none_excluded(self) -> None:
        r = MergeReview(verdict="conflict", review_cost_usd=None)
        d = r.to_dict()
        assert "review_cost_usd" not in d


class TestMergeOverlap:
    """Coverage for MergeOverlap dataclass."""

    def test_basic_construction(self) -> None:
        o = MergeOverlap(file="shared.py", merged_by=["task-a", "task-b"])
        assert o.file == "shared.py"
        assert o.merged_by == ["task-a", "task-b"]

    def test_default_merged_by_is_empty_list(self) -> None:
        o = MergeOverlap(file="solo.py")
        assert o.merged_by == []


class TestGetBaseBranchAdditional:
    """Additional get_base_branch edge cases."""

    def test_master_branch(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "maestro_cli.worktree.subprocess.run",
            lambda *args, **kwargs: _mock_run(stdout="master\n"),
        )
        assert get_base_branch(tmp_path) == "master"

    def test_custom_branch_name(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "maestro_cli.worktree.subprocess.run",
            lambda *args, **kwargs: _mock_run(stdout="release/v2.0\n"),
        )
        assert get_base_branch(tmp_path) == "release/v2.0"

    def test_completely_empty_stdout(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "maestro_cli.worktree.subprocess.run",
            lambda *args, **kwargs: _mock_run(stdout=""),
        )
        with pytest.raises(ValueError, match="unable to determine current git branch"):
            get_base_branch(tmp_path)


class TestCreateWorktreeAdditional:
    """Additional create_worktree edge cases."""

    def test_matrix_expanded_task_id(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Matrix-expanded task IDs like 'task@key=val,key2=val2' get sanitized."""
        _reset_git_env(monkeypatch, tmp_path)
        calls: list[list[str]] = []

        def mock_fn(*args: Any, **kwargs: Any) -> MagicMock:
            calls.append(list(args[0]))
            return _mock_run()

        monkeypatch.setattr("maestro_cli.worktree.subprocess.run", mock_fn)

        worktree_path = create_worktree(
            tmp_path, "lint@lang=python,strict=true", base_branch="main",
        )

        assert worktree_path == tmp_path / ".maestro-worktrees" / "lint@lang=python,strict=true"
        assert calls[0][5] == "maestro/lint-lang-python-strict-true"

    def test_task_id_with_dots_and_colons(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        calls: list[list[str]] = []

        def mock_fn(*args: Any, **kwargs: Any) -> MagicMock:
            calls.append(list(args[0]))
            return _mock_run()

        monkeypatch.setattr("maestro_cli.worktree.subprocess.run", mock_fn)

        create_worktree(tmp_path, "ns:task.v2", base_branch="main")

        assert calls[0][5] == "maestro/ns-task-v2"

    def test_prints_creation_message(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "maestro_cli.worktree.subprocess.run",
            lambda *args, **kwargs: _mock_run(),
        )

        create_worktree(tmp_path, "task-msg", base_branch="main")

        captured = capsys.readouterr()
        assert "[maestro] creating worktree" in captured.out
        assert "task-msg" in captured.out


class TestCleanupWorktreeAdditional:
    """Additional cleanup_worktree edge cases."""

    def test_prints_cleanup_message(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "maestro_cli.worktree.subprocess.run",
            lambda *args, **kwargs: _mock_run(),
        )

        cleanup_worktree(tmp_path, "task-clean", tmp_path / "wt")

        captured = capsys.readouterr()
        assert "[maestro] cleaning up worktree" in captured.out

    def test_worktree_remove_succeeds_branch_delete_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        calls: list[list[str]] = []

        def mock_fn(*args: Any, **kwargs: Any) -> MagicMock:
            calls.append(list(args[0]))
            if args[0][1] == "branch":
                raise OSError("branch delete failed")
            return _mock_run()

        monkeypatch.setattr("maestro_cli.worktree.subprocess.run", mock_fn)

        cleanup_worktree(tmp_path, "task-123", tmp_path / "wt")

        assert len(calls) == 2
        assert calls[0][1:3] == ["worktree", "remove"]
        assert calls[1][1:3] == ["branch", "-D"]

    def test_sanitizes_branch_name_for_cleanup(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _reset_git_env(monkeypatch, tmp_path)
        calls: list[list[str]] = []

        def mock_fn(*args: Any, **kwargs: Any) -> MagicMock:
            calls.append(list(args[0]))
            return _mock_run()

        monkeypatch.setattr("maestro_cli.worktree.subprocess.run", mock_fn)

        cleanup_worktree(tmp_path, "task@special.chars", tmp_path / "wt")

        assert calls[1] == ["git", "branch", "-D", "maestro/task-special-chars"]


class TestSanitizeBranchNameAdditional:
    """Additional _sanitize_branch_name edge cases."""

    def test_all_special_chars(self) -> None:
        assert _sanitize_branch_name("!@#$%^&*()") == "----------"

    def test_hyphens_preserved(self) -> None:
        assert _sanitize_branch_name("a-b-c") == "a-b-c"

    def test_numbers_preserved(self) -> None:
        assert _sanitize_branch_name("task123") == "task123"

    def test_mixed_case_preserved(self) -> None:
        assert _sanitize_branch_name("MyTask") == "MyTask"

    def test_leading_special_char(self) -> None:
        assert _sanitize_branch_name(".hidden") == "-hidden"

    def test_trailing_special_char(self) -> None:
        assert _sanitize_branch_name("task.") == "task-"


class TestMergeWorktreeEdgeCases:
    """Edge cases for merge_worktree."""

    def test_checkout_failure_with_no_output(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Checkout failure with empty stderr and stdout returns 'unknown git error'."""
        _reset_git_env(monkeypatch, tmp_path)
        responses = iter(
            [
                _mock_run(stdout="src/app.py | 2 +-\n"),
                _mock_run(returncode=1),  # checkout with no output
            ]
        )

        monkeypatch.setattr(
            "maestro_cli.worktree.subprocess.run",
            lambda *args, **kwargs: next(responses),
        )

        result = merge_worktree(tmp_path, "task-empty", tmp_path / "wt", "main")

        assert result.status == "error"
        assert result.error == "unknown git error"

    def test_merge_records_files_in_ledger(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """After successful merge, files are recorded in the ledger."""
        _reset_git_env(monkeypatch, tmp_path)
        reset_merge_ledger()
        responses = iter(
            [
                _mock_run(stdout="x.py | 1 +\ny.py | 2 +-\n"),
                _mock_run(),                          # checkout
                _mock_run(),                          # merge --no-commit
                _mock_run(),                          # commit
                _mock_run(stdout="aaa111\n"),          # rev-parse
            ]
        )

        monkeypatch.setattr(
            "maestro_cli.worktree.subprocess.run",
            lambda *args, **kwargs: next(responses),
        )

        result = merge_worktree(tmp_path, "task-ledger", tmp_path / "wt", "main")

        assert result.status == "merged"
        # Now check ledger recorded the files
        overlaps = _get_overlaps(["x.py"])
        assert len(overlaps) == 1
        assert overlaps[0].merged_by == ["task-ledger"]

    def test_worktree_path_parameter_is_ignored(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """The worktree_path parameter is kept for API compat but not used."""
        _reset_git_env(monkeypatch, tmp_path)
        reset_merge_ledger()
        monkeypatch.setattr(
            "maestro_cli.worktree.subprocess.run",
            lambda *args, **kwargs: _mock_run(stdout=""),
        )

        # Pass a bogus worktree_path — should not matter
        result = merge_worktree(
            tmp_path, "task-ignored", Path("/nonexistent/path"), "main",
        )

        assert result.status == "empty"


class TestDualVerification:
    """Tests for dual verification: agent claims vs actual git diff."""

    # -- _extract_claimed_files tests --

    def test_extract_modified_file(self) -> None:
        stdout = "I modified src/foo.py to fix the bug."
        result = _extract_claimed_files(stdout)
        assert "src/foo.py" in result

    def test_extract_created_backtick_file(self) -> None:
        stdout = "I created `utils/bar.js` with the helper functions."
        result = _extract_claimed_files(stdout)
        assert "utils/bar.js" in result

    def test_extract_updated_multiple_files(self) -> None:
        stdout = "I updated models.py and also updated views.py for consistency."
        result = _extract_claimed_files(stdout)
        assert "models.py" in result
        assert "views.py" in result

    def test_extract_backtick_near_action_verb(self) -> None:
        stdout = "After refactoring the code, the changes are in `auth/login.py` now."
        result = _extract_claimed_files(stdout)
        assert "auth/login.py" in result

    def test_extract_no_action_verbs_empty(self) -> None:
        stdout = "The system looks good, everything is working fine."
        result = _extract_claimed_files(stdout)
        assert result == set()

    def test_extract_empty_string_empty(self) -> None:
        result = _extract_claimed_files("")
        assert result == set()

    def test_extract_windows_backslash_in_backtick_normalized(self) -> None:
        # Backslash paths in backticks near action verbs are extracted
        # via the _FILE_MENTION_RE pattern and normalized
        stdout = r"I modified the file `src\utils\helper.py` for logging."
        result = _extract_claimed_files(stdout)
        # If extracted, backslashes should be normalized to forward slashes
        for f in result:
            assert "\\" not in f

    def test_extract_deleted_file(self) -> None:
        stdout = "I deleted old/config.yaml since it was unused."
        result = _extract_claimed_files(stdout)
        assert "old/config.yaml" in result

    def test_extract_refactored_backtick_file(self) -> None:
        stdout = "I refactored the login module in `auth/login.py` for clarity."
        result = _extract_claimed_files(stdout)
        assert "auth/login.py" in result

    def test_extract_quoted_single_quotes(self) -> None:
        stdout = "I edited 'config/settings.yaml' to update the defaults."
        result = _extract_claimed_files(stdout)
        assert "config/settings.yaml" in result

    def test_extract_multiple_action_verbs_one_line(self) -> None:
        stdout = "I modified api.py and also created tests/test_api.py for coverage."
        result = _extract_claimed_files(stdout)
        assert "api.py" in result
        assert "tests/test_api.py" in result

    # -- _normalize_path tests --

    def test_normalize_strips_leading_dot_slash(self) -> None:
        assert _normalize_path("./src/foo.py") == "src/foo.py"

    def test_normalize_converts_backslashes(self) -> None:
        assert _normalize_path("src\\utils\\bar.py") == "src/utils/bar.py"

    # -- verify_worktree_output tests --

    def test_verify_perfect_match(self) -> None:
        files = ["src/foo.py", "src/bar.py"]
        stdout = "I modified src/foo.py and also modified src/bar.py."
        result = verify_worktree_output(files, stdout)
        assert result.verified is True
        assert result.overlap_ratio == 1.0
        assert result.unclaimed_files == []
        assert result.phantom_files == []

    def test_verify_no_overlap(self) -> None:
        files = ["src/foo.py"]
        stdout = "I modified totally/different.py."
        result = verify_worktree_output(files, stdout)
        assert result.verified is False
        assert result.overlap_ratio == 0.0

    def test_verify_partial_overlap_above_threshold(self) -> None:
        files = ["src/a.py", "src/b.py", "src/c.py"]
        stdout = "I modified src/a.py and modified src/b.py."
        result = verify_worktree_output(files, stdout, threshold=0.5)
        assert result.verified is True
        assert result.overlap_ratio > 0.5

    def test_verify_partial_overlap_below_threshold(self) -> None:
        files = ["src/a.py", "src/b.py", "src/c.py", "src/d.py", "src/e.py"]
        stdout = "I modified src/a.py."
        result = verify_worktree_output(files, stdout, threshold=0.5)
        assert result.verified is False
        assert result.overlap_ratio < 0.5

    def test_verify_unclaimed_files_detected(self) -> None:
        files = ["src/a.py", "src/b.py", "src/c.py"]
        stdout = "I modified src/a.py."
        result = verify_worktree_output(files, stdout)
        assert len(result.unclaimed_files) > 0
        # b.py and c.py not mentioned
        unclaimed_basenames = {f.rsplit("/", 1)[-1] for f in result.unclaimed_files}
        assert "b.py" in unclaimed_basenames or "c.py" in unclaimed_basenames

    def test_verify_phantom_files_detected(self) -> None:
        files = ["src/a.py"]
        stdout = "I modified src/a.py and also modified ghost/phantom.py."
        result = verify_worktree_output(files, stdout)
        assert len(result.phantom_files) > 0
        assert any("phantom.py" in f for f in result.phantom_files)

    def test_verify_empty_diff_empty_stdout(self) -> None:
        result = verify_worktree_output([], "")
        assert result.verified is True
        assert result.overlap_ratio == 1.0

    def test_verify_diff_files_no_stdout(self) -> None:
        files = ["src/foo.py"]
        result = verify_worktree_output(files, "")
        assert result.verified is False
        assert result.files_in_diff == ["src/foo.py"]
        assert result.files_claimed == []

    def test_verify_custom_threshold(self) -> None:
        files = ["src/a.py", "src/b.py"]
        stdout = "I modified src/a.py."
        # With threshold=0.3, partial overlap should pass
        result = verify_worktree_output(files, stdout, threshold=0.3)
        assert result.verified is True
        # With threshold=0.9, partial overlap should fail
        result2 = verify_worktree_output(files, stdout, threshold=0.9)
        assert result2.verified is False

    def test_verify_basename_matching(self) -> None:
        """Agent says 'foo.py' but diff has 'src/foo.py' -- should match."""
        files = ["src/foo.py"]
        stdout = "I modified foo.py."
        result = verify_worktree_output(files, stdout)
        # Basename matching should provide some overlap
        assert result.overlap_ratio > 0.0

    def test_verify_all_files_unclaimed(self) -> None:
        files = ["src/a.py", "src/b.py"]
        stdout = "Everything looks good, no specific files to report."
        result = verify_worktree_output(files, stdout)
        assert result.verified is False
        assert len(result.unclaimed_files) == 2
        assert result.files_claimed == []

    def test_verify_all_files_phantom(self) -> None:
        files: list[str] = []
        stdout = "I modified ghost/a.py and also modified ghost/b.py."
        result = verify_worktree_output(files, stdout)
        assert result.verified is False
        assert len(result.phantom_files) == 2
        assert result.files_in_diff == []

    def test_verify_result_type(self) -> None:
        result = verify_worktree_output(["a.py"], "I modified a.py.")
        assert isinstance(result, DualVerificationResult)

    def test_verify_files_sorted(self) -> None:
        files = ["z.py", "a.py", "m.py"]
        stdout = "I modified z.py and modified a.py and modified m.py."
        result = verify_worktree_output(files, stdout)
        assert result.files_in_diff == sorted(result.files_in_diff)
        assert result.files_claimed == sorted(result.files_claimed)
