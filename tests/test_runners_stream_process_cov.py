from __future__ import annotations

import subprocess
import threading
from typing import Callable

import pytest

from maestro_cli.runners import _stream_process


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeLog:
    """Minimal stand-in for io.TextIOWrapper: only needs write()/flush()."""

    def __init__(self) -> None:
        self.buffer: list[str] = []

    def write(self, text: str) -> int:
        self.buffer.append(text)
        return len(text)

    def flush(self) -> None:  # pragma: no cover - trivial
        pass


class _RaisingIterable:
    """Iterable whose iterator raises a chosen exception partway through.

    Drives the `except (ValueError, OSError)` branch in the drain threads
    (lines 3760-3761 / 3775-3776).
    """

    def __init__(self, lines: list[str], exc: BaseException) -> None:
        self._lines = lines
        self._exc = exc

    def __iter__(self) -> "_RaisingIterable":
        self._it = iter(self._lines)
        return self

    def __next__(self) -> str:
        try:
            return next(self._it)
        except StopIteration:
            raise self._exc


class _FakeProc:
    """Controllable stand-in for subprocess.Popen.

    `stdout` / `stderr` are arbitrary iterables (lists work fine). `wait`
    returns immediately by default; behaviour can be customised per test via
    `wait_behaviour`, a callable taking the requested timeout.
    """

    def __init__(
        self,
        stdout: object,
        stderr: object,
        returncode: int = 0,
        wait_behaviour: Callable[[float | None], None] | None = None,
        pid: int = 4321,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.pid = pid
        self._wait_behaviour = wait_behaviour
        self.killed = False

    def wait(self, timeout: float | None = None) -> int:
        if self._wait_behaviour is not None:
            self._wait_behaviour(timeout)
        return self.returncode

    def kill(self) -> None:  # pragma: no cover - exercised via _kill_process_tree mock
        self.killed = True


def _join_all_threads() -> None:
    """Best-effort wait for daemon drain threads to finish before asserting."""
    for t in threading.enumerate():
        if t is threading.current_thread():
            continue
        t.join(timeout=2)


# ---------------------------------------------------------------------------
# stdout tail trimming + line_callback exception swallowing (3754, 3758-3761)
# ---------------------------------------------------------------------------


def test_stdout_tail_trims_to_limit() -> None:
    """More lines than stdout_tail_lines triggers last_lines.pop(0) (line 3754)."""
    lines = [f"line-{i}\n" for i in range(10)]
    proc = _FakeProc(stdout=lines, stderr=[])
    log = _FakeLog()

    rc, stdout_tail, stderr_tail = _stream_process(
        proc,  # type: ignore[arg-type]
        log,  # type: ignore[arg-type]
        timeout_sec=5,
        stdout_tail_lines=3,
    )

    assert rc == 0
    # Only the last 3 of 10 lines survive in the tail.
    assert stdout_tail == "line-7\nline-8\nline-9\n"
    assert stderr_tail == ""
    # Every line was still written to the log, in order.
    assert "".join(log.buffer) == "".join(lines)


def test_line_callback_exception_is_swallowed() -> None:
    """A raising line_callback must not crash the drain thread (lines 3758-3759)."""
    received: list[str] = []

    def _callback(line: str) -> None:
        received.append(line)
        raise RuntimeError("callback boom")

    proc = _FakeProc(stdout=["alpha\n", "beta\n"], stderr=[])
    log = _FakeLog()

    rc, stdout_tail, _ = _stream_process(
        proc,  # type: ignore[arg-type]
        log,  # type: ignore[arg-type]
        timeout_sec=5,
        line_callback=_callback,
    )

    assert rc == 0
    # Callback was invoked for every line despite raising each time.
    assert received == ["alpha", "beta"]
    assert stdout_tail == "alpha\nbeta\n"


def test_stdout_drain_swallows_value_error() -> None:
    """A ValueError raised while iterating stdout hits the except branch (3760-3761)."""
    stdout = _RaisingIterable(["one\n", "two\n"], ValueError("pipe closed"))
    proc = _FakeProc(stdout=stdout, stderr=[])
    log = _FakeLog()

    rc, stdout_tail, _ = _stream_process(
        proc,  # type: ignore[arg-type]
        log,  # type: ignore[arg-type]
        timeout_sec=5,
    )

    # Lines yielded before the exception are still captured; no crash.
    assert rc == 0
    assert stdout_tail == "one\ntwo\n"


# ---------------------------------------------------------------------------
# stderr tail trimming + drain exception swallowing (3774-3776)
# ---------------------------------------------------------------------------


def test_stderr_tail_trims_to_limit() -> None:
    """More stderr lines than stderr_tail_lines triggers last_stderr.pop(0) (3774)."""
    err_lines = [f"err-{i}\n" for i in range(6)]
    proc = _FakeProc(stdout=[], stderr=err_lines)
    log = _FakeLog()

    rc, _, stderr_tail = _stream_process(
        proc,  # type: ignore[arg-type]
        log,  # type: ignore[arg-type]
        timeout_sec=5,
        stderr_tail_lines=2,
    )

    assert rc == 0
    assert stderr_tail == "err-4\nerr-5\n"
    # Each stderr line is logged with the [stderr] prefix.
    assert "[stderr] err-0\n" in log.buffer


def test_stderr_drain_swallows_os_error() -> None:
    """An OSError raised while iterating stderr hits the except branch (3775-3776)."""
    stderr = _RaisingIterable(["bad\n"], OSError("broken pipe"))
    proc = _FakeProc(stdout=[], stderr=stderr)
    log = _FakeLog()

    rc, _, stderr_tail = _stream_process(
        proc,  # type: ignore[arg-type]
        log,  # type: ignore[arg-type]
        timeout_sec=5,
    )

    assert rc == 0
    assert stderr_tail == "bad\n"


# ---------------------------------------------------------------------------
# deadline_ref polling loop: normal exit + timeout (3787-3799)
# ---------------------------------------------------------------------------


def test_deadline_ref_normal_exit_breaks_loop() -> None:
    """deadline_ref path where proc.wait succeeds breaks the poll loop (3795-3797)."""
    proc = _FakeProc(stdout=["done\n"], stderr=[], returncode=0)
    log = _FakeLog()
    # Deadline far in the future so `remaining > 0` and wait() returns cleanly.
    deadline = [time_far_future()]

    rc, stdout_tail, _ = _stream_process(
        proc,  # type: ignore[arg-type]
        log,  # type: ignore[arg-type]
        timeout_sec=5,
        deadline_ref=deadline,
    )

    assert rc == 0
    assert stdout_tail == "done\n"


def test_deadline_ref_continue_then_exit() -> None:
    """First wait() raises TimeoutExpired (continue), second returns (3798-3799 + break)."""
    calls = {"n": 0}

    def _wait_behaviour(timeout: float | None) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise subprocess.TimeoutExpired(cmd="x", timeout=timeout or 0)
        # second call: return normally

    proc = _FakeProc(
        stdout=["hi\n"],
        stderr=[],
        returncode=0,
        wait_behaviour=_wait_behaviour,
    )
    log = _FakeLog()
    deadline = [time_far_future()]

    rc, stdout_tail, _ = _stream_process(
        proc,  # type: ignore[arg-type]
        log,  # type: ignore[arg-type]
        timeout_sec=5,
        deadline_ref=deadline,
    )

    assert rc == 0
    assert calls["n"] >= 2  # looped at least once before exiting
    assert stdout_tail == "hi\n"


def test_deadline_ref_timeout_kills_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """remaining <= 0 triggers kill + 124 return (lines 3789-3794)."""
    killed: list[object] = []
    monkeypatch.setattr(
        "maestro_cli.runners._kill_process_tree",
        lambda proc: killed.append(proc),
    )

    proc = _FakeProc(stdout=["x\n"], stderr=["e\n"], returncode=0)
    log = _FakeLog()
    # Already-expired deadline forces remaining <= 0 on the first loop turn.
    deadline = [0.0]

    rc, stdout_tail, stderr_tail = _stream_process(
        proc,  # type: ignore[arg-type]
        log,  # type: ignore[arg-type]
        timeout_sec=7,
        deadline_ref=deadline,
    )

    assert rc == 124
    assert "Task timed out after 7s" in stdout_tail
    assert killed and killed[0] is proc


# ---------------------------------------------------------------------------
# reader stuck after exit: force-kill path (3813-3817)
# ---------------------------------------------------------------------------


def test_reader_stuck_forces_kill(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the stdout reader never signals done, the force-kill branch runs (3815-3817).

    We replace threading.Thread so the stdout drain thread is never actually
    started, leaving `reader_done` unset after proc.wait() returns. The stderr
    thread is a real no-op thread so the function still joins cleanly.
    """
    real_thread = threading.Thread
    started: list[str] = []

    class _NoStartThread:
        """Stand-in for the stdout reader that never runs (reader_done stays unset)."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            self._target = kwargs.get("target")

        def start(self) -> None:
            started.append("stdout")

        def join(self, timeout: float | None = None) -> None:
            pass

    def _thread_factory(*args: object, **kwargs: object) -> object:
        target = kwargs.get("target")
        # The first Thread created in _stream_process is the stdout reader.
        if getattr(target, "__name__", "") == "_drain_stdout":
            return _NoStartThread(*args, **kwargs)
        return real_thread(*args, **kwargs)

    monkeypatch.setattr("maestro_cli.runners.threading.Thread", _thread_factory)

    killed: list[object] = []
    monkeypatch.setattr(
        "maestro_cli.runners._kill_process_tree",
        lambda proc: killed.append(proc),
    )

    proc = _FakeProc(stdout=["never-read\n"], stderr=[], returncode=0)
    log = _FakeLog()

    rc, stdout_tail, _ = _stream_process(
        proc,  # type: ignore[arg-type]
        log,  # type: ignore[arg-type]
        timeout_sec=5,
    )

    # reader_done never set -> force-kill branch executed.
    assert killed and killed[0] is proc
    assert rc == 0
    # stdout reader never ran, so no tail captured.
    assert stdout_tail == ""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def time_far_future() -> float:
    import time as _time

    return _time.monotonic() + 3600.0
