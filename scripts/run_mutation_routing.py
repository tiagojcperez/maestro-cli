"""Run a full cosmic-ray mutation cycle for routing.py.

Steps:
    1. cosmic-ray init  — creates a fresh session
    2. cosmic-ray exec  — executes all mutations
    3. cosmic-ray dump  — parses results, prints score, writes reports

Output (stdout, parseable by maestro watch):
    Mutation score: XX.XX%
    Killed: N  Survived: N  Incompetent: N  Total: N

Side effects:
    - routing-mutations.sqlite   (cosmic-ray session — gitignored)
    - mutation-survivors.txt     (surviving mutant details for the next watch iteration)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_CONFIG = "mutations/cosmic-ray-routing.toml"
_SESSION = "mutations/routing-mutations.sqlite"
_SURVIVORS_FILE = "mutations/mutation-survivors.txt"

_ROOT = Path(__file__).parent.parent


def _env() -> dict[str, str]:
    """Build environment with src/ on PYTHONPATH so cosmic-ray can find maestro_cli."""
    env = os.environ.copy()
    src = str(_ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src}{os.pathsep}{existing}" if existing else src
    return env


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print(f"[mutation] {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=check, text=True, encoding="utf-8", env=_env())


def main() -> None:
    # Step 1: initialise fresh session (--force overwrites any existing session file)
    _run(["cosmic-ray", "init", "--force", str(_ROOT / _CONFIG), str(_ROOT / _SESSION)])

    # Step 2: execute all mutations
    result = _run(
        ["cosmic-ray", "exec", str(_ROOT / _CONFIG), str(_ROOT / _SESSION)],
        check=False,
    )
    if result.returncode not in (0, 1):  # 1 = some tests failed (expected)
        print(f"[mutation] cosmic-ray exec exited with {result.returncode}", file=sys.stderr)

    # Step 3: dump session and parse results
    dump_result = subprocess.run(
        ["cosmic-ray", "dump", str(_ROOT / _SESSION)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=_env(),
    )

    killed = 0
    survived = 0
    incompetent = 0
    survivors: list[dict] = []  # type: ignore[type-arg]

    for line in dump_result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        if not isinstance(record, list) or len(record) < 2:
            continue

        job_spec = record[0] if isinstance(record[0], dict) else {}
        job_result = record[1] if isinstance(record[1], dict) else {}

        # cosmic-ray 8.x uses "test_outcome"; incompetent = worker failed/timeout
        worker_outcome = job_result.get("worker_outcome", "")
        test_outcome = job_result.get("test_outcome", "")

        mutations = job_spec.get("mutations", [])
        mutation_info = mutations[0] if mutations else {}

        if worker_outcome not in ("normal", ""):
            incompetent += 1  # timeout, exception, etc.
        elif test_outcome == "killed":
            killed += 1
        elif test_outcome == "survived":
            survived += 1
            survivors.append(mutation_info)
        else:
            incompetent += 1

    total_meaningful = killed + survived
    total_all = total_meaningful + incompetent

    if total_all == 0:
        # Print first few raw dump lines so we can diagnose the format
        raw_lines = [l for l in dump_result.stdout.splitlines() if l.strip()]
        print("DEBUG: dump stdout (first 5 lines):", file=sys.stderr)
        for ln in raw_lines[:5]:
            print(f"  {ln[:200]}", file=sys.stderr)
        print("ERROR: cosmic-ray generated 0 mutations. Check module-path in config.", file=sys.stderr)
        print("Mutation score: 0.00%")
        print("Killed: 0  Survived: 0  Incompetent: 0  Total: 0")
        sys.exit(1)

    score = (killed / total_meaningful * 100.0) if total_meaningful > 0 else 0.0

    # Print parseable metric line (watch extracts this)
    print(f"Mutation score: {score:.2f}%")
    print(f"Killed: {killed}  Survived: {survived}  Incompetent: {incompetent}  Total: {total_all}")

    # Write survivors report for the next iteration's engine task
    survivors_path = _ROOT / _SURVIVORS_FILE
    with survivors_path.open("w", encoding="utf-8") as f:
        if not survivors:
            f.write("No surviving mutants — all mutations killed!\n")
        else:
            f.write(f"# Surviving Mutants ({len(survivors)} total)\n\n")
            for s in survivors:
                op = s.get("operator_name", "unknown")
                module = s.get("module_path", "unknown")
                start = s.get("start_pos", "?")
                end = s.get("end_pos", "?")
                occurrence = s.get("occurrence", 0)
                f.write(f"- {op}  [{module} line {start} → {end}  occurrence={occurrence}]\n")

    print(f"[mutation] Survivors written to {survivors_path}")


if __name__ == "__main__":
    main()
