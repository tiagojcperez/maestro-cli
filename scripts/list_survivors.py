"""Read the last cosmic-ray session and write mutation-survivors.txt.

Used as pre_command in the watch plan so the engine task can see which
mutants survived before it writes new tests.

If no session file exists yet, writes a "no previous run" placeholder.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_SESSION = "mutations/routing-mutations.sqlite"
_SURVIVORS_FILE = "mutations/mutation-survivors.txt"


def main() -> None:
    root = Path(__file__).parent.parent
    session_path = root / _SESSION
    survivors_path = root / _SURVIVORS_FILE

    if not session_path.exists():
        with survivors_path.open("w", encoding="utf-8") as f:
            f.write("# No previous mutation run found.\n")
            f.write("# This is the first iteration — write broad coverage tests.\n")
        print("[mutation] No session found — wrote placeholder survivors file.")
        return

    dump_result = subprocess.run(
        ["cosmic-ray", "dump", str(session_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    if dump_result.returncode != 0:
        with survivors_path.open("w", encoding="utf-8") as f:
            f.write(f"# Error reading session: {dump_result.stderr.strip()}\n")
        print(f"[mutation] dump error: {dump_result.stderr.strip()}", file=sys.stderr)
        return

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
        worker_outcome = job_result.get("worker_outcome", "")
        test_outcome = job_result.get("test_outcome", "")
        if worker_outcome in ("normal", "") and test_outcome == "survived":
            mutations = job_spec.get("mutations", [])
            if mutations:
                survivors.append(mutations[0])

    with survivors_path.open("w", encoding="utf-8") as f:
        if not survivors:
            f.write("# No surviving mutants from last run — all mutations were killed.\n")
            f.write("# Write edge-case tests to defend against future regressions.\n")
        else:
            f.write(f"# Surviving Mutants ({len(survivors)} total)\n\n")
            f.write("Each line: operator  [module  line start→end  occurrence=N]\n\n")
            for s in survivors:
                op = s.get("operator_name", "unknown")
                module = s.get("module_path", "unknown")
                start = s.get("start_pos", "?")
                end = s.get("end_pos", "?")
                occurrence = s.get("occurrence", 0)
                f.write(f"- {op}  [{module}  line {start}→{end}  occurrence={occurrence}]\n")

    print(f"[mutation] {len(survivors)} surviving mutants written to {survivors_path}")


if __name__ == "__main__":
    main()
