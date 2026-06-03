"""Parse cosmic-ray dump output and print a mutation score.

Usage:
    cosmic-ray dump <session.sqlite> | py scripts/mutation_score.py

Output (parseable by maestro watch stdout_regex):
    Mutation score: XX.XX%
    Killed: N  Survived: N  Incompetent: N  Total: N
"""
from __future__ import annotations

import json
import sys


def main() -> None:
    killed = 0
    survived = 0
    incompetent = 0

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        # cosmic-ray 8.x dump format: [job_spec, job_result]
        # job_spec:   {"job_id": "...", "mutations": [...]}
        # job_result: {"worker_outcome": "normal"|"timeout"|..., "test_outcome": "killed"|"survived"|...}
        if not isinstance(record, list) or len(record) < 2:
            continue

        job_result = record[1]
        if not isinstance(job_result, dict):
            continue

        worker_outcome = job_result.get("worker_outcome", "")
        test_outcome = job_result.get("test_outcome", "")

        if worker_outcome not in ("normal", ""):
            incompetent += 1
        elif test_outcome == "killed":
            killed += 1
        elif test_outcome == "survived":
            survived += 1
        else:
            incompetent += 1

    total_meaningful = killed + survived
    total = total_meaningful + incompetent

    if total_meaningful == 0:
        score = 0.0
    else:
        score = (killed / total_meaningful) * 100.0

    print(f"Mutation score: {score:.2f}%")
    print(f"Killed: {killed}  Survived: {survived}  Incompetent: {incompetent}  Total: {total}")


if __name__ == "__main__":
    main()
