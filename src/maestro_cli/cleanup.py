from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path


def cleanup_runs(
    run_root: Path,
    keep: int = 10,
    older_than_days: int | None = None,
    dry_run: bool = False,
) -> list[Path]:
    if not run_root.exists():
        return []

    run_dirs = sorted(
        (d for d in run_root.iterdir() if d.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    now = datetime.now(UTC)
    to_delete: list[Path] = []

    for idx, run_dir in enumerate(run_dirs):
        if idx < keep:
            continue

        if older_than_days is not None:
            mtime = datetime.fromtimestamp(run_dir.stat().st_mtime, tz=UTC)
            age = now - mtime
            if age < timedelta(days=older_than_days):
                continue

        to_delete.append(run_dir)

    if not dry_run:
        for run_dir in to_delete:
            shutil.rmtree(run_dir)

    return to_delete
