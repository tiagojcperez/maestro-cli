from __future__ import annotations

import hashlib
import fnmatch
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

INDEX_SCHEMA_VERSION = 1
_HASH_CHUNK_SIZE = 1024 * 1024
_HEAD_HASH_BYTES = 64 * 1024
_FIRST_LINES_DEFAULT = 8
_MAX_FILES = 10_000
_DEFAULT_IGNORE_DIRS: set[str] = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    "__pycache__",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".maestro-cache",
}
_DEFAULT_IGNORE_FILES: set[str] = {
    ".DS_Store",
    "Thumbs.db",
}
_DEFAULT_EXCLUDES: tuple[str, ...] = tuple(
    sorted(_DEFAULT_IGNORE_DIRS | _DEFAULT_IGNORE_FILES | {"*.pyc", "*.pyo", "*.tmp"})
)


@dataclass
class FileEntry:
    path: str
    size_bytes: int
    mtime_ns: int
    sha256: str
    language: str | None = None
    sha256_head: str | None = None
    first_lines: list[str] = field(default_factory=list)

    @property
    def rel_path(self) -> str:
        return self.path

    @property
    def size(self) -> int:
        return self.size_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "sha256": self.sha256,
            "language": self.language,
            "sha256_head": self.sha256_head,
            "first_lines": list(self.first_lines),
            # Backward-compatible keys used by earlier cache payloads.
            "rel_path": self.path,
            "size": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> FileEntry:
        lines = raw.get("first_lines", [])
        return cls(
            path=str(raw.get("path", raw.get("rel_path", ""))),
            size_bytes=int(raw.get("size_bytes", raw.get("size", 0))),
            mtime_ns=int(raw.get("mtime_ns", 0)),
            sha256=str(raw.get("sha256", "")),
            language=str(raw.get("language")) if raw.get("language") is not None else None,
            sha256_head=str(raw.get("sha256_head")) if raw.get("sha256_head") is not None else None,
            first_lines=[str(item) for item in lines] if isinstance(lines, list) else [],
        )


# Backward compatibility with previous class name.
IndexedFile = FileEntry


@dataclass
class WorkspaceIndex:
    schema_version: int = INDEX_SCHEMA_VERSION
    workspace_root: str = ""
    snapshot_id: str = ""
    content_id: str = ""
    file_count: int = 0
    total_size_bytes: int = 0
    files: list[FileEntry] = field(default_factory=list)
    tree: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def entries(self) -> list[FileEntry]:
        return self.files

    @property
    def root_hash(self) -> str:
        return self.content_id or self.snapshot_id

    @property
    def total_bytes(self) -> int:
        return self.total_size_bytes

    @property
    def tree_summary(self) -> str:
        return build_tree_summary(self.files)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "workspace_root": self.workspace_root,
            "snapshot_id": self.snapshot_id,
            "content_id": self.content_id,
            "root_hash": self.root_hash,
            "file_count": self.file_count,
            "total_size_bytes": self.total_size_bytes,
            "total_bytes": self.total_size_bytes,
            "files": [entry.to_dict() for entry in self.files],
            "entries": [entry.to_dict() for entry in self.files],
            "tree_summary": self.tree_summary,
            "tree": self.tree,
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> WorkspaceIndex:
        files_raw = raw.get("files", raw.get("entries", []))
        files: list[FileEntry] = []
        if isinstance(files_raw, list):
            for item in files_raw:
                if isinstance(item, dict):
                    files.append(FileEntry.from_dict(item))

        tree = raw.get("tree", {})
        if not isinstance(tree, dict):
            tree = _build_tree(files)

        errors_raw = raw.get("errors", [])
        errors = [str(item) for item in errors_raw] if isinstance(errors_raw, list) else []

        return cls(
            schema_version=int(raw.get("schema_version", INDEX_SCHEMA_VERSION)),
            workspace_root=str(raw.get("workspace_root", "")),
            snapshot_id=str(raw.get("snapshot_id", "")),
            content_id=str(raw.get("content_id", raw.get("root_hash", ""))),
            file_count=int(raw.get("file_count", len(files))),
            total_size_bytes=int(raw.get("total_size_bytes", raw.get("total_bytes", 0))),
            files=files,
            tree=tree,
            errors=errors,
        )


def default_index_cache_dir(workspace_root: str | Path) -> Path:
    root = Path(workspace_root).resolve()
    return root / ".maestro-cache" / "index"


def file_sha256(path: Path, chunk_size: int = _HASH_CHUNK_SIZE) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _hash_file_head(path: Path, max_bytes: int = _HEAD_HASH_BYTES) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        hasher.update(f.read(max_bytes))
    return hasher.hexdigest()


def _read_first_lines(path: Path, max_lines: int = _FIRST_LINES_DEFAULT) -> list[str]:
    if max_lines <= 0:
        return []
    lines: list[str] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for _ in range(max_lines):
                line = f.readline()
                if line == "":
                    break
                lines.append(line.rstrip("\r\n"))
    except OSError:
        return []
    return lines


def _infer_language(path: str | Path) -> str | None:
    suffix = Path(path).suffix.lower()
    mapping = {
        ".py": "python",
        ".js": "javascript",
        ".cjs": "javascript",
        ".mjs": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".jsx": "javascript",
        ".json": "json",
        ".md": "markdown",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".toml": "toml",
        ".ini": "ini",
        ".cfg": "ini",
        ".xml": "xml",
        ".html": "html",
        ".css": "css",
        ".sh": "shell",
        ".ps1": "powershell",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".kt": "kotlin",
        ".swift": "swift",
        ".rb": "ruby",
        ".php": "php",
        ".sql": "sql",
        ".c": "c",
        ".h": "c",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".hpp": "cpp",
    }
    return mapping.get(suffix)


def _should_exclude(path: str | Path, excludes: set[str] | list[str] | tuple[str, ...] | None = None) -> bool:
    candidate = Path(path)
    posix = candidate.as_posix().lstrip("./")
    name = candidate.name
    patterns = excludes if excludes is not None else _DEFAULT_EXCLUDES
    for pattern in patterns:
        token = str(pattern).strip()
        if not token:
            continue
        if any(ch in token for ch in "*?[]"):
            if fnmatch.fnmatch(posix, token) or fnmatch.fnmatch(name, token):
                return True
            continue
        if token == name:
            return True
        if token in candidate.parts:
            return True
    return False


def _normalize_rel_path(path: Path) -> str:
    return path.as_posix()


def _collect_workspace_metadata(
    workspace_root: Path,
    *,
    include_hidden: bool,
    ignore_dirs: set[str],
    ignore_files: set[str],
    excludes: set[str],
    max_files: int,
) -> tuple[list[tuple[Path, str, int, int]], list[str]]:
    files: list[tuple[Path, str, int, int]] = []
    errors: list[str] = []
    hit_limit = False

    for dirpath, dirnames, filenames in os.walk(workspace_root, topdown=True, followlinks=False):
        current = Path(dirpath)
        rel_dir = _normalize_rel_path(current.relative_to(workspace_root)) if current != workspace_root else ""

        filtered_dirs: list[str] = []
        for dirname in sorted(dirnames):
            rel_dirname = f"{rel_dir}/{dirname}" if rel_dir else dirname
            if dirname in ignore_dirs:
                continue
            if not include_hidden and dirname.startswith("."):
                continue
            if _should_exclude(rel_dirname, excludes):
                continue
            filtered_dirs.append(dirname)
        dirnames[:] = filtered_dirs

        for filename in sorted(filenames):
            rel_name = f"{rel_dir}/{filename}" if rel_dir else filename
            if filename in ignore_files:
                continue
            if not include_hidden and filename.startswith("."):
                continue
            if _should_exclude(rel_name, excludes):
                continue

            abs_path = current / filename
            if abs_path.is_symlink() or not abs_path.is_file():
                continue

            try:
                stat = abs_path.stat()
            except OSError as exc:
                errors.append(f"{abs_path}: {exc}")
                continue

            try:
                rel_path = _normalize_rel_path(abs_path.relative_to(workspace_root))
            except ValueError:
                continue

            mtime_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)))
            files.append((abs_path, rel_path, int(stat.st_size), mtime_ns))
            if len(files) >= max_files:
                hit_limit = True
                break
        if hit_limit:
            break

    files.sort(key=lambda item: item[1])
    if hit_limit:
        errors.append(f"Workspace index hit file cap {max_files}; results truncated.")
    return files, errors


def _snapshot_id(files: list[tuple[Path, str, int, int]]) -> str:
    hasher = hashlib.sha256()
    hasher.update(str(len(files)).encode("utf-8"))
    hasher.update(b"\0")

    for _, rel_path, size, mtime_ns in files:
        hasher.update(rel_path.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(str(size).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(str(mtime_ns).encode("utf-8"))
        hasher.update(b"\0")

    return hasher.hexdigest()


def _content_id(files: list[FileEntry]) -> str:
    hasher = hashlib.sha256()
    hasher.update(str(len(files)).encode("utf-8"))
    hasher.update(b"\0")

    for entry in sorted(files, key=lambda item: item.path):
        hasher.update(entry.path.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(entry.sha256.encode("utf-8"))
        hasher.update(b"\0")

    return hasher.hexdigest()


def _build_tree(files: list[FileEntry]) -> dict[str, Any]:
    root: dict[str, Any] = {"dirs": {}, "files": []}

    for entry in files:
        parts = [part for part in entry.path.split("/") if part]
        if not parts:
            continue

        node = root
        for dirname in parts[:-1]:
            dirs = node.setdefault("dirs", {})
            child = dirs.get(dirname)
            if not isinstance(child, dict):
                child = {"dirs": {}, "files": []}
                dirs[dirname] = child
            node = child
        node.setdefault("files", []).append(parts[-1])

    return _sort_tree_node(root)


def _sort_tree_node(node: dict[str, Any]) -> dict[str, Any]:
    files = node.get("files", [])
    dirs = node.get("dirs", {})

    sorted_files = sorted(str(item) for item in files) if isinstance(files, list) else []

    sorted_dirs: dict[str, Any] = {}
    if isinstance(dirs, dict):
        for dirname in sorted(dirs):
            child = dirs[dirname]
            if isinstance(child, dict):
                sorted_dirs[str(dirname)] = _sort_tree_node(child)

    return {
        "dirs": sorted_dirs,
        "files": sorted_files,
    }


def build_tree_summary(files: list[FileEntry], *, max_lines: int = 200, max_depth: int = 4) -> str:
    if not files:
        return ""
    tree = _build_tree(files)
    lines: list[str] = []

    def _walk(node: dict[str, Any], prefix: str, depth: int) -> None:
        if len(lines) >= max_lines:
            return
        dirs = node.get("dirs", {})
        file_names = node.get("files", [])
        if isinstance(dirs, dict):
            for dirname in sorted(dirs):
                lines.append(f"{prefix}{dirname}/")
                if len(lines) >= max_lines:
                    return
                if depth < max_depth and isinstance(dirs[dirname], dict):
                    _walk(dirs[dirname], f"{prefix}  ", depth + 1)
        if isinstance(file_names, list):
            for filename in sorted(str(item) for item in file_names):
                lines.append(f"{prefix}{filename}")
                if len(lines) >= max_lines:
                    return

    _walk(tree, "", 0)
    if len(lines) >= max_lines:
        lines.append("...")
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return raw if isinstance(raw, dict) else None


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def load_cached_workspace_index(
    workspace_root: str | Path,
    *,
    snapshot_id: str | None = None,
    cache_dir: str | Path | None = None,
) -> WorkspaceIndex | None:
    root = Path(workspace_root).resolve()
    index_dir = Path(cache_dir).resolve() if cache_dir is not None else default_index_cache_dir(root)

    selected_snapshot = snapshot_id
    if selected_snapshot is None:
        latest_raw = _read_json(index_dir / "latest.json")
        if latest_raw is None:
            return None
        selected_snapshot = str(latest_raw.get("snapshot_id", "")).strip()
        if not selected_snapshot:
            return None

    index_raw = _read_json(index_dir / f"{selected_snapshot}.json")
    if index_raw is None:
        return None

    index = WorkspaceIndex.from_dict(index_raw)
    if index.schema_version != INDEX_SCHEMA_VERSION:
        return None
    if Path(index.workspace_root).resolve() != root:
        return None
    return index


def store_workspace_index(index: WorkspaceIndex, *, cache_dir: str | Path | None = None) -> Path:
    root = Path(index.workspace_root).resolve()
    index_dir = Path(cache_dir).resolve() if cache_dir is not None else default_index_cache_dir(root)
    index_path = index_dir / f"{index.snapshot_id}.json"

    _write_json_atomic(index_path, index.to_dict())
    _write_json_atomic(
        index_dir / "latest.json",
        {
            "schema_version": INDEX_SCHEMA_VERSION,
            "snapshot_id": index.snapshot_id,
        },
    )
    return index_path


def build_workspace_index(
    workspace_root: str | Path,
    *,
    cache_dir: str | Path | None = None,
    force_rebuild: bool = False,
    include_hidden: bool = True,
    excludes: list[str] | set[str] | tuple[str, ...] | None = None,
    max_files: int = _MAX_FILES,
    ignore_dirs: set[str] | None = None,
    ignore_files: set[str] | None = None,
) -> WorkspaceIndex:
    root = Path(workspace_root).resolve()
    if not root.exists() or not root.is_dir():
        raise NotADirectoryError(f"Workspace root does not exist or is not a directory: {root}")

    dirs_to_ignore = set(_DEFAULT_IGNORE_DIRS)
    if ignore_dirs:
        dirs_to_ignore.update(ignore_dirs)

    files_to_ignore = set(_DEFAULT_IGNORE_FILES)
    if ignore_files:
        files_to_ignore.update(ignore_files)

    exclude_patterns = set(_DEFAULT_EXCLUDES)
    if excludes:
        exclude_patterns.update(str(item) for item in excludes if str(item).strip())

    metadata, errors = _collect_workspace_metadata(
        root,
        include_hidden=include_hidden,
        ignore_dirs=dirs_to_ignore,
        ignore_files=files_to_ignore,
        excludes=exclude_patterns,
        max_files=max_files,
    )
    snapshot = _snapshot_id(metadata)

    if not force_rebuild:
        cached = load_cached_workspace_index(root, snapshot_id=snapshot, cache_dir=cache_dir)
        if cached is not None:
            if errors:
                cached.errors = list(dict.fromkeys(cached.errors + errors))
            return cached

    indexed_files: list[FileEntry] = []
    hash_errors = list(errors)
    total_size_bytes = 0

    for abs_path, rel_path, size, mtime_ns in metadata:
        try:
            digest = file_sha256(abs_path)
        except OSError as exc:
            hash_errors.append(f"{abs_path}: {exc}")
            continue

        indexed_files.append(
            FileEntry(
                path=rel_path,
                size_bytes=size,
                mtime_ns=mtime_ns,
                sha256=digest,
                language=_infer_language(rel_path),
                sha256_head=_hash_file_head(abs_path),
                first_lines=_read_first_lines(abs_path),
            )
        )
        total_size_bytes += size

    indexed_files.sort(key=lambda item: item.path)
    index = WorkspaceIndex(
        schema_version=INDEX_SCHEMA_VERSION,
        workspace_root=str(root),
        snapshot_id=snapshot,
        content_id=_content_id(indexed_files),
        file_count=len(indexed_files),
        total_size_bytes=total_size_bytes,
        files=indexed_files,
        tree=_build_tree(indexed_files),
        errors=hash_errors,
    )

    store_workspace_index(index, cache_dir=cache_dir)
    return index


def quick_root_hash(
    workspace_root: str | Path,
    *,
    include_hidden: bool = True,
    excludes: list[str] | set[str] | tuple[str, ...] | None = None,
    max_files: int = _MAX_FILES,
    ignore_dirs: set[str] | None = None,
    ignore_files: set[str] | None = None,
) -> str:
    root = Path(workspace_root).resolve()
    dirs_to_ignore = set(_DEFAULT_IGNORE_DIRS)
    if ignore_dirs:
        dirs_to_ignore.update(ignore_dirs)
    files_to_ignore = set(_DEFAULT_IGNORE_FILES)
    if ignore_files:
        files_to_ignore.update(ignore_files)

    exclude_patterns = set(_DEFAULT_EXCLUDES)
    if excludes:
        exclude_patterns.update(str(item) for item in excludes if str(item).strip())

    metadata, _ = _collect_workspace_metadata(
        root,
        include_hidden=include_hidden,
        ignore_dirs=dirs_to_ignore,
        ignore_files=files_to_ignore,
        excludes=exclude_patterns,
        max_files=max_files,
    )
    return _snapshot_id(metadata)


def load_cached_index(
    workspace_root: str | Path,
    *,
    snapshot_id: str | None = None,
    cache_dir: str | Path | None = None,
) -> WorkspaceIndex | None:
    return load_cached_workspace_index(workspace_root, snapshot_id=snapshot_id, cache_dir=cache_dir)


def save_index(index: WorkspaceIndex, *, cache_dir: str | Path | None = None) -> Path:
    return store_workspace_index(index, cache_dir=cache_dir)


def get_workspace_index(
    workspace_root: str | Path,
    *,
    cache_dir: str | Path | None = None,
    include_hidden: bool = True,
    excludes: list[str] | set[str] | tuple[str, ...] | None = None,
    max_files: int = _MAX_FILES,
    ignore_dirs: set[str] | None = None,
    ignore_files: set[str] | None = None,
) -> WorkspaceIndex:
    return build_workspace_index(
        workspace_root,
        cache_dir=cache_dir,
        force_rebuild=False,
        include_hidden=include_hidden,
        excludes=excludes,
        max_files=max_files,
        ignore_dirs=ignore_dirs,
        ignore_files=ignore_files,
    )


__all__ = [
    "INDEX_SCHEMA_VERSION",
    "FileEntry",
    "IndexedFile",
    "WorkspaceIndex",
    "_DEFAULT_EXCLUDES",
    "_MAX_FILES",
    "_hash_file_head",
    "_infer_language",
    "_read_first_lines",
    "_should_exclude",
    "build_workspace_index",
    "build_tree_summary",
    "default_index_cache_dir",
    "file_sha256",
    "get_workspace_index",
    "load_cached_index",
    "load_cached_workspace_index",
    "quick_root_hash",
    "save_index",
    "store_workspace_index",
]
