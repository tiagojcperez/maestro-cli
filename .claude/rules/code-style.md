# Rule: Code Style

## Scope
All Python files in `src/maestro_cli/`.

## Requirements

### Imports
- `from __future__ import annotations` MUST be the first import in every file
- Standard library imports first, then third-party, then local (separated by blank lines)
- Local imports use relative syntax: `from .models import PlanSpec`

### Naming
- Module-level functions: `snake_case`
- Private/internal functions: `_snake_case` (single underscore prefix)
- Constants: `_UPPER_SNAKE_CASE` (private) or `UPPER_SNAKE_CASE` (public)
- Dataclass fields: `snake_case`
- No classes except `@dataclass` and `Exception` subclasses

### Types
- PEP 604 union syntax: `str | None` (NEVER `Optional[str]` or `Union[str, None]`)
- Built-in generics: `list[str]`, `dict[str, str]`, `set[str]` (NOT `List`, `Dict`, `Set`)
- `Literal[...]` for finite value sets
- `field(default_factory=...)` for mutable defaults in dataclasses
- Full annotations on all function signatures (parameters AND return type)

### Strings
- f-strings for all interpolation (NEVER `.format()` or `%`)
- `encoding="utf-8"` on every `open()`, `read_text()`, `write_text()` call

### Paths
- `pathlib.Path` for all file/directory operations (NEVER `os.path`)
- Exception: `os.environ` for environment variables, `os.name` for platform detection

### Formatting
- Console output: `print(f"[maestro] ...")`
- No trailing whitespace
- No unused imports
- No commented-out code blocks
