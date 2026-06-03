# Rule: Type Safety

## Scope
All Python files in `src/maestro_cli/`.

## Requirements

### Annotations
- Every function MUST have full type annotations on all parameters and return type
- Dataclass fields MUST have type annotations
- Variables may omit annotations when the type is obvious from context

### Literal Types
- Enum-like values use `Literal` types (defined in `models.py`):
  - `EngineName = Literal["codex", "claude", "gemini", "copilot", "qwen", "ollama", "llama"]`
  - `ExecutionProfile = Literal["plan", "safe", "yolo"]`
  - `TaskStatus = Literal["success", "failed", "soft_failed", "skipped", "dry_run"]`
- When adding new enum values, update the Literal type FIRST

### Dataclass Patterns
- All data models are `@dataclass` (never plain dicts for structured data)
- Mutable defaults: `field(default_factory=list)`, `field(default_factory=dict)`
- Serialization via explicit `to_dict()` methods (not `dataclasses.asdict()`)

### Null Safety
- Optional fields: `X | None = None`
- Check for None before accessing optional values
- Never use `assert` for None checks in production paths (use explicit if/raise)

### No Any
- Avoid `Any` type except at serialization boundaries (e.g., `to_dict() -> dict[str, Any]`)
- YAML raw data from `yaml.safe_load()` may use `Any` during parsing
- Parsed data must be converted to typed structures immediately
