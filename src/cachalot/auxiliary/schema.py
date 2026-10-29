"""
A deterministic validator for the JSON-schema subset the auxiliary tasks use.

Constrained decoding makes a schema violation unlikely, not impossible, and a
model's output is untrusted input either way, so every structured result is
checked here before any task reads it. The subset: type (object, array, string,
integer, number, boolean), properties, required, additionalProperties: false,
items, enum, minItems, maxItems, minLength, maxLength, minimum, maximum.
Unknown keywords are ignored, never treated as satisfied constraints.
"""

from __future__ import annotations

from typing import Any

_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
}


def _type_ok(value: Any, kind: str) -> bool:
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    cls = _TYPES.get(kind)
    return cls is not None and isinstance(value, cls)


def validate(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Return a list of problems; empty means valid."""
    problems: list[str] = []
    kind = schema.get("type")
    if kind is not None and not _type_ok(value, kind):
        return [f"{path}: expected {kind}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        problems.append(f"{path}: {value!r} not in {schema['enum']}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            problems.append(f"{path}: shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            problems.append(f"{path}: longer than {schema['maxLength']}")
    if _type_ok(value, "number"):
        if "minimum" in schema and value < schema["minimum"]:
            problems.append(f"{path}: below {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            problems.append(f"{path}: above {schema['maximum']}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            problems.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            problems.append(f"{path}: more than {schema['maxItems']} items")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for i, item in enumerate(value):
                problems.extend(validate(item, item_schema, f"{path}[{i}]"))
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                problems.append(f"{path}: missing {name!r}")
        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in props:
                    problems.append(f"{path}: unexpected {name!r}")
        for name, sub in props.items():
            if name in value:
                problems.extend(validate(value[name], sub, f"{path}.{name}"))
    return problems
