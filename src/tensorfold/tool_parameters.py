"""Schema-aware decoding of Qwen XML parameter text."""

from __future__ import annotations

import ast
import json
from collections.abc import Sequence
from typing import Any
from urllib.parse import unquote

_UNIONS = ("anyOf", "oneOf", "allOf")
_KINDS = frozenset({"array", "object", "boolean", "integer", "number", "null", "string"})


def parameter_schemas(tools: Sequence[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    """Each offered tool's parameter schemas by lowercase name; a spec that is not an object reads as untyped."""

    result = {}
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        function = tool["function"] if isinstance(tool.get("function"), dict) else tool
        parameters = function.get("parameters") or function.get("input_schema") or {}
        properties = parameters.get("properties") or {} if isinstance(parameters, dict) else {}
        refs: dict[str, dict[str, Any] | None] = {}
        result[str(function.get("name", "")).lower()] = {
            name: _resolved(schema, parameters, refs) for name, schema in properties.items() if isinstance(schema, dict)
        } if isinstance(properties, dict) else {}
    return result


def _pointer(root: Any, ref: str) -> Any:
    """The node a local JSON pointer (``#/$defs/a~1b``) names in ``root``, else None."""

    if not ref.startswith("#"):
        return None
    node = root
    for part in unquote(ref[1:]).split("/")[1:]:
        if "~" in part.replace("~0", "").replace("~1", ""):
            return None                                     # a tilde that escapes nothing
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            return None
    return node


def _resolved(schema: dict[str, Any], root: Any, refs: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    """``schema`` with its ``$ref`` (and its unions' refs) inlined as an ``allOf`` branch, each ref resolved once.

    A ref that points nowhere or back into itself (None in ``refs`` while it resolves) constrains nothing."""

    out = {k: ([_resolved(b, root, refs) if isinstance(b, dict) else b for b in v] if k in _UNIONS and isinstance(v, list)
               else v) for k, v in schema.items() if k != "$ref"}
    ref = schema.get("$ref")
    if isinstance(ref, str):
        if ref not in refs:
            refs[ref] = None
            target = _pointer(root, ref)
            refs[ref] = _resolved(target, root, refs) if isinstance(target, dict) else None
        if refs[ref] is not None:
            out["allOf"] = [*(out.get("allOf") if isinstance(out.get("allOf"), list) else []), refs[ref]]
    return out


def _kind_of(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    return {int: "integer", float: "number", str: "string", list: "array", dict: "object"}.get(type(value), "null")


def _widened(kinds: set[str]) -> frozenset[str]:
    return frozenset(kinds | {"integer"} if "number" in kinds else kinds)     # an integer is a number


def schema_kinds(schema: dict[str, Any]) -> frozenset[str] | None:
    """The JSON kinds a schema takes, from its type or type list, enum, const and unions (refs inlined by
    ``parameter_schemas``), each a constraint that meets the others; None when nothing constrains it."""

    constraints: list[frozenset[str]] = []
    kind = schema.get("type")
    named = _KINDS & ({kind} if isinstance(kind, str) else set(kind) if isinstance(kind, list) else set())
    if named:                                   # a type that names no JSON kind constrains nothing
        constraints.append(_widened(named))
    if isinstance(schema.get("enum"), list):
        constraints.append(frozenset(_kind_of(v) for v in schema["enum"]))
    if "const" in schema:
        constraints.append(frozenset({_kind_of(schema["const"])}))
    for key in ("anyOf", "oneOf"):
        branches = [schema_kinds(b) for b in schema.get(key) or [] if isinstance(b, dict)]
        if branches and None not in branches:
            constraints.append(frozenset().union(*branches))
    for branch in schema.get("allOf") or []:
        kinds = schema_kinds(branch) if isinstance(branch, dict) else None
        if kinds is not None:
            constraints.append(kinds)
    return frozenset.intersection(*constraints) if constraints else None


def typed_parameter(schema: dict[str, Any]) -> bool:
    kinds = schema_kinds(schema)
    return kinds is not None and bool(kinds - {"string"})


def closed_json(text: str) -> str | None:
    """``text`` with the arrays and objects it left open closed, or None if it doesn't only stop short of them."""

    closers, in_string, escaped = [], False, False
    for ch in text:
        if in_string:
            escaped, in_string = (False, True) if escaped else (ch == "\\", ch != '"')
        elif ch == '"':
            in_string = True
        elif ch in "[{":
            closers.append("]" if ch == "[" else "}")
        elif ch in "]}" and (not closers or closers.pop() != ch):
            return None
    return text.rstrip() + "".join(reversed(closers)) if closers and not in_string else None


_PY_WORDS = {"true": True, "false": False, "none": None, "null": None}


def _python_literal(text: str) -> Any:
    """``text`` as a Python literal (``True``, ``['a']``, ``{'k': None}``), tuples as lists; ValueError if not one."""

    word = _PY_WORDS.get(text.strip().lower(), ...)
    if word is not ...:
        return word

    def lists(v: Any) -> Any:
        if isinstance(v, (list, tuple)):
            return [lists(x) for x in v]
        if isinstance(v, dict):
            return {k: lists(x) for k, x in v.items()}
        return v

    try:
        return lists(ast.literal_eval(text.strip()))
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError) as exc:
        raise ValueError(str(exc)) from None


def decode_parameter(value: str, schema: dict[str, Any], *, python: bool = True) -> Any:
    """``value`` as the schema's type when it spells one, else the text; ``python`` also reads Python's spelling."""

    kinds = schema_kinds(schema)
    if kinds is None or not kinds - {"string"}:
        return value
    taken = kinds - {"string"}                  # a quoted string keeps the text the model wrote

    def valid(v: Any) -> bool:
        return _kind_of(v) in taken

    # a model can end an object or array value one closer short (#87): the value closed is what it meant
    for text in (value, closed_json(value) if taken & {"array", "object"} else None):
        if text is None:
            continue
        try:
            parsed = json.loads(text)
            json.dumps(parsed, allow_nan=False)
        except (ValueError, TypeError):
            continue
        return parsed if valid(parsed) else value
    # not JSON: Python's spelling (True, None, single-quoted strings), as vLLM's Qwen parsers also accept
    if not python:
        return value
    try:
        parsed = _python_literal(value)
        json.dumps(parsed, allow_nan=False)
    except (ValueError, TypeError):
        return value
    return parsed if valid(parsed) else value
