"""Pythonic tool calls (Liquid's LFM2): ``<|tool_call_start|>[name(key="value", n=2), other()]<|tool_call_end|>``.

A block holds a Python list of calls with keyword arguments. Values are Python literals, or JSON where the chat
template wrote an object or array argument with ``tojson`` (so ``true``, ``false`` and ``null`` read as JSON's). A
name need not be a Python identifier (``web-search(...)``): each top-level call's name is swapped for a placeholder
before Python's parser reads the list. Positional arguments take the tool's parameters in its schema's order.
"""

from __future__ import annotations

import ast
import json
import uuid
from typing import Any

CALL_OPEN, CALL_CLOSE = "<|tool_call_start|>", "<|tool_call_end|>"
_NAMES = {"true": True, "false": False, "null": None, "True": True, "False": False, "None": None}


def _tool_name(tool: dict[str, Any]) -> str:
    fn = tool.get("function") if isinstance(tool, dict) else None
    return str((fn or tool).get("name") or "").strip() if isinstance(tool, dict) else ""


def _parameters(tool: dict[str, Any]) -> list[str]:
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    props = (fn.get("parameters") or {}).get("properties") or {}
    return list(props) if isinstance(props, dict) else []


def _value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in _NAMES:
        return _NAMES[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_value(e) for e in node.elts]
    if isinstance(node, ast.Dict):
        if any(k is None for k in node.keys):
            raise ValueError("a ** expansion in a tool call argument")
        return {_value(k): _value(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _value(node.operand)
        if isinstance(inner, (int, float)) and not isinstance(inner, bool):
            return -inner if isinstance(node.op, ast.USub) else inner
    raise ValueError(f"a tool call argument that is not a literal: {ast.dump(node)[:80]}")


def _placeholders(body: str) -> tuple[str, list[str]]:
    """``body`` with each top-level call's name replaced by ``_tfN`` (strings and brackets skipped), and the names."""

    out: list[str] = []
    names: list[str] = []
    depth, i, n = 0, 0, len(body)
    listed = body.startswith("[")
    expect = not listed                                     # a bare call, or in the list after "[" or ","
    while i < n:
        ch = body[i]
        if ch in "'\"":
            end = i + 1
            while end < n and body[end] != ch:
                end += 2 if body[end] == "\\" else 1
            out.append(body[i:end + 1])
            i = end + 1
            continue
        if expect and not ch.isspace() and ch not in "])":
            end = body.find("(", i)
            if end < 0:
                raise ValueError("a tool call without its argument list")
            names.append(body[i:end].strip())
            out.append(f"_tf{len(names) - 1}")
            i, expect = end, False
            continue
        if ch in "([{":
            depth += 1
            expect = depth == 1 and ch == "["
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 1 and listed:
            expect = True
        out.append(ch)
        i += 1
    return "".join(out), names


def parse_block(body: str, tools: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """The calls of one block's text (between the markers) as (name as written, arguments); ValueError if malformed."""

    text, names = _placeholders(body.strip())
    try:
        tree = ast.parse(text, mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"not a Python list of calls: {exc}") from None
    items = tree.elts if isinstance(tree, ast.List) else [tree]
    order = {_tool_name(t).lower(): _parameters(t) for t in tools}
    calls: list[tuple[str, dict[str, Any]]] = []
    for item in items:
        if not (isinstance(item, ast.Call) and isinstance(item.func, ast.Name) and item.func.id.startswith("_tf")):
            raise ValueError("a list item that is not a call")
        name = names[int(item.func.id[3:])]
        params = order.get(name.lower(), [])
        if len(item.args) > len(params):
            raise ValueError(f"{name}: more positional arguments than the tool has parameters")
        args = {params[j]: _value(a) for j, a in enumerate(item.args)}
        for kw in item.keywords:
            if kw.arg is None:
                raise ValueError(f"{name}: a ** expansion")
            args[kw.arg] = _value(kw.value)
        calls.append((name, args))
    return calls


def parse_tool_calls(text: str, tools: list[dict[str, Any]], *,
                     max_calls: int | None = None) -> tuple[str, list[dict[str, Any]] | None]:
    """(the text outside the blocks, OpenAI tool calls or None); a block that does not parse, or calls a tool not
    offered, stays in the text."""

    if not tools:
        return text, None
    known = {_tool_name(t).lower(): _tool_name(t) for t in tools}
    calls: list[dict[str, Any]] = []
    residue: list[str] = []
    at = 0
    while (start := text.find(CALL_OPEN, at)) >= 0:
        end = text.find(CALL_CLOSE, start + len(CALL_OPEN))
        if end < 0:
            break
        residue.append(text[at:start])
        at = end + len(CALL_CLOSE)
        try:
            found = parse_block(text[start + len(CALL_OPEN):end], tools)
            for _, args in found:
                json.dumps(args, allow_nan=False)
        except (ValueError, TypeError):
            found = []
        if not found or any(name.lower() not in known for name, _ in found):
            if max_calls is None:
                residue.append(text[start:at])
            continue
        for name, args in found:
            if max_calls is not None and len(calls) >= max_calls:
                break
            calls.append({"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                          "function": {"name": known[name.lower()],
                                       "arguments": json.dumps(args, ensure_ascii=False, separators=(",", ":"))}})
    residue.append(text[at:])
    return "".join(residue).strip(), calls or None


def close_call(text: str, tools: list[dict[str, Any]]) -> str:
    """What closes the block ``text`` ends inside when an end token came first: "" unless the closed block parses
    whole into calls of offered tools."""

    start = text.rfind(CALL_OPEN)
    if not tools or start < 0 or text.find(CALL_CLOSE, start) >= 0:
        return ""
    body = text[start + len(CALL_OPEN):].rstrip()
    for tail in ("", "]", ")]"):
        content, calls = parse_tool_calls(CALL_OPEN + body + tail + CALL_CLOSE, tools)
        if calls and not content:
            return tail + CALL_CLOSE
    return ""
