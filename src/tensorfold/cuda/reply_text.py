"""The CUDA server's reply text: streamed decoding, think blocks and tool-call parsing (the Mac lane server's rules)."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from tensorfold.server import pythonic_calls
from tensorfold.server.stopping import StopPolicy
from tensorfold.server.text import hide_tool_calls  # noqa: F401  (the CUDA server's, with the Mac's four formats)
from tensorfold.server.tools import parse_glm_tool_call_block, tool_call_blocks, tool_spec_name
from tensorfold.tool_parameters import closed_json, decode_parameter, parameter_schemas

_CALL_OPEN, _CALL_CLOSE = "<tool_call>", "</tool_call>"
_TOOL_FUNCTION_BLOCK_RE = re.compile(r"^\s*<function=([^>\s]+)>\s*(.*?)\s*</function>\s*$", re.IGNORECASE | re.DOTALL)
_TOOL_PARAMETER_BLOCK_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.IGNORECASE | re.DOTALL)



class StreamDecoder:
    """Decode a shared token window to preserve leading spaces and byte boundaries, deferring incomplete characters.

    With ``close``, the think block's close token, ``close_at`` is where that token's text begins (-1 before it
    comes), for ``split_thinking``; None without it."""

    def __init__(self, tok, skip: tuple[int, ...] = (), close: int | None = None):
        self.tok, self.skip, self.close = tok, frozenset(skip), close
        self.ids: list[int] = []
        self.text = ""
        self.prefix = 0             # window start
        self.read = 0               # tokens already reflected in ``text``
        self.close_at: int | None = None if close is None else -1

    def _decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)

    def add(self, new: list[int]) -> str:
        self.ids.extend(t for t in new if t not in self.skip)
        if self.close_at == -1 and self.close in new:           # once a reply: the text before its first close
            self.close_at = len(self._decode(self.ids[:self.ids.index(self.close)]))
        before = self._decode(self.ids[self.prefix:self.read])
        after = self._decode(self.ids[self.prefix:])
        if len(after) > len(before) and not after.endswith("\ufffd"):
            self.text += after[len(before):]
            self.prefix, self.read = self.read, len(self.ids)
        return self.text

    def final(self) -> str:
        """Everything, including a trailing partial character (as decoding it all at once gives)."""

        before = self._decode(self.ids[self.prefix:self.read])
        return self.text + self._decode(self.ids[self.prefix:])[len(before):]



class StopStrings:
    """A request's ``stop`` strings, decided token by token, so a reply cuts at the same token however its tokens arrive."""

    def __init__(self, strings: tuple[str, ...], tok, skip: tuple[int, ...] = ()):
        self.strings, self.tok, self.skip = strings, tok, frozenset(skip)
        # a new match lies in the last (its UTF-8 length) tokens, plus eight as the Mac's ``StopPolicy`` keeps
        self.tail = max((len(s.encode()) for s in strings), default=0) + 8

    def hit(self, tokens: list[int]) -> bool:
        """Whether the text through the newest token holds a stop string (earlier tokens were checked already)."""

        ids = [t for t in tokens[-self.tail:] if t not in self.skip]
        text = self.tok.decode(ids, skip_special_tokens=False)
        return any(stop in text for stop in self.strings)

    def visible(self, text: str, *, partial: bool = False) -> str:
        """The text before the first match; ``partial`` also holds back an end that may begin one (the Mac's rule)."""

        return StopPolicy.visible(self, text, partial=partial)


def parse_tool_calls(text: str, tools: list[dict[str, Any]], *, max_calls: int | None = None) -> tuple[str, list[dict[str, Any]] | None]:
    """Qwen ``<function=name><parameter=k>v</parameter></function>``, GLM ``name<arg_key>..``, JSON or LFM2 calls.

    A reply may name the other format's opener in its prose: LFM2's Pythonic list is read when the reply holds its
    opener, unless a ``<tool_call>`` block in it parses to calls."""

    if not tools:
        return text, None
    if pythonic_calls.CALL_OPEN not in text:
        return _parse_tagged(text, tools, max_calls)
    if _CALL_OPEN in text.lower():
        content, calls = _parse_tagged(text, tools, max_calls)
        if calls:
            return content, calls
    return pythonic_calls.parse_tool_calls(text, tools, max_calls=max_calls)


def _parse_tagged(text: str, tools: list[dict[str, Any]],
                  max_calls: int | None) -> tuple[str, list[dict[str, Any]] | None]:
    """The ``<tool_call>`` blocks' calls (Qwen, GLM or JSON inside)."""

    known = {tool_spec_name(t).lower(): tool_spec_name(t) for t in tools}
    schemas = parameter_schemas(tools)
    calls: list[dict[str, Any]] = []
    residue: list[str] = []
    cursor = 0
    for start, end, body in tool_call_blocks(text):
        residue.append(text[cursor:start])
        cursor = end
        if max_calls is not None and len(calls) >= max_calls:
            continue
        block = body.strip()
        name, args = None, {}
        try:
            payload = json.loads(block)
            if isinstance(payload, dict):
                fn = payload.get("function") if isinstance(payload.get("function"), dict) else payload
                name = fn.get("name")
                args = fn.get("arguments", fn.get("parameters", {}))
                if isinstance(args, str):
                    args = json.loads(args) if args.strip() else {}
        except (json.JSONDecodeError, AttributeError):
            m = _TOOL_FUNCTION_BLOCK_RE.match(block)
            if m:
                if max_calls is not None and _TOOL_PARAMETER_BLOCK_RE.sub("", m.group(2)).strip():
                    continue
                name = m.group(1).strip()
                # typed parameters (array, object, number...) decode per the tool's schema, as the Mac server does
                props = schemas.get(name.lower(), {})
                args = {key: decode_parameter(value, props.get(key, {}))
                        for key, value in ((p.group(1).strip(), p.group(2))
                                           for p in _TOOL_PARAMETER_BLOCK_RE.finditer(m.group(2)))}
            else:
                glm = parse_glm_tool_call_block(block, tools, complete=max_calls is not None)
                if glm is not None:
                    name, args = glm
        if not name or str(name).lower() not in known:
            if max_calls is None:
                residue.append(text[start:end])
            continue
        if max_calls is not None:
            try:
                if not isinstance(args, dict):
                    continue
                json.dumps(args, allow_nan=False)
            except (ValueError, TypeError):
                continue
        calls.append({"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                      "function": {"name": known[str(name).lower()],
                                   "arguments": json.dumps(args, ensure_ascii=False, separators=(",", ":"))}})
    residue.append(text[cursor:])
    return "".join(residue).strip(), calls or None


# -- chat template -------------------------------------------------------------------------


def _arguments_close(body: str) -> str:
    """What a call's arguments lack to close: ``</function>``, a GLM value's ``</arg_value>``, or JSON's brackets."""

    if body.lower().startswith("<function="):
        return "" if body.lower().endswith("</function>") else "</function>"
    value = body.rfind("<arg_value>")
    if value >= 0 and body.rfind("</arg_value>") < value:
        return "</arg_value>"
    if not body or body[0] not in "{[":
        return ""
    closed = closed_json(body)
    return closed[len(body):] if closed is not None else ""


def close_call(text: str, tools: list[dict[str, Any]]) -> str:
    """What closes the call ``text`` ends inside, for a reply the model's end token ended before ``</tool_call>``:
    "" unless the closed call parses whole, one call to an offered tool and nothing but its markup (upstream
    TensorFold 1.0.9, #761). An opener whose closer follows is a whole call already."""

    if not tools:
        return ""
    low, at, start = text.lower(), 0, None
    if text.rfind(pythonic_calls.CALL_OPEN) > low.rfind(_CALL_OPEN):     # the reply ends in LFM2's list
        return pythonic_calls.close_call(text, tools)
    while (opener := low.find(_CALL_OPEN, at)) >= 0:
        closer = low.find(_CALL_CLOSE, opener + len(_CALL_OPEN))
        if closer < 0:
            start = opener
            break
        at = closer + len(_CALL_CLOSE)
    if start is None:
        return ""
    block = text[start + len(_CALL_OPEN):]
    tail = _arguments_close(block.strip())
    content, calls = parse_tool_calls(_CALL_OPEN + block + tail + _CALL_CLOSE, tools, max_calls=1)
    return tail + _CALL_CLOSE if calls and len(calls) == 1 and not content.strip() else ""
