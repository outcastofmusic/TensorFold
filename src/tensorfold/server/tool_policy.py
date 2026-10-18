import re
from typing import Any

from tensorfold.server.errors import RequestError
from tensorfold.server.pythonic_calls import CALL_CLOSE, CALL_OPEN

# Envelopes whose opener is not an XML-like ``<name:tool_call>`` tag, as (opener, closer, payload leads): LFM2's
# Pythonic list, Gemma 4's ``call:NAME{..}`` (the 26B writes a bare ``:NAME{..}``) and DeepSeek-V4's DSML invokes.
# With leads, the opener hides only when its payload (after whitespace) starts with one: another family's reply
# may name the opener in prose. LFM2's payload can be a bare call, which no lead tells apart from prose.
_FIXED = ((CALL_OPEN, CALL_CLOSE, ()),
          ("<|tool_call>", "<tool_call|>", ("call:", ":")),
          ("<｜DSML｜tool_calls>", "</｜DSML｜tool_calls>", ("<｜DSML｜invoke",)))


class _ToolTextFilter:
    """Hide tool envelopes while retaining only a partial tag between prose deltas."""

    def __init__(self):
        self.pending = ""
        self.hidden = None              # the closer the hidden envelope ends at
        self.opened = None              # (closer, leads) of an opener whose payload has not shown a lead yet
        self.held: list[str] = []       # that opener and the whitespace after it, shown if no lead follows

    def feed(self, text: str) -> str:
        self.pending += text
        out = []
        while self.pending:
            if self.hidden:
                close = self.hidden
                end = self.pending.lower().find(close)
                if end < 0:
                    self.pending = self.pending[-(len(close) - 1):]
                    break
                self.pending = self.pending[end + len(close):]
                self.hidden = None
                continue
            if self.opened:
                closer, leads = self.opened
                payload = self.pending.lstrip()
                self.held.append(self.pending[:len(self.pending) - len(payload)])
                self.pending = payload
                if any(lead.startswith(payload) for lead in leads):   # whitespace or a partial lead: wait for more
                    break
                self.opened = None
                if payload.startswith(leads):
                    self.hidden = closer
                else:                                                  # prose that names the opener
                    out.extend(self.held)
                self.held = []
                continue
            start = self.pending.find("<")
            if start < 0:
                out.append(self.pending)
                self.pending = ""
                break
            out.append(self.pending[:start])
            self.pending = self.pending[start:]
            envelope = next((e for e in _FIXED if e[0].startswith(self.pending[:len(e[0])])), None)
            if envelope is not None:
                opener, closer, leads = envelope
                if len(self.pending) < len(opener):            # a partial opener: wait for the rest
                    break
                self.pending = self.pending[len(opener):]
                if leads:
                    self.opened, self.held = (closer.lower(), leads), [opener]
                else:
                    self.hidden = closer.lower()
                continue
            end = self.pending.find(">")
            if end < 0:
                name = self.pending[1:]
                prefix, separator, suffix = name.partition(":")
                possible = (not name or re.fullmatch(r"[A-Za-z_][\w.-]*", name)
                            or (separator and re.fullmatch(r"[A-Za-z_][\w.-]*", prefix)
                                and "tool_call".startswith(suffix.lower())))
                if possible:
                    break
                out.append("<")
                self.pending = self.pending[1:]
                continue
            name = self.pending[1:end]
            if re.fullmatch(r"(?:[A-Za-z_][\w.-]*:)?tool_call", name, flags=re.I):
                self.hidden = f"</{name.lower()}>"
            else:
                out.append(self.pending[:end + 1])
            self.pending = self.pending[end + 1:]
        return "".join(out)

    def finish(self) -> str:
        tail = "" if self.hidden or self.opened else self.pending
        self.pending, self.opened, self.held = "", None, []
        return tail


class ToolCallPolicy:
    def __init__(self, body: dict[str, Any]):
        parallel = body.get("parallel_tool_calls")
        parallel = True if parallel is None else parallel
        if type(parallel) is not bool:
            raise RequestError("parallel_tool_calls must be a boolean")
        self.single = not parallel
        self._text = _ToolTextFilter() if self.single else None

    @property
    def max_calls(self) -> int | None:
        return 1 if self.single else None

    def limit(self, calls):
        return calls[:1] if self.single and calls else calls

    def delta(self, delta):
        if self.single:
            if isinstance(delta, str):
                return self._text.feed(delta)
            if isinstance(delta, dict):
                result = {k: v for k, v in delta.items() if k != "tool_calls"}
                if isinstance(result.get("content"), str):
                    content = self._text.feed(result.pop("content"))
                    if content:
                        result["content"] = content
                return result
        return delta

    def flush(self) -> str:
        return self._text.finish() if self.single else ""

    def content(self, text: str, *, finished: bool = True) -> str:
        if self.single:
            filtered = _ToolTextFilter()
            return filtered.feed(text) + (filtered.finish() if finished else "")
        return text

    def finish(self, reply, tools, parse):
        if not tools:
            return reply
        content, calls = parse(str(reply.get("content") or ""), tools, max_calls=self.max_calls)
        calls = calls or reply.get("tool_calls")
        if not calls:
            return {**reply, "content": self.content(content)} if self.single else reply
        result = {**reply, "content": self.content(content), "tool_calls": self.limit(calls), "finish_reason": "tool_calls"}
        if self.single:
            result["tool_calls_streamed"] = False
        return result
