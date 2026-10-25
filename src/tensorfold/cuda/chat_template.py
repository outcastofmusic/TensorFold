"""The CUDA server's chat template: the checkpoint's own Jinja template, rendered as Hugging Face does."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from tensorfold.cuda.request_text import ESC, RequestText
from tensorfold.server.messages import _normalize_tool_call_arguments, late_system_role, normalize_messages
from tensorfold.server.request_options import effort_levels


def _generation_tag():
    """Hugging Face's ``{% generation %}...{% endgeneration %}`` (it marks assistant tokens for training masks):
    rendered as its body, as ``apply_chat_template`` does when no assistant mask is asked for."""

    import jinja2.ext

    class Generation(jinja2.ext.Extension):
        tags = frozenset({"generation"})

        def parse(self, parser):
            lineno = next(parser.stream).lineno
            body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
            return jinja2.nodes.Scope(body, lineno=lineno)

    return Generation


class ChatTemplate:
    """The model's own Jinja chat template, rendered the way Hugging Face's apply_chat_template does."""

    def __init__(self, model_dir: Path):
        import jinja2
        import jinja2.ext
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        cfg = json.loads((model_dir / "tokenizer_config.json").read_text())
        source_path = model_dir / "chat_template.jinja"
        source = source_path.read_text() if source_path.exists() else cfg["chat_template"]

        def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
            return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

        def raise_exception(message):
            raise jinja2.exceptions.TemplateError(message)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                            extensions=[jinja2.ext.loopcontrols, _generation_tag()])
        env.filters["tojson"] = tojson
        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
        self.template = env.from_string(source)
        self.efforts = effort_levels(source)          # the reasoning efforts it names (the Mac reads the same ones)
        self.thinking_switch = "enable_thinking" in source or "thinking_mode" in source   # thinking can go off
        self.specials = {k: (v.get("content") if isinstance(v, dict) else v)
                         for k, v in cfg.items() if k in ("bos_token", "eos_token", "pad_token", "unk_token")}
        self.late_system = late_system_role(
            lambda messages: self.template.render(**self.specials, messages=messages, add_generation_prompt=False))

    def render(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None,
               enable_thinking: bool, extra: dict[str, Any] | None = None, allow_images: bool = False,
               request: RequestText | None = None) -> str:
        """The rendered prompt. With ``request``, a control-token spelling in the request's strings renders as an
        escape (``cuda.request_text``), and only the template's own spellings stay literal."""

        messages = _normalize_tool_call_arguments(normalize_messages(messages, late_system=self.late_system,
                                                                     allow_images=allow_images))
        kwargs = dict(self.specials, messages=messages, tools=tools or None, add_generation_prompt=True,
                      enable_thinking=enable_thinking)
        kwargs.update(extra or {})
        if request is None or not request.holds([messages, tools, extra]):
            return self.template.render(**kwargs)
        import jinja2

        marked = request.seal(self.template.render(**dict(kwargs, messages=request.mark(messages),
                                                          tools=request.mark(tools) or None,
                                                          **request.mark(extra or {}))))
        if ESC not in marked:
            return marked               # the request spelt no added token, and no boundary stayed
        try:
            same = request.unmark(marked) == self.template.render(**kwargs)
        except jinja2.exceptions.TemplateError:
            same = False
        if not same:
            # the template tests the request's text for a spelling (Qwen's ``<tool_response>`` user turns): it saw
            # the escape instead, and the escaped rendering stands
            print("[tensorfold] the chat template branched on a control-token spelling in the request; "
                  "it rendered the spelling as plain text", flush=True)
        return marked
