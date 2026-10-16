"""GLM request routes validate context before streaming and render an empty think block without a reasoning-effort line when thinking is off."""

from __future__ import annotations

from typing import Any, Callable

from tensorfold.cuda import warm
from tensorfold.cuda.server import App, PreparedRequest, RequestError
from tensorfold.families.glm5_next.prompts import clear_thinking, thinking_off
from tensorfold.server.errors import CONTEXT_LIMIT
from tensorfold.server.text import is_title_request


class ThinkingOffTemplate:
    """The checkpoint template as GLM-5.3's thinking-off template renders it (``prompts.thinking_off``), with earlier
    turns' reasoning kept unless the request's ``chat_template_kwargs.clear_thinking`` says otherwise
    (``prompts.clear_thinking``)."""

    def __init__(self, inner, clear: bool | None = None) -> None:
        self.inner = inner
        self.efforts = getattr(inner, "efforts", frozenset())
        self.thinking_switch = getattr(inner, "thinking_switch", False)
        self.clear = clear_thinking() if clear is None else clear

    def render(self, messages, *, tools, enable_thinking, extra=None) -> str:
        extra = {"clear_thinking": self.clear, **(extra or {})}       # a request's own value wins
        text = self.inner.render(messages, tools=tools, enable_thinking=enable_thinking, extra=extra)
        return text if enable_thinking else thinking_off(text)


class GlmApp(App):
    reads_ignore_eos = True             # ``run`` hands it to the engine's request
    warmer: warm.Warmer | None = None   # the next turn's background prefill (``warm``), on one rank

    def __init__(self, engine, model_dir, served: str, **kwargs: Any) -> None:
        super().__init__(engine, model_dir, served, **kwargs)
        self.template = ThinkingOffTemplate(self.template)
        self.warmer, self.warming = None, warm.enabled() and hasattr(engine, "warm")
        keeping = getattr(engine, "disk", None) is not None          # TENSORFOLD_GLM_STATES: states also go to disk
        if (self.warming or keeping) and getattr(engine, "comm", None) is None:
            turns = self._turns()
            self.warmer = warm.Warmer(engine.warm, lambda: turns.take(True), turns.give, turns.wanted,
                                      after=engine.persist if keeping else None)

    def check(self, body: dict[str, Any], *, prepared: PreparedRequest | None = None) -> str | None:
        """Validate the rendered prompt plus max_tokens against the engine context limit before streaming."""

        problem = self._check_fields(body)
        limit = getattr(self.engine, "limit", None)
        if problem or limit is None:
            return problem or super().check(body, prepared=prepared)
        if prepared is None:
            try:
                prepared = self._prepare(body, "messages" in body)
            except RequestError as exc:
                return str(exc)
        prompt = len(prepared.prompt)
        asked = body.get("max_tokens") or body.get("max_completion_tokens")
        need = prompt + (int(asked) if asked else 1)
        if need <= limit:
            return super().check(body, prepared=prepared)
        detail = f"{prompt} prompt tokens plus max_tokens {int(asked)}" if asked else f"a {prompt}-token prompt"
        # OpenAI's wording, so prepare refuses it as context_length_exceeded (clients compact on it)
        return (f"{CONTEXT_LIMIT} {limit} tokens: this request needs a {need}-token context ({detail}), which exceeds "
                f"the context window this server was started for; shorten the prompt or reply"
                f"{self._restart(need, ' both ranks')}")

    def run(self, body: dict[str, Any], chat: bool, emit: Callable[[dict[str, Any]], bool], *,
            prepared: PreparedRequest | None = None, cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
        model = str(body.get("model") or "")
        self.engine.request.policy = body.get("tf_policy") or (model.split("@", 1)[1] if "@" in model else None)
        self.engine.request.stop_eos = not bool(body.get("ignore_eos", False))
        try:
            result = super().run(body, chat, emit, prepared=prepared, cancelled=cancelled)
        finally:
            if self.warmer is not None:
                self.warmer.finished()
        if self.warmer is not None and chat and result["finish"] in ("stop", "tool_calls") \
                and body.get("priority") != "background" and not is_title_request(body["messages"], body.get("tools")):
            self.warmer.queue(lambda: self._next_turn(body, result) if self.warming else None)
        return result

    def _next_turn(self, body: dict[str, Any], result: dict[str, Any]) -> tuple[list[int], int] | None:
        """(the next turn's shared prompt, the length of this turn's kept state), or None when it adds nothing."""

        def render(messages: list[dict[str, Any]]) -> list[int]:
            return self._prepare({**body, "messages": messages}, True).prompt

        try:
            prompt = warm.next_turn_prefix(render, list(body["messages"]), warm.reply_message(result))
        except RequestError:                          # the template refuses the conversation with the reply in it
            return None
        born = result["prompt_tokens"] - 1            # where the turn's own state was kept (``_run_once``)
        return (prompt, born) if len(prompt) > result["prompt_tokens"] else None
