"""The HTTP side of the LFM2 decision engine: ``/v1/systemone`` (llama-server's System One request and response) and
``/v1/decisions`` (the base App's, scored by the engine's ``score_labels``); chat, completions, Responses and
Messages requests are refused, since d1 writes no text."""

from __future__ import annotations

from typing import Any

from tensorfold.cuda.server import App
from tensorfold.server.decisions import DecisionError
from tensorfold.server.errors import RequestError
from tensorfold.vision.images import ImageInputError, load_images, split_images

_MEDIA = ("input_audio", "audio", "video_url", "video")
REFUSAL = "d1 is a decision model and writes no text: use POST /v1/systemone or /v1/decisions"


class DecisionApp(App):
    no_delay = True          # a decision's reply follows its headers within milliseconds (cuda.http)

    def _pictures(self, body: dict[str, Any]) -> tuple[Any, list]:
        """The state with its pictures taken out (as llama-server reads a System One state), and the pictures.

        ``files`` and ``images`` hold data URLs; a chat-style state holds ``image_url`` content parts. Everything else
        in the state stays as written, so the rendered prompt is llama-server's.
        """

        state = body["state"]
        for key in ("files", "images"):
            if body.get(key) is not None and not isinstance(body[key], list):
                raise DecisionError(f"{key} must be an array of data URLs")
        if body.get("videos"):
            raise DecisionError("videos are not supported")
        urls = list(body.get("files") or []) + list(body.get("images") or [])
        messages = state.get("messages") if isinstance(state, dict) else state
        if isinstance(messages, list) and messages:
            kept = []
            for message in messages:
                content = message.get("content") if isinstance(message, dict) else None
                if not isinstance(content, list):
                    kept.append(message)
                    continue
                parts = []
                for part in content:
                    kind = part.get("type") if isinstance(part, dict) else None
                    if kind == "image_url":
                        url = part.get("image_url")
                        urls.append(url.get("url") if isinstance(url, dict) else url)
                    elif kind in _MEDIA:
                        raise DecisionError(f"{kind} is not supported: a decision state takes images")
                    else:
                        parts.append(part)
                kept.append({**message, "content": parts})
            state = {**state, "messages": kept} if isinstance(state, dict) else kept
        if not urls:
            return state, []
        if len(urls) > self.image_limits.max_images:
            raise DecisionError(f"at most {self.image_limits.max_images} pictures a request")
        try:
            _, sources = split_images([{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url}} for url in urls]}], limits=self.image_limits)
            return state, [image.to_pil() for image in load_images(sources, limits=self.image_limits)]
        except ImageInputError as exc:
            raise DecisionError(str(exc)) from exc

    def systemone(self, body: dict[str, Any]) -> dict[str, Any]:
        if "state" not in body:
            raise DecisionError("a request needs a state (null when the pictures are the whole state)")
        if "questions" not in body:
            raise DecisionError("a request needs questions")
        state, images = self._pictures(body)
        return {"model": self.reply_model(body), **self.engine.decide(state, body["questions"], images)}

    def prepare(self, body, chat):
        raise RequestError(REFUSAL)
