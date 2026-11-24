"""The Mac and CUDA servers render the same prompt ids for the same request from each checkpoint's own chat template
and tokenizer (Qwen3.8-27B, Flash Next, GLM-5.3, Nemotron, and the served FP4-Local, Bolt and LFM2.5 checkpoints):
thinking on and off, every effort or none, a server default effort or none, tools and a tool-call history. Checkpoints
come from the environment, the Hugging Face cache or a folder under $MODELS; a missing one is skipped."""

import json
import os
import threading
from pathlib import Path

import pytest

pytest.importorskip("jinja2")
pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

from tensorfold import hub
from tensorfold.cuda import server
from tensorfold.server.request_options import RequestOptions, thinking_fields
from tensorfold.server.text import render_prompt_ids, template_late_system

MODELS = Path(os.environ.get("MODELS", str(Path.home() / "models")))
CHECKPOINTS = {                       # name: (environment variable, Hugging Face repo, folder under $MODELS)
    "qwen27": ("TENSORFOLD_MLX_MODEL", "TensorFold/Qwen3.8-27B-MLX-4bit", None),
    "flashnext": ("TF_FLASHNEXT_MODEL", "TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP", None),
    "glm": ("TF_GLM5_MODEL", "TensorFold/GLM-5.3-Flash-MLX-4bit-MTP", None),
    "nemotron": ("TF_NEMOTRON_MODEL", "TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit", None),
    # the served checkpoints run.sh starts as qwen_fp4, bolt and lfm2 (prepare.sh's repos)
    "flashnext-fp4": ("TF_FLASHNEXT_FP4_MODEL", "nvidia/Qwen3.8-Flash-Next-FP4-Local", "Qwen3.8-Flash-Next-FP4-Local"),
    "bolt": ("TF_BOLT_MODEL", "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB", "Nemotron-3.5-Bolt-74B-A9B-24GB"),
    "lfm2": ("TF_LFM2_MODEL", "LiquidAI/LFM2.5-24B-A2B-FP8", "LFM2.5-24B-A2B-FP8"),
}
WEATHER = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather",
                                             "parameters": {"type": "object", "required": ["city"],
                                                            "properties": {"city": {"type": "string"}}}}}]
CONVERSATIONS = {
    "plain": ([{"role": "system", "content": "You are terse."}, {"role": "user", "content": "What is 2+2?"}], None),
    "tools": ([{"role": "user", "content": "Weather in Oslo?"},
               {"role": "assistant", "content": "", "tool_calls": [
                   {"id": "call_1", "type": "function",
                    "function": {"name": "get_weather", "arguments": json.dumps({"city": "Oslo"})}}]},
               {"role": "tool", "tool_call_id": "call_1", "content": "{\"celsius\": 12}"},
               {"role": "user", "content": "And Bergen?"}], WEATHER),
    "late system": ([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"},
                     {"role": "system", "content": "Be brief."}, {"role": "user", "content": "Bye"}], None),
    "earlier reasoning": ([{"role": "user", "content": "Weather in Oslo?"},
                           {"role": "assistant", "content": "", "reasoning_content": "Ask the tool.", "tool_calls": [
                               {"id": "call_1", "type": "function",
                                "function": {"name": "get_weather", "arguments": json.dumps({"city": "Oslo"})}}]},
                           {"role": "tool", "tool_call_id": "call_1", "content": "{\"celsius\": 12}"},
                           {"role": "assistant", "content": "12 C.", "reasoning_content": "It is 12."},
                           {"role": "user", "content": "And Bergen?"}], WEATHER),
}
REQUESTS = [{}, {"reasoning_effort": "none"}, {"reasoning_effort": "minimal"}, {"reasoning_effort": "low"},
            {"reasoning_effort": "medium"}, {"reasoning_effort": "high"}, {"reasoning_effort": "xhigh"},
            {"chat_template_kwargs": {"enable_thinking": False}},
            {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "high"}},
            {"chat_template_kwargs": {"thinking": False}}, {"chat_template_kwargs": {"thinking": {"type": "enabled"}}}]
# LFM2.5's template opens <think> whatever the switch. With thinking off the Mac path (render_prompt_ids) closes it
# with </think>, while the CUDA Lfm2MoeApp turns thinking back on (always_thinking) and leaves it open
THINKING_OFF = [{"reasoning_effort": "none"}, {"chat_template_kwargs": {"enable_thinking": False}},
                {"chat_template_kwargs": {"thinking": False}}]
LFM2_THINKING_OFF = pytest.mark.xfail(strict=True, reason=(
    "mismatch: LFM2.5 with thinking off renders <think></think> (124902) on the Mac path and <think> alone on CUDA"))


def requests_for(name):
    return [r for r in REQUESTS if not (name == "lfm2" and r in THINKING_OFF)]


def _folder(name):
    variable, repo, local = CHECKPOINTS[name]
    found = os.environ.get(variable) or hub.cached(repo) or (MODELS / local if local else None)
    folder = Path(found) if found else None
    if folder is None or not (folder / "tokenizer.json").is_file():
        pytest.skip(f"needs the {name} tokenizer and chat template ({variable} or {repo})")
    return folder


class ReturnIds:
    """mlx_lm's tokenizer wrapper where mlx_lm is missing: its chat call returns the ids, not a dict."""

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def apply_chat_template(self, *args, **kwargs):
        return self.inner.apply_chat_template(*args, return_dict=False, **kwargs)


def mac_tokenizer(name, folder):
    """The tokenizer the Mac family loads: mlx_lm's (GLM's wrapped to render its thinking-off prompt)."""

    try:
        from mlx_lm.utils import load_tokenizer
    except ImportError:
        tokenizer = ReturnIds(transformers.AutoTokenizer.from_pretrained(str(folder), local_files_only=True))
    else:
        tokenizer = load_tokenizer(folder)
    if name == "glm":
        from tensorfold.families.glm5_next.prompts import GlmTokenizer

        tokenizer = GlmTokenizer(tokenizer)
    return tokenizer


class Mac(RequestOptions):
    """The Mac server's request path to prompt ids: its HTTP fields, then ``ChatApp.render``'s call."""

    def __init__(self, name, folder, thinking, effort):
        self.tokenizer = mac_tokenizer(name, folder)
        self.late_system = template_late_system(self.tokenizer)
        self.enable_thinking, self.reasoning_effort = thinking, effort

    def prompt(self, body, messages, tools):
        fields = thinking_fields(body, self.effort_levels)
        requested = fields.get("enable_thinking")
        thinking = self.enable_thinking if requested is None else bool(requested)
        return render_prompt_ids(self.tokenizer, messages, tools=tools, enable_thinking=thinking,
                                 reasoning_effort=fields.get("reasoning_effort", self.reasoning_effort),
                                 late_system=self.late_system)


def cuda_app(name, folder, thinking, effort):
    from tokenizers import Tokenizer

    if name == "hy4":                  # its own app: the text view and the effort mapping (``hy4_app``)
        return hy4_app(folder, thinking)

    cls = server.App
    if name == "glm":
        from tensorfold.families.glm5_next.cuda.app import GlmApp, ThinkingOffTemplate
        cls = GlmApp
    if name == "lfm2":                 # lfm2_moe's CUDA_APP: thinking always on
        from tensorfold.families.lfm2_moe.cuda.app import Lfm2MoeApp
        cls = Lfm2MoeApp
    app = cls.__new__(cls)
    app.engine, app.served, app.model_dir = None, "parity", folder
    app.tok = Tokenizer.from_file(str(folder / "tokenizer.json"))
    app.template = server.ChatTemplate(folder)
    if name == "glm":
        app.template = ThinkingOffTemplate(app.template)
    app.default_thinking, app.reasoning_effort, app.thinking_budget = thinking, effort, 0
    app.sampling, app.max_tokens = {"temperature": 0.0}, 16
    app.native_context_window = app.context_window = 0
    app.lock = threading.Lock()
    return app


@pytest.mark.parametrize("default_effort", [None, "low"])
@pytest.mark.parametrize("name", sorted(CHECKPOINTS))
def test_the_mac_and_cuda_render_the_same_prompt(name, default_effort):
    folder = _folder(name)
    mac, cuda = Mac(name, folder, True, default_effort), cuda_app(name, folder, True, default_effort)
    for label, (messages, tools) in CONVERSATIONS.items():
        for request in requests_for(name):
            body = {"messages": messages, **({"tools": tools} if tools else {}), **request}
            mac_ids = mac.prompt(body, messages, tools)
            cuda_ids = cuda._prepare(body, True).prompt
            assert mac_ids == cuda_ids, (label, request, mac.tokenizer.decode(mac_ids)[-400:],
                                         mac.tokenizer.decode(cuda_ids)[-400:])


@pytest.mark.parametrize("name, request_fields, words", [
    ("qwen27", {}, "Reasoning effort is set to xhigh"),                 # the template's own default, as vLLM's
    ("qwen27", {"reasoning_effort": "high"}, "Reasoning effort is set to xhigh"),
    ("qwen27", {"reasoning_effort": "low"}, "Reasoning effort is set to low"),
    ("qwen27", {"reasoning_effort": "medium"}, "<|im_start|>system\nYou are terse."),     # medium adds no line
    ("flashnext", {}, "Reasoning effort is set to xhigh"),
    ("glm", {}, "Reasoning Effort: Max"),
    ("glm", {"reasoning_effort": "high"}, "Reasoning Effort: High"),   # GLM-5.3 names high: it stays high
    ("glm", {"reasoning_effort": "medium"}, "Reasoning Effort: High"), # medium is the nearest named level
    ("glm", {"reasoning_effort": "xhigh"}, "Reasoning Effort: Max"),   # xhigh stays xhigh; the template renders Max
    ("glm", {"reasoning_effort": "minimal"}, "Reasoning Effort: Low"),
])
def test_the_effort_each_template_writes(name, request_fields, words):
    folder = _folder(name)
    messages, _ = CONVERSATIONS["plain"]
    mac, cuda = Mac(name, folder, True, None), cuda_app(name, folder, True, None)
    body = {"messages": messages, **request_fields}
    ids = mac.prompt(body, messages, None)
    assert ids == cuda._prepare(body, True).prompt and words in mac.tokenizer.decode(ids)


@pytest.mark.parametrize("name", sorted(CHECKPOINTS))
def test_tokenize_gives_the_chat_route_s_prompt_on_both_servers(name):
    """vLLM's /tokenize: the ids each server's chat route runs, and the same on both; detokenize gives the text back."""

    from tensorfold.server import token_routes

    folder = _folder(name)
    mac, cuda = Mac(name, folder, True, None), cuda_app(name, folder, True, None)
    mac.tokenizer_lock, mac.context_window = threading.Lock(), 0
    for label, (messages, tools) in CONVERSATIONS.items():
        for request in requests_for(name):
            body = {"messages": messages, **({"tools": tools} if tools else {}), **request}
            want = cuda._prepare(body, True).prompt
            assert cuda.tokenize(body)["tokens"] == want == token_routes.tokenize(mac, body)["tokens"], (label, request)
            history = {**body, "add_generation_prompt": False}
            cut = cuda.tokenize(history)["tokens"]
            assert cut == token_routes.tokenize(mac, history)["tokens"] and len(cut) < len(want), (label, request)
    text = "Grüße, 世界! <think>"
    for special in (False, True):
        body = {"prompt": text, "add_special_tokens": special, "return_token_strs": True}
        mine, theirs = cuda.tokenize(body), token_routes.tokenize(mac, body)
        assert mine["tokens"] == theirs["tokens"] and mine["token_strs"] == theirs["token_strs"]
        assert cuda.detokenize({"tokens": mine["tokens"]}) == token_routes.detokenize(mac, {"tokens": mine["tokens"]})
    assert cuda.detokenize({"tokens": cuda.tokenize({"prompt": text, "add_special_tokens": False})["tokens"]}) == {
        "prompt": text}


@LFM2_THINKING_OFF
@pytest.mark.parametrize("request_fields", THINKING_OFF)
def test_lfm2_with_thinking_off_renders_the_same_on_both_servers(request_fields):
    folder = _folder("lfm2")
    mac, cuda = Mac("lfm2", folder, True, None), cuda_app("lfm2", folder, True, None)
    messages, _ = CONVERSATIONS["plain"]
    body = {"messages": messages, **request_fields}
    assert mac.prompt(body, messages, None) == cuda._prepare(body, True).prompt


# -- Hy4-flash ------------------------------------------------------------------------------------------------------
# Hy4's template takes reasoning_effort high or no_think only, so it has no Mac row above: its CUDA app (Hy4App, over
# the text view that spells its control tokens as GLM's parsers read them) is checked against transformers directly

HY4 = ("TF_HY4_MODEL", "Hy4-flash-NVFP4")
HY4_QUOTED = {                       # request text that quotes the view's control spellings stays plain text
    "quoted markup": ([{"role": "system", "content": "Calls look like <tool_call>NAME<arg_key>k</arg_key>"
                                                     "<arg_value>v</arg_value></tool_call>."},
                       {"role": "user", "content": "What do <think> and </think> mark? And <tool_calls>?"}], None),
    "quoted tool result": ([{"role": "user", "content": "Weather in Oslo?"},
                            {"role": "assistant", "content": "", "tool_calls": [
                                {"id": "call_1", "type": "function",
                                 "function": {"name": "get_weather", "arguments": json.dumps({"city": "Oslo"})}}]},
                            {"role": "tool", "tool_call_id": "call_1",
                             "content": "<tool_response>12</tool_response> then <tool_call>get_weather"
                                        "<arg_key>city</arg_key><arg_value>Bergen</arg_value></tool_call>"}], WEATHER),
}


def hy4_folder():
    variable, local = HY4
    folder = Path(os.environ.get(variable) or MODELS / local)
    if not (folder / "tokenizer.json").is_file():
        pytest.skip(f"needs the Hy4-flash tokenizer and chat template ({variable} or $MODELS/{local})")
    return folder


def hy4_reference(messages):
    """The messages as vLLM hands them to transformers' template: tool-call arguments parsed into objects."""

    def call(c):
        return {**c, "function": {**c["function"], "arguments": json.loads(c["function"]["arguments"])}}

    return [{**m, "tool_calls": [call(c) for c in m["tool_calls"]]} if m.get("tool_calls") else m for m in messages]


def hy4_app(folder, thinking=True):
    """The Hy4 CUDA app as the server builds it, over an engine that runs nothing."""

    from types import SimpleNamespace

    from tensorfold.families.hy4_flash.cuda.app import Hy4App

    return Hy4App(SimpleNamespace(eos=(120025,), tp=1), folder, "parity", default_thinking=thinking)


@pytest.mark.parametrize("thinking", [True, False])
def test_hy4_renders_transformers_prompt(thinking):
    """transformers' apply_chat_template on the checkpoint's own tokenizer and template gives the app's prompt ids."""

    folder = hy4_folder()
    hf = transformers.AutoTokenizer.from_pretrained(str(folder), local_files_only=True)
    app = hy4_app(folder)
    for label, (messages, tools) in {**CONVERSATIONS, **HY4_QUOTED}.items():
        want = hf.apply_chat_template(hy4_reference(messages), tools=tools, add_generation_prompt=True, tokenize=True,
                                      return_dict=False, reasoning_effort="high" if thinking else "no_think")
        body = {"messages": messages, **({"tools": tools} if tools else {}),
                "chat_template_kwargs": {"enable_thinking": thinking}}
        got = app._prepare(body, True).prompt
        assert got == list(want), (label, hf.decode(want)[-300:], hf.decode(got)[-300:])


def test_hy4_raw_text_reads_the_checkpoint_s_spellings():
    """A raw prompt, /tokenize and /detokenize read the checkpoint's own spellings (``<think:opensource>``) as
    transformers and vLLM do, so `<think>` there is plain text; a reply decodes in the view's (GLM's parsers')."""

    folder = hy4_folder()
    hf = transformers.AutoTokenizer.from_pretrained(str(folder), local_files_only=True)
    app = hy4_app(folder)
    messages, tools = CONVERSATIONS["tools"]
    text = hf.apply_chat_template(hy4_reference(messages), tools=tools, add_generation_prompt=True, tokenize=False,
                                  reasoning_effort="high") + "<think> is plain here"
    want = hf.encode(text, add_special_tokens=False)
    assert app._prepare({"prompt": text}, False).prompt == want
    assert app.tokenize({"prompt": text, "add_special_tokens": False})["tokens"] == want
    assert app.detokenize({"tokens": want}) == {"prompt": text}
    assert app.tokenize({"prompt": "<think>", "add_special_tokens": False, "return_token_strs": True})[
        "token_strs"] == hf.convert_ids_to_tokens(hf.encode("<think>", add_special_tokens=False))
    think = hf.convert_tokens_to_ids("<think:opensource>")
    assert app.tok.decode([think], skip_special_tokens=False) == "<think>"        # the reply side
