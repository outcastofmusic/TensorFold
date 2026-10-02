"""continue_final_message (vLLM's field): the reply continues the last assistant turn instead of opening a new one."""

import json

import pytest

pytest.importorskip("tokenizers")
pytest.importorskip("jinja2")

from tensorfold.cuda import server
from tests.test_cuda_admission import http_server, post
from tests.test_cuda_stop_strings import TOK, Engine, ids, make_app, reply

# Qwen's shape: an assistant turn writes its think block closed, and the generation prompt leaves one open
QWEN_LIKE = ("{% for m in messages %}<|im_start|>{{ m.role }}\n"
             "{% if m.role == 'assistant' %}<think>\n{{ (m.reasoning_content or '')|trim }}\n</think>\n\n{% endif %}"
             "{{ m.content|trim }}<|im_end|>\n{% endfor %}"
             "{% if add_generation_prompt %}<|im_start|>assistant\n{% if enable_thinking %}<think>\n{% endif %}{% endif %}")
THINKING = {"chat_template_kwargs": {"enable_thinking": True}}


class RecordingEngine(Engine):
    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        self.prompts = [*getattr(self, "prompts", []), TOK.decode(prompt)]
        return super().generate(prompt, max_tokens, sampling, on_tokens, draft)


def qwen_app(tmp_path, answer):
    app = make_app(tmp_path, RecordingEngine(ids(answer)))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": QWEN_LIKE}))
    app.template = server.ChatTemplate(tmp_path)
    return app


def cut_thinking(content=""):
    return [{"role": "user", "content": "Is 91 prime?"},
            {"role": "assistant", "reasoning_content": "91 = 7 * 13.\n\n", "content": content}]


@pytest.mark.parametrize("stream", [False, True])
def test_the_prompt_ends_inside_the_last_assistant_turn(tmp_path, stream):
    app = qwen_app(tmp_path, "No.")
    with http_server(app) as port:
        reply(port, True, stream, messages=cut_thinking(), continue_final_message=True, **THINKING)
    assert app.engine.prompts == ["<|im_start|>user\nIs 91 prime?<|im_end|>\n"
                                  "<|im_start|>assistant\n<think>\n91 = 7 * 13.\n</think>\n\n"]


def test_a_partial_answer_is_continued_where_it_stops(tmp_path):
    app = qwen_app(tmp_path, " 7 * 13.")
    with http_server(app) as port:
        reply(port, True, False, messages=cut_thinking("No, it is"), continue_final_message=True, **THINKING)
    assert app.engine.prompts[0].endswith("</think>\n\nNo, it is")


@pytest.mark.parametrize("stream", [False, True])
def test_a_reply_after_a_closed_think_block_is_all_answer(tmp_path, stream):
    app = qwen_app(tmp_path, "No, 91 = 7 * 13.")
    with http_server(app) as port:
        content, reasoning, *_ = reply(port, True, stream, messages=cut_thinking(), continue_final_message=True,
                                       **THINKING)
    assert (content, reasoning) == ("No, 91 = 7 * 13.", "")


def test_without_the_field_a_new_assistant_turn_opens(tmp_path):
    app = qwen_app(tmp_path, "more thought</think>No.")
    with http_server(app) as port:
        content, reasoning, *_ = reply(port, True, False, messages=cut_thinking("Maybe."), **THINKING)
    assert app.engine.prompts[0].endswith("Maybe.<|im_end|>\n<|im_start|>assistant\n<think>\n")
    assert (content, reasoning) == ("No.", "more thought")


@pytest.mark.parametrize("messages", [[{"role": "user", "content": "x"}],
                                      [{"role": "user", "content": "x"}, {"role": "assistant", "content": ["y"]}]])
def test_continuing_needs_a_final_assistant_message_with_text(tmp_path, messages):
    app = qwen_app(tmp_path, "z")
    with http_server(app) as port:
        status, text = post(port, {"messages": messages, "continue_final_message": True}, True)
    assert status == 400 and "continue_final_message" in text
