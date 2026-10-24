"""A parameter declared by a union (anyOf, oneOf, allOf, a type list) or a local ``$ref`` decodes as JSON when its
schema takes a non-string kind, as a plain ``"type": "object"`` one does; a quoted string, text that is not JSON and
a reference that points nowhere or loops keep the model's text (upstream f8064d99, dc2a724a, 539c0de2, b0a182d9)."""

from __future__ import annotations

import json

import pytest

from tensorfold.tool_parameters import decode_parameter, parameter_schemas

DEFS = {"Point": {"type": "object"}, "Ids": {"$ref": "#/$defs/List"}, "List": {"type": "array"},
        "Loop": {"$ref": "#/$defs/Loop"}, "Named": {"description": "no kinds"}, "a/b": {"type": "integer"}}


def schema_of(prop: dict) -> dict:
    tool = {"type": "function", "function": {"name": "f", "parameters": {
        "type": "object", "properties": {"p": prop}, "$defs": DEFS}}}
    return parameter_schemas([tool])["f"]["p"]


@pytest.mark.parametrize("prop, text, want", [
    ({"anyOf": [{"type": "object"}, {"type": "null"}]}, '{"x": 1}', {"x": 1}),
    ({"oneOf": [{"type": "array"}, {"type": "string"}]}, "[1, 2]", [1, 2]),
    ({"type": ["integer", "null"]}, "7", 7),
    ({"type": ["integer", "null"]}, "null", None),
    ({"allOf": [{"type": "number"}, {"type": "integer"}]}, "3", 3),
    ({"allOf": [{"type": "number"}, {"type": "integer"}]}, "3.5", "3.5"),      # not an integer: the text
    ({"$ref": "#/$defs/Point"}, '{"x": 1}', {"x": 1}),
    ({"$ref": "#/$defs/Ids"}, "[3]", [3]),                                     # a reference to a reference
    ({"$ref": "#/$defs/a~1b"}, "4", 4),                                        # a JSON pointer's escape
    ({"$ref": "#/$defs/Loop"}, "[3]", "[3]"),                                   # a cycle names no kinds
    ({"$ref": "#/$defs/Missing"}, "[3]", "[3]"),
    ({"type": "array", "allOf": [{"$ref": "#/$defs/Named"}]}, "[3]", [3]),      # no kinds constrain nothing
    ({"enum": [1, 2, 3]}, "2", 2),
    ({"anyOf": [{"type": "string"}, {"type": "integer"}]}, '"5"', '"5"'),       # a quoted string keeps its text
    ({"anyOf": [{"type": "string"}, {"type": "object"}]}, "plain words", "plain words"),
    ({"anyOf": [{"type": "string"}, {"enum": ["a"]}]}, "[1]", "[1]"),           # every branch a string
    ({"type": "object"}, '{"x": 1}', {"x": 1}),
    ({"type": "string"}, "[1]", "[1]"),
])
def test_a_union_or_reference_decodes_as_its_kinds(prop, text, want):
    assert decode_parameter(text, schema_of(prop)) == want


def test_the_cuda_parser_returns_a_union_parameter_as_json():
    from tensorfold.cuda.reply_text import parse_tool_calls

    tools = [{"type": "function", "function": {"name": "edit", "parameters": {"type": "object", "properties": {
        "changes": {"anyOf": [{"type": "array", "items": {"type": "object"}}, {"type": "null"}]}}}}}]
    text = '<tool_call>\n<function=edit>\n<parameter=changes>\n[{"line": 3}]\n</parameter>\n</function>\n</tool_call>'
    _, calls = parse_tool_calls(text, tools)
    assert json.loads(calls[0]["function"]["arguments"]) == {"changes": [{"line": 3}]}
