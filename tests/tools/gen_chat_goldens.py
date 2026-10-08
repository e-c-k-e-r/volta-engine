#!/usr/bin/env python3
# Regenerates tests/data/chat_template_cases.json: conversations rendered by the reference Jinja2 (transformers'
# environment: sandboxed, trim_blocks, lstrip_blocks, loop controls, json.dumps-based tojson) with the two checkpoint
# templates stored next to it. tests/2_parity/test_jinja.cpp renders the same inputs with the engine's interpreter.
#   python3 -I tests/tools/gen_chat_goldens.py
import json, os, sys
import jinja2
import jinja2.ext
import jinja2.exceptions
from jinja2.sandbox import ImmutableSandboxedEnvironment

here = os.path.dirname(os.path.abspath(__file__))
data = os.path.join(here, "..", "data")

def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

def raise_exception(msg):
    raise jinja2.exceptions.TemplateError(msg)

env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
env.filters["tojson"] = tojson
env.globals["raise_exception"] = raise_exception

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "search", "description": "Web search — returns top results",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "default": 5}}}}},
]

U = lambda t: {"role": "user", "content": t}
A = lambda t, **k: dict({"role": "assistant", "content": t}, **k)
S = lambda t: {"role": "system", "content": t}
T = lambda t, **k: dict({"role": "tool", "content": t}, **k)
TC = lambda name, args, i=None: dict({"type": "function", "function": {"name": name, "arguments": args}}, **({"id": i} if i else {}))

CONVS = {
    "user": ([U("What is the capital of France?")], None),
    "system_user": ([S("You are terse."), U("Hi")], None),
    "multiturn": ([S("Be helpful."), U("Hi"), A("Hello!", reasoning_content="greet back"), U("Capital of Japan?")], None),
    "unicode": ([U("日本語の質問 — café \U0001F600 \"quoted\" back\\slash")], None),
    "parts": ([{"role": "user", "content": [{"type": "text", "text": "part one "}, {"type": "text", "text": "part two"}]}], None),
    "tools": ([S("Use tools."), U("Weather in Paris?")], TOOLS),
    "tools_nosys": ([U("Weather in Paris?")], TOOLS),
    "toolcall": ([U("Weather in Paris?"),
        A("Let me check.", reasoning_content="need the tool", tool_calls=[TC("get_weather", {"city": "Paris", "unit": "celsius"}, "call_1")]),
        T("18 C, cloudy", tool_call_id="call_1"),
        ], TOOLS),
    "toolcall_final": ([U("Weather in Paris?"),
        A("", reasoning_content="need the tool", tool_calls=[TC("get_weather", {"city": "Paris"}, "call_1")]),
        T("18 C, cloudy", tool_call_id="call_1"),
        A("It is 18 C and cloudy.", reasoning_content="done"),
        U("And Rome?")], TOOLS),
    "parallel": ([U("Weather in Paris and Rome?"),
        A("", reasoning_content="two calls", tool_calls=[TC("get_weather", {"city": "Paris"}, "c1"), TC("get_weather", {"city": "Rome"}, "c2")]),
        T("18 C", tool_call_id="c1"), T("24 C", tool_call_id="c2")], TOOLS),
    "args_types": ([U("search"),
        A("ok", tool_calls=[TC("search", {"query": "a \"b\"\nc", "limit": 3, "opts": {"x": [1, 2.5, None, True]}}, "c1")]),
        T("none", tool_call_id="c1")], TOOLS),
}

OPTS = [
    {},
    {"enable_thinking": False},
    {"reasoning_effort": "low"},
    {"reasoning_effort": "medium"},
    {"reasoning_effort": "high"},
    {"preserve_thinking": False},
    {"clear_thinking": False},
]

out = {"templates": {}}
for name in ("qwen3_5", "glm5_next"):
    path = os.path.join(data, name + "_chat_template.jinja")
    tpl = env.from_string(open(path, encoding="utf-8").read())
    cases = []
    for cname, (messages, tools) in CONVS.items():
        for opts in OPTS:
            for gen in (True, False):
                kw = dict(opts, messages=messages, add_generation_prompt=gen)
                if tools:
                    kw["tools"] = tools
                case = {"name": cname, "vars": kw}
                try:
                    case["expected"] = tpl.render(**kw)
                except Exception as e:
                    case["error"] = str(e)
                cases.append(case)
    out["templates"][name] = cases
with open(os.path.join(data, "chat_template_cases.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)
print({k: len(v) for k, v in out["templates"].items()}, "cases;",
      sum(1 for v in out["templates"].values() for c in v if "error" in c), "expected errors")
