#!/usr/bin/env python3
# Black-box checks of the chat APIs against a running volta_server (OpenAI /v1/chat/completions, and /v1/messages when
# the server routes it). Usage: python3 -I api_checks.py http://127.0.0.1:8000
import json, sys, urllib.request, urllib.error, socket, time

BASE = sys.argv[1].rstrip("/")
failures = 0

def check(cond, what, extra=""):
    global failures
    print(("[ok]   " if cond else "[FAIL] ") + what + ((" :: " + str(extra)[:300]) if (extra and not cond) else ""))
    failures += 0 if cond else 1

def post(path, body, stream=False, timeout=120):
    if isinstance(body, dict) and "temperature" not in body:
        body = dict(body, temperature=0)  # deterministic checks; the server default follows the checkpoint
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    if not stream:
        return r.status, r.read().decode()
    return r.status, r

def sse(resp):
    events = []
    for raw in resp:
        line = raw.decode().rstrip("\n")
        if line.startswith("data: "):
            payload = line[6:]
            events.append(payload if payload == "[DONE]" else json.loads(payload))
    return events

TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}}]

# models, health
code, body = urllib.request.urlopen(BASE + "/v1/models").status, json.loads(urllib.request.urlopen(BASE + "/v1/models").read())
check(code == 200 and body["data"][0]["object"] == "model", "GET /v1/models")
IS_GLM = "GLM" in body["data"][0]["id"]

# plain completion, thinking off by default
code, body = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Capital of France? One word."}], "max_tokens": 16})
j = json.loads(body)
check(code == 200 and "Paris" in j["choices"][0]["message"]["content"] and j["choices"][0]["finish_reason"] == "stop", "chat completion", body)
check("reasoning_content" not in j["choices"][0]["message"] and j["usage"]["completion_tokens"] > 0, "no reasoning by default, usage present")

# reasoning on
code, body = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "What is 2+2?"}], "max_tokens": 400, "reasoning_effort": "low"})
j = json.loads(body)
m = j["choices"][0]["message"]
# GLM may skip its reasoning when the effort is low: only the answer is required there
check(code == 200 and (m.get("reasoning_content") or IS_GLM) and "4" in (m["content"] or ""), "reasoning_content separated from content", body)

# length finish
code, body = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Count to 100."}], "max_tokens": 5})
check(json.loads(body)["choices"][0]["finish_reason"] == "length", "finish_reason length", body)

# stop string
code, body = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Count 1 to 9, comma separated, no spaces."}], "max_tokens": 40, "stop": ["5"]})
c = json.loads(body)["choices"][0]["message"]["content"]
check("5" not in c and c.startswith("1,2,3,4"), "stop string", c)

# streaming
code, resp = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Count 1 to 5, comma separated."}], "max_tokens": 40, "stream": True, "stream_options": {"include_usage": True}}, stream=True)
ev = sse(resp)
text = "".join(e["choices"][0]["delta"].get("content", "") for e in ev if e != "[DONE]" and e["choices"])
check(ev[-1] == "[DONE]" and "1, 2, 3, 4, 5" in text, "stream content", text)
check(ev[0]["choices"][0]["delta"].get("role") == "assistant", "stream starts with the role chunk")
check(any(e != "[DONE]" and e["choices"] and e["choices"][0]["finish_reason"] for e in ev), "stream finish_reason chunk")
check(any(e != "[DONE]" and "usage" in e and e["usage"]["total_tokens"] > 0 for e in ev), "stream usage chunk")

# tool call, then the result back
code, body = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "What is the weather in Paris for 3 days? Use the tool."}], "tools": TOOLS, "max_tokens": 300})
j = json.loads(body)
ch = j["choices"][0]
calls = ch["message"].get("tool_calls") or []
check(ch["finish_reason"] == "tool_calls" and len(calls) == 1 and calls[0]["function"]["name"] == "get_weather", "tool call", body)
args = json.loads(calls[0]["function"]["arguments"]) if calls else {}
check(args.get("city") == "Paris" and args.get("days") == 3, "tool arguments typed per schema", args)
code, body = post("/v1/chat/completions", {"messages": [
    {"role": "user", "content": "Weather in Paris?"},
    {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "18C and cloudy"}], "tools": TOOLS, "max_tokens": 80})
check("18" in json.loads(body)["choices"][0]["message"]["content"], "tool result is used", body)

code, resp = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}], "tools": TOOLS, "max_tokens": 300, "stream": True}, stream=True)
ev = [e for e in sse(resp) if e != "[DONE]"]
tcs = [e["choices"][0]["delta"]["tool_calls"][0] for e in ev if e["choices"] and "tool_calls" in e["choices"][0]["delta"]]
check(len(tcs) == 1 and tcs[0]["function"]["name"] == "get_weather" and ev[-1]["choices"][0]["finish_reason"] == "tool_calls", "streamed tool call")

# errors
for bad, what in (({"messages": []}, "empty messages"), ({"nope": 1}, "missing messages"), ({"messages": [{"role": "user", "content": "x"}], "temperature": 9}, "bad temperature"),
                  ({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]}, "image part")):
    code, body = post("/v1/chat/completions", bad)
    check(code == 400 and "error" in json.loads(body), "400 for " + what, body)
code, body = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "word " * 20000}], "max_tokens": 4})
check(code == 400 and json.loads(body)["error"]["type"] == "context_length_exceeded", "context length exceeded", body[:200])
try:
    urllib.request.urlopen(BASE + "/v1/missing")
    check(False, "404 on unknown path")
except urllib.error.HTTPError as e:
    check(e.code == 404, "404 on unknown path")

# ---- Anthropic /v1/messages
A_TOOLS = [{"name": "get_weather", "description": "Get weather for a city", "cache_control": {"type": "ephemeral"},
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}]
code, body = post("/v1/messages", {"model": "claude-x", "max_tokens": 16, "system": [{"type": "text", "text": "Answer in one word.", "cache_control": {"type": "ephemeral"}}],
    "metadata": {"user_id": "u"}, "messages": [{"role": "user", "content": [{"type": "text", "text": "Capital of France?"}]}]})
j = json.loads(body)
check(code == 200 and j["type"] == "message" and j["role"] == "assistant" and j["content"][0]["type"] == "text" and "Paris" in j["content"][0]["text"] and j["stop_reason"] == "end_turn", "messages completion", body)
check(j["usage"]["input_tokens"] > 0 and j["usage"]["output_tokens"] > 0 and j["model"] == "claude-x", "messages usage and model echo")

code, body = post("/v1/messages", {"max_tokens": 5, "messages": [{"role": "user", "content": "Count to 100."}]})
check(json.loads(body)["stop_reason"] == "max_tokens", "stop_reason max_tokens", body)
code, body = post("/v1/messages", {"max_tokens": 40, "stop_sequences": ["5"], "messages": [{"role": "user", "content": "Count 1 to 9, comma separated, no spaces."}]})
j = json.loads(body)
check(j["stop_reason"] == "stop_sequence" and j["stop_sequence"] == "5" and j["content"][0]["text"].startswith("1,2,3,4"), "stop_sequences", body)

code, body = post("/v1/messages", {"max_tokens": 400, "thinking": {"type": "enabled", "budget_tokens": 2000}, "messages": [{"role": "user", "content": "What is 2+2?"}]})
j = json.loads(body)
types = [b["type"] for b in j["content"]]
check(types[:2] == ["thinking", "text"] and j["content"][0]["thinking"] and "4" in j["content"][1]["text"], "thinking block then text", body)

code, resp = post("/v1/messages", {"max_tokens": 40, "stream": True, "messages": [{"role": "user", "content": "Count 1 to 5, comma separated."}]}, stream=True)
names = []; text = ""; cur = None
for raw in resp:
    line = raw.decode().rstrip("\n")
    if line.startswith("event: "): names.append(line[7:])
    elif line.startswith("data: "):
        d = json.loads(line[6:])
        if d["type"] == "content_block_delta" and d["delta"]["type"] == "text_delta": text += d["delta"]["text"]
check(names[0] == "message_start" and names[1] == "content_block_start" and names[-3:] == ["content_block_stop", "message_delta", "message_stop"], "anthropic SSE event order", names[:4] + names[-3:])
check("1, 2, 3, 4, 5" in text, "anthropic SSE text", text)

code, body = post("/v1/messages", {"max_tokens": 300, "tools": A_TOOLS, "messages": [{"role": "user", "content": "What is the weather in Paris for 3 days? Use the tool."}]})
j = json.loads(body)
uses = [b for b in j["content"] if b["type"] == "tool_use"]
check(j["stop_reason"] == "tool_use" and len(uses) == 1 and uses[0]["name"] == "get_weather" and uses[0]["input"] == {"city": "Paris", "days": 3}, "tool_use block", body)
tid = uses[0]["id"] if uses else "toolu_x"
code, body = post("/v1/messages", {"max_tokens": 80, "tools": A_TOOLS, "messages": [
    {"role": "user", "content": "Weather in Paris?"},
    {"role": "assistant", "content": [{"type": "tool_use", "id": tid, "name": "get_weather", "input": {"city": "Paris"}}]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid, "content": [{"type": "text", "text": "18C and cloudy"}]}]}]})
check("18" in "".join(b.get("text", "") for b in json.loads(body)["content"]), "tool_result is used", body)

code, resp = post("/v1/messages", {"max_tokens": 300, "stream": True, "tools": A_TOOLS, "messages": [{"role": "user", "content": "What is the weather in Paris for 3 days? Use the tool."}]}, stream=True)
blocks = {}; stop_reason = None
for raw in resp:
    line = raw.decode().rstrip("\n")
    if not line.startswith("data: "): continue
    d = json.loads(line[6:])
    if d["type"] == "content_block_start": blocks[d["index"]] = {"block": d["content_block"], "json": ""}
    elif d["type"] == "content_block_delta" and d["delta"]["type"] == "input_json_delta": blocks[d["index"]]["json"] += d["delta"]["partial_json"]
    elif d["type"] == "message_delta": stop_reason = d["delta"]["stop_reason"]
tu = [b for b in blocks.values() if b["block"]["type"] == "tool_use"]
check(stop_reason == "tool_use" and len(tu) == 1 and json.loads(tu[0]["json"]) == {"city": "Paris", "days": 3}, "streamed tool_use input_json_delta", blocks)

code, body = post("/v1/messages/count_tokens", {"messages": [{"role": "user", "content": "Capital of France?"}]})
check(code == 200 and json.loads(body)["input_tokens"] > 5, "count_tokens", body)
code, body = post("/v1/messages", {"max_tokens": 5, "messages": []})
j = json.loads(body)
check(code == 400 and j["type"] == "error" and j["error"]["type"] == "invalid_request_error", "anthropic error shape", body)

# ---- prefix cache: a long shared system prompt is snapshotted once and later requests resume from it
metrics = json.loads(urllib.request.urlopen(BASE + "/metrics").read())
if "prefix_cache" in metrics:
    facts = " ".join("Fact %d: the code for district %d is %d." % (i, i, i * 37 % 1000) for i in range(260))
    def ask(q):
        code, body = post("/v1/chat/completions", {"messages": [{"role": "system", "content": "You answer from these facts, briefly. " + facts}, {"role": "user", "content": q}], "max_tokens": 12})
        return json.loads(body)["choices"][0]["message"]["content"]
    m0 = json.loads(urllib.request.urlopen(BASE + "/metrics").read())["prefix_cache"]
    h0 = m0["hits"]
    a1 = ask("What is the code for district 7? Number only.")
    a2 = ask("What is the code for district 100? Number only.")
    a3 = ask("What is the code for district 7? Number only.")
    m1 = json.loads(urllib.request.urlopen(BASE + "/metrics").read())["prefix_cache"]
    h1 = m1["hits"]
    check(h1 - h0 >= 2, "prefix cache hits on a shared system prompt", (h0, h1))
    check((m1["hit_tokens"] - m0["hit_tokens"]) / max(1, h1 - h0) > 3000, "hits resume deep into the shared system prompt", (m0, m1))
    check("259" in a1 and a1 == a3, "answer after a prefix hit equals the cold answer", (a1, a2, a3))

# client disconnect frees the slot
s = socket.create_connection((BASE.split("//")[1].split(":")[0], int(BASE.rsplit(":", 1)[1])))
payload = json.dumps({"messages": [{"role": "user", "content": "Write a very long story."}], "max_tokens": 3000, "stream": True}).encode()
s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload)
s.recv(4096); s.close()
time.sleep(1.5)
active = json.loads(urllib.request.urlopen(BASE + "/metrics").read())["active_slots"]
check(active == 0, "disconnect cancels the generation", active)

print("FAILED: %d" % failures if failures else "ALL API CHECKS PASSED")
sys.exit(1 if failures else 0)
