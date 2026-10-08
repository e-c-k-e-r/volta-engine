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

# plain completion, thinking off by default
code, body = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Capital of France? One word."}], "max_tokens": 16})
j = json.loads(body)
check(code == 200 and "Paris" in j["choices"][0]["message"]["content"] and j["choices"][0]["finish_reason"] == "stop", "chat completion", body)
check("reasoning_content" not in j["choices"][0]["message"] and j["usage"]["completion_tokens"] > 0, "no reasoning by default, usage present")

# reasoning on
code, body = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "What is 2+2?"}], "max_tokens": 400, "reasoning_effort": "low"})
j = json.loads(body)
m = j["choices"][0]["message"]
check(code == 200 and m.get("reasoning_content") and "4" in (m["content"] or ""), "reasoning_content separated from content", body)

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
