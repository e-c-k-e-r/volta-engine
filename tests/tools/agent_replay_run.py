#!/usr/bin/env python3
"""Runs replay turns (agent_replay_build.py) against a running volta_server, one at a time, greedy: per turn the prompt and generated tokens, time to first
token, decode tokens per second, speculation rounds and suffix rounds / accepted / proposed from /api/stats, and the generated text.

  agent_replay_run.py TURNS.json OUT.json [--url http://127.0.0.1:8000] [--max-tokens 400] [--limit N] [--pause S]
"""
import argparse, json, time, urllib.request


def http_json(url, body=None, timeout=600):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def spec_stats(base):
    try:
        return http_json(base + "/api/stats", timeout=10).get("speculation", {})
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("turns")
    ap.add_argument("out")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pause", type=float, default=0.0, help="seconds to idle between turns (lets the cards cool)")
    a = ap.parse_args()
    model = http_json(a.url + "/v1/models")["data"][0]["id"]
    turns = json.load(open(a.turns))
    if a.limit:
        turns = turns[:a.limit]
    res = []
    for i, t in enumerate(turns):
        body = {"model": model, "messages": t["messages"], "max_tokens": a.max_tokens, "stream": True, "stream_options": {"include_usage": True},
                "enable_thinking": False, "temperature": 0.0}
        req = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        s0 = spec_stats(a.url)
        t0 = time.perf_counter()
        first = last = None
        text, usage = "", None
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:") or line.endswith("[DONE]"):
                    continue
                j = json.loads(line[5:])
                if j.get("usage"):
                    usage = j["usage"]
                for c in j.get("choices", []):
                    d = c.get("delta", {}).get("content")
                    if d:
                        now = time.perf_counter()
                        first = first or now
                        last = now
                        text += d
        end = time.perf_counter()
        s1 = spec_stats(a.url)
        comp = usage["completion_tokens"] if usage else 0
        dec = (comp - 1) / (last - first) if first and last and last > first and comp > 1 else 0.0
        x0, x1 = s0.get("suffix", {}), s1.get("suffix", {})
        res.append({"i": i, "seg": t["seg"], "prompt": usage["prompt_tokens"] if usage else 0, "gen": comp,
                    "ttft": round(first - t0, 2) if first else None, "tok_s": round(dec, 1), "total_s": round(end - t0, 2),
                    "rounds": s1.get("rounds", 0) - s0.get("rounds", 0), "sfx_rounds": x1.get("rounds", 0) - x0.get("rounds", 0),
                    "sfx_acc": x1.get("accepted", 0) - x0.get("accepted", 0), "sfx_prop": x1.get("proposed", 0) - x0.get("proposed", 0), "text": text})
        print(i, {k: v for k, v in res[-1].items() if k != "text"}, flush=True)
        if a.pause:
            time.sleep(a.pause)
    json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
