#!/usr/bin/env python3
"""Speculation under steering: acceptance and decode speed per condition against a running server (steer=on, a speculator configured).

Every condition sends the same prompts one after another (greedy, thinking off unless --think) and reports the drafted tokens taken
(delta of /metrics spec_accepted / spec_proposals) and the mean decode tok/s of the requests (usage.decode_tok_s):
  base     no steering
  vector   steer_vector NAME (--vector), at its recommended scale
  clamp    steer_probe with the gate forced (steer_gate_threshold -1), edit window and strength of the server
  penalty  think_penalty P (--penalty) with thinking on (reasoning rows only)
  steer_spec_bench.py --url http://localhost:8011 --prompts eval_general.txt --vector terse --penalty 8
"""
import argparse, json, statistics, sys, urllib.request


def post(url, path, body=None):
    req = urllib.request.Request(url + path, json.dumps(body).encode() if body is not None else None, {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def run(url, prompts, extra, max_tokens, think):
    m0 = post(url, "/metrics")
    speeds, toks = [], 0
    for p in prompts:
        body = {"messages": [{"role": "user", "content": p}], "max_tokens": max_tokens, "temperature": 0, "enable_thinking": think}
        body.update(extra)
        d = post(url, "/v1/chat/completions", body)
        speeds.append(d["usage"]["decode_tok_s"])
        toks += d["usage"]["completion_tokens"]
    m1 = post(url, "/metrics")
    prop = m1["spec_proposals"] - m0["spec_proposals"]
    acc = m1["spec_accepted"] - m0["spec_accepted"]
    return acc / prop if prop else float("nan"), statistics.mean(speeds), toks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8011")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--vector", default="")
    ap.add_argument("--penalty", type=float, default=0.0)
    ap.add_argument("--think-prompts", help="prompts for the penalty condition (thinking on); default --prompts")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--limit", type=int, default=8)
    a = ap.parse_args()
    prompts = [l.strip() for l in open(a.prompts) if l.strip()][:a.limit]
    tprompts = [l.strip() for l in open(a.think_prompts or a.prompts) if l.strip()][:a.limit]
    conds = [("base", prompts, {}, False)]
    if a.vector:
        conds.append((f"vector {a.vector}", prompts, {"steer_vector": a.vector}, False))
    conds.append(("clamp (forced)", prompts, {"steer_probe": True, "steer_gate_threshold": -1.0}, False))
    if a.penalty:
        conds.append(("think base", tprompts, {}, True))
        conds.append((f"think penalty {a.penalty:g}", tprompts, {"think_penalty": a.penalty}, True))
    for name, ps, extra, think in conds:
        acc, tps, toks = run(a.url, ps, extra, a.max_tokens if not think else 3 * a.max_tokens, think)
        print(f"{name:<22} accepted {acc:.3f} decode {tps:6.1f} tok/s  ({toks} tokens)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
