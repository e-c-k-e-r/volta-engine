#!/usr/bin/env python3
"""Refusal rate per prompt group under always-on clamp files (steer_vector=NAME, no probe) against a running server (steer=on, steer_dir).

  steer_clamp_eval.py --url http://localhost:8011 --groups tech=eval_tech.txt,ethic=eval_ethic.txt,benign=eval_benign.txt --vectors tech_a,ethic_a,ethic_orth

Each group is a file of one-line prompts. Per vector (and the unsteered baseline, "none"): share of replies that open with a refusal, repetition
and doubled words. Writes every reply to --jsonl.
"""
import argparse, json, sys, urllib.request
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from steer_sweep import STOCK, stock, repetition, doubled


def ask(url, prompt, max_tokens, vector):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0, "enable_thinking": False}
    if vector != "none":
        body["steer_vector"] = vector
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    if "choices" not in d:
        raise RuntimeError(str(d))
    return d["choices"][0]["message"].get("content") or ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8011")
    ap.add_argument("--groups", required=True, help="NAME=FILE,... prompt files")
    ap.add_argument("--vectors", required=True, help="steer_vector names, comma separated (none is always run first)")
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--jsonl", default="steer_clamp_eval.jsonl")
    a = ap.parse_args()
    groups = {}
    for g in a.groups.split(","):
        name, path = g.split("=")
        groups[name] = [l.strip() for l in open(path) if l.strip()]
    out = open(a.jsonl, "w")
    for vec in ["none"] + a.vectors.split(","):
        line = f"{vec:>14}"
        for name, prompts in groups.items():
            res = []
            for p in prompts:
                t = ask(a.url, p, a.max_tokens, vec)
                res.append({"vector": vec, "group": name, "prompt": p, "text": t, "stock": stock(t, STOCK), "rep": repetition(t), "doubled": doubled(t)})
                out.write(json.dumps(res[-1]) + "\n")
            n = len(res)
            line += f" | {name} refuse {sum(r['stock'] for r in res) / n:.2f} rep {sum(r['rep'] for r in res) / n:.2f} dbl {sum(r['doubled'] for r in res) / n:.1f}"
        print(line, flush=True)


if __name__ == "__main__":
    sys.exit(main())
