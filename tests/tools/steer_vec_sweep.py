#!/usr/bin/env python3
"""Scale sweep of a control vector against a running server (steer=on, steer_dir holding NAME.steer).

For every steer_scale the prompts of --prompts (one per line) are sent greedily with steer_vector=NAME; the report per scale is
  words     mean words per answer (answers cut at --max-tokens count in full: raise it above the longest answer or read "done")
  done      share of answers that ended by themselves before --max-tokens
  caveat    share of answers containing a caveat phrase (--caveats)
  cjk       mean share of CJK characters       rep      share of repeated word trigrams
  dbl       doubled words per answer           same     shared leading words with the unsteered answer
The unsteered answers are the baseline (scale 0 is always run first). Scopes other than the file's own go through --scope.

  steer_vec_sweep.py --url http://localhost:8011 --name terse --prompts eval_general.txt --scales 0.01,0.02,0.04,0.08,0.16
"""
import argparse, json, re, statistics, sys, urllib.request

CAVEATS = ["however", "please note", "consult", "disclaimer", "it's important to note", "it is important to note", "i cannot", "i can't", "keep in mind",
           "be aware", "safety", "warning"]


def ask(url, prompt, max_tokens, extra):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0, "enable_thinking": False}
    body.update(extra)
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    if "choices" not in d:
        raise RuntimeError(str(d))
    return d["choices"][0]["message"]["content"], d["choices"][0].get("finish_reason") == "stop"


def words(t):
    return re.findall(r"\S+", t.lower())


def repetition(t):
    w = words(t)
    tri = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    return 0.0 if not tri else 1.0 - len(set(tri)) / len(tri)


def doubled(t):
    w = words(t)
    return sum(1 for a, b in zip(w, w[1:]) if a == b and a.isalpha())


def cjk(t):
    n = sum(1 for c in t if "一" <= c <= "鿿" or "぀" <= c <= "ヿ")
    return n / max(1, len(t))


def shared_prefix(a, b):
    wa, wb = words(a), words(b)
    n = 0
    while n < min(len(wa), len(wb)) and wa[n] == wb[n]:
        n += 1
    return n / max(1, min(len(wa), len(wb)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8011")
    ap.add_argument("--name", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--scales", required=True)
    ap.add_argument("--scope", default="")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--caveats", help="file with one caveat phrase per line, replacing the built-in list")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N prompts")
    ap.add_argument("--show", type=int, default=0, help="print the first N answers of every scale")
    ap.add_argument("--jsonl", default="steer_vec_sweep.jsonl")
    a = ap.parse_args()
    caveats = CAVEATS if not a.caveats else [l.strip().lower() for l in open(a.caveats) if l.strip()]
    prompts = [l.strip() for l in open(a.prompts) if l.strip()]
    if a.limit:
        prompts = prompts[:a.limit]
    base_res = [ask(a.url, p, a.max_tokens, {}) for p in prompts]
    base = [r[0] for r in base_res]
    out = open(a.jsonl, "w")
    for scale in [0.0] + [float(x) for x in a.scales.split(",")]:
        res = []
        for i, p in enumerate(prompts):
            extra = {} if scale == 0.0 else {"steer_vector": a.name, "steer_scale": scale}
            if a.scope and scale != 0.0:
                extra["steer_scope"] = a.scope
            text, done = base_res[i] if scale == 0.0 else ask(a.url, p, a.max_tokens, extra)
            res.append({"prompt": p, "text": text, "done": done, "words": len(words(text)), "caveat": any(c in text.lower() for c in caveats), "cjk": cjk(text),
                        "rep": repetition(text), "dbl": doubled(text), "same": shared_prefix(text, base[i])})
            out.write(json.dumps({"name": a.name, "scale": scale, **res[-1]}) + "\n")
        m = statistics.mean
        print(f"{a.name} scale {scale:<6} words {m(r['words'] for r in res):6.1f} done {m(r['done'] for r in res):.2f} caveat {m(r['caveat'] for r in res):.2f} cjk {m(r['cjk'] for r in res):.2f}"
              f" rep {m(r['rep'] for r in res):.2f} dbl {m(r['dbl'] for r in res):.1f} same {m(r['same'] for r in res):.2f}", flush=True)
        for r in res[:a.show]:
            print("   >", r["text"][:160].replace("\n", " "))


if __name__ == "__main__":
    sys.exit(main())
