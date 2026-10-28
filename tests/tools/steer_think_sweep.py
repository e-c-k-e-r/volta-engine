#!/usr/bin/env python3
"""Reasoning-length sweep of the reflection-marker penalty against a running server (steer=on).

A fixed, seeded set of multi-step word problems with integer answers (generated here, nothing bundled) is sent with thinking enabled for every
think_penalty value. Per value:
  think    mean words in the reasoning block            marks   marker words (wait / hmm / alternatively) per 100 reasoning words
  done     share of answers that ended before --max-tokens (an answer cut off inside the reasoning counts as wrong)
  correct  share of answers whose final text contains the expected number as its last number
Penalty 0 is the baseline and runs first. The server must return the reasoning separately (reasoning_content).

  steer_think_sweep.py --url http://localhost:8011 --penalties 2,4,8,16 --problems 12 --max-tokens 2500
"""
import argparse, json, random, re, statistics, sys, urllib.request

MARKS = re.compile(r"\b(wait|hmm|alternatively)\b", re.I)


def problems(n, seed):
    rnd = random.Random(seed)
    out = []
    for i in range(n):
        k = i % 3
        if k == 0:
            a, b, c, d = rnd.randint(12, 49), rnd.randint(6, 19), rnd.randint(30, 99), rnd.randint(3, 9)
            q = f"A school buys {a} boxes of pencils with {b} pencils in each box, then gives away {c} pencils and splits the rest equally among {d} classes, with {(a * b - c) % d} pencils left over. How many pencils does each class get?"
            ans = (a * b - c) // d
        elif k == 1:
            s, t, u = rnd.randint(40, 90), rnd.randint(2, 6), rnd.randint(30, 70)
            h = rnd.randint(2, 5)
            q = f"A train travels at {s} km/h for {t} hours, stops, then travels at {u} km/h for {h} more hours. How many kilometres did it travel in total?"
            ans = s * t + u * h
        else:
            x, y, z = rnd.randint(5, 20), rnd.randint(3, 12), rnd.randint(2, 9)
            q = f"There are {x} shelves with {y} books on each. {z} shelves are emptied and their books are spread evenly over the remaining shelves, and the number of books divides evenly. How many books are on each remaining shelf?"
            while (x * y) % (x - z) != 0 or x - z <= 0:
                z = rnd.randint(1, x - 1)
                y = rnd.randint(3, 12)
                x = x
            q = f"There are {x} shelves with {y} books on each. {z} shelves are emptied and their books are spread evenly over the remaining shelves. How many books are on each remaining shelf?"
            ans = (x * y) // (x - z)
        out.append((q + " Give the final number at the end.", ans))
    return out


def ask(url, prompt, max_tokens, penalty):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0, "enable_thinking": True}
    if penalty:
        body["think_penalty"] = penalty
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    if "choices" not in d:
        raise RuntimeError(str(d))
    m = d["choices"][0]["message"]
    return m.get("reasoning_content") or "", m.get("content") or "", d["choices"][0].get("finish_reason") == "stop"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8011")
    ap.add_argument("--penalties", default="2,4,8,16")
    ap.add_argument("--problems", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=2500)
    ap.add_argument("--jsonl", default="steer_think_sweep.jsonl")
    a = ap.parse_args()
    probs = problems(a.problems, a.seed)
    out = open(a.jsonl, "w")
    for pen in [0.0] + [float(x) for x in a.penalties.split(",")]:
        res = []
        for q, ans in probs:
            think, text, done = ask(a.url, q, a.max_tokens, pen)
            nums = re.findall(r"-?\d[\d,]*", text.replace("$", ""))
            ok = done and bool(nums) and nums[-1].replace(",", "") == str(ans)
            tw = len(think.split())
            res.append({"penalty": pen, "q": q, "answer": ans, "think_words": tw, "marks": len(MARKS.findall(think)), "done": done, "correct": ok, "text": text[-120:]})
            out.write(json.dumps(res[-1]) + "\n")
        m = statistics.mean
        words = sum(r["think_words"] for r in res)
        print(f"penalty {pen:<5} think {m(r['think_words'] for r in res):7.1f} marks/100w {100 * sum(r['marks'] for r in res) / max(1, words):5.2f}"
              f" done {m(r['done'] for r in res):.2f} correct {m(r['correct'] for r in res):.2f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
