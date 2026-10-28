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
    """Counting and number-theory questions with an exact answer: easy to check, and hard enough to make the model hesitate and recheck."""
    rnd = random.Random(seed)
    out = []
    for i in range(n):
        k = i % 4
        if k == 0:
            hi, a, b = rnd.randint(300, 900), rnd.choice([4, 6, 7, 9]), rnd.choice([5, 8, 10, 11])
            ans = sum(1 for x in range(1, hi + 1) if (x % a == 0) != (x % b == 0))
            q = f"How many integers from 1 to {hi} are divisible by exactly one of {a} and {b} (divisible by one of them but not by both)?"
        elif k == 1:
            m = rnd.choice([7, 9, 11, 13, 17, 19])
            r1 = rnd.randint(2, m - 1) ** 2 % m or 1
            ans = sum(1 for x in range(1, 200) if x * x % m == r1)
            q = f"How many integers x with 1 <= x <= 199 satisfy x*x leaving remainder {r1} when divided by {m}?"
        elif k == 2:
            n_, d = rnd.randint(20, 60), rnd.randint(3, 6)
            ans = sum(1 for a_ in range(1, n_ + 1) for b_ in range(a_ + 1, n_ + 1) if (a_ + b_) % d == 0)
            q = f"How many pairs of different integers a < b chosen from 1 to {n_} have a sum divisible by {d}?"
        else:
            lo, hi = rnd.randint(100, 400), rnd.randint(500, 900)
            ans = sum(1 for x in range(lo, hi + 1) if sum(int(c) for c in str(x)) % 7 == 0)
            q = f"How many integers from {lo} to {hi} (both included) have a digit sum divisible by 7?"
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
