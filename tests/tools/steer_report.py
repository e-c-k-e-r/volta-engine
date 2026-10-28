#!/usr/bin/env python3
"""Prints a sweep file (the .jsonl of steer_sweep.py, steer_vec_sweep.py or steer_think_sweep.py) as readable text, one block per prompt.

  steer_report.py FILE --list                               which settings the file holds
  steer_report.py FILE [--set target|control] [--strength -8] [--layers 34-42] [--scope all]
                       [--scale 1.0] [--fired|--not-fired] [--stock|--no-stock] [--limit 10] [--width 300]

Clamp sweeps (steer_sweep.py): score, whether the gate fired, whether the answer starts with a stock opening (and whether it did unprobed), the share
of leading words kept from the unprobed answer, and the answer; files written since the base_text field was added also show the unprobed answer.
Vector sweeps (steer_vec_sweep.py): per prompt the answers of every scale in one block, scale 0 being the unsteered answer.
Penalty sweeps (steer_think_sweep.py): per question the reasoning length, markers and correctness per penalty.
"""
import argparse, json, sys, textwrap


def short(t, n):
    t = " ".join(t.split())
    return t if len(t) <= n else t[:n] + " ..."


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--set")
    ap.add_argument("--strength", type=float)
    ap.add_argument("--layers")
    ap.add_argument("--scope")
    ap.add_argument("--scale", type=float, action="append", help="show only these scales (vector sweeps; scale 0 is always shown)")
    ap.add_argument("--fired", action="store_true")
    ap.add_argument("--not-fired", action="store_true")
    ap.add_argument("--stock", action="store_true", help="only answers that open with a stock reply")
    ap.add_argument("--no-stock", action="store_true", help="only answers that do not")
    ap.add_argument("--limit", type=int, default=0, help="stop after N prompts")
    ap.add_argument("--width", type=int, default=300)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.file) if l.strip()]
    if not rows:
        print("empty file")
        return 1
    kind = "vec" if "scale" in rows[0] and "name" in rows[0] else "think" if "think_words" in rows[0] else "clamp"

    if a.list:
        if kind == "clamp":
            seen = {}
            for r in rows:
                seen.setdefault((r["strength"], r["layers"], r["scope"], r["hedge"]), {"target": 0, "control": 0})[r["set"]] += 1
            for (st, ly, sc, he), n in seen.items():
                print(f"strength {st:g} layers {ly} scope {sc} hedge {he:g}: {n['target']} target rows, {n['control']} control rows")
        elif kind == "vec":
            print(f"vector {rows[0]['name']}: scales", sorted({r["scale"] for r in rows}), f"{len({r['prompt'] for r in rows})} prompts")
        else:
            print("penalties", sorted({r["penalty"] for r in rows}), f"{len({r['q'] for r in rows})} questions")
        return 0

    shown = 0
    if kind == "clamp":
        for r in rows:
            if a.set and r["set"] != a.set: continue
            if a.strength is not None and r["strength"] != a.strength: continue
            if a.layers and r["layers"] != a.layers: continue
            if a.scope and r["scope"] != a.scope: continue
            if a.fired and not r["fired"]: continue
            if a.not_fired and r["fired"]: continue
            if a.stock and not r["stock"]: continue
            if a.no_stock and r["stock"]: continue
            score = f"{r['score']:.2f}" if r.get("score") is not None else "-"
            print(f"[{r['set']}] score {score} {'fired' if r['fired'] else 'clean'} | stock opening: unprobed {'yes' if r['base_stock'] else 'no'}, now {'yes' if r['stock'] else 'no'}"
                  f" | same {r['same']:.2f} rep {r['rep']:.2f} dbl {r['doubled']} | strength {r['strength']:g} layers {r['layers']} scope {r['scope']}")
            print("  Q   :", short(r["prompt"], 120))
            if r.get("base_text") is not None:
                print("  base:", short(r["base_text"], a.width))
            print("  now :", short(r["text"], a.width))
            print()
            shown += 1
            if a.limit and shown >= a.limit: break
    elif kind == "vec":
        prompts = []
        for r in rows:
            if r["prompt"] not in prompts: prompts.append(r["prompt"])
        for p in prompts:
            print("Q:", short(p, 120))
            for r in rows:
                if r["prompt"] != p: continue
                if a.scale and r["scale"] != 0.0 and r["scale"] not in a.scale: continue
                print(f"  scale {r['scale']:<4g} ({r['words']} words{', finished' if r.get('done') else ''}): {short(r['text'], a.width)}")
            print()
            shown += 1
            if a.limit and shown >= a.limit: break
    else:
        qs = []
        for r in rows:
            if r["q"] not in qs: qs.append(r["q"])
        for q in qs:
            print("Q:", short(q, 140), "| expected", next(r["answer"] for r in rows if r["q"] == q))
            for r in rows:
                if r["q"] == q:
                    print(f"  penalty {r['penalty']:<4g} reasoning {r['think_words']:>5} words, markers {r['marks']}, {'finished' if r['done'] else 'cut off'}, {'correct' if r['correct'] else 'wrong'}: {short(r['text'], 80)}")
            print()
            shown += 1
            if a.limit and shown >= a.limit: break
    return 0


if __name__ == "__main__":
    sys.exit(main())
