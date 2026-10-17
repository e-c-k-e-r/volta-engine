#!/usr/bin/env python3
"""Builds replay turns from a recorded Claude Code session (a ~/.claude/projects/*/*.jsonl transcript): segments of consecutive assistant turns, each turn's
prompt being the rendered history before it (tool calls and results as plain text, results clipped) so the server writes the next turn. Segments start at
fixed places in the session and keep one fixed history start, so the prefix cache serves their later turns.

  agent_replay_build.py SESSION.jsonl OUT.json [--starts 0.15,0.5,0.8] [--per-segment 10] [--history-chars 34000]
"""
import argparse, json


def clip(s, n=3500):
    s = str(s)
    return s if len(s) <= n else s[:n // 2] + "\n...[clipped]...\n" + s[-n // 2:]


def load_messages(path):
    msgs = []
    for line in open(path):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        m = d.get("message", {})
        role, c = m.get("role"), m.get("content")
        if role not in ("user", "assistant"):
            continue
        if isinstance(c, str):
            if c.strip() and not c.startswith("<command") and "system-reminder" not in c[:40]:
                msgs.append((role, clip(c, 6000)))
            continue
        if not isinstance(c, list):
            continue
        parts = []
        for x in c:
            if not isinstance(x, dict):
                continue
            t = x.get("type")
            if t == "text" and x.get("text", "").strip():
                parts.append(clip(x["text"], 6000))
            elif t == "tool_use":
                parts.append("Tool call: %s\n%s" % (x.get("name"), clip(json.dumps(x.get("input", {}), ensure_ascii=False), 4000)))
            elif t == "tool_result":
                cc = x.get("content")
                if isinstance(cc, list):
                    cc = " ".join(y.get("text", "") for y in cc if isinstance(y, dict))
                parts.append("Tool result:\n" + clip(cc))
        if parts:
            msgs.append((role, "\n\n".join(parts)))
    merged = []
    for r, t in msgs:
        if merged and merged[-1][0] == r:
            merged[-1] = (r, merged[-1][1] + "\n\n" + t)
        else:
            merged.append((r, t))
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("out")
    ap.add_argument("--starts", default="0.15,0.5,0.8")
    ap.add_argument("--per-segment", type=int, default=10)
    ap.add_argument("--history-chars", type=int, default=34000)
    a = ap.parse_args()
    merged = load_messages(a.session)
    targets = [i for i, (r, t) in enumerate(merged) if r == "assistant" and i >= 2 and 80 < len(t) < 2500]
    turns = []
    for si, frac in enumerate(float(x) for x in a.starts.split(",")):
        s = int(len(targets) * frac)
        seg = targets[s:s + a.per_segment]
        if not seg:
            continue
        tot, lo = 0, seg[0]
        while lo > 0 and tot < a.history_chars:
            lo -= 1
            tot += len(merged[lo][1])
        if merged[lo][0] != "user":
            lo += 1
        for ti in seg:
            hist = [{"role": r, "content": t} for r, t in merged[lo:ti]]
            if hist and hist[0]["role"] != "user":
                hist = hist[1:]
            turns.append({"seg": si, "messages": hist, "target_chars": len(merged[ti][1]), "target": merged[ti][1]})
    json.dump(turns, open(a.out, "w"))
    print("messages %d, targets %d, turns %d" % (len(merged), len(targets), len(turns)))


if __name__ == "__main__":
    main()
