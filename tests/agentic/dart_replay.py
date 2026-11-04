#!/usr/bin/env python3
"""Offline turn replay for per-turn thinking: would a quick answer without reasoning have taken the same action as the reasoned one?

  dart_replay.py --dump DIR --url URL --out OUT.jsonl [--drafts 2] [--temperature T] [--max-tokens 4096] [--limit N] [--shard I/N]
  dart_replay.py --report OUT.jsonl

--dump is a server's dump_requests directory from Claude Code sessions run with thinking on. Requests are chained into sessions by their
message lists (a turn's list extends the previous turn's), and a turn's reference action is the assistant message the next request carries
back. Every turn with a reference is sent again with thinking disabled, --drafts times (sampled with the server's defaults unless
--temperature is given, as the reference was), and each draft's action is compared with
the others and with the reference. An action is the list of tool calls, normalised per tool: Read / Glob / Grep by their path and pattern,
Bash by its command with whitespace collapsed, Edit by file and the stripped old and new strings, Write by file and stripped content,
TodoWrite by name alone; a reply without tool calls is the action "text" (the end of the task). The report also compares coarser forms:
the kind of step (look, change, run) with the files it touches, and the kind of step alone.
"""
import argparse, glob, json, os, re, sys, urllib.request


def load(dump):
    reqs = []
    for p in glob.glob(os.path.join(dump, "req_*.json")):
        n = int(re.search(r"req_(\d+)\.json$", p).group(1))
        try:
            body = json.load(open(p))
        except Exception:
            continue
        if not body.get("tools") or not isinstance(body.get("messages"), list):
            continue   # side calls (titles, summaries) carry no tools
        reqs.append((n, body))
    reqs.sort(key=lambda x: x[0])
    return reqs


def canon(m):
    """A message without the parts a client rewrites between turns (cache markers, thinking signatures)"""
    c = m.get("content")
    if isinstance(c, str):
        c = [{"type": "text", "text": c}]
    out = []
    for b in c or []:
        b = {k: v for k, v in b.items() if k not in ("cache_control", "signature")}
        if b.get("type") == "text":
            b["text"] = b.get("text", "").strip()
        out.append(b)
    return {"role": m.get("role"), "content": out}


def turns(reqs):
    """(request body, reference assistant message) for every request a later request continues"""
    seen = [(n, b, [canon(m) for m in b["messages"]]) for n, b in reqs]
    out = []
    for i, (n, b, msgs) in enumerate(seen):
        for _, b2, msgs2 in seen[i + 1:]:
            if len(msgs2) > len(msgs) and msgs2[:len(msgs)] == msgs and msgs2[len(msgs)]["role"] == "assistant":
                out.append((n, b, b2["messages"][len(msgs)]))
                break
    return out


def blocks(content):
    return [{"type": "text", "text": content}] if isinstance(content, str) else (content or [])


def norm_call(name, inp):
    ws = lambda s: " ".join(str(s).split())
    if name == "Read": return ("Read", inp.get("file_path"))
    if name == "Bash": return ("Bash", ws(inp.get("command", "")))
    if name == "Edit": return ("Edit", inp.get("file_path"), ws(inp.get("old_string", "")), ws(inp.get("new_string", "")))
    if name == "MultiEdit": return ("MultiEdit", inp.get("file_path"), tuple((ws(e.get("old_string", "")), ws(e.get("new_string", ""))) for e in inp.get("edits", [])))
    if name == "Write": return ("Write", inp.get("file_path"), ws(inp.get("content", "")))
    if name in ("Glob", "Grep"): return (name, inp.get("pattern"), inp.get("path"))
    if name == "TodoWrite": return ("TodoWrite",)
    return (name, json.dumps(inp, sort_keys=True))


def action(content):
    calls = [norm_call(b.get("name"), b.get("input") or {}) for b in blocks(content) if b.get("type") == "tool_use"]
    return tuple(calls) if calls else (("text",),)


def kind(act):
    return "+".join(sorted({c[0] for c in act}))


def draft(url, body, temperature, max_tokens):
    b = dict(body)
    b["thinking"] = {"type": "disabled"}
    b["stream"] = False
    if temperature is not None:
        b["temperature"] = temperature
    b["max_tokens"] = min(int(b.get("max_tokens", max_tokens)), max_tokens)
    req = urllib.request.Request(url + "/v1/messages", json.dumps(b).encode(), {"Content-Type": "application/json", "x-api-key": "sk-local", "anthropic-version": "2023-06-01"})
    d = json.load(urllib.request.urlopen(req, timeout=900))
    return d.get("content", []), d.get("usage", {})


LOOK = {"ls", "cat", "head", "tail", "wc", "grep", "rg", "find", "sed", "awk", "sort", "uniq", "cut", "diff", "file", "stat", "tree", "echo", "pwd", "less", "git"}
FILE_RE = re.compile(r"[\w./-]*\.(?:py|c|h|js|json|sql|sqlite|db|csv|log|txt|md|conf|sh|cfg|ini)\b|[\w-]+/[\w./-]+")


def coarse_call(c):
    """The kind of step and what it touches: (look | change | run | todo | other, file names)"""
    name = c[0]
    base = lambda p: os.path.basename(str(p or "").rstrip("/")) or str(p)
    if name in ("Read", "Glob", "Grep"): return ("look", frozenset([base(c[1])] if name == "Read" else [str(c[1])]))
    if name in ("Write", "Edit", "MultiEdit"): return ("change", frozenset([base(c[1])]))
    if name == "TodoWrite": return ("todo", frozenset())
    if name == "text": return ("text", frozenset())
    if name == "Bash":
        cmd = c[1]
        files = frozenset(base(f) for f in FILE_RE.findall(cmd) if not f.startswith("/tmp/work") or base(f) not in ("work", ""))
        if "<<" in cmd or re.search(r"(^|[;&|]\s*)(sed -i|mv|cp|rm|mkdir|touch|tee|patch)\b", cmd) or re.search(r">\s*[\w./-]+", cmd.replace("2>&1", "")):
            return ("change", files)
        progs = {seg.split()[0] for seg in re.split(r"[;&|]+", cmd) if seg.split()}
        progs = {p for p in progs if p not in ("cd",)}
        if progs and progs <= LOOK and not re.search(r"git (commit|merge|add|checkout|rebase|reset)", cmd): return ("look", files)
        return ("run", files)
    return ("other", frozenset([name]))


def coarse(act):
    """Per kind of step, the union of the files its calls touch (two Reads equal one cat of both files)"""
    by = {}
    for c in act:
        k, files = coarse_call(tuple(c))
        by[k] = by.get(k, frozenset()) | files
    return tuple(sorted(by.items(), key=repr))


def tally(rows, key):
    n = len(rows)
    agree = [r for r in rows if all(key(d) == key(r["drafts"][0]) for d in r["drafts"])]
    agree_ref = [r for r in agree if key(r["drafts"][0]) == key(r["ref"])]
    any_ref = [r for r in rows if any(key(d) == key(r["ref"]) for d in r["drafts"])]
    return n, len(agree), len(agree_ref), len(any_ref)


def report(path):
    rows = [json.loads(l) for l in open(path)]
    n = len(rows)
    agree = [r for r in rows if r["agree"]]
    agree_ref = [r for r in agree if r["draft_eq_ref"][0]]
    any_ref = [r for r in rows if any(r["draft_eq_ref"])]
    print(f"{n} turns from {len({r['session'] for r in rows})} sessions")
    print(f"drafts agree with each other: {len(agree)} ({100 * len(agree) / max(1, n):.0f}%)")
    print(f"  of those, equal to the thinking reference: {len(agree_ref)} ({100 * len(agree_ref) / max(1, len(agree)):.0f}%)")
    print(f"any draft equal to the reference: {len(any_ref)} ({100 * len(any_ref) / max(1, n):.0f}%)")
    for label, key in (("coarse (kind of step and the files it touches)", coarse), ("kind of step only", lambda act: tuple(sorted({c[0] for c in coarse(act)})))):
        t, ag, ar, anyr = tally(rows, key)
        print(f"{label}: drafts agree {ag} ({100 * ag / max(1, t):.0f}%), of those equal to the reference {ar} ({100 * ar / max(1, ag):.0f}%), any draft equal {anyr} ({100 * anyr / max(1, t):.0f}%)")
    by = {}
    for r in rows:
        k = r["ref_kind"]
        s = by.setdefault(k, [0, 0, 0])
        s[0] += 1; s[1] += r["agree"]; s[2] += r["agree"] and r["draft_eq_ref"][0]
    print("per reference action kind: turns, drafts agree, agree and equal to the reference")
    for k, (t, a, e) in sorted(by.items(), key=lambda x: -x[1][0]):
        print(f"  {k:24s} {t:4d} {a:4d} {e:4d}")
    by_pos = {}
    for r in rows:
        k = min(r["turn"], 10)
        s = by_pos.setdefault(k, [0, 0, 0])
        s[0] += 1; s[1] += r["agree"]; s[2] += r["agree"] and r["draft_eq_ref"][0]
    print("per turn index (10 = 10 and later): turns, drafts agree, agree and equal")
    for k in sorted(by_pos):
        t, a, e = by_pos[k]
        print(f"  {k:3d} {t:4d} {a:4d} {e:4d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump"); ap.add_argument("--url"); ap.add_argument("--out"); ap.add_argument("--report")
    ap.add_argument("--drafts", type=int, default=2); ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=4096); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="I/N: replay every N-th turn starting at I (one client per server slot)")
    a = ap.parse_args()
    if a.report:
        report(a.report)
        return
    reqs = load(a.dump)
    ts = turns(reqs)
    # sessions: a turn whose message list starts with another turn's belongs to that turn's session
    first = {}
    sess_of, idx_of = {}, {}
    for n, b, _ in ts:
        key = json.dumps(canon(b["messages"][0]), sort_keys=True)
        sess_of[n] = None
        for m, (sid, msgs) in list(first.items()):
            if len(b["messages"]) > len(msgs) and [canon(x) for x in b["messages"][:len(msgs)]] == msgs:
                sess_of[n] = sid
        if sess_of[n] is None:
            sess_of[n] = n
        first[n] = (sess_of[n], [canon(x) for x in b["messages"]])
    counts = {}
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["req"] for l in open(a.out)}
    out = open(a.out, "a")
    for k, (n, b, ref) in enumerate(ts):
        counts[sess_of[n]] = counts.get(sess_of[n], 0) + 1
        idx_of[n] = counts[sess_of[n]] - 1
        if a.limit and k >= a.limit: break
        si, sn = (int(x) for x in a.shard.split("/"))
        if n in done or k % sn != si: continue
        ref_act = action(ref.get("content"))
        acts, usage = [], []
        for _ in range(a.drafts):
            c, u = draft(a.url, b, a.temperature, a.max_tokens)
            acts.append(action(c)); usage.append(u.get("output_tokens", 0))
        rec = {"req": n, "session": sess_of[n], "turn": idx_of[n], "ref_kind": kind(ref_act), "ref": ref_act, "drafts": acts,
               "agree": all(x == acts[0] for x in acts), "draft_eq_ref": [x == ref_act for x in acts], "draft_tokens": usage}
        out.write(json.dumps(rec) + "\n"); out.flush()
        print(f"req {n:4d} turn {idx_of[n]:2d} {kind(ref_act):16s} agree {rec['agree']!s:5s} eq_ref {rec['draft_eq_ref']}", flush=True)


if __name__ == "__main__":
    main()
