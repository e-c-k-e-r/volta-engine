#!/usr/bin/env python3
"""Agentic benchmark: the Claude Code CLI works through small tasks against any server that speaks the Anthropic messages API.

Every run happens in a bubblewrap sandbox: read-only system, an empty home, a scratch copy of the task's repository as the only writable
directory. A task is tasks/<id>/ with task.json (prompt, tier, tags), repo/ (the starting files), verify.sh (exit 0 = pass, run in the
work directory afterwards; files from hidden/ are copied in first) and solution.sh (reference solution, used by the self-test).

  bench.py selftest                                   every verifier fails on the starting repo and passes after solution.sh
  bench.py run --base-url URL --model NAME --out DIR  [--tier quick] [--only ID,ID] [--repeats N] [--thinking low|medium|high|off]
                                                      [--max-turns N] [--timeout S] [--api-key KEY]
  bench.py summary DIR                                pass rate, time, turns and tokens per task and overall

A run directory holds meta.json (client version, server URL, model, settings, git revision), results.jsonl (one record per task run)
and <task>/<rep>/ with the Claude Code event stream (events.jsonl) and the final work directory.
"""
import argparse, json, os, shutil, statistics, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
THINKING = {"off": "0", "low": "4000", "medium": "10000", "high": "31999"}


def claude_paths():
    exe = shutil.which("claude")
    if not exe:
        sys.exit("claude CLI not found on PATH")
    real = os.path.realpath(exe)
    return exe, [os.path.dirname(exe), os.path.dirname(real)]


def sandbox_cmd(work, inner, net, extra_env=None, ro_extra=()):
    """bwrap command line: read-only root, empty /home, /mnt hidden, `work` mounted writable at /tmp/work."""
    home = os.path.expanduser("~")
    cmd = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/home", "--tmpfs", "/mnt", "--tmpfs", "/tmp", "--tmpfs", "/run",
           "--bind", work, "/tmp/work", "--chdir", "/tmp/work", "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--die-with-parent", "--clearenv"]
    if not net:
        cmd += ["--unshare-net"]
    for p in ro_extra:
        if p.startswith(home):
            cmd += ["--ro-bind", p, p]
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin:" + os.path.join(home, ".local/bin"), "HOME": home, "LANG": "C.UTF-8", "TERM": "dumb",
           "TMPDIR": "/tmp", "CLAUDE_CONFIG_DIR": "/tmp/claude-config"}
    env.update(extra_env or {})
    for k, v in env.items():
        cmd += ["--setenv", k, v]
    return cmd + inner


def run_checked(cmd, timeout):
    """-> (returncode, stdout, stderr, timed_out)"""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout)
        return p.returncode, out, err, False
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, 9)
        out, err = p.communicate()
        return -9, out or "", err or "", True


def load_tasks(tier=None, only=None):
    tasks = []
    for tid in sorted(os.listdir(os.path.join(HERE, "tasks"))):
        d = os.path.join(HERE, "tasks", tid)
        if not os.path.exists(d + "/task.json"):
            continue
        t = json.load(open(d + "/task.json"))
        t["dir"] = d
        if tier and t["tier"] != tier:
            continue
        if only and tid not in only:
            continue
        tasks.append(t)
    return tasks


def verify(task, work):
    for f in os.listdir(task["dir"] + "/hidden") if os.path.isdir(task["dir"] + "/hidden") else []:
        shutil.copy(os.path.join(task["dir"], "hidden", f), os.path.join(work, f))
    shutil.copy(task["dir"] + "/verify.sh", os.path.join(work, ".verify.sh"))
    rc, out, err, to = run_checked(sandbox_cmd(work, ["bash", "/tmp/work/.verify.sh"], net=False), 120)
    return rc == 0, (out + err)[-2000:]


def selftest(args):
    bad = 0
    for t in load_tasks(args.tier, args.only and args.only.split(",")):
        with tempfile.TemporaryDirectory() as w:
            shutil.copytree(t["dir"] + "/repo", w, dirs_exist_ok=True)
            before, _ = verify(t, w)
            shutil.copy(t["dir"] + "/solution.sh", w + "/.solution.sh")
            rc, out, err, _ = run_checked(sandbox_cmd(w, ["bash", "/tmp/work/.solution.sh"], net=False), 60)
            os.unlink(w + "/.solution.sh")
            after, log = verify(t, w)
        ok = (not before) and after and rc == 0
        print(f"{'ok  ' if ok else 'FAIL'} {t['id']:20s} fails-before={not before} passes-after={after} solution-rc={rc}")
        if not ok:
            bad += 1
            print(log, err)
    return 1 if bad else 0


def parse_events(path):
    m = {"turns": 0, "tool_calls": 0, "tool_errors": 0, "think_chars": 0, "text_chars": 0, "input_tokens": 0, "output_tokens": 0,
         "cache_read_tokens": 0, "is_error": None, "result": ""}
    for line in open(path):
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        t = ev.get("type")
        if t == "assistant":
            for b in ev.get("message", {}).get("content", []):
                if b.get("type") == "thinking": m["think_chars"] += len(b.get("thinking", ""))
                elif b.get("type") == "text": m["text_chars"] += len(b.get("text", ""))
                elif b.get("type") == "tool_use": m["tool_calls"] += 1
        elif t == "user":
            c = ev.get("message", {}).get("content", [])
            if isinstance(c, list):
                for b in c:
                    if b.get("type") == "tool_result" and b.get("is_error"): m["tool_errors"] += 1
        elif t == "result":
            u = ev.get("usage", {})
            m.update(turns=ev.get("num_turns", 0), is_error=ev.get("is_error"), result=str(ev.get("result", ""))[:500],
                     input_tokens=u.get("input_tokens", 0), output_tokens=u.get("output_tokens", 0), cache_read_tokens=u.get("cache_read_input_tokens", 0),
                     subtype=ev.get("subtype"), claude_duration_s=ev.get("duration_ms", 0) / 1000.0)
    return m


def git_rev():
    try:
        return subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def run(args):
    exe, ro = claude_paths()
    os.makedirs(args.out, exist_ok=True)
    version = subprocess.run([exe, "--version"], capture_output=True, text=True).stdout.strip()
    meta = {"client": version, "base_url": args.base_url, "model": args.model, "thinking": args.thinking, "max_turns": args.max_turns, "timeout_s": args.timeout,
            "repeats": args.repeats, "tier": args.tier, "revision": git_rev(), "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=2)
    env = {"ANTHROPIC_BASE_URL": args.base_url, "ANTHROPIC_API_KEY": args.api_key, "ANTHROPIC_MODEL": args.model, "ANTHROPIC_SMALL_FAST_MODEL": args.model,
           "DISABLE_TELEMETRY": "1", "DISABLE_ERROR_REPORTING": "1", "DISABLE_AUTOUPDATER": "1", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
           "API_TIMEOUT_MS": str(args.timeout * 1000)}
    if args.thinking in THINKING:
        env["MAX_THINKING_TOKENS"] = THINKING[args.thinking]
    results = open(os.path.join(args.out, "results.jsonl"), "a")
    for t in load_tasks(args.tier, args.only and args.only.split(",")):
        for rep in range(args.repeats):
            rdir = os.path.join(args.out, t["id"], str(rep))
            shutil.rmtree(rdir, ignore_errors=True)
            work = os.path.join(rdir, "work")
            shutil.copytree(t["dir"] + "/repo", work)
            cmd = ["claude", "-p", t["prompt"], "--output-format", "stream-json", "--verbose", "--max-turns", str(args.max_turns),
                   "--permission-mode", "bypassPermissions", "--model", args.model]
            t0 = time.time()
            rc, out, err, timed_out = run_checked(sandbox_cmd(work, cmd, net=True, extra_env=env, ro_extra=ro), args.timeout)
            wall = time.time() - t0
            open(os.path.join(rdir, "events.jsonl"), "w").write(out)
            open(os.path.join(rdir, "stderr.txt"), "w").write(err[-4000:])
            m = parse_events(os.path.join(rdir, "events.jsonl"))
            passed, vlog = verify(t, work)
            rec = {"task": t["id"], "rep": rep, "passed": passed, "wall_s": round(wall, 1), "claude_rc": rc, "timed_out": timed_out, **m}
            results.write(json.dumps(rec) + "\n")
            results.flush()
            print(f"{'PASS' if passed else 'FAIL'} {t['id']:20s} rep {rep} {wall:6.1f}s turns {m['turns']:3d} tools {m['tool_calls']:3d} (err {m['tool_errors']}) "
                  f"out {m['output_tokens']:6d} tok{' TIMEOUT' if timed_out else ''}", flush=True)
    summary(args.out)


def summary(out):
    rows = [json.loads(l) for l in open(os.path.join(out, "results.jsonl"))]
    meta = json.load(open(os.path.join(out, "meta.json")))
    print(f"\n{meta['model']} @ {meta['base_url']}  client {meta['client']}  thinking {meta['thinking']}  revision {meta['revision']}")
    print(f"{'task':20s} {'pass':>7s} {'wall med s':>11s} {'turns':>6s} {'tools':>6s} {'out tok':>8s} {'think/(think+text)':>19s}")
    ids = sorted({r["task"] for r in rows})
    for tid in ids + [None]:
        rs = [r for r in rows if tid is None or r["task"] == tid]
        th = sum(r["think_chars"] for r in rs)
        tx = sum(r["text_chars"] for r in rs)
        print(f"{tid or 'ALL':20s} {sum(r['passed'] for r in rs):3d}/{len(rs):<3d} {statistics.median(r['wall_s'] for r in rs):11.1f} "
              f"{statistics.mean(r['turns'] for r in rs):6.1f} {statistics.mean(r['tool_calls'] for r in rs):6.1f} "
              f"{statistics.mean(r['output_tokens'] for r in rs):8.0f} {th / max(th + tx, 1):19.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("selftest")
    st.add_argument("--tier"); st.add_argument("--only")
    r = sub.add_parser("run")
    r.add_argument("--base-url", required=True); r.add_argument("--model", required=True); r.add_argument("--out", required=True)
    r.add_argument("--api-key", default="sk-local"); r.add_argument("--tier"); r.add_argument("--only")
    r.add_argument("--repeats", type=int, default=1); r.add_argument("--thinking", default="default", choices=["default", "off", "low", "medium", "high"])
    r.add_argument("--max-turns", type=int, default=30); r.add_argument("--timeout", type=int, default=900)
    s = sub.add_parser("summary"); s.add_argument("dir")
    a = ap.parse_args()
    if a.cmd == "selftest": sys.exit(selftest(a))
    if a.cmd == "run": run(a)
    if a.cmd == "summary": summary(a.dir)


if __name__ == "__main__":
    main()
