#!/usr/bin/env python3
"""Generates parametrized task instances (the same layout as tasks/<id>/, see bench.py) from seeds.

  gen_tasks.py --out DIR [--per-family N] [--seed S] [--families a,b] [--tier quick|standard|hard]

Families: py-bugfix, py-feature, c-fix, js-fix, sql-query, git-conflict, log-analysis, sh-text. Expected values in tests and verifiers are
computed from a reference implementation at generation time, so an instance is a pure function of (family, index, seed).
"""
import argparse, json, os, random, shutil, sqlite3, stat, subprocess, textwrap


def put(d, name, body, mode=None):
    p = os.path.join(d, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if isinstance(body, bytes):
        open(p, "wb").write(body)
    else:
        open(p, "w").write(textwrap.dedent(body).lstrip("\n"))
    if mode: os.chmod(p, mode)


def emit(out, tid, tier, prompt, repo, verify, solution, hidden=None, tags=()):
    d = os.path.join(out, tid)
    shutil.rmtree(d, ignore_errors=True)
    for n, b in repo.items(): put(d + "/repo", n, b)
    for n, b in (hidden or {}).items(): put(d + "/hidden", n, b)
    put(d, "verify.sh", "#!/bin/bash\n" + textwrap.dedent(verify).lstrip("\n"), 0o755)
    put(d, "solution.sh", "#!/bin/bash\n" + textwrap.dedent(solution).lstrip("\n"), 0o755)
    json.dump({"id": tid, "tier": tier, "prompt": prompt, "tags": list(tags)}, open(d + "/task.json", "w"), indent=2)


# ---------------------------------------------------------------- py-bugfix
BUGS = {
    "moving_average": ('''
def {f}(xs, n):
    out = []
    for i in range(len(xs) - n + 1):
        out.append(sum(xs[i:i + n]) / n)
    return out
''', "range(len(xs) - n + 1)", "range(len(xs) - n)",
     lambda r: ([r.randint(-9, 30) for _ in range(r.randint(4, 9))], r.randint(1, 3))),
    "median": ('''
def {f}(xs):
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2
''', "return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2", "return s[m]",
     lambda r: ([r.randint(0, 50) for _ in range(r.randint(3, 9))],)),
    "run_length": ('''
def {f}(s):
    out = []
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out.append((s[i], j - i))
        i = j
    return out
''', "while i < len(s):", "while i < len(s) - 1:",
     lambda r: ("".join(r.choice("aab") * r.randint(1, 4) for _ in range(r.randint(2, 6))),)),
    "bsearch": ('''
def {f}(xs, target):
    lo, hi = 0, len(xs) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if xs[mid] == target:
            return mid
        if xs[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
''', "while lo <= hi:", "while lo < hi:",
     lambda r: (sorted(r.sample(range(0, 60), r.randint(4, 9))), r.randint(0, 60))),
    "flatten": ('''
def {f}(x):
    out = []
    for item in x:
        if isinstance(item, list):
            out.extend({f}(item))
        else:
            out.append(item)
    return out
''', "out.extend({f}(item))", "out.extend(item)",
     lambda r: ([r.randint(0, 9), [r.randint(0, 9), [r.randint(0, 9), [r.randint(0, 9)]]], r.randint(0, 9)],)),
}
NAMES = {"moving_average": ["moving_average", "rolling_mean", "window_avg"], "median": ["median", "middle_value"], "run_length": ["run_length", "rle_encode", "compress_runs"],
         "bsearch": ["bsearch", "find_index", "binary_find"], "flatten": ["flatten", "flat", "unnest"]}


def py_bugfix(out, i, rng, tier):
    key = sorted(BUGS)[i % len(BUGS)]
    good, goodfrag, badfrag, gen = BUGS[key]
    f = rng.choice(NAMES[key])
    mod = rng.choice(["utils", "helpers", "calc", "seqtools"])
    gsrc = good.format(f=f).lstrip("\n")
    bsrc = gsrc.replace(goodfrag.format(f=f), badfrag.format(f=f))
    ns = {}
    exec(gsrc, ns)
    bns = {}
    exec(bsrc, bns)
    cases = []
    while len(cases) < 5 or not any(ns[f](*c) != bns[f](*c) for c in cases):    # the tests must expose the bug
        cases = [gen(rng) for _ in range(5)]
    body = "\n".join(f"    def test_{k}(self):\n        self.assertEqual({f}(*{args!r}), {ns[f](*args)!r})\n" for k, args in enumerate(cases))
    test = f"import unittest\nfrom {mod} import {f}\n\n\nclass T(unittest.TestCase):\n{body}\n\nif __name__ == \"__main__\":\n    unittest.main()\n"
    emit(out, f"py-bugfix-{i:03d}", tier, f"The tests in test_{mod}.py fail. Find the bug in {mod}.py and fix it. Do not change the tests.",
         {f"{mod}.py": bsrc, f"test_{mod}.py": test},
         f"python3 -m unittest -q test_{mod} >/dev/null 2>&1 || exit 1\n",
         f"cat > {mod}.py <<'EOF'\n{gsrc}EOF\n", tags=("bugfix", "python"))


# ---------------------------------------------------------------- py-feature
FEATURES = {
    "parse_duration": ("parse_duration(text) -> int: seconds in a string like '1h30m', '45s', '2d4h' (units d, h, m, s; any order of magnitude, each at most once); a bare number is seconds; empty string is 0",
                       '''
import re
def parse_duration(text):
    if not text: return 0
    if text.isdigit(): return int(text)
    mult = {"d": 86400, "h": 3600, "m": 60, "s": 1}
    return sum(int(n) * mult[u] for n, u in re.findall(r"(\\d+)([dhms])", text))
''', lambda r: [f"{r.randint(0, 3)}d{r.randint(0, 23)}h", f"{r.randint(1, 59)}m{r.randint(0, 59)}s", str(r.randint(0, 999)), "", f"{r.randint(1, 9)}h"]),
    "chunk": ("chunk(items, n) -> list of lists: consecutive pieces of at most n items; the last piece may be shorter; n < 1 raises ValueError",
              '''
def chunk(items, n):
    if n < 1: raise ValueError("n")
    return [list(items[i:i + n]) for i in range(0, len(items), n)]
''', lambda r: [(list(range(r.randint(0, 11))), r.randint(1, 4)) for _ in range(5)]),
    "word_wrap": ("word_wrap(text, width) -> list of lines: greedy wrapping on spaces, no line longer than width unless a single word is longer; words are separated by single spaces",
                  '''
def word_wrap(text, width):
    lines, cur = [], ""
    for w in text.split():
        if not cur: cur = w
        elif len(cur) + 1 + len(w) <= width: cur += " " + w
        else:
            lines.append(cur); cur = w
    if cur: lines.append(cur)
    return lines
''', lambda r: [(" ".join(r.choice(["alpha", "be", "gamma", "delta", "x", "epsilon"]) for _ in range(r.randint(3, 12))), r.randint(6, 14)) for _ in range(5)]),
    "snake_to_camel": ("snake_to_camel(name) -> str: 'user_id_map' -> 'userIdMap'; leading and trailing underscores are kept; repeated underscores collapse inside the name",
                       '''
def snake_to_camel(name):
    lead = len(name) - len(name.lstrip("_")); trail = len(name) - len(name.rstrip("_"))
    core = name.strip("_")
    parts = [p for p in core.split("_") if p]
    s = parts[0] + "".join(p.capitalize() for p in parts[1:]) if parts else ""
    return "_" * lead + s + "_" * trail
''', lambda r: [r.choice(["", "_"]) + "_".join(r.choice(["user", "id", "map", "get", "x", "value"]) for _ in range(r.randint(1, 4))) + r.choice(["", "_", "__"]) for _ in range(5)]),
}


def py_feature(out, i, rng, tier):
    key = sorted(FEATURES)[i % len(FEATURES)]
    spec, ref, gen = FEATURES[key]
    ns = {}
    exec(textwrap.dedent(ref), ns)
    fn = ns[key]
    cases = gen(rng)
    lines = []
    for c in cases:
        args = c if isinstance(c, tuple) else (c,)
        lines.append(f"assert {key}(*{args!r}) == {fn(*args)!r}, {args!r}")
    hidden = "from impl import " + key + "\n" + "\n".join(lines) + "\n"
    emit(out, f"py-feature-{i:03d}", tier,
         f"Implement {spec} in impl.py, and add unit tests for it in test_impl.py (unittest). Run the tests.",
         {"impl.py": f"# implement the function described in the task here\n", "README.md": "Small utility module.\n"},
         "python3 hidden_check.py || exit 1\ngrep -q \"unittest\" test_impl.py || exit 1\npython3 -m unittest -q test_impl >/dev/null 2>&1 || exit 1\n",
         f"cat > impl.py <<'EOF'\n{textwrap.dedent(ref).lstrip()}EOF\ncat > test_impl.py <<'EOF'\nimport unittest\nfrom impl import {key}\nclass T(unittest.TestCase):\n    def test_x(self):\n        self.assertEqual({key}(*{(cases[0] if isinstance(cases[0], tuple) else (cases[0],))!r}), {fn(*(cases[0] if isinstance(cases[0], tuple) else (cases[0],)))!r})\nEOF\n",
         hidden={"hidden_check.py": hidden}, tags=("feature", "python", "tests"))


# ---------------------------------------------------------------- c-fix
C_BUGS = [
    ("sum_array", "long sum_array(const int *a, int n)", "long s = 0; for (int i = 0; i <= n; i++) s += a[i]; return s;", "long s = 0; for (int i = 0; i < n; i++) s += a[i]; return s;",
     lambda r: (lambda a: (a, sum(a)))([r.randint(1, 20) for _ in range(r.randint(3, 8))])),
    ("count_char", "int count_char(const char *s, char c)", "int k = 0; for (; *s; s++) if (*s == c) k++; return k + 1;", "int k = 0; for (; *s; s++) if (*s == c) k++; return k;",
     lambda r: (lambda s: (s, s.count("a")))("".join(r.choice("abc") for _ in range(r.randint(5, 14))))),
    ("max_of", "int max_of(const int *a, int n)", "int m = 0; for (int i = 0; i < n; i++) if (a[i] > m) m = a[i]; return m;", "int m = a[0]; for (int i = 1; i < n; i++) if (a[i] > m) m = a[i]; return m;",
     lambda r: (lambda a: (a, max(a)))([-r.randint(1, 20) for _ in range(r.randint(3, 7))])),
]


def c_fix(out, i, rng, tier):
    name, sig, bad, good, gen = C_BUGS[i % len(C_BUGS)]
    data, expect = gen(rng)
    if isinstance(data, list):
        arr = "{" + ", ".join(map(str, data)) + "}"
        call = f"{name}(a, {len(data)})"
        decl = f"    int a[] = {arr};\n"
    else:
        decl = f'    const char *a = "{data}";\n'
        call = f"{name}(a, 'a')"
    lib = lambda body: f"#include \"lib.h\"\n\n{sig}\n{{\n    {body}\n}}\n"
    header = f"#ifndef LIB_H\n#define LIB_H\n{sig};\n#endif\n"
    main = f'#include <stdio.h>\n#include "lib.h"\n\nint main(void)\n{{\n{decl}    long got = (long){call};\n    if (got != {expect}L) {{ printf("FAIL: got %ld, want {expect}\\n", got); return 1; }}\n    printf("ok\\n");\n    return 0;\n}}\n'
    mk = "CC = gcc\nCFLAGS = -Wall -O1\n\ntest: main\n\t./main\n\nmain: main.c lib.c lib.h\n\t$(CC) $(CFLAGS) -o main main.c lib.c\n\nclean:\n\trm -f main\n"
    emit(out, f"c-fix-{i:03d}", tier, "`make test` fails. Find the bug in the C code and fix it. Do not change main.c.",
         {"lib.c": lib(bad), "lib.h": header, "main.c": main, "Makefile": mk},
         "make -s test >/dev/null 2>&1 || exit 1\n", f"cat > lib.c <<'EOF'\n{lib(good)}EOF\n", tags=("bugfix", "c"))


# ---------------------------------------------------------------- js-fix
JS_BUGS = [
    ("sortNumbers", "function sortNumbers(a) { return [...a].sort(); }", "function sortNumbers(a) { return [...a].sort((x, y) => x - y); }", lambda r: [r.randint(1, 120) for _ in range(7)], "sorted ascending numerically"),
    ("sumPositive", "function sumPositive(a) { return a.filter(x => x > 0).reduce((s, x) => s + x); }", "function sumPositive(a) { return a.filter(x => x > 0).reduce((s, x) => s + x, 0); }", lambda r: [-r.randint(1, 9) for _ in range(4)] + [-1], "sum of positive numbers, 0 when there are none"),
    ("dedupe", "function dedupe(a) { return a.filter((x, i) => a.indexOf(x) === i + 1 || a.indexOf(x) === i && false); }", "function dedupe(a) { return a.filter((x, i) => a.indexOf(x) === i); }", lambda r: [r.randint(1, 5) for _ in range(9)], "the array without duplicates, first occurrences kept in order"),
]


def js_fix(out, i, rng, tier):
    name, bad, good, gen, doc = JS_BUGS[i % len(JS_BUGS)]
    data = gen(rng)
    ref = {"sortNumbers": sorted(data), "sumPositive": sum(x for x in data if x > 0), "dedupe": list(dict.fromkeys(data))}[name]
    lib = lambda src: f"{src}\n\nmodule.exports = {{ {name} }};\n"
    test = f"const test = require('node:test');\nconst assert = require('node:assert');\nconst {{ {name} }} = require('./lib');\n\ntest('{name}', () => {{\n  assert.deepStrictEqual({name}({json.dumps(data)}), {json.dumps(ref)});\n}});\n"
    emit(out, f"js-fix-{i:03d}", tier, f"`node --test` fails. {name} should return {doc}. Fix the bug in lib.js; do not change the test.",
         {"lib.js": lib(bad), "lib.test.js": test, "package.json": '{"name": "demo", "version": "1.0.0"}\n'},
         "node --test >/dev/null 2>&1 || exit 1\n", f"cat > lib.js <<'EOF'\n{lib(good)}EOF\n", tags=("bugfix", "javascript"))


# ---------------------------------------------------------------- sql-query
SQL_Q = [
    ("Which customer has the highest total of order amounts? Write only the customer name to answer.txt.",
     "SELECT c.name FROM customers c JOIN orders o ON o.customer_id = c.id GROUP BY c.id ORDER BY SUM(o.amount) DESC, c.name LIMIT 1"),
    ("How many orders were placed in the month with the most orders (month as YYYY-MM)? Write '<month> <count>' to answer.txt.",
     "SELECT substr(placed, 1, 7) m, COUNT(*) n FROM orders GROUP BY m ORDER BY n DESC, m LIMIT 1"),
    ("What is the average order amount of customers from the city with the most customers? Write the number rounded to 2 decimals to answer.txt.",
     "SELECT ROUND(AVG(o.amount), 2) FROM orders o JOIN customers c ON c.id = o.customer_id WHERE c.city = (SELECT city FROM customers GROUP BY city ORDER BY COUNT(*) DESC, city LIMIT 1)"),
]


def sql_query(out, i, rng, tier):
    q, sql = SQL_Q[i % len(SQL_Q)]
    import tempfile
    db = os.path.join(tempfile.mkdtemp(), "shop.db")
    con = sqlite3.connect(db)
    con.executescript("CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT, city TEXT); CREATE TABLE orders(id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL, placed TEXT);")
    names = ["Ada", "Bo", "Cy", "Di", "Eli", "Fay", "Gus", "Hal", "Ivy", "Jo", "Kit", "Lou"]
    cities = ["Oslo", "Lima", "Kyiv", "Pune"]
    for k, n in enumerate(rng.sample(names, 9), 1): con.execute("INSERT INTO customers VALUES (?,?,?)", (k, n, rng.choice(cities)))
    for k in range(1, rng.randint(40, 70)): con.execute("INSERT INTO orders VALUES (?,?,?,?)", (k, rng.randint(1, 9), round(rng.uniform(5, 300), 2), f"2024-{rng.randint(1, 6):02d}-{rng.randint(1, 28):02d}"))
    con.commit()
    ans = con.execute(sql).fetchall()[0]
    con.close()
    want = " ".join(str(x) for x in ans)
    emit(out, f"sql-query-{i:03d}", tier, "shop.db is a SQLite database (tables customers and orders). " + q,
         {"shop.db": open(db, "rb").read(), "NOTES.md": "Use the sqlite3 command line or Python.\n"},
         f"[ \"$(tr -s ' \\n' ' ' < answer.txt | sed 's/ *$//')\" = {json.dumps(want)} ] || exit 1\n", f"echo {json.dumps(want)} > answer.txt\n", tags=("sql", "analysis"))


# ---------------------------------------------------------------- git-conflict
def git(cwd, *a):
    subprocess.run(["git", "-C", cwd, *a], check=True, capture_output=True, env={**os.environ, "GIT_AUTHOR_NAME": "gen", "GIT_AUTHOR_EMAIL": "g@x", "GIT_COMMITTER_NAME": "gen", "GIT_COMMITTER_EMAIL": "g@x",
                                                                         "GIT_CONFIG_GLOBAL": "/dev/null"})


def git_conflict(out, i, rng, tier):
    tid = f"git-conflict-{i:03d}"
    d = os.path.join(out, tid, "repo")
    shutil.rmtree(os.path.join(out, tid), ignore_errors=True)
    os.makedirs(d)
    a, b = rng.sample(["retries", "timeout", "workers", "backoff", "limit"], 2)
    va, vb = rng.choice([2, 4, 5, 6, 7, 8, 9]), rng.choice(list(range(10, 30)) + list(range(31, 99)))
    put(d, "settings.conf", f"[main]\nname = demo\n{a} = 3\n{b} = 30\nmode = fast\n")
    git(d, "init", "-q", "-b", "main"); git(d, "add", "."); git(d, "commit", "-q", "-m", "base")
    git(d, "checkout", "-q", "-b", "feature")
    put(d, "settings.conf", f"[main]\nname = demo\n{a} = {va}\n{b} = 30\nmode = fast\n"); git(d, "commit", "-qam", f"feature: raise {a}")
    git(d, "checkout", "-q", "main")
    put(d, "settings.conf", f"[main]\nname = demo\n{a} = 3\n{b} = {vb}\nmode = fast\n")
    put(d, "NOTES.md", "main moved on\n"); git(d, "add", "."); git(d, "commit", "-qm", f"main: change {b}")
    emit_json = {"id": tid, "tier": tier, "prompt": f"Merge the branch 'feature' into 'main'. Both branches changed settings.conf (feature: {a}, main: {b}); keep both changes. Commit the merge.", "tags": ["git", "merge"]}
    json.dump(emit_json, open(os.path.join(out, tid, "task.json"), "w"), indent=2)
    put(os.path.join(out, tid), "verify.sh", f"#!/bin/bash\ngrep -q '^{a} = {va}$' settings.conf && grep -q '^{b} = {vb}$' settings.conf || exit 1\n! grep -q '<<<<' settings.conf || exit 1\n"
        f"[ \"$(git rev-list --merges --count HEAD)\" -ge 1 ] || exit 1\n[ -z \"$(git status --porcelain | grep -v -e '[.]verify[.]sh' -e '[.]solution[.]sh')\" ] || exit 1\n", 0o755)
    put(os.path.join(out, tid), "solution.sh", f"#!/bin/bash\nexport GIT_AUTHOR_NAME=s GIT_AUTHOR_EMAIL=s@x GIT_COMMITTER_NAME=s GIT_COMMITTER_EMAIL=s@x\ngit merge feature >/dev/null 2>&1\n"
        f"sed -i -e '/^<<<<<<<\\|^=======\\|^>>>>>>>/d' settings.conf\ngrep -q '^{a} = {va}$' settings.conf || sed -i 's/^{a} = 3$/{a} = {va}/' settings.conf\n"
        f"python3 - <<'EOF'\nimport re\ns = open('settings.conf').read()\nlines = []\nfor l in s.splitlines():\n    if l.startswith('{a} = ') and l != '{a} = {va}': continue\n    if l.startswith('{b} = ') and l != '{b} = {vb}': continue\n    lines.append(l)\nopen('settings.conf', 'w').write('\\n'.join(lines) + '\\n')\nEOF\n"
        f"git add -A && git commit -qm merge >/dev/null\n", 0o755)


# ---------------------------------------------------------------- log-analysis
def log_analysis(out, i, rng, tier):
    n = rng.randint(700, 1400)
    routes = ["/api/users", "/api/orders", "/api/items", "/health", "/api/login"]
    lines, worst = [], (0, "")
    codes = {}
    for k in range(n):
        rid = f"r{rng.randrange(16 ** 6):06x}"
        status = rng.choices([200, 200, 200, 201, 404, 500, 503], k=1)[0]
        ms = int(rng.lognormvariate(4.2, 0.6))
        if ms > worst[0]: worst = (ms, rid)
        codes[status] = codes.get(status, 0) + 1
        lines.append(f"2024-03-{1 + k * 28 // n:02d}T{(k * 7) % 24:02d}:{(k * 13) % 60:02d}:{(k * 17) % 60:02d}Z {rng.choice(routes)} {status} {ms}ms id={rid} ua=\"{rng.choice(['curl/8.1', 'Mozilla/5.0', 'okhttp/4.9'])}\"")
    q = i % 2
    if q == 0:
        prompt, want = "access.log has one request per line. Which request id had the longest latency? Write only the id to answer.txt.", worst[1]
    else:
        top = max((c for c in codes if c >= 400), key=lambda c: (codes[c], c))
        prompt, want = "access.log has one request per line. Among the status codes of 400 and above, which occurs most often? Write '<code> <count>' to answer.txt.", f"{top} {codes[top]}"
    emit(out, f"log-analysis-{i:03d}", tier, prompt, {"access.log": "\n".join(lines) + "\n"},
         f"[ \"$(tr -s ' \\n' ' ' < answer.txt | sed 's/ *$//')\" = {json.dumps(want)} ] || exit 1\n", f"echo {json.dumps(want)} > answer.txt\n", tags=("analysis", "long-context"))


# ---------------------------------------------------------------- sh-text
def sh_text(out, i, rng, tier):
    depts = ["eng", "ops", "sales", "hr"]
    rows = [(rng.choice(depts), rng.choice(["ann", "bob", "cat", "dan", "eve"]), rng.randint(40, 190) * 1000) for _ in range(rng.randint(15, 30))]
    tot = {}
    for d, _, s in rows: tot.setdefault(d, []).append(s)
    want = "\n".join(f"{d} {len(v)} {sum(v) // len(v)}" for d, v in sorted(tot.items())) + "\n"
    emit(out, f"sh-text-{i:03d}", tier,
         "staff.csv has the columns dept,name,salary (with a header line). Write a shell script summary.sh that prints one line per department, sorted by department name: '<dept> <headcount> <average salary, integer division>'. Run it and save its output as summary.txt.",
         {"staff.csv": "dept,name,salary\n" + "".join(f"{d},{n},{s}\n" for d, n, s in rows)},
         f"bash summary.sh > /tmp/o.txt 2>/dev/null; cmp -s /tmp/o.txt summary.txt || exit 1\n[ \"$(cat summary.txt)\" = \"$(printf %b {json.dumps(want.rstrip())})\" ] || exit 1\n",
         "cat > summary.sh <<'EOF'\n#!/bin/bash\ntail -n +2 staff.csv | awk -F, '{n[$1]++; s[$1]+=$3} END {for (d in n) printf \"%s %d %d\\n\", d, n[d], int(s[d]/n[d])}' | sort\nEOF\nbash summary.sh > summary.txt\n",
         tags=("shell", "text"))


FAMILIES = {"py-bugfix": py_bugfix, "py-feature": py_feature, "c-fix": c_fix, "js-fix": js_fix, "sql-query": sql_query, "git-conflict": git_conflict,
            "log-analysis": log_analysis, "sh-text": sh_text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-family", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--tier", default="standard")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for fam in a.families.split(","):
        for i in range(a.per_family):
            FAMILIES[fam](a.out, i, random.Random(f"{fam}/{a.seed}/{i}"), a.tier)
    print(f"{len(a.families.split(',')) * a.per_family} tasks in {a.out}")


if __name__ == "__main__":
    main()
