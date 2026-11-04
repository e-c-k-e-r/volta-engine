#!/usr/bin/env python3
"""Generates parametrized task instances (the same layout as tasks/<id>/, see bench.py) from seeds.

  gen_tasks.py --out DIR [--per-family N] [--seed S] [--families a,b] [--tier quick|standard|hard]

Families: py-bugfix, py-feature, c-fix, js-fix, sql-query, git-conflict, log-analysis, sh-text, prose (writing tasks over a local document; loose verifiers), and harder ones:
py-multibug (two bugs in different modules of a small package, each exposed by the tests on its own), py-algo (a function with edge cases in its spec, hidden tests including
the errors it must raise), data-reconcile (matching two CSV files by stated rules), log-multi (joining two logs with retried requests). Expected values in tests and verifiers are
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


# ---------------------------------------------------------------- prose
STOP = set("the a an and or of to in is are was were be been it its this that these those for on with as by at from not but if then so we you they he she i do does did can could should would will may might must have has had which who what when where how than into out up down over under about more most some any each other such only also just there their them our your his her my me us no yes all one two".split())
_DOCS = None


def prose_sources():
    """candidate documents: markdown of the user's own projects and man pages, as (label, text) sorted by label"""
    global _DOCS
    if _DOCS is None:
        import glob, gzip, re
        home = os.path.expanduser("~/Programs")
        bad = ("node_modules", ".venv", "site-packages", "third_party", "vendor", "/build", "/models", "/.git/", "/dep/", "/ext/", "LICENSE")
        docs = []
        for p in sorted(glob.glob(home + "/*/docs/**/*.md", recursive=True) + glob.glob(home + "/*/*.md")):
            if any(b in p for b in bad): continue
            t = open(p, errors="ignore").read()
            if 600 <= len(t.split()) <= 5000: docs.append((p.replace(home + "/", ""), t))
        for p in sorted(glob.glob("/usr/share/man/man1/*.1.gz"))[::7]:
            try:
                raw = subprocess.run(["man", "-l", p], capture_output=True, text=True, env={**os.environ, "MANWIDTH": "100"}).stdout
            except OSError:
                continue
            t = re.sub(r".\x08", "", raw)
            if 600 <= len(t.split()) <= 4000: docs.append(("man/" + os.path.basename(p)[:-5], t))
        _DOCS = docs
    return _DOCS


PROSE_KINDS = [
    ("summarize", "Read input.md and write a summary of it as 5 to 8 bullet points to summary.md.", "summary.md", 40, 400),
    ("plain", "Read input.md and rewrite its first two sections in plain language for a reader with no background, in plain.md.", "plain.md", 80, 700),
    ("faq", "Read input.md and write five questions a new reader might ask about it, each followed by an answer taken from the text, to faq.md.", "faq.md", 120, 900),
    ("outline", "Read input.md and write a hierarchical outline of it (headings and sub-points) to outline.md.", "outline.md", 60, 600),
    ("review", "Read input.md as an editor: list the five most important unclear, redundant or contradictory passages, quote each, and suggest a rewrite, in review.md.", "review.md", 150, 1000),
]


def prose(out, i, rng, tier):
    import re, collections
    docs = prose_sources()
    label, text = docs[(i * 7 + rng.randrange(len(docs))) % len(docs)]
    kind, prompt, outfile, lo, hi = PROSE_KINDS[i % len(PROSE_KINDS)]
    words = [w for w in re.findall(r"[a-z][a-z-]{4,}", text.lower()) if w not in STOP]
    top = [w for w, _ in collections.Counter(words).most_common(12)]
    emit(out, f"prose-{i:03d}", tier, prompt, {"input.md": text},
         f"[ -s {outfile} ] || exit 1\nn=$(wc -w < {outfile}); [ $n -ge {lo} ] && [ $n -le {hi} ] || exit 1\n"
         f"hits=0; for w in {' '.join(top)}; do grep -qi -- \"$w\" {outfile} && hits=$((hits+1)); done; [ $hits -ge 3 ] || exit 1\n",
         f"python3 - <<'EOF'\nwords = {top!r}\nopen('{outfile}', 'w').write(' '.join((words * 40)[:{max(lo, 60) + 20}]) + '\\n')\nEOF\n",
         tags=("prose", kind, label))


# ---------------------------------------------------------------- hard families: several files, a spec with edge cases, a join of two sources
INV_GOOD = {
    "inventory/__init__.py": "",
    "inventory/parse.py": '''
def parse_line(line):
    """'sku,qty,price' -> (SKU upper-cased, int qty, float price); blank lines and lines starting with # give None"""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    sku, qty, price = [p.strip() for p in line.split(",")]
    return sku.upper(), int(qty), float(price)


def parse(text):
    return [r for r in (parse_line(l) for l in text.splitlines()) if r is not None]
''',
    "inventory/stock.py": '''
class Stock:
    """Lots per SKU, oldest first; shipping takes from the oldest lots (FIFO)."""

    def __init__(self):
        self.lots = {}

    def receive(self, sku, qty, price):
        self.lots.setdefault(sku, []).append([qty, price])

    def ship(self, sku, qty):
        """Returns the cost of the shipped units; raises ValueError (and ships nothing) when fewer than qty are on hand"""
        lots = self.lots.get(sku, [])
        if sum(q for q, _ in lots) < qty:
            raise ValueError(sku)
        cost = 0.0
        while qty > 0:
            take = min(qty, lots[0][0])
            cost += take * lots[0][1]
            lots[0][0] -= take
            qty -= take
            if lots[0][0] == 0:
                lots.pop(0)
        return cost

    def on_hand(self, sku):
        return sum(q for q, _ in self.lots.get(sku, []))

    def value(self):
        return sum(q * p for lots in self.lots.values() for q, p in lots)
''',
    "inventory/report.py": '''
from inventory.parse import parse
from inventory.stock import Stock


def run(receipts_text, orders):
    """receipts_text: 'sku,qty,price' lines; orders: (sku, qty) pairs, filled in order (SKUs case-insensitive).
    Returns cogs and value rounded to cents, the SKUs of the orders that could not be filled (in order) and the units on hand per SKU (sorted)."""
    s = Stock()
    for sku, qty, price in parse(receipts_text):
        s.receive(sku, qty, price)
    cogs, short = 0.0, []
    for sku, qty in orders:
        try:
            cogs += s.ship(sku.upper(), qty)
        except ValueError:
            short.append(sku.upper())
    return {"cogs": round(cogs, 2), "value": round(s.value(), 2), "short": short, "on_hand": {k: s.on_hand(k) for k in sorted(s.lots)}}
''',
}
INV_BUGS = [   # (file, good fragment, bad fragment)
    ("inventory/parse.py", "return sku.upper(), int(qty), float(price)", "return sku, int(qty), float(price)"),
    ("inventory/parse.py", 'if not line or line.startswith("#"):', "if not line:"),
    ("inventory/stock.py", "cost += take * lots[0][1]", "cost += qty * lots[0][1]"),
    ("inventory/stock.py", "if sum(q for q, _ in lots) < qty:", "if sum(q for q, _ in lots) <= qty:"),
    ("inventory/report.py", "cogs += s.ship(sku.upper(), qty)", "cogs += s.ship(sku, qty)"),
    ("inventory/report.py", "short.append(sku.upper())", "short.append(sku)"),
]


def inv_load(files):
    import sys, types
    mods = {}
    pkg = types.ModuleType("inventory"); pkg.__path__ = []
    saved = {k: sys.modules.get(k) for k in ("inventory", "inventory.parse", "inventory.stock", "inventory.report")}
    sys.modules["inventory"] = pkg
    try:
        for name in ("parse", "stock", "report"):
            m = types.ModuleType("inventory." + name)
            sys.modules["inventory." + name] = m
            setattr(pkg, name, m)
            exec(textwrap.dedent(files[f"inventory/{name}.py"]), m.__dict__)
            mods[name] = m
    finally:
        for k, v in saved.items():
            if v is None: sys.modules.pop(k, None)
            else: sys.modules[k] = v
    return mods["report"].run


def inv_case(rng):
    skus = rng.sample(["ab12", "cx9", "dd4", "ef70", "gh3"], 3)
    lines = ["# receipts"]
    for _ in range(rng.randint(5, 8)):
        sku = rng.choice(skus)
        lines.append(f"{rng.choice([sku, sku.upper()])}, {rng.randint(1, 9)}, {rng.choice([1.25, 2.5, 3.1, 4.75, 7.2])}")
        if rng.random() < 0.3: lines.append("")
    lines.insert(rng.randint(1, len(lines)), "# late delivery, re-counted")
    orders = [(rng.choice([s, s.upper()]), rng.randint(1, 12)) for s in (rng.choice(skus) for _ in range(rng.randint(3, 6)))]
    return "\n".join(lines) + "\n", orders


def py_multibug(out, i, rng, tier):
    files = {k: textwrap.dedent(v).lstrip("\n") for k, v in INV_GOOD.items()}
    while True:
        b1, b2 = rng.sample(range(len(INV_BUGS)), 2)
        if INV_BUGS[b1][0] != INV_BUGS[b2][0]: break
    def with_bugs(ids):
        f = dict(files)
        for b in ids:
            path, good, bad = INV_BUGS[b]
            assert good in f[path]
            f[path] = f[path].replace(good, bad)
        return f
    good_run, only1, only2 = inv_load(files), inv_load(with_bugs([b1])), inv_load(with_bugs([b2]))
    def differs(f, c, e):
        try:
            return f(*c) != e
        except Exception:
            return True
    for _ in range(500):   # cases where each bug on its own changes a result, so both have to be found
        cases = [inv_case(rng) for _ in range(6)]
        exp = [good_run(*c) for c in cases]
        if any(differs(only1, c, e) for c, e in zip(cases, exp)) and any(differs(only2, c, e) for c, e in zip(cases, exp)): break
    else:
        raise SystemExit("py-multibug: no exposing cases")
    body = "\n".join(f"    def test_run_{k}(self):\n        self.assertEqual(run({c[0]!r}, {c[1]!r}), {e!r})\n" for k, (c, e) in enumerate(zip(cases, exp)))
    test = f"import unittest\nfrom inventory.report import run\n\n\nclass T(unittest.TestCase):\n{body}\n\nif __name__ == \"__main__\":\n    unittest.main()\n"
    repo = with_bugs([b1, b2])
    repo["test_inventory.py"] = test
    repo["README.md"] = "Inventory costing: receipts are parsed (inventory/parse.py), lots are kept per SKU and shipped FIFO (inventory/stock.py), and report.run ties them together.\n"
    sol = "".join(f"cat > {p} <<'EOF'\n{files[p]}EOF\n" for p in ("inventory/parse.py", "inventory/stock.py", "inventory/report.py"))
    emit(out, f"py-multibug-{i:03d}", tier, "The tests in test_inventory.py fail. Find and fix the bugs in the inventory package (there may be more than one). Do not change the tests.",
         repo, "python3 -m unittest -q test_inventory >/dev/null 2>&1 || exit 1\n", sol, tags=("bugfix", "python", "multi-file"))


def topo_case(r, cycle):
    """A dependency map over a few names; with cycle, one edge back from an early task to a later one closes a loop"""
    names = r.sample(list("abcdefghij"), r.randint(3, 9))
    deps = {}
    for n in r.sample(names, len(names) - r.randint(0, 2)):
        earlier = [m for m in names if m < n]
        deps[n] = sorted(r.sample(earlier, r.randint(0, min(3, len(earlier)))))
    if cycle:
        a, b = sorted(r.sample(names, 2))
        deps.setdefault(b, []); deps[b] = sorted(set(deps[b]) | {a})
        deps.setdefault(a, []); deps[a] = sorted(set(deps[a]) | {b})
    return (deps,)


ALGOS = {
    "merge_intervals": ("merge_intervals(intervals) -> list: intervals are half-open [start, end) pairs of ints; drop empty ones (start >= end), merge the ones that overlap or touch "
                        "([1, 3) and [3, 5) become [1, 5)), and return the result as a list of [start, end] lists sorted by start",
                        '''
def merge_intervals(intervals):
    out = []
    for a, b in sorted((a, b) for a, b in intervals if a < b):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out
''', lambda r: [[(lambda a: [a, a + r.randint(-2, 6)])(r.randint(0, 30)) for _ in range(r.randint(0, 9))] for _ in range(14)]),
    "topo_order": ("topo_order(deps) -> list: deps maps a task name to the list of names it depends on (a name that only appears as a dependency is a task too); return every task once, "
                   "each after all of its dependencies, choosing the alphabetically smallest ready task at every step; raise ValueError when the dependencies contain a cycle",
                   '''
import heapq
def topo_order(deps):
    nodes = set(deps) | {d for ds in deps.values() for d in ds}
    need = {n: set(deps.get(n, [])) for n in nodes}
    users = {n: [] for n in nodes}
    for n, ds in need.items():
        for d in ds: users[d].append(n)
    ready = [n for n in nodes if not need[n]]
    heapq.heapify(ready)
    out = []
    while ready:
        n = heapq.heappop(ready)
        out.append(n)
        for u in users[n]:
            need[u].discard(n)
            if not need[u]: heapq.heappush(ready, u)
    if len(out) != len(nodes): raise ValueError("cycle")
    return out
''', lambda r: [topo_case(r, k % 4 == 3) for k in range(12)]),
    "eval_expr": ("eval_expr(text) -> int: evaluate an integer expression with +, -, *, / (division truncates toward zero, as in C), parentheses, unary minus and spaces, with the usual "
                  "precedence (* and / before + and -, left to right); raise ValueError for a malformed expression and ZeroDivisionError for a division by zero",
                  '''
def eval_expr(text):
    toks, i = [], 0
    while i < len(text):
        c = text[i]
        if c.isspace(): i += 1; continue
        if c.isdigit():
            j = i
            while j < len(text) and text[j].isdigit(): j += 1
            toks.append(int(text[i:j])); i = j; continue
        if c in "+-*/()": toks.append(c); i += 1; continue
        raise ValueError(c)
    pos = 0
    def peek(): return toks[pos] if pos < len(toks) else None
    def take():
        nonlocal pos
        if pos >= len(toks): raise ValueError("end")
        pos += 1
        return toks[pos - 1]
    def atom():
        t = take()
        if t == "-": return -atom()
        if t == "(":
            v = expr()
            if take() != ")": raise ValueError(")")
            return v
        if isinstance(t, int): return t
        raise ValueError(t)
    def term():
        v = atom()
        while peek() in ("*", "/"):
            op = take(); w = atom()
            if op == "*": v *= w
            else:
                if w == 0: raise ZeroDivisionError()
                q = abs(v) // abs(w)
                v = q if (v >= 0) == (w >= 0) else -q
        return v
    def expr():
        v = term()
        while peek() in ("+", "-"):
            op = take(); w = term()
            v = v + w if op == "+" else v - w
        return v
    v = expr()
    if pos != len(toks): raise ValueError("trailing")
    return v
''', lambda r: [r.choice(["7 - 10 / 3", "-7 / 2", "2 * (3 + 4) - -5", "((1+2)*(3-8))/4", "10 - 2 - 3", "-(4 - 9) * 2", "1 + 2 * 3 - 4 / 2",
                           "8 / -3", "(2+3", "4 / (2 - 2)", "3 + * 4", " 12 ", "-(-(-3))", "6 / 4 * 4", "100 / 7 / 2", "2 3"]) for _ in range(12)]),
    "justify": ("justify(text, width) -> list: full justification. Split text on whitespace into words and fill lines greedily (as many words as fit with single spaces); in every line "
                "but the last, widen the gaps so the line is exactly width long, giving the leftmost gaps the extra spaces; a line with a single word, and the last line, are left-aligned "
                "with single spaces and padded with spaces to width; a word longer than width gets a line of its own, unpadded; empty text gives []",
                '''
def justify(text, width):
    words, lines, cur = text.split(), [], []
    for w in words:
        if cur and len(" ".join(cur + [w])) > width:
            lines.append(cur); cur = []
        cur.append(w)
    if cur: lines.append(cur)
    out = []
    for k, ws in enumerate(lines):
        if len(ws) == 1 or k == len(lines) - 1:
            s = " ".join(ws)
            out.append(s + " " * max(0, width - len(s)))
            continue
        gaps, spaces = len(ws) - 1, width - sum(len(w) for w in ws)
        s = ""
        for g, w in enumerate(ws[:-1]):
            s += w + " " * (spaces // gaps + (1 if g < spaces % gaps else 0))
        out.append(s + ws[-1])
    return out
''', lambda r: [(" ".join(r.choice(["a", "to", "the", "quick", "extraordinary", "fox", "jumps", "over", "lazy", "dog", "x"]) for _ in range(r.randint(0, 14))), r.randint(6, 16)) for _ in range(12)]),
}


def py_algo(out, i, rng, tier):
    key = sorted(ALGOS)[i % len(ALGOS)]
    spec, ref, gen = ALGOS[key]
    ns = {}
    exec(textwrap.dedent(ref), ns)
    fn = ns[key]
    lines = []
    for c in gen(rng):
        args = c if isinstance(c, tuple) else (c,)
        try:
            want = ("ok", fn(*args))
        except (ValueError, ZeroDivisionError) as e:
            want = ("raise", type(e).__name__)
        lines.append(f"check({args!r}, {want!r})")
    hidden = (f"from impl import {key}\n"
              "def check(args, want):\n"
              "    try:\n"
              f"        got = ('ok', {key}(*args))\n"
              "    except (ValueError, ZeroDivisionError) as e:\n"
              "        got = ('raise', type(e).__name__)\n"
              "    assert got == want, (args, got, want)\n" + "\n".join(lines) + "\n")
    emit(out, f"py-algo-{i:03d}", tier, f"Implement {spec}. Put it in impl.py and add unit tests for it in test_impl.py (unittest), covering the edge cases. Run the tests.",
         {"impl.py": "# implement the function described in the task here\n", "README.md": "Small utility module.\n"},
         "python3 hidden_check.py || exit 1\ngrep -q \"unittest\" test_impl.py || exit 1\npython3 -m unittest -q test_impl >/dev/null 2>&1 || exit 1\n",
         f"cat > impl.py <<'EOF'\n{textwrap.dedent(ref).lstrip()}EOF\ncat > test_impl.py <<'EOF'\nimport unittest\nimport impl\nclass T(unittest.TestCase):\n    def test_exists(self):\n        self.assertTrue(callable(impl.{key}))\nEOF\n",
         hidden={"hidden_check.py": hidden}, tags=("feature", "python", "spec"))


def data_reconcile(out, i, rng, tier):
    import datetime
    base = datetime.date(2024, 5, 1)
    ledger, bank = [], []
    for k in range(rng.randint(14, 22)):
        amt = rng.choice([1250, 4000, 999, 15075, 2300, 4000, 999, 7710])
        d = base + datetime.timedelta(days=rng.randint(0, 25))
        ledger.append((f"L{k + 1}", d, amt))
        r = rng.random()
        if r < 0.75:
            bank.append((d + datetime.timedelta(days=rng.choice([0, 0, 1, 2, -1, 3])), amt if rng.random() < 0.9 else amt + rng.choice([1, -100])))
    for _ in range(rng.randint(1, 3)):
        bank.append((base + datetime.timedelta(days=rng.randint(0, 27)), rng.choice([999, 4000, 5555])))
    rng.shuffle(bank)
    bank = [(f"B{k + 1}", d, a) for k, (d, a) in enumerate(bank)]
    # reference: ledger rows in date order (then id number); each takes the unmatched bank row with the same amount and the smallest date distance (at most 2 days), ties to the earlier bank date, then the lower id
    used, unmatched_l = set(), []
    num = lambda s: int(s[1:])
    for lid, ld, la in sorted(ledger, key=lambda x: (x[1], num(x[0]))):
        cands = [(abs((bd - ld).days), bd, num(bid), bid) for bid, bd, ba in bank if bid not in used and ba == la and abs((bd - ld).days) <= 2]
        if cands: used.add(min(cands)[3])
        else: unmatched_l.append(lid)
    unmatched_b = [bid for bid, _, _ in bank if bid not in used]
    want = "ledger: " + " ".join(sorted(unmatched_l, key=num)) + "\nbank: " + " ".join(sorted(unmatched_b, key=num))
    fmt = lambda c: f"{c // 100}.{c % 100:02d}"
    lcsv = "id,date,amount\n" + "".join(f"{a},{d.isoformat()},{fmt(c)}\n" for a, d, c in ledger)
    bcsv = "id,booked,amount,memo\n" + "".join(f"{a},{d.strftime('%d/%m/%Y')},{fmt(c)},{rng.choice(['card', 'transfer', 'ref ' + str(rng.randint(100, 999))])}\n" for a, d, c in bank)
    prompt = ("ledger.csv and bank.csv list the same payments from two sides. Match them: take the ledger rows in date order (ties by id number); each matches the not yet matched bank row with "
              "exactly the same amount whose booking date is closest to the ledger date, at most 2 days apart (ties: the earlier booking date, then the lower id number). Note the bank dates are "
              "day/month/year. Write the ids that stay unmatched to answer.txt as two lines, 'ledger: <ids>' and 'bank: <ids>', ids in ascending id number, separated by single spaces.")
    emit(out, f"data-reconcile-{i:03d}", tier, prompt, {"ledger.csv": lcsv, "bank.csv": bcsv},
         f"python3 - <<'EOF' || exit 1\nimport sys\ngot = [' '.join(l.split()) for l in open('answer.txt').read().strip().splitlines() if l.strip()]\nwant = {want.splitlines()!r}\nsys.exit(0 if got == [' '.join(w.split()) for w in want] else 1)\nEOF\n",
         f"cat > answer.txt <<'EOF'\n{want}\nEOF\n", tags=("analysis", "data", "spec"))


def log_multi(out, i, rng, tier):
    ids = [f"req-{rng.randrange(16 ** 5):05x}" for _ in range(rng.randint(60, 110))]
    api, db, final = [], [], {}
    t = 0
    for rid in ids:
        for attempt in range(rng.choice([1, 1, 1, 2, 3])):
            t += rng.randint(1, 40)
            status = rng.choices([200, 500, 502, 404], [6, 2, 1, 1])[0]
            api.append((t, f"{t // 3600:02d}:{t // 60 % 60:02d}:{t % 60:02d} api {rid} status={status} route={rng.choice(['/pay', '/cart', '/user'])}"))
            final[rid] = status
            for _ in range(rng.randint(0, 4)):
                db.append((t + rng.randint(0, 3), f"db query id={rid} ms={rng.randint(2, 400)} table={rng.choice(['orders', 'users', 'items'])}"))
    db.sort(key=lambda x: x[0])
    total = {}
    for _, l in db:
        rid = l.split("id=")[1].split()[0]
        total[rid] = total.get(rid, 0) + int(l.split("ms=")[1].split()[0])
    failed = [r for r in final if final[r] >= 500]
    best = max(failed, key=lambda r: (total.get(r, 0), r))
    want = f"{best} {total.get(best, 0)}"
    prompt = ("api.log has one line per attempt of a request (a client retries a request under the same id, so an id can appear several times; the status of its last line is its final "
              "status). db.log has the database queries of every attempt, with the request id and the time each took. Among the requests whose final status is 500 or above, which one spent "
              "the most database time in total, over all of its attempts? Write '<id> <total ms>' to answer.txt.")
    emit(out, f"log-multi-{i:03d}", tier, prompt, {"api.log": "\n".join(l for _, l in api) + "\n", "db.log": "\n".join(l for _, l in db) + "\n"},
         f"[ \"$(tr -s ' \\n' ' ' < answer.txt | sed 's/ *$//')\" = {json.dumps(want)} ] || exit 1\n", f"echo {json.dumps(want)} > answer.txt\n", tags=("analysis", "join"))


FAMILIES = {"py-bugfix": py_bugfix, "py-feature": py_feature, "c-fix": c_fix, "js-fix": js_fix, "sql-query": sql_query, "git-conflict": git_conflict,
            "log-analysis": log_analysis, "sh-text": sh_text, "prose": prose,
            "py-multibug": py_multibug, "py-algo": py_algo, "data-reconcile": data_reconcile, "log-multi": log_multi}


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
