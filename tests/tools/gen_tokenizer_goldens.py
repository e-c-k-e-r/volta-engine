#!/usr/bin/env python3
# Regenerates tests/data/tokenizer_cases_<model>.json with the reference HF `tokenizers` output for a corpus of
# awkward texts (code, markdown, unicode, whitespace runs, CRLF, digits, contractions, marks, emoji).
#   <venv with tokenizers>/python tests/tools/gen_tokenizer_goldens.py $ENGINE_MODEL_ROOT [qwen3_5 | glm5_next | qwen4_exp]
# (the corpus samples the repository sources, so regenerating one model does not touch the others' files when named)
import json, os, random, sys
from tokenizers import Tokenizer

root = sys.argv[1]
only = sys.argv[2] if len(sys.argv) > 2 else None
here = os.path.dirname(os.path.abspath(__file__))
repo = os.path.join(here, "..", "..")
rng = random.Random(1234)

texts = [
    "", " ", "  ", "\n", "\n\n\n", " \n ", "a", " a", "  a", "\ta", "hello world", "Hello, World!", "don't I'll we've THEY'RE it's 'tis",
    "x = 1234567 + 89", "3.14159 2,718 1e-9 0xFF 100%", "multi-step -step --flag -O2 a-b_c", "foo.bar(baz)[0]->qux::quux",
    "line1\r\nline2\r\n\r\nline3\rline4", "tab\tseparated\tvalues\t\t", "trailing spaces   \nnext   ", "   leading", "end   ",
    "日本語のテキスト、これはテストです。", "中文 mixed English 한국어 العربية עברית ไทย", "café naïve façade résumé", "é ä क्ष กิ",
    "emoji \U0001F600\U0001F680 flags \U0001F1FA\U0001F1F8 zwj \U0001F468‍\U0001F469‍\U0001F467", "ＡＢＣ １２３ ０９", "٠١٢٣ ४५६", "x y z　w", "​﻿ zero width", "<|im_start|>user\nhi<|im_end|>",
    "<think>\n\n</think>\n\n", "```python\nfor i in range(10):\n    print(i)\n```", "# Title\n\n- item one\n- item two\n\n> quote", "{\"a\": [1, 2.5, null], \"b\": \"c\"}",
    "...!!!???---===", "'''quoted''' \"double\" `tick`", "a'b'c 'a' ''", "$100 €50 £30 ¥20", "~/.config/app/*.json", "/home/mrq/.claude/jobs/ba50c9cf/tmp/cc_work",
]
files = []
for d in ("src", "inc", "tests", ".agents"):
    for dp, _, fs in os.walk(os.path.join(repo, d)):
        for f in fs:
            if f.endswith((".cpp", ".h", ".cu", ".py", ".md", ".sh", ".txt")):
                files.append(os.path.join(dp, f))
files.sort()
for _ in range(90):
    p = rng.choice(files)
    try:
        data = open(p, encoding="utf-8").read()
    except Exception:
        continue
    if len(data) < 50:
        continue
    a = rng.randrange(0, len(data) - 40)
    texts.append(data[a:a + rng.randrange(20, 400)])
alphabet = " \t\n\r.,;:'\"!?-_/\\()[]{}<>@#$%^&*+=|~`0123456789abcXYZéüñ日本語한ก́‍😀"
for _ in range(60):
    texts.append("".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 60))))

for model, out in (("Qwen3.8-27B-exl3", "qwen3_5"), ("GLM-5.3-Flash-exl3-3.05bpw", "glm5_next"), ("Swift-Qwen3.8-Flash-Next-exl3-4.00bpw", "qwen4_exp")):
    if only and only != out:
        continue
    tok = Tokenizer.from_file(os.path.join(root, model, "tokenizer.json"))
    cases = [{"text": t, "ids": tok.encode(t, add_special_tokens=False).ids} for t in texts]
    with open(os.path.join(repo, "tests", "data", "tokenizer_cases_%s.json" % out), "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False)
    print(out, len(cases), "cases", sum(len(c["ids"]) for c in cases), "tokens")
