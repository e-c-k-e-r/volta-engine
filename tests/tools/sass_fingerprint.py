#!/usr/bin/env python3
# Order-independent fingerprint of `cuobjdump -sass` output on stdin: md5 over the sorted (function name, md5 of its
# instruction stream) pairs, and the HMMA count. Instruction text keeps opcodes and operands, drops addresses and encodings,
# so moving source files (which can reorder cubins in the final link) does not change it, but any change to a kernel does.
import hashlib, re, sys

funcs = {}
cur = None
buf = []
hmma = 0
insn = re.compile(r"^\s+/\*[0-9a-f]{4}\*/\s+(.*?);")
for line in sys.stdin:
    if "Function :" in line:
        if cur is not None:
            funcs[cur] = hashlib.md5("\n".join(buf).encode()).hexdigest()
        cur = line.split("Function :")[1].strip()
        buf = []
        continue
    m = insn.match(line)
    if m:
        text = re.sub(r"/\*.*?\*/", "", m.group(1)).strip()
        buf.append(text)
        if "HMMA" in text:
            hmma += 1
if cur is not None:
    funcs[cur] = hashlib.md5("\n".join(buf).encode()).hexdigest()
h = hashlib.md5()
for name in sorted(funcs):
    h.update((name + " " + funcs[name] + "\n").encode())
print(h.hexdigest(), "functions=%d" % len(funcs), "hmma=%d" % hmma)
