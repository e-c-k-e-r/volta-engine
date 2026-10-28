#!/usr/bin/env python3
"""Reads and edits the header of a control-vector file (.steer, see inc/engine/runtime/steer_file.h).

  steer_file.py info FILE...          hidden size, layers, scope, entries, recommended scale
  steer_file.py scale FILE X          set the recommended scale (version 2; a version 1 file is rewritten as version 2)
"""
import struct, sys

SCOPES = {1: "prompt", 2: "think", 4: "answer", 6: "generated", 7: "all"}


def read(path):
    b = open(path, "rb").read()
    if b[:4] != b"VSTR":
        raise SystemExit(f"{path}: not a control-vector file")
    ver, hidden, layers, scope, n = struct.unpack_from("<5i", b, 4)
    off = 24
    scale = 1.0
    if ver == 2:
        scale = struct.unpack_from("<f", b, off)[0]
        off += 4
    return b, ver, hidden, layers, scope, n, scale, off


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("info", "scale"):
        print(__doc__)
        return 2
    if sys.argv[1] == "info":
        for p in sys.argv[2:]:
            _, ver, hidden, layers, scope, n, scale, _ = read(p)
            print(f"{p}: version {ver}, hidden {hidden}, model layers {layers}, scope {SCOPES.get(scope, scope)}, {n} entries, scale {scale:g}")
        return 0
    path, x = sys.argv[2], float(sys.argv[3])
    b, ver, hidden, layers, scope, n, _, off = read(path)
    open(path, "wb").write(b"VSTR" + struct.pack("<5i", 2, hidden, layers, scope, n) + struct.pack("<f", x) + b[off:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
