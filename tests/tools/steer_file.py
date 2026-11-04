#!/usr/bin/env python3
"""Reads and edits the header of a control-vector file (.steer, see inc/engine/runtime/steer_file.h).

  steer_file.py info FILE...          hidden size, layers, scope, entries, recommended scale
  steer_file.py scale FILE X          set the recommended scale (a version 1 file is rewritten as version 2)
  steer_file.py merge OUT A B...      one file with the entries of A, B, ... (same hidden size; header, scope and scale from A; a layer holds at most 4
                                      edit or probe entries, router entries are not counted; version 3 when any input has router entries)
"""
import struct, sys

SCOPES = {1: "prompt", 2: "think", 4: "answer", 6: "generated", 7: "all"}
STEER_ROUTE = 9


def read(path):
    b = open(path, "rb").read()
    if b[:4] != b"VSTR":
        raise SystemExit(f"{path}: not a control-vector file")
    ver, hidden, layers, scope, n = struct.unpack_from("<5i", b, 4)
    off = 24
    scale = 1.0
    if ver >= 2:
        scale = struct.unpack_from("<f", b, off)[0]
        off += 4
    return b, ver, hidden, layers, scope, n, scale, off


def entries(b, ver, hidden, n, off):
    """(layer, kind, par, expert, vector bytes) per entry; version 3 carries an expert id per entry and no vector for router entries."""
    out = []
    for _ in range(n):
        layer, kind, par = struct.unpack_from("<iif", b, off)
        off += 12
        expert = -1
        if ver >= 3:
            expert = struct.unpack_from("<i", b, off)[0]
            off += 4
        vec = b""
        if kind != STEER_ROUTE:
            vec = b[off:off + 4 * hidden]
            off += 4 * hidden
        out.append((layer, kind, par, expert, vec))
    return out


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("info", "scale", "merge"):
        print(__doc__)
        return 2
    if sys.argv[1] == "info":
        for p in sys.argv[2:]:
            _, ver, hidden, layers, scope, n, scale, _ = read(p)
            print(f"{p}: version {ver}, hidden {hidden}, model layers {layers}, scope {SCOPES.get(scope, scope)}, {n} entries, scale {scale:g}")
        return 0
    if sys.argv[1] == "merge":
        out, files = sys.argv[2], sys.argv[3:]
        _, _, hidden0, layers0, scope0, _, scale0, _ = read(files[0])
        all_entries, per_layer = [], {}
        for f in files:
            b, ver, hidden, layers, _, n, _, off = read(f)
            if hidden != hidden0 or layers != layers0:
                raise SystemExit(f"{f}: hidden size or layer count differs from {files[0]}")
            for e in entries(b, ver, hidden, n, off):
                if e[1] != STEER_ROUTE:
                    per_layer[e[0]] = per_layer.get(e[0], 0) + 1
                    if per_layer[e[0]] > 4:
                        raise SystemExit(f"layer {e[0]} would hold more than 4 entries")
                all_entries.append(e)
        v3 = any(e[1] == STEER_ROUTE for e in all_entries)
        body = b"".join(struct.pack("<iif", l, k, p) + (struct.pack("<i", x) if v3 else b"") + v for l, k, p, x, v in all_entries)
        total = len(all_entries)
        open(out, "wb").write(b"VSTR" + struct.pack("<5i", 3 if v3 else 2, hidden0, layers0, scope0, total) + struct.pack("<f", scale0) + body)
        print(f"wrote {out}: {total} entries from {len(files)} files")
        return 0
    path, x = sys.argv[2], float(sys.argv[3])
    b, ver, hidden, layers, scope, n, _, off = read(path)
    open(path, "wb").write(b"VSTR" + struct.pack("<5i", max(ver, 2), hidden, layers, scope, n) + struct.pack("<f", x) + b[off:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
