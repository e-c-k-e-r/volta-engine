"""Writes <model>/engram_hash.bin: the tables DeepSeek-V4.1's Engram n-gram hashing needs (inference/engram.py), so the engine can hash
token ids without the tokenizer's normalizers or numpy's generator.

Layout (little endian):
  char[4] "DSEH", int32 version (1), vocab V, engram layers L, max_ngram_size N, heads, compressed pad id
  int32 layer_ids[L]
  int32 token_map[V]                    token id -> compressed id
  int64 multipliers[L][N]
  int64 primes[L][N - 1][heads]
  int64 offsets[L][(N - 1) * heads]     first table row of each (n-gram size, head) bucket range
For position p with compressed ids c[p - s] (s < N; the pad id before the sequence start), layer l:
  rolling = c[p] * mult[l][0]; for i in 1..N-1: rolling ^= c[p - i] * mult[l][i]; row[(i - 1) * heads + h] = rolling % primes[l][i - 1][h] + offsets[l][...]

Run with the exllamav3 venv: python tests/tools/dsv41/engram_hash_tables.py <model dir>"""
import json
import os
import struct
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engram as E  # noqa: E402
import model as M  # noqa: E402


def plain_hash(ids, token_map, mult, primes, offsets, pad, N, heads):
    """The hash as the engine computes it, one position at a time."""
    out = []
    comp = [int( token_map[t] ) for t in ids]
    for p in range(len(ids)):
        row = []
        for l in range(len(mult)):
            toks = [comp[p - s] if p - s >= 0 else int( pad ) for s in range(N)]
            rolling = toks[0] * int(mult[l][0])
            cols = []
            for i in range(1, N):
                rolling ^= toks[i] * int(mult[l][i])
                for h in range(heads):
                    cols.append(rolling % int(primes[l][i - 1][h]) + int(offsets[l][(i - 1) * heads + h]))
            row.append(cols)
        out.append(row)
    return np.array(out, dtype=np.int64)


def main():
    model_dir = sys.argv[1]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    cfg = json.load(open(os.path.join(HERE, "config.json")))
    names = set(M.ModelArgs.__dataclass_fields__)
    args = M.ModelArgs(**{k: v for k, v in cfg.items() if k in names}, max_batch_size=1, max_seq_len=64)
    layout = E.EngramLayout.from_args(args)
    state = E.NgramHashState(args, layout, tok)
    token_map = state.token_map.numpy().astype(np.int32)
    mult = state.multipliers.numpy().astype(np.int64)
    primes = state.primes.numpy().astype(np.int64)
    offsets = state.offsets.numpy().astype(np.int64)
    L, N, H = len(layout.layer_ids), layout.max_ngram_size, layout.n_heads
    assert mult.shape == (L, N) and primes.shape == (L, N - 1, H) and offsets.shape == (L, (N - 1) * H)

    # the plain hash must reproduce the reference module's
    ids = tok.encode("Hello, world! The quick brown fox jumps over the lazy dog. 1234 + 5678 = 6912.", add_special_tokens=True)
    ref = state(torch.tensor([ids]), 0)[0].numpy()
    mine = plain_hash(ids, token_map, mult, primes, offsets, int(state.pad_id), N, H)
    assert np.array_equal(ref, mine), "plain hash differs from NgramHashState"

    path = os.path.join(model_dir, "engram_hash.bin")
    with open(path, "wb") as f:
        f.write(b"DSEH")
        f.write(struct.pack("<6i", 1, len(token_map), L, N, H, int(state.pad_id)))
        f.write(np.array(layout.layer_ids, dtype=np.int32).tobytes())
        f.write(token_map.tobytes())
        f.write(mult.tobytes())
        f.write(primes.tobytes())
        f.write(offsets.tobytes())
    print(f"wrote {path}: vocab {len(token_map)}, layers {layout.layer_ids}, n-gram up to {N}, {H} heads; plain hash matches on {len(ids)} tokens")


if __name__ == "__main__":
    main()
