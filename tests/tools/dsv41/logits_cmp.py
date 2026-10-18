"""Teacher-forced comparison of test_dsv41's logit dump (dump=<file>, FP16 [positions, vocab]) with dsv41_ref.py's outputs (top8_ids / top8_vals /
lse / nll / ids): top-1 agreement, KL(reference || engine) over the reference's top 8 plus the rest of its mass as one bucket, and both perplexities.
  python tests/tools/dsv41/logits_cmp.py <dump> <dsv41_ref.py --out directory>"""
import math
import sys

import numpy as np

V = 129280


def main():
    dump, ref = sys.argv[1], sys.argv[2]
    ids = np.load(f"{ref}/ids.npy")
    n = len(ids)
    eng = np.memmap(dump, dtype=np.float16, mode="r", shape=(n, V))
    t_ids, t_vals, lse_r = np.load(f"{ref}/top8_ids.npy"), np.load(f"{ref}/top8_vals.npy"), np.load(f"{ref}/lse.npy")
    agree, kls, nll = 0, [], []
    for p in range(n):
        e = eng[p].astype(np.float64)
        lse_e = np.log(np.exp(e - e.max()).sum()) + e.max()
        agree += int(e.argmax() == t_ids[p, 0])
        pr = np.exp(t_vals[p].astype(np.float64) - lse_r[p])
        qe = np.exp(e[t_ids[p]] - lse_e)
        rest_p, rest_q = max(1.0 - pr.sum(), 1e-12), max(1.0 - qe.sum(), 1e-12)
        kls.append(float((pr * (np.log(pr) - np.log(np.maximum(qe, 1e-30)))).sum() + rest_p * (math.log(rest_p) - math.log(rest_q))))
        if p + 1 < n:
            nll.append(lse_e - e[ids[p + 1]])
    nll_r = np.load(f"{ref}/nll.npy")
    kls = np.array(kls)
    print(f"{n} positions: top-1 agreement {agree / n:.4f}, KL mean {kls.mean():.4f} median {np.median(kls):.4f} p99 {np.quantile(kls, 0.99):.4f}; "
          f"ppl engine {math.exp(np.mean(nll)):.4f} reference {math.exp(nll_r.mean()):.4f}")


if __name__ == "__main__":
    main()
