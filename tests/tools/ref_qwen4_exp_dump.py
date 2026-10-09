# Oracle dump for the Qwen3.8-Flash-Next port: runs the exllamav3 reference on a fixed prompt and saves module and block-internal
# activations (module index: 0 embed, 1 expand streams, 2 block 0, 3 PLE, 4..8 blocks 1..5; b2 = layer 0 GDN block, b6 = layer 3 QSA block)
# plus ids.npy and logits.npy to results/flash_next_ref/. Run from ~/Programs/exllamav3 with its venv: EXL3_NO_TRITON=1 taskset -c 0-9 .venv/bin/python <this file>
import sys, time, os, torch, numpy as np
sys.path.insert(0, "/home/mrq/Programs/exllamav3")
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, GreedySampler
OUT = "/home/mrq/Programs/volta-engine/results/flash_next_ref"
def main():
    os.makedirs(OUT, exist_ok = True)
    d = "/home/mrq/Documents/models/Swift-Qwen3.8-Flash-Next-exl3-4.00bpw"
    config = Config.from_directory(d)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = 4096)
    model.load(max_chunk_size = 2048)
    tok = Tokenizer.from_config(config)
    text = "<|im_start|>user\nThe quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.<|im_end|>\n<|im_start|>assistant\n"
    ids = tok.encode(text, add_bos = False)
    S = ids.shape[-1]
    np.save(f"{OUT}/ids.npy", ids.flatten().cpu().numpy().astype(np.int64))
    print("S", S, "modules", len(model.modules), [type(m).__name__ for m in model.modules[:8]], flush = True)
    seen = {}
    def wrap(i, m):
        f = m.forward
        def g(x, params, *a, **k):
            y = f(x, params, *a, **k)
            if i not in seen and hasattr(y, "shape") and y.dim() >= 2 and y.shape[1] in (S, S - 1):
                seen[i] = True
                t = y.detach().float().cpu().numpy()
                np.save(f"{OUT}/mod{i:03d}_{type(m).__name__}.npy", t)
            return y
        m.forward = g
    keep = list(range(0, 9)) + [len(model.modules) - 2, len(model.modules) - 1]
    for i in keep: wrap(i, model.modules[i])
    def grab(tag, y):
        ts = y if isinstance(y, (tuple, list)) else (y,)
        for j, t in enumerate(ts):
            if hasattr(t, "shape") and t.dim() >= 2 and t.shape[1] in (S, S - 1):
                np.save(f"{OUT}/{tag}_{j}.npy", t.detach().float().cpu().numpy())
    def wrapm(tag, obj, name):
        f = getattr(obj, name)
        done = {}
        def g(*a, **k):
            y = f(*a, **k)
            if not done:
                done[1] = True
                grab(tag, y)
            return y
        setattr(obj, name, g)
    for bi in (2, 6):
        b = model.modules[bi]
        wrapm(f"b{bi}_attn_hc_mix", b.attn_hc, "mix")
        wrapm(f"b{bi}_attn_out", b.attn, "forward")
        wrapm(f"b{bi}_attn_hc_apply", b.attn_hc, "apply_")
        wrapm(f"b{bi}_mlp_hc_mix", b.mlp_hc, "mix")
        wrapm(f"b{bi}_mlp_out", b.mlp, "forward")
        wrapm(f"b{bi}_mlp_hc_apply", b.mlp_hc, "apply_")
    gen = Generator(model, cache, tok)
    job = Job(input_ids = ids, max_new_tokens = 1, sampler = GreedySampler(), return_logits = True)
    gen.enqueue(job)
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if "logits" in r:
                np.save(f"{OUT}/logits.npy", r["logits"].float().flatten().cpu().numpy())
    print("saved", sorted(os.listdir(OUT)), flush = True)
if __name__ == "__main__":
    main()
