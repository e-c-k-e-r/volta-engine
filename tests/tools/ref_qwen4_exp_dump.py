# Oracle dump for the Qwen3.8-Flash-Next port: runs the exllamav3 reference on a fixed prompt and saves module and block-internal
# activations (module index: 0 embed, 1 expand streams, 2 block 0, 3 PLE, 4..8 blocks 1..5; b2 = layer 0 GDN block, b6 = layer 3 QSA block)
# plus ids.npy and logits.npy to results/flash_next_ref/. Run from ~/Programs/exllamav3 with its venv: EXL3_NO_TRITON=1 taskset -c 0-9 .venv/bin/python <this file>
import sys, time, os, torch, numpy as np
sys.path.insert(0, "/home/mrq/Programs/exllamav3")
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, GreedySampler
OUT = os.environ.get("FN_OUT", "/home/mrq/Programs/volta-engine/results/flash_next_ref")
LONG = int(os.environ.get("FN_LONG", "0"))   # N > 0: a ~N-token filler prompt (QSA selection kicks in past 2052 tokens); only ids and logits are saved
def main():
    os.makedirs(OUT, exist_ok = True)
    d = "/home/mrq/Documents/models/Swift-Qwen3.8-Flash-Next-exl3-4.00bpw"
    config = Config.from_directory(d)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = 12288)
    model.load(max_chunk_size = 2048)
    tok = Tokenizer.from_config(config)
    text = "<|im_start|>user\nThe quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.<|im_end|>\n<|im_start|>assistant\n"
    if LONG:
        body = "".join(f"Note {i}: the weather in district {i * 7 % 101} was mild on day {i % 365}. " for i in range(LONG))
        text = "<|im_start|>user\n" + body + "The secret access code is 84920. " + "".join(f"Note {i}: the weather in district {i * 7 % 101} was mild on day {i % 365}. " for i in range(LONG // 40)) + "\n\nWhat is the secret access code? Answer with the number only.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
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
    keep = [] if LONG else ( list(range(0, len(model.modules))) if os.environ.get("FN_ALL") else list(range(0, 9)) + [len(model.modules) - 2, len(model.modules) - 1] )
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
    for bi in (() if LONG else (2, 6)):
        b = model.modules[bi]
        for sm in b.attn.modules:
            wrapm(f"b{bi}_attn_sub_{sm.key.split('.')[-1]}", sm, "forward")
        wrapm(f"b{bi}_attn_hc_mix", b.attn_hc, "mix")
        wrapm(f"b{bi}_attn_out", b.attn, "forward")
        wrapm(f"b{bi}_attn_hc_apply", b.attn_hc, "apply_")
        wrapm(f"b{bi}_mlp_hc_mix", b.mlp_hc, "mix")
        wrapm(f"b{bi}_mlp_out", b.mlp, "forward")
        wrapm(f"b{bi}_mlp_hc_apply", b.mlp_hc, "apply_")
    # Layer 0's GDN core: conv output, gates and the delta-rule output (first call of each)
    import exllamav3.modules.gated_delta_net as gdn_mod
    first = {}
    orig_conv = gdn_mod.causal_conv1d_update
    def conv_w(*a, **k):
        y = orig_conv(*a, **k)
        if "conv" not in first:
            first["conv"] = True
            np.save(f"{OUT}/gdn0_conv_in.npy", k["mixed_qkv"].detach().float().cpu().numpy()) if "mixed_qkv" in k else None
            np.save(f"{OUT}/gdn0_conv_out.npy", y.detach().float().cpu().numpy())
        return y
    gdn_mod.causal_conv1d_update = conv_w
    orig_rule = gdn_mod.gated_delta_rule_fn
    ncall = [0]
    def rule_w(*a, **k):
        y = orig_rule(*a, **k)
        ncall[0] += 1
        if ncall[0] <= 8 or ncall[0] % 36 == 1:
            bt = k["beta"].detach().float(); gt = k["g"].detach().float()
            print(f"DBG2 ba absmax {None if a0.ba_weight_t is None else float(a0.ba_weight_t.float().abs().max())} filled {a0.ba_weight_filled} | rule call {ncall[0]}: beta shape {tuple(bt.shape)} min {bt.min().item():.4f} max {bt.max().item():.4f} std-over-tokens {bt.reshape(-1, bt.shape[-1]).std(0).max().item():.5f}; g std-over-tokens {gt.reshape(-1, gt.shape[-1]).std(0).max().item():.5f}", flush = True)
        if "rule" not in first:
            first["rule"] = True
            np.save(f"{OUT}/gdn0_rule_beta.npy", k["beta"].detach().float().cpu().numpy())
            np.save(f"{OUT}/gdn0_rule_g.npy", k["g"].detach().float().cpu().numpy())
            np.save(f"{OUT}/gdn0_rule_out.npy", y.detach().float().cpu().numpy())
        return y
    gdn_mod.gated_delta_rule_fn = rule_w
    a0 = model.modules[2].attn
    print("DBG ba_weight_filled", a0.ba_weight_filled, "ba_weight_t absmax", None if a0.ba_weight_t is None else float(a0.ba_weight_t.float().abs().max()),
          "b weight absmax", float(a0.b_proj.inner.get_weight_tensor().float().abs().max()), "shape", tuple(a0.b_proj.inner.get_weight_tensor().shape), flush = True)
    gen = Generator(model, cache, tok)
    job = Job(input_ids = ids, max_new_tokens = 1, sampler = GreedySampler(), return_logits = True)
    gen.enqueue(job)
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if "logits" in r:
                np.save(f"{OUT}/logits.npy", r["logits"].float().flatten().cpu().numpy())
    job2 = Job(input_ids = ids, max_new_tokens = 1, sampler = GreedySampler())
    gen.enqueue(job2)
    while gen.num_remaining_jobs():
        gen.iterate()
    print("saved", sorted(os.listdir(OUT)), flush = True)
if __name__ == "__main__":
    main()
