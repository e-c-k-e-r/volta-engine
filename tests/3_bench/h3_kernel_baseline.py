# Baseline of 1Cat's H3 kernels on the real TP4 shapes. Run with ~/Programs/v100-skinny/.venv-1cat13/bin/python (needs ~/Programs/1Cat-vLLM built).
# usage: CUDA_VISIBLE_DEVICES=0 taskset -c 0-9 python h3_kernel_baseline.py [M]
import sys, time, torch
sys.path.insert(0, "/home/mrq/Programs/1Cat-vLLM")
import vllm._h3_w8a16_C as g
import vllm._h3_flashattn_C as fa

M = int(sys.argv[1]) if len(sys.argv) > 1 else 29440
dev = "cuda"
print(torch.cuda.get_device_name(0), "M", M)

def timeit(fn, n=8):
    for _ in range(2): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n

# TP4 per-card shapes (N out, K in)
shapes = {"qkv": (5376, 5376), "out": (5376, 1792), "fc1": (7168, 5376), "fc2": (5376, 3584),
          "qkv_tp1": (21504, 5376)}
for name, (n, k) in shapes.items():
    x = torch.randn(M, k, device=dev, dtype=torch.float16)
    w = torch.randn(n, k, device=dev, dtype=torch.float16) * 0.02
    fl = 2.0 * M * n * k
    t_cb = timeit(lambda: x @ w.t())
    t_1c = timeit(lambda: g.gemm(x, w, False))
    wc = w.t().contiguous().t()
    try:
        plan = g.ColumnMajorGemmPlan(0, M, n, k, False)
        t_cm = timeit(lambda: plan.run(x, wc)) if plan.supported else float("nan")
    except Exception as ex:
        t_cm = float("nan")
    print(f"{name:8s} N={n:5d} K={k:5d}  cublas {t_cb:7.2f} ms {fl/t_cb/1e9:6.1f} TF | 1cat gemm {t_1c:7.2f} ms {fl/t_1c/1e9:6.1f} TF | colmajor {t_cm:7.2f} ms {fl/t_cm/1e9:6.1f} TF", flush=True)
    del x, w, wc

# attention: B=1, S tokens, H heads (per card at TP4: 14), D=128
for heads in (14, 56):
    q, k, v = [torch.randn(1, M, heads, 128, device=dev, dtype=torch.float16) for _ in range(3)]
    fl = 4.0 * M * M * 128 * heads
    for qt, kt in ((64, 0), (128, 64), (128, 128)):
        try:
            if kt == 0:
                t = timeit(lambda: fa.forward(q, k, v, 128 ** -0.5), n=3)
            else:
                t = timeit(lambda: fa.forward(q, k, v, 128 ** -0.5, kt, qt), n=3)
            print(f"attn H={heads:2d} qtile={qt:3d} ktile={kt:3d}: {t:8.1f} ms {fl/t/1e9:6.1f} TF", flush=True)
        except Exception as ex:
            print("attn fail", heads, qt, kt, str(ex)[:80])
    del q, k, v
print("peak mem GB", torch.cuda.max_memory_allocated() / 1e9)
