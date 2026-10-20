"""Goldens for the engine's DeepSeek-V4.1 vision tower (tests/2_parity/test_dsv41_vision.cpp): a deterministic test image, the checkpoint
reference's patches for it (image_processor.load_image), and the ViT + aligner output of vision.py with the checkpoint's weights.

  python tests/tools/dsv41/vision_golden.py <model dir> <out dir> [width height]

Writes <out>/image.png, patches.f32 (n_h * n_w rows of 588 floats), emb.f32 (llm_h * llm_w rows of the language hidden size) and
meta.txt (n_h n_w llm_h llm_w hidden)."""
import glob
import io
import json
import os
import struct
import sys
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from image_processor import load_image  # noqa: E402
from vision import Aligner, ViT  # noqa: E402


def make_image(w, h):
    y, x = np.mgrid[0:h, 0:w]
    img = np.stack([(x * 255 // max(w - 1, 1)), (y * 255 // max(h - 1, 1)), ((x + y) * 255 // max(w + h - 2, 1))], axis=-1).astype(np.uint8)
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([w // 8, h // 8, w // 2, h // 2], outline=(255, 255, 255), width=3)
    d.ellipse([w // 2, h // 3, w - w // 8, h - h // 8], fill=(20, 200, 90))
    d.line([0, h - 1, w - 1, 0], fill=(0, 0, 0), width=2)
    return im


def load_weights(model_dir, prefixes):
    from safetensors import safe_open
    out = {}
    for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            keys = [k for k in json.loads(fh.read(n)) if k != "__metadata__" and k.startswith(prefixes)]
        if not keys:
            continue
        with safe_open(f, framework="pt") as sf:
            for k in keys:
                out[k] = sf.get_tensor(k)
    return out


def main():
    model_dir, out_dir = sys.argv[1], sys.argv[2]
    w, h = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (500, 300)
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    vc = cfg["vision_config"]
    args = SimpleNamespace(
        vision_patch_size=vc["patch_size"], vision_dim=vc["hidden_size"], vision_n_heads=vc["num_attention_heads"],
        vision_inter_dim=vc["intermediate_size"], vision_rope_theta=float(vc["rope_theta"]), vision_n_layers=vc["num_hidden_layers"],
        vision_downsample_ratio=vc["downsample_ratio"], vision_max_n_token=vc["max_image_tokens"], vision_min_pixels=vc["min_pixels"],
        vision_max_wh_ratio=vc.get("max_wh_ratio"), dim=cfg["text_config"]["hidden_size"])
    os.makedirs(out_dir, exist_ok=True)
    im = make_image(w, h)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    open(os.path.join(out_dir, "image.png"), "wb").write(buf.getvalue())

    patches, n_h, n_w, llm_h, llm_w = load_image({"data": buf.getvalue()}, args)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(dev)   # the reference builds its rotary tables on the default device
    vit, aligner = ViT(args), Aligner(args)
    sd = load_weights(model_dir, ("vision.", "aligner."))
    vit.load_state_dict({k[len("vision."):]: v.float() for k, v in sd.items() if k.startswith("vision.")}, strict=True)
    aligner.load_state_dict({k[len("aligner."):]: v.float() for k, v in sd.items() if k.startswith("aligner.")}, strict=True)
    vit, aligner = vit.to(dev).to(torch.bfloat16), aligner.to(dev).to(torch.bfloat16)
    for m in vit.modules():   # the norms keep FP32 gains like the reference
        if hasattr(m, "weight") and m.__class__.__name__ == "RMSNorm":
            m.weight.data = m.weight.data.float()
    with torch.inference_mode():
        emb = aligner(vit(patches.to(dev), n_h, n_w), n_h, n_w).float().cpu().numpy()
    # the same in FP32: the BF16 reference is itself noisy on a few rows, and this is the truth both are measured against
    vit32, al32 = vit.float(), aligner.float()
    for m in vit32.modules():
        if hasattr(m, "weight") and m.__class__.__name__ == "RMSNorm":
            m.weight.data = m.weight.data.float()
    with torch.inference_mode():
        emb32 = al32(vit32(patches.to(dev).float(), n_h, n_w), n_h, n_w).float().cpu().numpy()
    cos = (emb * emb32).sum(1) / (np.linalg.norm(emb, axis=1) * np.linalg.norm(emb32, axis=1) + 1e-30)
    print(f"bf16 reference vs fp32: row cosine mean {cos.mean():.5f} median {np.median(cos):.5f} worst {cos.min():.4f}")
    emb32.astype(np.float32).tofile(os.path.join(out_dir, "emb32.f32"))
    patches.float().reshape(n_h * n_w, -1).numpy().astype(np.float32).tofile(os.path.join(out_dir, "patches.f32"))
    emb.astype(np.float32).tofile(os.path.join(out_dir, "emb.f32"))
    open(os.path.join(out_dir, "meta.txt"), "w").write(f"{n_h} {n_w} {llm_h} {llm_w} {emb.shape[1]}\n")
    print(f"image {w}x{h}: patch grid {n_h}x{n_w}, token grid {llm_h}x{llm_w}, embeddings {emb.shape}, |emb| mean {np.abs(emb).mean():.4f}")


if __name__ == "__main__":
    main()
