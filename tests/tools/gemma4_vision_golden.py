# Gemma 4 vision golden for test_gemma4_vision: exllamav3's preprocessing (pixel values in 0..1, raster patches) and image embeddings for one image
#   <exllamav3 venv>/bin/python -I tests/tools/gemma4_vision_golden.py <model dir> <image> <golden dir>
import sys, os, struct
import numpy as np, torch
sys.path.insert(0, os.environ.get("EXLLAMAV3_DIR", os.path.expanduser("~/Programs/exllamav3")))
from PIL import Image
from exllamav3 import Config, Model, Tokenizer
model_dir, image_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
cfg = Config.from_directory(model_dir)
vm = Model.from_config(cfg, component="vision")
vm.load(device="cuda:0", progressbar=False)
tok = Tokenizer.from_config(cfg)
img = Image.open(image_path)
pv, pos, n_soft, size = vm.preprocess(img)
pv.float().numpy().astype("float32").tofile(os.path.join(out, "patches.f32"))
mme = vm.get_image_embeddings(tok, img)
emb = mme.embeddings.float().numpy()
emb.astype("float32").tofile(os.path.join(out, "emb.f32"))
with open(os.path.join(out, "meta.txt"), "w") as f:
    f.write(f"{size[0]} {size[1]} {pv.shape[1]} {n_soft} {emb.shape[0]} {emb.shape[1]}\n")
print("size", size, "patches", tuple(pv.shape), "soft", n_soft, "emb", emb.shape, "max", float(np.abs(emb).max()))
