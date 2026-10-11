"""Golden for the H3 Lanczos resize (PIL): a random-but-smooth image, resized down and up.   usage: h3_image_golden.py OUT.safetensors"""
import sys
import numpy as np, torch
from PIL import Image
from safetensors.torch import save_file
rng = np.random.default_rng(3)
small = rng.integers(0, 255, size=(9, 16, 3), dtype=np.uint8)
src = np.array(Image.fromarray(small).resize((613, 347), Image.Resampling.BICUBIC))
down = np.array(Image.fromarray(src).resize((320, 192), Image.Resampling.LANCZOS))
up = np.array(Image.fromarray(src).resize((1280, 736), Image.Resampling.LANCZOS))
save_file({"src": torch.from_numpy(src).contiguous(), "down": torch.from_numpy(down).contiguous(), "up": torch.from_numpy(up).contiguous()}, sys.argv[1])
