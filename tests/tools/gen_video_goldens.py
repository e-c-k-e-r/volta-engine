#!/usr/bin/env python3
"""Writes tests/data/video_cases.json and tests/data/video_rgb.mp4: frame samples and canvases of transformers' video processors
(video_processing_qwen3_vl.py with cap_pixels_per_frame, video_processing_glm5_next.py, Gemma 4's default sampler; their functions restated
here without torch) and a
three-second clip of solid red, green and blue seconds (FFmpeg's command line)."""
import json, math, os, subprocess
import numpy as np

out = os.path.join(os.path.dirname(__file__), "..", "data")

# Qwen3VLVideoProcessor (video_preprocessor_config.json of the Qwen3.8 checkpoints)
Q = dict(fps=2, min_frames=4, max_frames=768, shortest=4096, longest=25165824, frame_tokens=768, factor=32, tp=2)
# Glm5NextVideoProcessor (processor_config.json of GLM-5.3-Flash)
G = dict(fps=2, max_frames=2048, min_tokens=16, max_tokens=240000, factor=28, tp=2)

def qwen_sample(total, fps):
	n = int(total / fps * Q["fps"])
	n = min(max(n, Q["min_frames"]), Q["max_frames"], total)
	return np.linspace(0, total - 1, n).round().astype(int).tolist()

def glm_sample(total, fps, duration):
	max_frame_idx = total - 1
	duration = duration or round(max_frame_idx / fps) + 1
	max_seconds = int(duration)
	extract_t = min(int(duration * G["fps"]), G["max_frames"])
	duration_per_frame = 1 / fps
	timestamps = [i * duration_per_frame for i in range(total)]
	if total < extract_t:
		frame_indices = np.linspace(0, total - 1, extract_t, dtype=int).tolist()
	else:
		frame_indices = []
		current_second = 0
		inv_fps = 1 / G["fps"]
		for frame_index in range(total):
			if timestamps[frame_index] >= current_second:
				current_second += inv_fps
				frame_indices.append(frame_index)
				if current_second >= max_seconds:
					break
	if len(frame_indices) < extract_t:
		start, end = (0, max(total - 1, 0)) if len(frame_indices) == 0 else (frame_indices[0], frame_indices[-1])
		frame_indices = np.linspace(start, end, extract_t, dtype=int).tolist()
	elif len(frame_indices) > extract_t:
		frame_indices = np.linspace(0, total - 1, extract_t, dtype=int).tolist()
	seen, uniq = set(), []
	for idx in frame_indices:
		if idx not in seen:
			seen.add(idx)
			uniq.append(idx)
	if len(uniq) & 1:
		uniq.append(uniq[-1])
	return uniq

def qwen_canvas(n, h, w):
	f, tf = Q["factor"], Q["tp"]
	per_frame = max(min(Q["frame_tokens"] * f * f, Q["longest"] // n), int(Q["shortest"] * 1.05))
	max_pixels, min_pixels = per_frame * n, Q["shortest"]
	if h < f or w < f:
		scale = max(f / h, f / w)
		h, w = int(h * scale), int(w * scale)
	hb, wb, tb = round(h / f) * f, round(w / f) * f, round(n / tf) * tf
	if tb * hb * wb > max_pixels:
		beta = math.sqrt((n * h * w) / max_pixels)
		hb, wb = max(f, math.floor(h / beta / f) * f), max(f, math.floor(w / beta / f) * f)
	elif tb * hb * wb < min_pixels:
		beta = math.sqrt(min_pixels / (n * h * w))
		hb, wb = math.ceil(h * beta / f) * f, math.ceil(w * beta / f) * f
	return hb, wb, hb, wb

def glm_canvas(n, height, width):
	factor, tf = G["factor"], G["tp"]
	ppt = tf * factor ** 2
	min_pixels, max_pixels = G["min_tokens"] * ppt, G["max_tokens"] * ppt
	align = lambda v, f: math.ceil(v / f) * f
	frames = max(tf, round(n / tf) * tf)
	ah, aw = align(height, factor), align(width, factor)
	if frames * ah * aw < min_pixels:
		scale = math.sqrt(min_pixels / (n * height * width))
		ah, aw = align(max(1, math.ceil(height * scale)), factor), align(max(1, math.ceil(width * scale)), factor)
	if frames * ah * aw > max_pixels:
		low, high, bh, bw = 1, height, factor, factor
		while low <= high:
			ch = (low + high) // 2
			cw = max(1, math.floor(width * ch / height))
			th, tw = align(ch, factor), align(cw, factor)
			if frames * th * tw <= max_pixels:
				bh, bw, low = th, tw, ch + 1
			else:
				high = ch - 1
		ah, aw = bh, bw
	scale = min(ah / height, aw / width)
	if n * height * width >= ppt * G["min_tokens"]:
		scale = min(1.0, scale)
	return ah, aw, max(1, min(ah, math.floor(height * scale))), max(1, min(aw, math.floor(width * scale)))

def gemma_sample(total, num_frames=32):
	# default_sample_indices_fn with Gemma4VideoProcessor's num_frames (it refuses a shorter video; the engine takes all of its frames)
	n = min(num_frames, total)
	return (np.arange(n, dtype=int) * total // n).tolist()

cases = {"qwen_sample": [], "glm_sample": [], "gemma_sample": [], "qwen_canvas": [], "glm_canvas": []}
for total in [300, 32, 20, 1000, 241]:
	cases["gemma_sample"].append({"total": total, "frames": gemma_sample(total)})
for total, fps in [(300, 30), (90, 30), (10, 30), (5000, 25), (61, 29.97), (3, 24), (100000, 30)]:
	cases["qwen_sample"].append({"total": total, "fps": fps, "frames": qwen_sample(total, fps)})
for total, fps, dur in [(300, 30, 10.0), (90, 30, 3.0), (10, 30, 0.34), (5000, 25, 200.0), (61, 29.97, 2.035), (7, 2, 3.5), (30000, 30, 1000.0), (300, 30, 0)]:
	cases["glm_sample"].append({"total": total, "fps": fps, "duration": dur, "frames": glm_sample(total, fps, dur)})
for n, h, w in [(20, 480, 640), (120, 1080, 1920), (4, 240, 320), (768, 720, 1280), (7, 100, 3000), (2, 20, 30), (64, 2160, 3840)]:
	cases["qwen_canvas"].append({"frames": n, "height": h, "width": w, "canvas": qwen_canvas(n, h, w)})
for n, h, w in [(20, 480, 640), (120, 1080, 1920), (4, 240, 320), (2048, 720, 1280), (2, 20, 30), (8, 3000, 500), (600, 2160, 3840)]:
	cases["glm_canvas"].append({"frames": n, "height": h, "width": w, "canvas": glm_canvas(n, h, w)})
with open(os.path.join(out, "video_cases.json"), "w") as f:
	json.dump(cases, f, indent=1)

# 64 x 48, 10 fps, three seconds: red, green, blue
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=red:s=64x48:r=10:d=1", "-f", "lavfi", "-i", "color=c=lime:s=64x48:r=10:d=1",
	"-f", "lavfi", "-i", "color=c=blue:s=64x48:r=10:d=1", "-filter_complex", "[0][1][2]concat=n=3:v=1[v]", "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
	"-crf", "10", "-movflags", "+faststart", os.path.join(out, "video_rgb.mp4")], check=True)
