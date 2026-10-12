#!/usr/bin/env python3
"""Decode-speed benchmark against a running volta_server (stdlib only).

Mirrors the workloads of 1Cat-vLLM's SM70 benchmarks so numbers can sit side by side:
  fixedN    one short sentence tiled to about N prompt tokens (very repetitive: speculation accepts almost everything)
  natural8k four questions behind about 8K tokens of the same filler, thinking off (their natural cohort)
Sampling is either their release contract (T=1, top-p 0.95, top-k 20, seed 0) or greedy.

Per request it reports the prompt tokens (from the usage block), time to first token, decode tok/s from the stream
(first token to last, the prefill excluded) and, from /api/stats before and after, the speculation rounds and tokens per round.
The GPUs' power limit, SM clock and the highest power draw seen during the run are recorded: compare runs only at the same cap.

  bench_serve.py --url http://127.0.0.1:8000 --cases fixed4k,fixed8k,natural8k --repeats 3 --out results/serve_cap250.json
  bench_serve.py --url ... --cases fixed32k --concurrency 4        (their C4 contract: four streams at once, aggregate tok/s)
  bench_serve.py --selftest                                        (against a built-in mock server, no GPU)
"""
import argparse, json, statistics, subprocess, sys, threading, time, urllib.request

SENTENCE = ("This fixed benchmark prompt is used to create a deterministic tokenized input "
            "for single-request decode measurement. ")
COPY_TASK = "\n\nRepeat the text above word for word, over and over, until you are stopped. No commentary."
FILLER_HEAD = "Context document. The following repeated reference text contains no instructions. Use the question at the end.\n"
QUESTIONS = [
	"一辆车先以每小时60公里行驶2小时，再以每小时80公里行驶1.5小时。总路程和全程平均速度分别是多少？说明计算过程。",
	"Write a Python function computing the longest increasing subsequence length in O(n log n). Include three edge-case tests.",
	"Explain how to prove the sum of the first n odd positive integers equals n squared. Give two different arguments.",
	"请解释为什么整数除法处理负数时需要特别小心，给出 Python 和 C 结果不同的例子，并说明余数的约束。",
]
CONTRACT = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": 0}
GREEDY = {"temperature": 0.0}


def http_json(url, body=None, timeout=600):
	req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
		headers={"Content-Type": "application/json"})
	with urllib.request.urlopen(req, timeout=timeout) as r:
		return json.loads(r.read())


class Counter:
	"""Prompt length control through /v1/messages/count_tokens (the chat template's own tokens included)."""
	def __init__(self, base, model):
		self.base, self.model = base, model

	def count(self, text):
		r = http_json(self.base + "/v1/messages/count_tokens",
			{"model": self.model, "messages": [{"role": "user", "content": text}]})
		return r["input_tokens"]

	def fit(self, make, target):
		"""make(n_words) -> text; returns the text whose token count is closest to target."""
		lo_t = self.count(make(1))
		n = max(2, int(target * 0.75))
		best = None
		for _ in range(6):
			text = make(n)
			t = self.count(text)
			if best is None or abs(t - target) < abs(best[1] - target):
				best = (text, t)
			if abs(t - target) <= 2:
				break
			n = max(1, int(n * (target - lo_t) / max(1, t - lo_t)))
		return best


def tile(n_words, sentence=SENTENCE):
	words = sentence.split()
	return " ".join(words[i % len(words)] for i in range(n_words))


def cases(counter, names):
	out = []
	for name in names:
		if name.startswith("fixed"):
			target = int(name[5:].rstrip("kK")) * (1024 if name.lower().endswith("k") else 1)
			# 1Cat's fixed fixture is raw token ids the model keeps repeating; the chat API has no raw prompt, so the tiled text is followed by a copy
			# instruction: the answer is the same repetition, and it runs to the token limit (without the instruction the model answers in a few tokens)
			text, t = counter.fit(lambda n: tile(n) + COPY_TASK, target)
			out.append((name, [{"role": "user", "content": text}], t))
		elif name == "natural8k":
			for i, q in enumerate(QUESTIONS):
				tail = "\nQuestion:\n" + q
				text, t = counter.fit(lambda n: FILLER_HEAD + tile(n) + tail, 8192)
				out.append((f"natural8k/{i}", [{"role": "user", "content": text}], t))
		else:
			sys.exit(f"unknown case {name}")
	return out


def stream_request(base, model, messages, sampling, max_tokens, thinking):
	body = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": True,
		"stream_options": {"include_usage": True}, "enable_thinking": thinking, **sampling}
	req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
		headers={"Content-Type": "application/json"})
	t0 = time.perf_counter()
	first = last = None
	events = 0
	usage = None
	with urllib.request.urlopen(req, timeout=3600) as r:
		for raw in r:
			line = raw.decode().strip()
			if not line.startswith("data:") or line.endswith("[DONE]"):
				continue
			j = json.loads(line[5:])
			if j.get("usage"):
				usage = j["usage"]
			for c in j.get("choices", []):
				d = c.get("delta", {})
				if d.get("content") or d.get("reasoning_content") or d.get("reasoning") or d.get("tool_calls"):
					now = time.perf_counter()
					first = first or now
					last = now
					events += 1
	end = time.perf_counter()
	comp = (usage or {}).get("completion_tokens", events)
	dec = (comp - 1) / (last - first) if first and last and last > first and comp > 1 else 0.0
	return {"prompt_tokens": (usage or {}).get("prompt_tokens"), "cached_tokens":
		((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0),
		"completion_tokens": comp, "ttft_s": (first - t0) if first else None, "decode_tok_s": dec, "total_s": end - t0}


class PowerWatch(threading.Thread):
	def __init__(self):
		super().__init__(daemon=True)
		self.stop = threading.Event()
		self.peak = {}

	def run(self):
		while not self.stop.is_set():
			try:
				o = subprocess.check_output(["nvidia-smi", "--query-gpu=index,power.draw", "--format=csv,noheader,nounits"], text=True, timeout=5)
				for l in o.strip().splitlines():
					i, p = [x.strip() for x in l.split(",")]
					self.peak[i] = max(self.peak.get(i, 0.0), float(p))
			except Exception:
				pass
			self.stop.wait(0.5)


def gpu_state():
	try:
		o = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,power.limit,clocks.max.sm,clocks.sm", "--format=csv,noheader,nounits"], text=True, timeout=5)
		return [l.strip() for l in o.strip().splitlines()]
	except Exception:
		return []


def spec(base):
	try:
		s = http_json(base + "/api/stats", timeout=10)
		return s.get("speculation", {}), s.get("totals", {}), s.get("server", {})
	except Exception:
		return {}, {}, {}


def run_case(base, model, name, messages, tokens, sampling, max_tokens, thinking, conc, repeats):
	rows = []
	for rep in range(repeats):
		s0, t0, _ = spec(base)
		res = [None] * conc
		def go(i):
			res[i] = stream_request(base, model, messages, sampling, max_tokens, thinking)
		th = [threading.Thread(target=go, args=(i,)) for i in range(conc)]
		wall = time.perf_counter()
		[t.start() for t in th]; [t.join() for t in th]
		wall = time.perf_counter() - wall
		s1, t1, _ = spec(base)
		rounds = s1.get("rounds", 0) - s0.get("rounds", 0)
		toks = t1.get("tokens_out", 0) - t0.get("tokens_out", 0)
		rows.append({
			"decode_tok_s": statistics.mean(r["decode_tok_s"] for r in res),
			"aggregate_tok_s": sum(r["decode_tok_s"] for r in res),
			"gen_tokens": statistics.mean(r["completion_tokens"] for r in res),
			"short": any(r["completion_tokens"] < max_tokens // 2 for r in res),
			"ttft_s": statistics.mean(r["ttft_s"] or 0 for r in res),
			"prompt_tokens": res[0]["prompt_tokens"], "completion_tokens": res[0]["completion_tokens"],
			"rounds": rounds, "tokens_per_round": (toks / rounds) if rounds else None,
			"accept_rate": ((s1.get("accepted", 0) - s0.get("accepted", 0)) / max(1, s1.get("proposed", 0) - s0.get("proposed", 0))) if rounds else None,
			"wall_s": wall})
	return {"case": name, "target_tokens": tokens, "runs": rows}


def report(results, meta):
	print(f"\n{meta['model']}  sampling={meta['sampling']}  max_tokens={meta['max_tokens']}  concurrency={meta['concurrency']}")
	for g in meta["gpus"]:
		print("  gpu", g)
	print(f"  peak draw per GPU (W): {meta.get('peak_w')}\n")
	print("| case | prompt tok | gen tok | TTFT s | decode tok/s | aggregate | tok/round | accept | runs |")
	print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
	for r in results:
		R = r["runs"]
		short = sum(1 for x in R if x["short"])
		R = [x for x in R if not x["short"]] or R   # a generation that ended early says nothing about the decode rate: leave it out
		m = lambda k: statistics.mean(x[k] for x in R if x[k] is not None) if any(x[k] is not None for x in R) else None
		f = lambda v, p=1: "-" if v is None else f"{v:.{p}f}"
		sd = statistics.pstdev([x["decode_tok_s"] for x in R]) if len(R) > 1 else 0
		print(f"| {r['case']} | {R[0]['prompt_tokens']} | {f(m('gen_tokens'), 0)} | {f(m('ttft_s'), 2)} | {f(m('decode_tok_s'))} ±{sd:.1f} | {f(m('aggregate_tok_s'))} | {f(m('tokens_per_round'), 2)} | {f(m('accept_rate'), 2)} | {len(R)}{f' (+{short} short)' if short else ''} |")


def selftest():
	"""A mock server that streams a fixed number of tokens, checks the plumbing without a GPU."""
	import http.server
	class H(http.server.BaseHTTPRequestHandler):
		def log_message(self, *a): pass
		def do_GET(self):
			b = json.dumps({"speculation": {"rounds": 0}, "totals": {}, "server": {}} if self.path == "/api/stats" else {"data": [{"id": "mock"}]}).encode()
			self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
		def do_POST(self):
			body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
			if self.path.endswith("count_tokens"):
				text = body["messages"][0]["content"]
				b = json.dumps({"input_tokens": 3 + int(len(text.split()) * 1.2)}).encode()
				self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b); return
			self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
			for i in range(body["max_tokens"]):
				self.wfile.write(("data: " + json.dumps({"choices": [{"delta": {"content": "x"}}]}) + "\n\n").encode()); self.wfile.flush(); time.sleep(0.002)
			self.wfile.write(("data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 4096, "completion_tokens": body["max_tokens"]}}) + "\n\n").encode())
			self.wfile.write(b"data: [DONE]\n\n")
	srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
	threading.Thread(target=srv.serve_forever, daemon=True).start()
	base = f"http://127.0.0.1:{srv.server_address[1]}"
	c = Counter(base, "mock")
	cs = cases(c, ["fixed4k", "natural8k"])
	assert len(cs) == 5 and all(abs(t - (4096 if n == "fixed4k" else 8192)) <= 40 for n, _, t in cs), [(n, t) for n, _, t in cs]
	r = run_case(base, "mock", cs[0][0], cs[0][1], cs[0][2], CONTRACT, 64, False, 2, 2)
	assert len(r["runs"]) == 2 and r["runs"][0]["decode_tok_s"] > 50, r
	print("selftest ok:", [(n, t) for n, _, t in cs][:3], "decode tok/s", round(r["runs"][0]["decode_tok_s"]))


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--url", default="http://127.0.0.1:8000")
	ap.add_argument("--cases", default="fixed4k,fixed8k,natural8k")
	ap.add_argument("--max-tokens", type=int, default=256)
	ap.add_argument("--repeats", type=int, default=3)
	ap.add_argument("--concurrency", type=int, default=1)
	ap.add_argument("--greedy", action="store_true", help="T=0 instead of their T=1 / top-p .95 / top-k 20 / seed 0")
	ap.add_argument("--thinking", action="store_true", help="thinking on (fixed cases default to off, as in their runs)")
	ap.add_argument("--out", help="JSON results file")
	ap.add_argument("--selftest", action="store_true")
	a = ap.parse_args()
	if a.selftest:
		return selftest()
	base = a.url.rstrip("/")
	model = http_json(base + "/v1/models")["data"][0]["id"]
	sampling = GREEDY if a.greedy else CONTRACT
	cs = cases(Counter(base, model), a.cases.split(","))
	watch = PowerWatch(); watch.start()
	gpus = gpu_state()
	_, _, srv = spec(base)
	# one short warm-up so graph capture and lazy workspaces are out of the timed runs
	stream_request(base, model, [{"role": "user", "content": "Say hello."}], GREEDY, 8, False)
	results = [run_case(base, model, n, m, t, sampling, a.max_tokens, a.thinking, a.concurrency, a.repeats) for n, m, t in cs]
	watch.stop.set()
	meta = {"model": model, "server": srv, "sampling": "greedy" if a.greedy else "contract", "max_tokens": a.max_tokens,
		"concurrency": a.concurrency, "gpus": gpus, "peak_w": watch.peak, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
	report(results, meta)
	if a.out:
		json.dump({"meta": meta, "results": results}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
	main()
