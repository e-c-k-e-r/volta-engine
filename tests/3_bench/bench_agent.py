#!/usr/bin/env python3
"""Agent-shaped load against a running volta_server (stdlib only): a main agent and parallel subagents, each a multi-turn conversation whose
prompt grows by a tool result every turn, so every request after the first resumes from the prefix cache.

  phase 1  the main agent takes --main-turns turns alone
  phase 2  --subagents subagents start together and take --sub-turns turns each (the main agent waits)
  phase 3  the main agent takes --main-turns more turns on its grown history

All agents share one system prompt (--system-tokens) and have their own task context (--context-tokens); a turn adds a tool result of
--result-tokens (+/- 40%) and asks for --max-tokens of reply (one in five asks for four times as much: a plan or a patch). The text is
generated from a seed, so a run is repeatable. Token counts are steered by the server's own usage figures (characters per token is
learned from the first answers).

Per request it records the time to the first token as the client sees it (queue wait included) and, from /api/stats before and after, the
server's split of the time before the first token: queue, prefix restore, prefill kernels, checkpoint stores, and what is left (other
slots' decode steps taking turns with the prefill).

  bench_agent.py --url http://127.0.0.1:8000 --subagents 5 --out results/agent.json
"""
import argparse, importlib.util, json, os, random, statistics, sys, threading, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("bench_serve", os.path.join(HERE, "bench_serve.py"))
bench_serve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench_serve)

WORDS = ("buffer cache slot stream kernel launch graph token prefix block queue rank expert layer router tensor shard "
         "decode prefill merge split fence barrier handle config option driver driver loader parser tokenizer window gap "
         "latency budget limit offset index table score vector matrix scale norm gate bias weight").split()


class Gen:
	"""Repeatable filler: code-like lines and log lines, sized in tokens from a learned characters-per-token ratio."""
	cpt = 3.0
	lock = threading.Lock()

	def __init__(self, seed):
		self.rng = random.Random(seed)

	def text(self, tokens):
		chars = int(tokens * Gen.cpt)
		out, n = [], 0
		while n < chars:
			r = self.rng
			if r.random() < 0.6:
				a, b, c = r.choice(WORDS), r.choice(WORDS), r.choice(WORDS)
				line = f"    int {a}_{b}_{r.randrange(1000)} = {c}_{r.choice(WORDS)}({a}, {r.randrange(4096)}) + {r.randrange(99)};  // {r.choice(WORDS)} {r.choice(WORDS)}"
			else:
				line = f"[{r.randrange(24):02d}:{r.randrange(60):02d}:{r.randrange(60):02d}] {r.choice(WORDS)} {r.choice(WORDS)} took {r.randrange(9000)} us (rank {r.randrange(4)}, layer {r.randrange(48)})"
			out.append(line)
			n += len(line) + 1
		return "\n".join(out)

	@classmethod
	def learn(cls, text_chars, prompt_tokens):
		if prompt_tokens and prompt_tokens > 2000:
			with cls.lock:
				cls.cpt = 0.7 * cls.cpt + 0.3 * (text_chars / prompt_tokens)


class Agent:
	def __init__(self, name, seed, system, context_tokens, args):
		self.name, self.args = name, args
		self.gen = Gen(seed)
		self.messages = [{"role": "system", "content": system},
			{"role": "user", "content": f"Task for {name}. Context follows.\n" + self.gen.text(context_tokens)}]
		self.turn = 0

	def step(self, base, model, log, lock):
		a = self.args
		if self.turn:
			tokens = int(a.result_tokens * self.gen.rng.uniform(0.6, 1.4))
			self.messages.append({"role": "user", "content": f"Tool result {self.turn}:\n" + self.gen.text(tokens)})
		self.turn += 1
		max_tokens = a.max_tokens * (4 if self.gen.rng.random() < 0.2 else 1)
		chars = sum(len(m["content"]) for m in self.messages)
		t0 = time.perf_counter()
		res = bench_serve.stream_request(base, model, self.messages, bench_serve.GREEDY, max_tokens, False)
		Gen.learn(chars, res["prompt_tokens"])
		# the reply joins the history as the next turn's assistant message (the text is not needed here: a fixed stand-in keeps runs comparable)
		self.messages.append({"role": "assistant", "content": "Noted; continuing with the next step. " + self.gen.text(max(8, (res["completion_tokens"] or 8) - 8))})
		rec = {"agent": self.name, "turn": self.turn, "t": t0, **res}
		with lock:
			log.append(rec)
		return rec


def run_phase(agents, turns, base, model, log, lock):
	def worker(ag):
		for _ in range(turns):
			ag.step(base, model, log, lock)
	ts = [threading.Thread(target=worker, args=(ag,)) for ag in agents]
	for t in ts: t.start()
	for t in ts: t.join()


def pct(v, p):
	v = sorted(x for x in v if x is not None)
	return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--url", default="http://127.0.0.1:8000")
	ap.add_argument("--model", default="x")
	ap.add_argument("--subagents", type=int, default=5)
	ap.add_argument("--main-turns", type=int, default=3)
	ap.add_argument("--sub-turns", type=int, default=6)
	ap.add_argument("--system-tokens", type=int, default=12000)
	ap.add_argument("--context-tokens", type=int, default=8000)
	ap.add_argument("--result-tokens", type=int, default=2000)
	ap.add_argument("--max-tokens", type=int, default=120)
	ap.add_argument("--seed", type=int, default=1)
	ap.add_argument("--out")
	a = ap.parse_args()

	stats0 = bench_serve.http_json(a.url + "/api/stats")
	system = Gen(a.seed).text(a.system_tokens)
	main_agent = Agent("main", a.seed + 1, system, a.context_tokens, a)
	subs = [Agent(f"sub{i}", a.seed + 10 + i, system, a.context_tokens, a) for i in range(a.subagents)]
	log, lock = [], threading.Lock()
	t0 = time.perf_counter()
	run_phase([main_agent], a.main_turns, a.url, a.model, log, lock)
	run_phase(subs, a.sub_turns, a.url, a.model, log, lock)
	run_phase([main_agent], a.main_turns, a.url, a.model, log, lock)
	wall = time.perf_counter() - t0
	stats1 = bench_serve.http_json(a.url + "/api/stats")

	# the server's own split of the time before the first token, for the requests of this run (the monitor keeps the last 40)
	done0 = stats0["totals"]["completed"]
	new = max(0, min(len(log), stats1["totals"]["completed"] - done0))
	recent = stats1["recent"][:new]
	print(f"{len(log)} requests, {wall:.0f} s wall; the server kept the split of {len(recent)}")
	print("\nclient view (ms)        mean    p50    p90    max")
	for key, label in (("ttft_s", "time to first token"), ("total_s", "request total")):
		v = [r[key] * 1000 for r in log if r.get(key)]
		print(f"  {label:<20} {statistics.mean(v):6.0f} {pct(v, .5):6.0f} {pct(v, .9):6.0f} {max(v):6.0f}")
	print("\nserver view (ms)        mean    p50    p90    max   (from admission to the first token, queue before it)")
	other = [r["ttft_ms"] - r["restore_ms"] - r["prefill_ms"] - r["store_ms"] for r in recent]
	for label, v in (("queue (before slot)", [r["queue_ms"] for r in recent]), ("ttft", [r["ttft_ms"] for r in recent]),
			("  prefix restore", [r["restore_ms"] for r in recent]), ("  prefill kernels", [r["prefill_ms"] for r in recent]),
			("  checkpoint stores", [r["store_ms"] for r in recent]), ("  other slots' steps", other)):
		if v: print(f"  {label:<20} {statistics.mean(v):6.0f} {pct(v, .5):6.0f} {pct(v, .9):6.0f} {max(v):6.0f}")
	p = [r["prompt_tokens"] for r in log if r.get("prompt_tokens")]
	c = [r["cached_tokens"] for r in log if r.get("prompt_tokens")]
	print(f"\nprompts {min(p)} to {max(p)} tokens, {100.0 * sum(c) / sum(p):.0f}% of them from the prefix cache")
	dec = [r["decode_tok_s"] for r in log if r["decode_tok_s"] and r["completion_tokens"] > 20]
	if dec: print(f"decode (requests over 20 tokens): mean {statistics.mean(dec):.0f} tok/s, p10 {pct(dec, .1):.0f}")
	sp = stats1["speculation"]
	if sp.get("rounds"): print(f"speculation: {sp['tokens_per_round']:.2f} tokens per round, {100 * sp['accept_rate']:.0f}% accepted")
	if a.out:
		with open(a.out, "w") as f:
			json.dump({"args": vars(a), "requests": log, "server_recent": recent, "wall_s": wall}, f, indent=1)


if __name__ == "__main__":
	main()
