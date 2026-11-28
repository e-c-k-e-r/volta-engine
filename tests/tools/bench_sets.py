#!/usr/bin/env python3
"""Writes the served-speed test sets of docs/PERFORMANCE.md in agent_replay_run.py's format (a list of {"seg", "messages"}):
tests/data/bench_gsm8k24.json and tests/data/bench_mbpp24.json, 24 problems each of GSM8K's test split and MBPP's full test split drawn with
random.Random(0), and tests/data/bench_chat20.json, 20 single-turn questions and writing requests. The two datasets are read from the Hugging
Face datasets server (openai/gsm8k main/test, google-research-datasets/mbpp full/test), or with --cache DIR from gsm8k_test.json and
mbpp_test.json there (lists of the rows).

  bench_sets.py [--cache DIR]
"""
import argparse, json, os, random, time, urllib.request

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

CHAT = [
	"Explain how a refrigerator moves heat out of its inside, for someone without a physics background.",
	"Write a short, friendly email to a neighbour asking them to water my plants for a week while I travel.",
	"What are the main differences between a Roth IRA and a traditional IRA?",
	"Write a 200-word story about a lighthouse keeper who finds a message in a bottle.",
	"Why does the moon always show the same face to the Earth?",
	"Give me a three-day itinerary for a first visit to Kyoto.",
	"Summarise the causes of the French Revolution in a few paragraphs.",
	"I keep procrastinating on a big project. What are some practical ways to get started?",
	"Compare electric cars and hybrid cars for someone who drives 40 km a day.",
	"Write a haiku sequence of four poems about the changing seasons in a city.",
	"How do vaccines train the immune system?",
	"Draft a polite message declining a job offer after accepting another one.",
	"What is the difference between weather and climate?",
	"Suggest five vegetarian dinners that can be cooked in under thirty minutes, with a short description of each.",
	"Explain what inflation is and why central banks raise interest rates to fight it.",
	"Write a product description for a handmade ceramic coffee mug.",
	"How should I prepare for a job interview for a project manager position?",
	"Why is the sky blue during the day and red at sunset?",
	"Write a short speech for a best friend's wedding, warm and a little funny.",
	"What are the pros and cons of learning a second language as an adult?",
]


def fetch_rows( ds, cfg, split ):
	out, off = [], 0
	while True:
		url = f"https://datasets-server.huggingface.co/rows?dataset={ds}&config={cfg}&split={split}&offset={off}&length=100"
		for _ in range( 5 ):
			try:
				with urllib.request.urlopen( urllib.request.Request( url, headers={ "User-Agent": "bench-set-builder" } ), timeout=60 ) as r:
					j = json.loads( r.read() )
				break
			except Exception:
				time.sleep( 2 )
		else:
			raise SystemExit( "fetch failed: " + url )
		out += [ x["row"] for x in j["rows"] ]
		if len( j["rows"] ) < 100:
			return out
		off += 100


def turns( prompts, seg ):
	return [ { "seg": seg, "messages": [ { "role": "user", "content": p } ] } for p in prompts ]


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument( "--cache", default=None )
	a = ap.parse_args()
	if a.cache:
		gsm = json.load( open( os.path.join( a.cache, "gsm8k_test.json" ) ) )
		mbpp = json.load( open( os.path.join( a.cache, "mbpp_test.json" ) ) )
	else:
		gsm = fetch_rows( "openai/gsm8k", "main", "test" )
		mbpp = fetch_rows( "google-research-datasets/mbpp", "full", "test" )
	g = sorted( random.Random( 0 ).sample( range( len( gsm ) ), 24 ) )
	m = sorted( random.Random( 0 ).sample( range( len( mbpp ) ), 24 ) )
	gp = [ gsm[i]["question"] + "\n\nSolve it step by step, and give the final answer as a number on the last line." for i in g ]
	mp = [ mbpp[i]["text"] + "\n\nYour code should pass these tests:\n" + "\n".join( mbpp[i]["test_list"] ) + "\n\nWrite the Python function." for i in m ]
	for name, prompts in ( ( "bench_gsm8k24.json", gp ), ( "bench_mbpp24.json", mp ), ( "bench_chat20.json", CHAT ) ):
		with open( os.path.join( DATA, name ), "w" ) as f:
			json.dump( turns( prompts, name.split( "_" )[1].split( "." )[0] ), f, indent=1 )
		print( name, len( prompts ) )


if __name__ == "__main__":
	main()
