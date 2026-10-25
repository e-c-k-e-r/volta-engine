#!/usr/bin/env python3
"""Plain-text prose for calibration from a directory of per-book Whisper transcripts (<dir>/<title>/whisper.json: {"<file>.mp3": {"segments": [{"text": ...}]}}).

  prose_corpus.py --src /mnt/tmp/Audiobooks --out prose.txt [--books 12] [--words 20000] [--seed 1]

Takes a seeded sample of books, a contiguous stretch of --words words from the middle of each (the first and last 5% hold titles, credits and
sponsor text), and writes them separated by blank lines. The output is for local calibration only and does not belong in the repository.
"""
import argparse, json, os, random


def book_text(path):
    d = json.load(open(path))
    out = []
    for key in sorted(d, key=lambda k: (len(k), k)):
        out.extend(s.get("text", "").strip() for s in d[key].get("segments", []))
    return " ".join(t for t in out if t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--books", type=int, default=12)
    ap.add_argument("--words", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    titles = sorted(t for t in os.listdir(a.src) if os.path.exists(os.path.join(a.src, t, "whisper.json")))
    rng.shuffle(titles)
    done, parts = 0, []
    for t in titles:
        if done >= a.books: break
        words = book_text(os.path.join(a.src, t, "whisper.json")).split()
        lo, hi = int(len(words) * 0.05), int(len(words) * 0.95)
        if hi - lo < a.words: continue
        at = rng.randint(lo, hi - a.words)
        parts.append(" ".join(words[at:at + a.words]))
        done += 1
        print(f"{t[:60]:60s} {len(words):7d} words, took {a.words} from {at}")
    open(a.out, "w").write("\n\n".join(parts) + "\n")
    print(f"{done} books, {sum(len(p.split()) for p in parts)} words -> {a.out}")


if __name__ == "__main__":
    main()
