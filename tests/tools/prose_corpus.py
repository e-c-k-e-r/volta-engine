#!/usr/bin/env python3
"""Plain-text prose for calibration from a directory of per-book Whisper transcripts (<dir>/<title>/whisper.json: {"<file>.mp3": {"segments": [{"text": ...}]}}).

  prose_corpus.py --src /mnt/tmp/Audiobooks --out prose.txt [--books 12] [--words 20000] [--seed 1]

Takes a seeded sample of books, a contiguous stretch of --words words from the middle of each (the first and last 5% hold titles, credits and
sponsor text), and writes them separated by blank lines. The output is for local calibration only and does not belong in the repository.
"""
import argparse, collections, json, os, random


def book_text(path):
    """-> (text, mean word score, share of words scored below 0.5, share of 4-grams that occur more than 3 times)"""
    d = json.load(open(path))
    out, scores = [], []
    for key in sorted(d, key=lambda k: (len(k), k)):
        for seg in d[key].get("segments", []):
            out.append(seg.get("text", "").strip())
            scores.extend(w["score"] for w in seg.get("words", []) if "score" in w)
    text = " ".join(t for t in out if t)
    words = text.split()
    grams = collections.Counter(tuple(words[i:i + 4]) for i in range(len(words) - 3))
    rep = sum(c for c in grams.values() if c > 3) / max(1, sum(grams.values()))
    return text, (sum(scores) / len(scores) if scores else 0.0), (sum(x < 0.5 for x in scores) / len(scores) if scores else 1.0), rep


def has_loop(words, n=6, times=4):
    """a run of one n-word phrase repeated back to back `times` or more times: the signature of a Whisper loop"""
    for i in range(len(words) - n * times):
        if all(words[i + k * n:i + k * n + n] == words[i:i + n] for k in range(1, times)):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--books", type=int, default=12)
    ap.add_argument("--words", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--min-score", type=float, default=0.90, help="minimum mean word confidence of a book")
    ap.add_argument("--max-low", type=float, default=0.02, help="maximum share of words below 0.5 confidence")
    ap.add_argument("--max-rep", type=float, default=0.03, help="maximum share of 4-grams that occur more than 3 times")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    titles = sorted(t for t in os.listdir(a.src) if os.path.exists(os.path.join(a.src, t, "whisper.json")))
    rng.shuffle(titles)
    done, parts = 0, []
    for t in titles:
        if done >= a.books: break
        text, score, low, rep = book_text(os.path.join(a.src, t, "whisper.json"))
        words = text.split()
        lo, hi = int(len(words) * 0.05), int(len(words) * 0.95)
        if hi - lo < a.words: continue
        if score < a.min_score or low > a.max_low or rep > a.max_rep:
            print(f"skip {t[:50]:50s} score {score:.3f} low {low:.3f} rep {rep:.3f}")
            continue
        for _ in range(8):
            at = rng.randint(lo, hi - a.words)
            if not has_loop(words[at:at + a.words]): break
        else:
            print(f"skip {t[:50]:50s} repeated loops in every sampled stretch")
            continue
        parts.append(" ".join(words[at:at + a.words]))
        done += 1
        print(f"{t[:50]:50s} {len(words):7d} words, score {score:.3f}, took {a.words} from {at}")
    open(a.out, "w").write("\n\n".join(parts) + "\n")
    print(f"{done} books, {sum(len(p.split()) for p in parts)} words -> {a.out}")


if __name__ == "__main__":
    main()
