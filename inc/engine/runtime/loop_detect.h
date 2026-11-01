#pragma once

// Streaming detection of generation loops: the output repeating the same span of tokens, exactly or nearly, with nothing else in between.
// A port of exllamav3's generator/loop_detect.py (upstream 805caee9): every token's k-gram is hashed (rolling hash) and looked up in a table of
// recent k-gram positions; the distance to an earlier occurrence is a candidate period. A run follows one period (drifting by tol * period for
// insertions and deletions, matching up to max_mult copies back) and ends after `gap` unmatched tokens. A run is a loop once it spans min_reps
// periods and min_len tokens (exact repetition: exact_reps and exact_len), enough of it matched, its recent match rate is high, it is not
// transcribing prompt content not generated before (answer options being compared), and, over a tiny vocabulary, it is (near) exact.
// O(1) per token; the tables are allocated once from max_period.

#include <cstdint>
#include <vector>

namespace engine::runtime {

class LoopDetector {
public:
	struct Params {
		int k = 6;               // k-gram length in tokens
		int max_period = 32768;  // longest repeating unit detected
		int min_reps = 8;        // periods an approximate loop spans before it is reported
		int min_len = 8192;      // ... and tokens
		int exact_reps = 4;      // periods an exact loop spans before it is reported
		int exact_len = 1024;    // ... and tokens
		float tol = 0.05f;       // period drift between copies, as a fraction of the period
		int gap = 24;            // consecutive unmatched tokens that end a run
		int max_mult = 3;        // copies back a k-gram may match
		float min_frac = 0.5f;   // matched share of a run
		float min_recent = 0.75f;// recent match rate (exponential average over about one period)
		int lowdiv_len = 2400;   // length a near-exact loop over a tiny vocabulary must reach
		float prompt_frac = 0.25f;// share of a run's unmatched tokens that must be new prompt content for it to count as transcription
		int max_runs = 8;        // candidate periods tracked at once
		int ways_pos = 4;        // recent positions remembered per k-gram
	};

	LoopDetector();
	explicit LoopDetector( const Params& p );

	// Indexes the prompt's k-grams (Bloom filter); call once before feeding generated tokens
	void prime( const int* prompt, int n );
	// Observes one generated token; true once a loop has been found (latches)
	bool feed( int token );
	bool detected() const { return detected_at_ >= 0; }
	int64_t detected_at() const { return detected_at_; }   // index of the generated token at which it was found
	int period() const { return period_; }

private:
	bool in_prompt( uint64_t h ) const;
	bool diverse( int64_t t, int p ) const;
	bool is_loop( int r, int64_t t );

	Params p_;
	uint64_t mask_ = 0, bk_ = 0, h_ = 0;
	std::vector<uint64_t> key_;
	std::vector<int64_t> newest_, pos_;
	std::vector<int8_t> head_;
	std::vector<uint64_t> ring_, hashes_;
	std::vector<int64_t> tokens_;
	int64_t t_ = -1;
	std::vector<char> on_, exempt_;
	std::vector<int64_t> start_, last_, sourced_last_;
	std::vector<int> per_, matched_, unmatched_, sourced_;
	std::vector<double> recent_;
	std::vector<uint8_t> bloom_;
	uint64_t bloom_mask_ = 0;
	int64_t detected_at_ = -1;
	int period_ = 0;
};

} // namespace engine::runtime
