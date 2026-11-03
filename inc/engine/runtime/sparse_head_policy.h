#pragma once

#include <cstdint>
#include <vector>

namespace engine::runtime {

// Host side of the sparse drafter head (modules/sparse_head, after exllamav3's SparseHead): which vocabulary columns the draft steps use, and
// whether they use them at all.
//  - counts: every generated token adds 1, every prompt token prompt_weight; the counts halve every decay_tokens generated tokens
//  - subset: the top `capacity` ids by count, an incumbent keeping its place until a challenger's count exceeds its own by swap_margin; rebuilt
//    every `interval` generated tokens once min_evidence has been seen, and only extracted again when enough of it changed (an extraction costs
//    a few full-head passes)
//  - use: a target token outside the subset is a draft miss the sparse head caused; the sparse head pays while the miss rate stays below
//    2 * (time saved per round) / (round time), measured from the draft steps of both modes (a full round now and then refreshes the full
//    head's time), with hysteresis around that break-even rate
class SparseHeadPolicy {
public:
	struct Params {
		int capacity = 16384;
		int interval = 2048;
		int decay_tokens = 8192;
		float swap_margin = 0.25f;
		float prompt_weight = 0.2f;
		float min_evidence = 256.0f;
		float min_change = 0.02f;   // share of the subset that must change before it is extracted again
		int full_probe_rounds = 256;   // sparse rounds between full rounds (refreshes the full head's step time)
	};

	void init( int vocab, const Params& p );
	bool active() const { return vocab_ > 0; }

	void observe_prompt( const int* tokens, int n );
	void observe_generated( int token );   // a committed token: counts it and, once a subset exists, scores it as a hit or a miss

	// True when a new subset should be extracted; ranked() then gives the ids, most important first, and set_subset() records what was kept
	bool rebuild_due();
	std::vector<int> ranked() const;
	void set_subset( const std::vector<int>& kept );
	int subset_size() const { return subset_n_; }

	// Timing of a round's draft steps (ms per step) in the mode it ran in, and of the whole round
	void draft_step_time( bool sparse, double ms );
	void round_time( double ms, int draft_steps );
	// Whether the next round drafts through the subset
	bool use_sparse();

	// Counts kept between runs: load() at start (false when the file is absent or for another vocabulary), save() now and then and at exit
	bool load( const char* path );
	bool save( const char* path ) const;
	bool save_due();   // true every 4096 generated tokens

	float miss_rate() const { return static_cast<float>( miss_ ); }
	double full_step_ms() const { return t_full_; }
	double sparse_step_ms() const { return t_sparse_; }

private:
	Params p_;
	int vocab_ = 0;
	std::vector<float> counts_;
	std::vector<uint8_t> in_;
	int subset_n_ = 0;
	double evidence_ = 0.0;
	int since_rebuild_ = 0, since_decay_ = 0;
	bool built_ = false;
	double miss_ = 0.0;
	int scored_ = 0;
	double t_full_ = 0.0, t_sparse_ = 0.0, t_round_ = 0.0, steps_ = 0.0;
	bool on_ = false;
	int sparse_rounds_ = 0;
	int since_save_ = 0;
};

} // namespace engine::runtime
