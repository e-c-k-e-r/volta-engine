#pragma once

#include "engine/runtime/sampling.h"

#include <mutex>
#include <unordered_map>
#include <vector>

namespace engine::runtime {

// The target's top-k logits at the positions a speculation round emitted, kept per slot while tap_spool is on and written with the request's
// taps when it finishes, so a drafter can be trained against the target's distribution instead of its tokens alone.
class TargetSpool {
public:
	struct Row {
		int pos = 0;                      // the position these logits predict
		Candidates c;
	};
	static TargetSpool& get() {
		static TargetSpool g;
		return g;
	}
	void enable() { on_ = true; }
	bool on() const { return on_; }
	void add( int slot, int pos, const Candidates& c ) {
		std::lock_guard<std::mutex> lock( mtx_ );
		rows_[slot].push_back( Row{ pos, c } );
	}
	std::vector<Row> take( int slot ) {
		std::lock_guard<std::mutex> lock( mtx_ );
		std::vector<Row> out;
		out.swap( rows_[slot] );
		return out;
	}
private:
	bool on_ = false;
	std::mutex mtx_;
	std::unordered_map<int, std::vector<Row>> rows_;
};

} // namespace engine::runtime
