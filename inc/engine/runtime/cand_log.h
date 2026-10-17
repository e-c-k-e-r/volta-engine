#pragma once

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>

namespace engine::runtime {

// Debug log of the DFlash2 drafter's candidates (spec_cand_log): one record per speculation round and one per finished request (its whole token list),
// so a candidate re-ranker can be trained and tested offline. Binary, little endian:
//   round:   int32 0xC0DE0001, slot, pos, anchor, active, valid, then per valid position k: chosen, target, float vals[16], int32 ids[16]
//            (valid = the positions up to the first mismatch: the target's token is known there)
//   request: int32 0xC0DE0002, slot, n_tokens, then n_tokens int32 (prompt and output)
class CandLog {
public:
	static CandLog& get() {
		static CandLog g;
		return g;
	}
	bool open( const std::string& path ) {
		std::lock_guard<std::mutex> lock( mtx_ );
		if ( f_ ) return true;
		f_ = std::fopen( path.c_str(), "wb" );
		return f_ != nullptr;
	}
	bool on() const { return f_ != nullptr; }
	void write( const void* p, size_t n ) {
		std::lock_guard<std::mutex> lock( mtx_ );
		if ( f_ ) std::fwrite( p, 1, n, f_ );
	}
	void flush() {
		std::lock_guard<std::mutex> lock( mtx_ );
		if ( f_ ) std::fflush( f_ );
	}
private:
	std::FILE* f_ = nullptr;
	std::mutex mtx_;
};

} // namespace engine::runtime
