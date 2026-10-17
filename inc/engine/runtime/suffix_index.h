#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace engine::runtime {

// Token history of one request with an index of its K-grams: the draft source that proposes the tokens which followed the last earlier
// occurrence of the current suffix. Text that copies (file contents, tool-call arguments, identifiers) is predicted without a model pass.
class SuffixIndex {
public:
	// Rebuilds from nothing on a new request
	void reset() {
		toks_.clear();
		last_.clear();
		next_ = -1;
		k_ = 0;
	}

	// Appends the tokens of `all` beyond what the index holds. `k` is the shortest suffix that counts as a match
	void sync( const std::vector<int>& all, int k ) {
		if ( k != k_ || all.size() < toks_.size() ) {
			reset();
			k_ = k;
		}
		for ( size_t i = toks_.size(); i < all.size(); ++i ) append( all[i] );
	}

	// Up to `max_n` tokens that followed the last earlier occurrence of the current K-token suffix, and how long that match is (up to
	// `max_len`). Returns the number of tokens written to `out`.
	int propose( int max_n, int max_len, int* out, int* match_len ) const {
		*match_len = 0;
		if ( next_ < 0 ) return 0;
		const int n = static_cast<int>( toks_.size() );
		int len = 0;
		while ( len < max_len && next_ - 1 - len >= 0 && toks_[next_ - 1 - len] == toks_[n - 1 - len] ) ++len;
		*match_len = len;
		int count = 0;
		while ( count < max_n && next_ + count < n ) {
			out[count] = toks_[next_ + count];
			++count;
		}
		return count;
	}

private:
	void append( int tok ) {
		toks_.push_back( tok );
		next_ = -1;
		const int n = static_cast<int>( toks_.size() );
		if ( k_ <= 0 || n < k_ ) return;
		uint64_t h = 1469598103934665603ull;
		for ( int i = n - k_; i < n; ++i ) h = ( h ^ static_cast<uint64_t>( static_cast<uint32_t>( toks_[i] ) ) ) * 1099511628211ull;
		auto it = last_.find( h );
		if ( it != last_.end() ) next_ = it->second;
		last_[h] = n;
	}

	std::vector<int> toks_;
	std::unordered_map<uint64_t, int> last_;   // hash of a K-gram -> index of the token after its latest occurrence
	int next_ = -1;                            // index after the earlier occurrence of the current suffix
	int k_ = 0;
};

} // namespace engine::runtime
