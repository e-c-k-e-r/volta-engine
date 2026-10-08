#pragma once
#include "engine/config.h"
#include "engine/runtime/options.h"
#include <cstdio>
#include <cstdlib>

namespace engine::runtime {

// Per-request choice between a speculative round and a plain M=1 decode step. A round emits `tokens` in
// `ms`; a plain step emits one in `dense_ms`. Speculation is kept while its measured rate beats the plain
// rate. The verify forward is weight-bound (its cost is nearly flat in rows), so the depth is never worth
// trimming; the loss case is low acceptance (high-entropy text, T > 0), where a round costs more than the
// tokens it returns. In plain mode a probe round runs every `probe_` steps (backing off to kMaxProbe while
// probes keep losing) so a return to predictable text is picked up within a few tokens.
class SpecController {
public:
	// False = take one plain step instead of a round.
	bool use_spec() {
		if ( !dense_mode_ ) return true;
		if ( ++dense_run_ >= probe_ ) {
			dense_run_ = 0;
			return true;
		}
		return false;
	}

	void observe_spec( int tokens, double ms ) {
		// Running mean until the window fills, then an EMA: tokens per round is heavy-tailed (1..8), so a
		// few early rounds must not decide the mode.
		const double floor_a = dense_mode_ ? 0.3 : 0.1;
		const double a = MAX( floor_a, 1.0 / ( rounds_ + 1 ) );
		e_tok_ = rounds_ ? ( 1.0 - a ) * e_tok_ + a * tokens : static_cast<double>( tokens );
		e_ms_ = rounds_ ? ( 1.0 - a ) * e_ms_ + a * ms : ms;
		++rounds_;
		if ( !enabled() ) return;
		if ( dense_n_ >= kDenseSamples ) {
			evaluate();
		}
		else if ( rounds_ >= kWarmRounds && e_tok_ < kLowTokens ) {
			// Acceptance looks poor but the plain rate is unmeasured: take plain steps to measure it.
			dense_mode_ = true;
			probe_ = kProbe;
			dense_run_ = 0;
		}
	}

	void observe_dense( double ms ) {
		dense_ms_ = dense_n_ ? 0.7 * dense_ms_ + 0.3 * ms : ms;
		++dense_n_;
		if ( enabled() && rounds_ && dense_n_ >= kDenseSamples ) evaluate();
	}

	bool dense_mode() const { return dense_mode_; }

private:
	static constexpr int kWarmRounds = 12;
	static constexpr int kDenseSamples = 3;
	static constexpr int kProbe = 8;
	static constexpr int kMaxProbe = 64;
	static constexpr double kLowTokens = 1.6;
	static constexpr double kMargin = 1.15; // plain must win by this factor to displace speculation

	static bool enabled() {
		return runtime::options().spec_ctrl;
	}

	void evaluate() {
		const double spec_rate = e_tok_ / e_ms_;
		const double dense_rate = 1.0 / dense_ms_;
		if ( runtime::options().spec_ctrl_log ) {
			std::fprintf( stderr, "[ctrl] spec %.2f tok / %.2f ms = %.1f tok/s, dense %.2f ms = %.1f tok/s, mode %d\n", e_tok_, e_ms_, spec_rate * 1000.0, dense_ms_, dense_rate * 1000.0, dense_mode_ ? 1 : 0 );
		}
		if ( dense_rate > spec_rate * kMargin ) {
			// A probe that still loses backs the interval off; the first loss enters plain mode.
			probe_ = dense_mode_ ? ( probe_ * 2 > kMaxProbe ? kMaxProbe : probe_ * 2 ) : kProbe;
			dense_mode_ = true;
		}
		else {
			dense_mode_ = false;
			probe_ = kProbe;
		}
		dense_run_ = 0;
	}

	bool dense_mode_ = false;
	int dense_run_ = 0;
	int probe_ = kProbe;
	int rounds_ = 0;
	int dense_n_ = 0;
	double e_tok_ = 0.0, e_ms_ = 1.0, dense_ms_ = 1.0;
};

} // namespace engine::runtime
