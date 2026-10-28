#pragma once

#include <string>

namespace engine::models {

// Defaults of the steering probe per model_type (steer_gate_threshold, steer_edit_strength and steer_edit_layers say "auto" until set). Tuned with
// tests/tools/steer_sweep.py on one checkpoint and bit width per model; docs/STEERING.md has the measurements.
struct SteerDefaults {
	double gate_threshold;
	double edit_strength;
	int lo_num, lo_den, hi_num, hi_den;   // the edit window, first and last layer as fractions of the layer count
};

inline SteerDefaults steer_defaults( const std::string& model_type ) {
	if ( model_type == "qwen4_exp" ) return { 0.60, 8.0, 1, 3, 2, 3 };   // Qwen3.8-Flash-Next, 2 bpw experts
	return { 0.65, -8.0, 17, 32, 21, 32 };                                // Qwen3.8-27B, EXL3 4.0 bpw; also the fallback
}

} // namespace engine::models
