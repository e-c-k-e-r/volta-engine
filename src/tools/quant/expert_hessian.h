#pragma once

// The down projection Hessians of routed experts, upstream's recipe: every expert sees every calibration row. The rows are the router's input
// rows of a MoE layer as `route_dump` writes them (per prefill chunk: int32 {layer, rows, first position, K}, the rows' expert ids, int32
// rows x K, and the rows, FP16 rows x hidden); the activation of expert e is act( x W_gate ) * ( x W_up ) with the model's SwiGLU (GLM:
// silu( min( g, limit ) ) * clamp( u, -limit, limit )), and its Hessian the sum of a a^T over the rows, intermediate x intermediate.
// With the dump's routing weights (<dump>.w, FP32 rows x K per record) the Hessians can be weighted by routing as exllamav3 1.6.7's
// --moe_affinity_eps does (conversion/moe_affinity.py, MIT): for expert e with weight p_t on the rows routed to it (0 elsewhere), S = sum p_t over
// the N rows,
//   gate / up:  H_e = eps (S / N) H_in + sum_t p_t x_t x_t^T / N      (H_in: the layer input's mean Hessian)
//   down:       H_e = eps (S / N) sum_t a_t a_t^T + sum_t p_t a_t a_t^T
#include "engine/backend/types.h"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace engine::tools::quant {

class RouteRows {
public:
	// One pass over the record headers; the rows are read per layer. path: one dump, or several comma-separated (their rows in that order)
	bool open( const std::string& path, int hidden );
	int64_t rows( int layer ) const;
	// The layer's rows (at most max_rows, 0 = all) as FP16 bits, rows x hidden
	bool read( int layer, int64_t max_rows, std::vector<uint16_t>& out ) const;
	// How many of the layer's rows route to each expert (ids at or above `experts` are ignored)
	bool counts( int layer, int experts, std::vector<int64_t>& out ) const;
	// The expert ids and routing weights (rows x K each) of the rows read() returns; false without the weights file
	bool routes( int layer, int64_t max_rows, std::vector<int>& ids, std::vector<float>& w, int& K ) const;
	bool has_weights() const { return has_w_; }

private:
	struct Rec {
		int64_t offset;   // of the rows' data
		int rows;
		int k;            // expert ids per row, int32 before the rows' data
		int64_t w_offset; // of the routing weights in <path>.w
		int file;         // which dump of the list
	};
	std::vector<std::string> paths_;
	int hidden_ = 0;
	bool has_w_ = false;
	std::map<int, std::vector<Rec>> by_layer_;
};

// Device scratch for the activations; one instance per device
class ExpertHessian {
public:
	~ExpertHessian();
	// H (intermediate x intermediate, fp32, device) = sum over the rows of a a^T; x: rows x hidden FP16 (device), gate / up: hidden x
	// intermediate fp32 (device, input-major as the quantizer takes weights). With row_w (device, one routing weight per row, 0 where the
	// expert is not routed; w_sum their sum) the affinity-weighted down Hessian above instead
	bool compute( const void* x, int64_t rows, int hidden, const float* gate, const float* up, int inter, float limit, float* H,
		backend::stream_t stream, const float* row_w = nullptr, double w_sum = 0.0, float eps = 0.0f );
	// The affinity-weighted gate / up Hessian (hidden x hidden, device) of one expert: `routed` rows of x (device indices) with their weights
	// (device), out of `rows`; H_in the layer input's mean Hessian (device)
	bool affinity_input( const void* x, int64_t rows, int hidden, const int* routed, const float* routed_w, int n_routed, double w_sum, float eps,
		const float* H_in, float* H, backend::stream_t stream );

private:
	void* blas_ = nullptr;
	void* w16_ = nullptr;       // gate and up in FP16, hidden x intermediate each
	size_t w16_bytes_ = 0;
	float* g_ = nullptr;        // chunk x intermediate, twice (gate, up)
	size_t g_bytes_ = 0;
	void* aff_ = nullptr;       // the weighted sum (intermediate x intermediate), or gathered rows (chunk x hidden)
	size_t aff_bytes_ = 0;
};

} // namespace engine::tools::quant
