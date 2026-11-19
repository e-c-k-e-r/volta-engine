#pragma once

// The down projection Hessians of routed experts, upstream's recipe: every expert sees every calibration row. The rows are the router's input
// rows of a MoE layer as `route_dump` writes them (per prefill chunk: int32 {layer, rows, first position, K}, the rows' expert ids, int32
// rows x K, and the rows, FP16 rows x hidden); the activation of expert e is act( x W_gate ) * ( x W_up ) with the model's SwiGLU (GLM:
// silu( min( g, limit ) ) * clamp( u, -limit, limit )), and its Hessian the sum of a a^T over the rows, intermediate x intermediate.
#include "engine/backend/types.h"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace engine::tools::quant {

class RouteRows {
public:
	// One pass over the record headers; the rows are read per layer
	bool open( const std::string& path, int hidden );
	int64_t rows( int layer ) const;
	// The layer's rows (at most max_rows, 0 = all) as FP16 bits, rows x hidden
	bool read( int layer, int64_t max_rows, std::vector<uint16_t>& out ) const;

private:
	struct Rec {
		int64_t offset;   // of the rows' data
		int rows;
	};
	std::string path_;
	int hidden_ = 0;
	std::map<int, std::vector<Rec>> by_layer_;
};

// Device scratch for the activations; one instance per device
class ExpertHessian {
public:
	~ExpertHessian();
	// H (intermediate x intermediate, fp32, device) = sum over the rows of a a^T; x: rows x hidden FP16 (device), gate / up: hidden x
	// intermediate fp32 (device, input-major as the quantizer takes weights)
	bool compute( const void* x, int64_t rows, int hidden, const float* gate, const float* up, int inter, float limit, float* H,
		backend::stream_t stream );

private:
	void* blas_ = nullptr;
	void* w16_ = nullptr;       // gate and up in FP16, hidden x intermediate each
	size_t w16_bytes_ = 0;
	float* g_ = nullptr;        // chunk x intermediate, twice (gate, up)
	size_t g_bytes_ = 0;
};

} // namespace engine::tools::quant
