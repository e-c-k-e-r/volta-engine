// CPU backend: routed MoE reference. Router (split-K logits + the Slang router_select), route dispatch, per-expert
// gate/up (trellis) -> SwiGLU -> down (trellis) through the module API, weighted sum in route order. Slow but correct: no
// persistent kernel, tickets or run tables; the observable results of the module entry points match the CUDA path.
// cpu_kernels.h first: the Slang prelude defines int4, which the backend shim only declares when the prelude is absent
#include "cpu_kernels.h"
#include "engine/modules/moe.h"
#include "engine/modules/swiglu.h"
#include "engine/backend/numeric.h"
#include <algorithm>
#include <vector>
#include <cstring>

namespace engine::modules {

namespace {

constexpr int kTopK = 8;           // routes per row of the dispatch / forward (as on the CUDA path)
constexpr int kRouterStride = 288; // partials stride per k slice, fixed by router_select

} // namespace

void MoERouter::route_topk( const pod::TensorView& x, const pod::TensorView& router_weights, const pod::TensorView& e_score_bias, int* d_topk_indices, backend::half_t* d_topk_weights, int hidden_size, int num_experts, int top_k, float routed_scaling_factor, backend::stream_t, int rows ) {
	ENGINE_ASSERT( rows >= 1, "MoERouter::route_topk: rows out of range" );
	ENGINE_ASSERT( top_k <= 16 && num_experts <= kRouterStride && hidden_size % 128 == 0, "MoERouter::route_topk: top_k / num_experts / hidden_size out of range" );
	const int slices = hidden_size / 128;
	std::vector<float> partials( static_cast<size_t>( rows ) * slices * kRouterStride, 0.0f );
	const backend::half_t* xp = x.ptr<const backend::half_t>();
	const backend::half_t* wp = router_weights.ptr<const backend::half_t>();
	// Logits per k slice of 128 dims, FP32 accumulation; router_select sums the slices in ascending order
	for ( int r = 0; r < rows; ++r ) {
		for ( int ks = 0; ks < slices; ++ks ) {
			float* dst = partials.data() + ( static_cast<size_t>( r ) * slices + ks ) * kRouterStride;
			for ( int j = 0; j < 128; ++j ) {
				const int d = ks * 128 + j;
				const float xv = backend::to_float( xp[static_cast<size_t>( r ) * hidden_size + d] );
				const backend::half_t* wr = wp + static_cast<size_t>( d ) * num_experts;
				for ( int e = 0; e < num_experts; ++e ) dst[e] += xv * backend::to_float( wr[e] );
			}
		}
	}
	const bool has_bias = e_score_bias.data != nullptr;
	RouterParams p{ partials.data(), const_cast<half*>( e_score_bias.ptr<const half>() ), d_topk_indices, reinterpret_cast<half*>( d_topk_weights ), num_experts, top_k, slices, slices, routed_scaling_factor, has_bias ? 1 : 0 };
	impl::cpu_dispatch( ::router_select, &p, static_cast<uint32_t>( rows ) );
}

void MoERouter::combine( const backend::half_t* expert_parts, const backend::half_t* topk_weights, const backend::half_t* shared_out, backend::half_t* output, int hidden_size, int top_k, backend::stream_t ) {
	for ( int d = 0; d < hidden_size; ++d ) {
		float sum = shared_out ? backend::to_float( shared_out[d] ) : 0.0f;
		for ( int k = 0; k < top_k; ++k ) {
			sum = backend::to_float( expert_parts[static_cast<size_t>( k ) * hidden_size + d] ) * backend::to_float( topk_weights[k] ) + sum;
		}
		output[d] = backend::from_float_half( sum );
	}
}

void MoERouter::accumulate( backend::half_t* accum_out, const backend::half_t* expert_out, backend::half_t weight, int hidden_size, backend::stream_t ) {
	const float w = backend::to_float( weight );
	for ( int d = 0; d < hidden_size; ++d ) {
		accum_out[d] = backend::from_float_half( backend::to_float( expert_out[d] ) * w + backend::to_float( accum_out[d] ) );
	}
}

void MoERouter::accumulate( pod::TensorView& accum_out, const pod::TensorView& expert_out, backend::half_t weight, int hidden_size, backend::stream_t stream ) {
	accumulate( accum_out.ptr<backend::half_t>(), expert_out.ptr<const backend::half_t>(), weight, hidden_size, stream );
}

// Histogram and stable scatter of the routes by expert: expert_count/cursors [num_experts + 1] (last bin = overflow), token_sorted /
// weight_sorted in route order per expert, active_experts = header + one item per (expert, chunk of <= chunk_rows)
void MoERouter::exl3_moe_dispatch( const int* selected_experts, const backend::half_t* routing_weights, int64_t* expert_count, int64_t* token_sorted, backend::half_t* weight_sorted, int* cursors, int num_experts, int routing_first, int routing_last, int4* active_experts, backend::stream_t, int rows, int chunk_rows, uint32_t* heat ) {
	const int n = kTopK * rows;
	std::vector<int> hist( static_cast<size_t>( num_experts ) + 1, 0 );
	for ( int i = 0; i < n; ++i ) {
		const int e = selected_experts[i];
		const int el = ( e >= routing_first && e < routing_last ) ? ( e - routing_first ) : num_experts;
		++hist[el];
	}
	int run = 0;
	for ( int e = 0; e <= num_experts; ++e ) {
		cursors[e] = run;
		expert_count[e] = hist[e];
		run += hist[e];
		if ( heat && e < num_experts ) heat[e] += static_cast<uint32_t>( hist[e] );
	}
	if ( active_experts ) {
		int items = 0;
		for ( int e = 0; e < num_experts; ++e ) {
			const int nchunks = ( hist[e] + chunk_rows - 1 ) / chunk_rows;
			for ( int c = 0; c < nchunks; ++c ) {
				active_experts[1 + items++] = int4{ e, cursors[e] + c * chunk_rows, std::min( chunk_rows, hist[e] - c * chunk_rows ), 0 };
			}
		}
		active_experts[0] = int4{ items, 0, 0, 0 };
	}
	std::vector<int> seen( static_cast<size_t>( num_experts ) + 1, 0 );
	for ( int i = 0; i < n; ++i ) {
		const int e = selected_experts[i];
		if ( e >= routing_first && e < routing_last ) {
			const int el = e - routing_first;
			const int pos = cursors[el] + seen[el]++;
			token_sorted[pos] = i / kTopK;
			weight_sorted[pos] = routing_weights[i];
		}
	}
}

// Pointer tables are a CUDA-side convenience (device array of per-expert pointers); the CPU path reads LinearWeight directly
void MoE::init_device_ptrs( pod::MoELayerWeights& weights, pod::MemoryArena&, backend::stream_t ) {
	weights.device_ptrs = nullptr;
}

pod::MoEScratch MoE::alloc_scratch( pod::MemoryArena&, int, int, int, int, int ) {
	return pod::MoEScratch{};
}

// out[r] = sum over routes ( r, j ) with expert in [start_expert, end_expert) of w * down( swiglu( gate( x[r] ), up( x[r] ) ) )
// Routes are batched per expert (one dequantization per weight and expert, not per route); the weighted sum runs in FP32 in route
// order and is rounded to FP16 once.
void MoE::forward( const pod::TensorView& x, const pod::MoELayerWeights& weights, const int* d_topk_indices, const backend::half_t* d_topk_weights, pod::TensorView& out, int start_expert, int end_expert, const pod::MoEScratch&, float act_limit, backend::stream_t stream ) {
	ENGINE_ASSERT( weights.type == pod::QuantType::TRELLIS || weights.type == pod::QuantType::FP16, "MoE::forward (CPU): TRELLIS or FP16 experts only" );
	const int rows = ( out.ndim >= 2 && out.shape[0] > 0 ) ? out.shape[0] : 1;
	const int hidden = out.shape[out.ndim - 1];
	const int E = weights.num_local_experts;
	const pod::LinearWeight& ref = weights.up[0];
	const int inter = ref.type == pod::QuantType::TRELLIS ? ref.w2.shape[0] : ref.w0.shape[1];
	const backend::half_t* xp = x.ptr<const backend::half_t>();
	std::vector<backend::half_t> route_out( static_cast<size_t>( rows ) * kTopK * hidden );
	for ( int le = 0; le < E; ++le ) {
		std::vector<int> hits; // route ids r * kTopK + j routed to this expert, in route order
		for ( int i = 0; i < rows * kTopK; ++i ) {
			if ( d_topk_indices[i] >= start_expert && d_topk_indices[i] < end_expert && d_topk_indices[i] - start_expert == le ) hits.push_back( i );
		}
		const int cnt = static_cast<int>( hits.size() );
		if ( cnt == 0 ) continue;
		std::vector<backend::half_t> xs( static_cast<size_t>( cnt ) * hidden ), g( static_cast<size_t>( cnt ) * inter ), u( g.size() ), a( g.size() ), d( static_cast<size_t>( cnt ) * hidden );
		for ( int h = 0; h < cnt; ++h ) std::memcpy( xs.data() + static_cast<size_t>( h ) * hidden, xp + static_cast<size_t>( hits[h] / kTopK ) * hidden, hidden * sizeof( backend::half_t ) );
		pod::TensorView tx = pod::make_tensor_2d( xs.data(), pod::DType::FP16, cnt, hidden );
		pod::TensorView tg = pod::make_tensor_2d( g.data(), pod::DType::FP16, cnt, inter );
		pod::TensorView tu = pod::make_tensor_2d( u.data(), pod::DType::FP16, cnt, inter );
		pod::TensorView ta = pod::make_tensor_2d( a.data(), pod::DType::FP16, cnt, inter );
		pod::TensorView td = pod::make_tensor_2d( d.data(), pod::DType::FP16, cnt, hidden );
		Linear::forward( tx, weights.gate[le], tg, stream );
		Linear::forward( tx, weights.up[le], tu, stream );
		SwiGLU::forward( tg, tu, ta, stream, act_limit );
		Linear::forward( ta, weights.down[le], td, stream );
		for ( int h = 0; h < cnt; ++h ) std::memcpy( route_out.data() + static_cast<size_t>( hits[h] ) * hidden, d.data() + static_cast<size_t>( h ) * hidden, hidden * sizeof( backend::half_t ) );
	}
	std::vector<float> acc( static_cast<size_t>( hidden ) );
	for ( int r = 0; r < rows; ++r ) {
		std::fill( acc.begin(), acc.end(), 0.0f );
		for ( int j = 0; j < kTopK; ++j ) {
			const int e = d_topk_indices[r * kTopK + j];
			if ( e < start_expert || e >= end_expert ) continue;
			ENGINE_ASSERT( e - start_expert < E, "MoE::forward (CPU): expert index beyond num_local_experts" );
			const float w = backend::to_float( d_topk_weights[r * kTopK + j] );
			const backend::half_t* d = route_out.data() + static_cast<size_t>( r * kTopK + j ) * hidden;
			for ( int c = 0; c < hidden; ++c ) acc[c] += w * backend::to_float( d[c] );
		}
		backend::half_t* orow = out.ptr<backend::half_t>() + static_cast<size_t>( r ) * hidden;
		for ( int c = 0; c < hidden; ++c ) orow[c] = backend::from_float_half( acc[c] );
	}
}

} // namespace engine::modules
