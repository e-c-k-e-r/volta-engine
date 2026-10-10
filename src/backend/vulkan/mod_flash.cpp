// Vulkan backend: the Qwen4Exp (Flash-Next) modules on the Slang kernels of src/kernels/slang/flash.slang: gated residual, the n-gram layer
// (PLE), the QSA indexer, DSA selection, the sparse-list attention, the softmax router and the routed MoE. The MoE reads the routes back to the
// host to batch the tokens per expert (the CPU form), so it cannot run inside a graph capture: use ENGINE_DEBUG_NO_GRAPHS=1 with it.
#include "mod_common.h"
#include "engine/backend/runtime.h"
#include "engine/backend/tiers.h"
#include "engine/modules/attention.h"
#include "engine/modules/dsa.h"
#include "engine/modules/gated_residual.h"
#include "engine/modules/linear.h"
#include "engine/modules/moe.h"
#include "engine/modules/ple.h"
#include "engine/modules/qsa.h"
#include "engine/modules/swiglu.h"
#include "engine/runtime/options.h"
#include <cmath>
#include <vector>

// The option table lives in the host library, which the module-only smoke binaries do not link: bind it weakly (no option = defaults)
namespace engine::runtime {
__attribute__(( weak )) Options& options();
}

namespace engine::modules {

namespace {
namespace vki = ::engine::backend::impl;

inline uint32_t tiles( int n, int tile ) { return static_cast<uint32_t>( ( n + tile - 1 ) / tile ); }
}

// ---------------------------------------------------------------- gated residual

void GatedResidual::init_streams( const backend::half_t* in, float* x, int rows, int hidden, int hc, backend::stream_t stream ) {
	vki::GrInitParams p{ vki::dp( in ), vki::dp( x ), hidden, hc };
	vki::launch( "gr_init", p, stream, tiles( hidden, 256 ), rows );
}

void GatedResidual::norm( float* x, const backend::half_t* norm_w, float* normed, backend::half_t* normed_hi, backend::half_t* normed_lo, int rows, int hidden, int hc, float eps,
	const float* prev_post, const backend::half_t* prev_y, backend::stream_t stream ) {
	vki::GrNormParams p{ vki::dp( x ), vki::dp( norm_w ), vki::dp( normed ), vki::dp( prev_post ), vki::dp( prev_y ), vki::dp( normed_hi ), vki::dp( normed_lo ),
		hidden, hc, eps, prev_y != nullptr ? 1 : 0, normed_hi != nullptr ? 1 : 0 };
	vki::launch( "gr_norm", p, stream, rows, hc );
}

void GatedResidual::mix(
	const float* x, const backend::half_t* norm_w, const backend::half_t* down, const backend::half_t* up, const backend::half_t* inject,
	backend::half_t* mixed, float* post, float* scratch, int rows, int hidden, int hc, int rank, float eps, backend::stream_t stream,
	const float* prev_post, const backend::half_t* prev_y
) {
	const int flat = hc * hidden;
	float* normed = scratch;
	float* t = scratch + static_cast<size_t>( rows ) * flat;
	norm( const_cast<float*>( x ), norm_w, normed, nullptr, nullptr, rows, hidden, hc, eps, prev_post, prev_y, stream );
	const int has_inject = inject != nullptr ? 1 : 0;
	vki::GrGateParams g{ vki::dp( normed ), vki::dp( down ), vki::dp( inject ), vki::dp( t ), vki::dp( post ), flat, rank, hc, has_inject };
	vki::launch( "gr_gate", g, stream, rank + ( has_inject ? hc : 0 ), rows );
	vki::GrUpMixParams u{ vki::dp( normed ), vki::dp( up ), vki::dp( t ), vki::dp( mixed ), hidden, hc, rank, rows };
	vki::launch( "gr_up_mix", u, stream, tiles( hidden, 64 ), rows );
}

void GatedResidual::gemm_f32( const backend::half_t* a, const backend::half_t* w, float* out, int m, int n, int k, bool accumulate, backend::stream_t stream ) {
	vki::GrGemmParams p{ vki::dp( a ), vki::dp( w ), vki::dp( out ), m, n, k, accumulate ? 1 : 0 };
	vki::launch( "gr_gemm_f32", p, stream, tiles( n, 64 ), m );
}

void GatedResidual::gate_act( const float* z, int ld, backend::half_t* t_hi, backend::half_t* t_lo, float* post, int rows, int rank, int hc, bool has_inject, backend::stream_t stream ) {
	vki::GrGateActParams p{ vki::dp( z ), vki::dp( t_hi ), vki::dp( t_lo ), vki::dp( post ), ld, rank, hc, has_inject ? 1 : 0 };
	vki::launch( "gr_gate_act", p, stream, rows );
}

void GatedResidual::up_epilogue( const float* u, const float* normed, backend::half_t* mixed, int rows, int hidden, int hc, backend::stream_t stream ) {
	vki::GrUpEpilogueParams p{ vki::dp( u ), vki::dp( normed ), vki::dp( mixed ), hidden, hc };
	vki::launch( "gr_up_epilogue", p, stream, tiles( hidden, 256 ), rows );
}

void GatedResidual::add_sigmoid_gated( backend::half_t* y, const backend::half_t* s, const backend::half_t* x, const backend::half_t* w, int rows, int hidden, backend::stream_t stream ) {
	vki::GrSigGatedParams p{ vki::dp( y ), vki::dp( s ), vki::dp( x ), vki::dp( w ), hidden };
	vki::launch( "gr_sigmoid_gated", p, stream, rows );
}

void GatedResidual::stream_norm( const float* x, const backend::half_t* w, backend::half_t* out, int rows, int hidden, int hc, float eps, backend::stream_t stream ) {
	vki::GrStreamNormParams p{ vki::dp( x ), vki::dp( w ), vki::dp( out ), hidden, hc, eps };
	vki::launch( "gr_stream_norm", p, stream, rows, hc );
}

void GatedResidual::add_broadcast( const backend::half_t* h, const backend::half_t* e, float* streams, int rows, int hidden, int hc, backend::stream_t stream ) {
	vki::GrBroadcastParams p{ vki::dp( h ), vki::dp( e ), vki::dp( streams ), hidden, hc };
	vki::launch( "gr_add_broadcast", p, stream, tiles( hidden, 256 ), rows, hc );
}

void GatedResidual::apply( float* x, const float* post, const backend::half_t* y, int rows, int hidden, int hc, backend::stream_t stream ) {
	vki::GrApplyParams p{ vki::dp( x ), vki::dp( post ), vki::dp( y ), hidden, hc };
	vki::launch( "gr_apply", p, stream, tiles( hidden, 256 ), rows, hc );
}

// ---------------------------------------------------------------- PLE

void PLE::gate_and_norm( const backend::half_t* key, const backend::half_t* value, const float* streams, const backend::half_t* nk, const backend::half_t* nq, const backend::half_t* nc,
	float* gated, backend::half_t* normed, int rows, int hidden, int hc, float eps, float gate_scale, backend::stream_t stream ) {
	vki::PleGateParams p{ vki::dp( key ), vki::dp( value ), vki::dp( streams ), vki::dp( nk ), vki::dp( nq ), vki::dp( nc ), vki::dp( gated ), vki::dp( normed ), hidden, hc, eps, gate_scale };
	vki::launch( "ple_gate", p, stream, rows, hc );
}

void PLE::conv_apply( const backend::half_t* normed, const backend::half_t* conv_w, const backend::half_t* ring, const pod::BatchRows* d_rows, const float* gated, float* streams,
	int rows, int channels, int dilation, backend::stream_t stream ) {
	vki::PleConvParams p{ vki::dp( normed ), vki::dp( conv_w ), vki::dp( ring ), vki::rows_slot( d_rows ), vki::rows_pos( d_rows ), vki::dp( gated ), vki::dp( streams ), channels, dilation };
	vki::launch( "ple_conv", p, stream, tiles( channels, 256 ), rows );
}

void PLE::ring_update( const backend::half_t* normed, backend::half_t* ring, const pod::BatchRows* d_rows, int rows, int channels, backend::stream_t stream ) {
	vki::PleRingParams p{ vki::dp( normed ), vki::dp( ring ), vki::rows_slot( d_rows ), vki::rows_pos( d_rows ), channels };
	vki::launch( "ple_ring", p, stream, tiles( channels, 256 ), rows );
}

// ---------------------------------------------------------------- QSA / DSA select / sparse attention

void QSA::prep( const backend::half_t* qk, const backend::half_t* q_norm_w, backend::half_t* q_out, backend::half_t* ring, int layer, int rows, const pod::BatchRows* d_rows,
	size_t ring_slot_stride, int rot_dim, float theta, float eps, backend::stream_t stream ) {
	vki::QsaPrepParams p{ vki::dp( qk ), vki::dp( q_norm_w ), vki::dp( q_out ), vki::dp( ring ), vki::rows_slot( d_rows ), vki::rows_pos( d_rows ), layer,
		static_cast<int64_t>( ring_slot_stride ), rot_dim, theta, eps };
	vki::launch( "qsa_prep", p, stream, rows, HEADS + 1 );
}

void QSA::pool_update( const backend::half_t* ring, backend::half_t* pool_plane, const backend::half_t* k_norm_w, int layer, int rows, const pod::BatchRows* d_rows,
	size_t ring_slot_stride, size_t pool_slot_stride, int max_pools, int rot_dim, float theta, float eps, backend::stream_t stream ) {
	vki::QsaPoolParams p{ vki::dp( ring ), vki::dp( pool_plane ), vki::dp( k_norm_w ), vki::rows_slot( d_rows ), vki::rows_pos( d_rows ), layer,
		static_cast<int64_t>( ring_slot_stride ), static_cast<int64_t>( pool_slot_stride ), max_pools, rot_dim, theta, eps };
	vki::launch( "qsa_pool_update", p, stream, rows );
}

void QSA::score( const backend::half_t* q, const backend::half_t* pool_plane, backend::half_t* scores, int score_stride, int layer, int max_pools, int rows,
	const pod::BatchRows* d_rows, size_t pool_slot_stride, backend::stream_t stream ) {
	vki::QsaScoreParams p{ vki::dp( q ), vki::dp( pool_plane ), vki::dp( scores ), vki::rows_slot( d_rows ), vki::rows_pos( d_rows ), score_stride, layer, max_pools,
		static_cast<int64_t>( pool_slot_stride ), 1.0f / std::sqrt( static_cast<float>( DIM ) ) };
	vki::launch( "qsa_score", p, stream, tiles( max_pools, 64 ), rows );
}

void DSA::select( const backend::half_t* scores, int score_stride, int* idx, int* nkeys, int rows, const pod::BatchRows* d_rows, backend::stream_t stream ) {
	vki::SelectParams p{ vki::dp( scores ), vki::dp( idx ), vki::dp( nkeys ), vki::rows_pos( d_rows ), score_stride, KPAD };
	vki::launch( "dsa_select", p, stream, rows );
}

void Attention::forward_sparse( const pod::KVRef& kv, const pod::TensorView& q, const pod::TensorView& gate, pod::TensorView& out, size_t slot_stride, int layer_idx,
	const pod::BatchRows* d_rows, int count, int num_q_heads, int num_kv_heads, int head_dim, int max_seq, const int* idx, const int* nkeys, int idx_stride, backend::stream_t stream ) {
	if ( kv.kq != pod::KVQ::F16 || kv.vq != pod::KVQ::F16 ) ENGINE_UNSUPPORTED( "Attention::forward_sparse (quantized KV cache)" );
	ENGINE_ASSERT( head_dim <= 256 && num_kv_heads > 0 && num_q_heads % num_kv_heads == 0, "Attention::forward_sparse (Vulkan): head_dim <= 256 and GQA only" );
	vki::SparseAttnParams p{ vki::dp( q.data ), vki::dp( kv.k ), vki::dp( kv.v ), vki::dp( gate.data ), vki::dp( out.data ), vki::rows_slot( d_rows ), vki::rows_pos( d_rows ),
		vki::dp( idx ), vki::dp( nkeys ), static_cast<int64_t>( slot_stride ), layer_idx, num_q_heads, num_kv_heads, head_dim, max_seq, idx_stride, gate.data != nullptr ? 1 : 0,
		1.0f / std::sqrt( static_cast<float>( head_dim ) ) };
	vki::launch( "sparse_attn", p, stream, num_q_heads, count );
}

// ---------------------------------------------------------------- router and MoE

void MoERouter::route_softmax_topk( const pod::TensorView& x, const pod::TensorView& router_weights, int* d_topk_indices, backend::half_t* d_topk_weights, int hidden_size, int num_experts,
	int top_k, backend::stream_t stream, int rows ) {
	ENGINE_ASSERT( rows >= 1 && top_k <= 16 && top_k <= num_experts && num_experts <= 512, "MoERouter::route_softmax_topk (Vulkan): rows / top_k / num_experts out of range" );
	static const int round_fp16 = ( engine::runtime::options != nullptr && engine::runtime::options().router_fp16 ) ? 1 : 0;
	vki::RouterParams p{ vki::dp( x.data ), vki::dp( router_weights.data ), vki::dp( d_topk_indices ), vki::dp( d_topk_weights ), hidden_size, num_experts, top_k, round_fp16 };
	vki::launch( "router_softmax_topk", p, stream, rows );
}

// Pointer tables are a CUDA-side convenience; the Vulkan path reads the LinearWeight array directly
void MoE::init_device_ptrs( pod::MoELayerWeights& weights, pod::MemoryArena&, backend::stream_t ) {
	weights.device_ptrs = nullptr;
}

bool MoE::fused_decode( const pod::MoELayerWeights&, const pod::MoEScratch&, int ) { return false; }

pod::MoEScratch MoE::alloc_scratch( pod::MemoryArena&, int, int, int, int, int ) {
	return pod::MoEScratch{};
}

// out[r] = sum over the routes (r, j) with the expert in [start_expert, end_expert) of w * down( swiglu( gate( x[r] ), up( x[r] ) ) ). The routes are read back to
// the host and batched per expert (one trellis dequantization per weight and expert); the weighted sum runs on the device in route order.
void MoE::forward( const pod::TensorView& x, const pod::MoELayerWeights& weights, const int* d_topk_indices, const backend::half_t* d_topk_weights, pod::TensorView& out, int start_expert, int end_expert,
	const pod::MoEScratch&, float act_limit, backend::stream_t stream ) {
	ENGINE_ASSERT( weights.type == pod::QuantType::TRELLIS || weights.type == pod::QuantType::FP16, "MoE::forward (Vulkan): TRELLIS or FP16 experts only" );
	const int rows = ( out.ndim >= 2 && out.shape[0] > 0 ) ? out.shape[0] : 1;
	const int hidden = out.shape[out.ndim - 1];
	const int E = weights.num_local_experts;
	const int topk = weights.top_k;
	const pod::LinearWeight& ref = weights.up[0];
	const int inter = ref.type == pod::QuantType::TRELLIS ? ref.w2.shape[0] : ref.w0.shape[1];
	std::vector<int> idx( static_cast<size_t>( rows ) * topk );
	backend::memcpy_async( idx.data(), d_topk_indices, idx.size() * sizeof( int ), backend::Copy::D2H, stream );
	backend::stream_sync( stream );
	// Scratch slots 0 - 2 belong to the trellis linear
	const size_t hb = static_cast<size_t>( hidden ) * sizeof( backend::half_t ), ib = static_cast<size_t>( inter ) * sizeof( backend::half_t );
	backend::half_t* route_out = static_cast<backend::half_t*>( vki::scratch( 8, static_cast<size_t>( rows ) * topk * hb, stream ) );
	const backend::half_t* xp = x.ptr<const backend::half_t>();
	std::vector<std::vector<int>> hits( E );
	for ( int i = 0; i < rows * topk; ++i ) {
		const int e = idx[i];
		if ( e >= start_expert && e < end_expert ) {
			ENGINE_ASSERT( e - start_expert < E, "MoE::forward (Vulkan): expert index beyond num_local_experts" );
			hits[e - start_expert].push_back( i );
		}
	}
	for ( int le = 0; le < E; ++le ) {
		const int cnt = static_cast<int>( hits[le].size() );
		if ( cnt == 0 ) continue;
		backend::half_t* xs = static_cast<backend::half_t*>( vki::scratch( 9, cnt * hb, stream ) );
		backend::half_t* g = static_cast<backend::half_t*>( vki::scratch( 10, cnt * ib, stream ) );
		backend::half_t* u = static_cast<backend::half_t*>( vki::scratch( 11, cnt * ib, stream ) );
		backend::half_t* a = static_cast<backend::half_t*>( vki::scratch( 12, cnt * ib, stream ) );
		backend::half_t* d = static_cast<backend::half_t*>( vki::scratch( 13, cnt * hb, stream ) );
		for ( int h = 0; h < cnt; ++h ) backend::memcpy_async( xs + static_cast<size_t>( h ) * hidden, xp + static_cast<size_t>( hits[le][h] / topk ) * hidden, hb, backend::Copy::D2D, stream );
		pod::TensorView tx = pod::make_tensor_2d( xs, pod::DType::FP16, cnt, hidden );
		pod::TensorView tg = pod::make_tensor_2d( g, pod::DType::FP16, cnt, inter );
		pod::TensorView tu = pod::make_tensor_2d( u, pod::DType::FP16, cnt, inter );
		pod::TensorView ta = pod::make_tensor_2d( a, pod::DType::FP16, cnt, inter );
		pod::TensorView td = pod::make_tensor_2d( d, pod::DType::FP16, cnt, hidden );
		Linear::forward( tx, weights.gate[le], tg, stream );
		Linear::forward( tx, weights.up[le], tu, stream );
		SwiGLU::forward( tg, tu, ta, stream, act_limit );
		Linear::forward( ta, weights.down[le], td, stream );
		for ( int h = 0; h < cnt; ++h ) backend::memcpy_async( route_out + static_cast<size_t>( hits[le][h] ) * hidden, d + static_cast<size_t>( h ) * hidden, hb, backend::Copy::D2D, stream );
	}
	vki::MoeCombineParams p{ vki::dp( route_out ), vki::dp( d_topk_weights ), vki::dp( d_topk_indices ), vki::dp( out.data ), hidden, topk, start_expert, end_expert };
	vki::launch( "moe_combine_routes", p, stream, tiles( hidden, 256 ), rows );
}

} // namespace engine::modules
