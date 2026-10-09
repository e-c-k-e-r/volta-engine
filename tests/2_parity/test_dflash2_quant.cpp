// Trellis quantization parity for the DFlash2 drafter, against its own unquantized checkpoint.
//
// Every other DFlash2 parity gate builds a synthetic backbone with dense FP16 weights, so the
// quantized path -- the codes, suh, svh and the optional mul1 flag -- was never checked against
// real numbers. That is the whole reason the drafter can be structurally correct and still accept
// poorly: a mis-decoded `fc` corrupts every context row the drafter reads.
//
// The reference here is $ENGINE_MODEL_ROOT/Qwen3.8-27B-DFlash2, the same drafter before
// quantization (81 BF16 tensors, `fc.weight` plus the 35 block projections unquantized). Its weights
// have max |w| = 20.875, comfortably inside FP16's 65504, and BF16 carries 7 mantissa bits against
// FP16's 10, so BF16 -> FP16 is exact for normals. No new kernel is needed to read it: the loader
// already converts BF16 -> FP16 on load, and LinearWeight::FP16 runs the same LinearPlainGEMV the
// unquantized drafter projections use.
//
// What is compared is the GEMV *output*, not the weights. Dequantizing on the host would duplicate
// the Trellis decoder a second time and could agree with itself while both are wrong; running the
// real kernel over both weights compares the thing the drafter actually consumes.
//
// A layout or scale bug does not show up as a small error here, it shows up as cosine near zero, so
// the gate is on relative error against the dense output's own RMS with a wide bound (5 bpw Trellis
// is lossy by construction) plus a cosine floor that only a decoded-but-wrong weight would miss.

#include "engine/core/arena.h"
#include "engine/core/tensor.h"
#include "engine/loader/safetensors.h"
#include "engine/modules/linear.h"
#include "engine/runtime/test_driver.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace engine::runtime;

namespace {

constexpr double REL_TOL = 0.06;   // 6% of the dense output RMS; 5 bpw Trellis is lossy
constexpr double COS_FLOOR = 0.99;  // a mis-decoded weight lands near 0, never near 1

struct Stat {
	double max_abs = 0.0;
	double rms_dense = 0.0;
	double rms_err = 0.0;
	double cos = 0.0;
	bool finite = true;
};

Stat compare( const std::vector<half>& dense, const std::vector<half>& quant ) {
	Stat s;
	ENGINE_ASSERT( dense.size() == quant.size(), "compare: size mismatch" );
	double dd = 0.0, ee = 0.0, dp = 0.0, qq = 0.0;
	for ( size_t i = 0; i < dense.size(); ++i ) {
		const double a = __half2float( dense[i] );
		const double b = __half2float( quant[i] );
		if ( !std::isfinite( a ) || !std::isfinite( b ) ) s.finite = false;
		dd += a * a;
		qq += b * b;
		dp += a * b;
		const double e = b - a;
		ee += e * e;
		s.max_abs = fmax( s.max_abs, fabs( e ) );
	}
	s.rms_dense = sqrt( dd / dense.size() );
	s.rms_err = sqrt( ee / dense.size() );
	s.cos = ( dd > 0.0 && qq > 0.0 ) ? dp / ( sqrt( dd * qq ) ) : 0.0;
	return s;
}

// Run one projection through both weights on the same activations.
Stat check_projection(
	engine::loader::SafeTensorsRepo& bf16_repo,
	engine::loader::SafeTensorsRepo& exl3_repo,
	pod::MemoryArena& arena,
	const std::string& base,      // "fc", "layers.0.self_attn.q_proj", ...
	const std::string& weight_key,// "<base>.weight" in the unquantized checkpoint
	int n,                        // output features
	int k,                        // contracted features
	const char* stage
) {
	int bits = 5;
	pod::LinearWeight qw = exl3_repo.load_linear( base.c_str(), arena, bits );

	// The unquantized tensor is stored [n, k]; LinearWeight::FP16 wants the same bytes viewed as
	// [k, n] (Linear::forward reads n from shape[1] and k from shape[0]).
	pod::TensorView w = bf16_repo.load( weight_key.c_str(), arena, nullptr, pod::DType::FP16 );
	ENGINE_ASSERT( w.data != nullptr, "missing unquantized weight" );
	ENGINE_ASSERT( w.numel() == static_cast<size_t>( n ) * k, "unquantized weight has wrong numel" );
	pod::TensorView w_kn = pod::make_tensor_2d( w.data, pod::DType::FP16, k, n );
	pod::LinearWeight dw = pod::LinearWeight::make_fp16( w_kn );

	const int m = 8;
	std::vector<half> hx( static_cast<size_t>( m ) * k );
	std::mt19937 rng( 1234 + static_cast<unsigned>( k ) );
	std::normal_distribution<float> nd( 0.0f, 1.0f );
	for ( auto& v : hx ) v = __float2half_rn( nd( rng ) );

	int xs[2] = { m, k };
	pod::TensorView xd = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, xs, 2 );
	CUDA_CHECK( cudaMemcpy( xd.data, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice ) );
	const pod::TensorView x = pod::make_tensor_2d( xd.data, pod::DType::FP16, m, k );

	int os[2] = { m, n };
	pod::TensorView odd = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, os, 2 );
	pod::TensorView oqd = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, os, 2 );
	pod::TensorView od = pod::make_tensor_2d( odd.data, pod::DType::FP16, m, n );
	pod::TensorView oq = pod::make_tensor_2d( oqd.data, pod::DType::FP16, m, n );

	engine::modules::Linear::forward( x, dw, od, nullptr );
	engine::modules::Linear::forward( x, qw, oq, nullptr );
	CUDA_CHECK( cudaDeviceSynchronize() );
	CUDA_CHECK( cudaGetLastError() );

	std::vector<half> vd( static_cast<size_t>( m ) * n ), vq( vd.size() );
	CUDA_CHECK( cudaMemcpy( vd.data(), od.data, vd.size() * 2, cudaMemcpyDeviceToHost ) );
	CUDA_CHECK( cudaMemcpy( vq.data(), oq.data, vq.size() * 2, cudaMemcpyDeviceToHost ) );

	Stat s = compare( vd, vq );
	const double rel = ( s.rms_dense > 0.0 ) ? s.rms_err / s.rms_dense : 1.0;
	std::printf( "  %-8s %-28s n=%-6d k=%-6d rel=%.4f cos=%.5f maxabs=%.3e%s\n",
		stage, base.c_str(), n, k, rel, s.cos, s.max_abs, s.finite ? "" : "  NON-FINITE" );

	// A weight that failed to decode still produces finite numbers, so finiteness alone proves
	// nothing; only the two error metrics do.
	ENGINE_ASSERT( s.finite, "quantized projection produced non-finite output" );
	ENGINE_ASSERT( s.cos >= COS_FLOOR, "quantized projection lost correlation with the dense weight" );
	ENGINE_ASSERT( rel <= REL_TOL, "quantized projection deviates from the unquantized weight" );
	return s;
}

// The tensors the quantizer left alone must be bit-identical across the two checkpoints. They are
// the ones no projection test touches: the conv base kernels, the four norms per layer, the selector
// codebooks and hidden_projection. If the export perturbed any of them the drafter would be wrong
// for a reason no amount of GEMV parity would show.
int check_shared_tensors(
	engine::loader::SafeTensorsRepo& bf16_repo,
	engine::loader::SafeTensorsRepo& exl3_repo,
	pod::MemoryArena& arena,
	const std::vector<std::string>& names
) {
	int checked = 0, bad = 0;
	for ( const std::string& name : names ) {
		pod::TensorView a = bf16_repo.load( name.c_str(), arena, nullptr, pod::DType::FP16 );
		pod::TensorView b = exl3_repo.load( name.c_str(), arena, nullptr, pod::DType::FP16 );
		ENGINE_ASSERT( a.data != nullptr && b.data != nullptr, "shared tensor missing from a checkpoint" );
		ENGINE_ASSERT( a.numel() == b.numel(), "shared tensor changed numel between checkpoints" );
		const size_t n = a.numel();
		std::vector<half> va( n ), vb( n );
		CUDA_CHECK( cudaMemcpy( va.data(), a.data, n * 2, cudaMemcpyDeviceToHost ) );
		CUDA_CHECK( cudaMemcpy( vb.data(), b.data, n * 2, cudaMemcpyDeviceToHost ) );
		size_t diff = 0;
		double worst = 0.0;
		for ( size_t i = 0; i < n; ++i ) {
			const double x = __half2float( va[i] ), y = __half2float( vb[i] );
			if ( x != y ) {
				++diff;
				worst = fmax( worst, fabs( x - y ) );
			}
		}
		if ( diff ) {
			std::printf( "  MISMATCH %-46s %zu/%zu differ, worst %.4e\n",
				name.c_str(), diff, n, worst );
			++bad;
		}
		++checked;
	}
	std::printf( "  %d unquantized tensors compared, %d mismatched\n", checked, bad );
	ENGINE_ASSERT( bad == 0, "unquantized tensors differ between the two checkpoints" );
	return checked;
}

} // namespace

int main() {
	std::printf( "========================================================\n" );
	std::printf( "   volta-engine Tier 2: DFlash2 Trellis vs BF16 weights\n" );
	std::printf( "========================================================\n" );
	CUDA_CHECK( cudaSetDevice( 0 ) );

	engine::loader::SafeTensorsRepo bf16_repo, exl3_repo;
	ENGINE_ASSERT( bf16_repo.open( test::QWEN_DRAFT_BF16_DIR ), "cannot open unquantized drafter" );
	ENGINE_ASSERT( exl3_repo.open( test::model_path( "Qwen3.8-27B-DFlash2-EXL3-5.0bpw" ) ), "cannot open EXL3 drafter" );

	pod::MemoryArena arena;
	engine::core::arena_create( arena, 0, 8ULL * 1024 * 1024 * 1024 );

	// `fc` is the one that matters most: it produces every context row the drafter attends over.
	// 5 taps x 5120 -> 5120.
	std::printf( "\n[*] context projection\n" );
	check_projection( bf16_repo, exl3_repo, arena, "fc", "fc.weight", 5120, 25600, "context" );

	std::printf( "\n[*] block projections\n" );
	const int H = 5120, QH = 32, KVH = 8, HD = 128, INTER = 17408;
	for ( int l = 0; l < 5; ++l ) {
		char p[128];
		std::snprintf( p, sizeof( p ), "layers.%d.self_attn.q_proj", l );
		check_projection( bf16_repo, exl3_repo, arena, p, std::string( p ) + ".weight", QH * HD, H, "attn" );
		std::snprintf( p, sizeof( p ), "layers.%d.self_attn.k_proj", l );
		check_projection( bf16_repo, exl3_repo, arena, p, std::string( p ) + ".weight", KVH * HD, H, "attn" );
		std::snprintf( p, sizeof( p ), "layers.%d.self_attn.v_proj", l );
		check_projection( bf16_repo, exl3_repo, arena, p, std::string( p ) + ".weight", KVH * HD, H, "attn" );
		std::snprintf( p, sizeof( p ), "layers.%d.self_attn.o_proj", l );
		check_projection( bf16_repo, exl3_repo, arena, p, std::string( p ) + ".weight", H, QH * HD, "attn" );
		std::snprintf( p, sizeof( p ), "layers.%d.mlp.gate_proj", l );
		check_projection( bf16_repo, exl3_repo, arena, p, std::string( p ) + ".weight", INTER, H, "mlp" );
		std::snprintf( p, sizeof( p ), "layers.%d.mlp.up_proj", l );
		check_projection( bf16_repo, exl3_repo, arena, p, std::string( p ) + ".weight", INTER, H, "mlp" );
		std::snprintf( p, sizeof( p ), "layers.%d.mlp.down_proj", l );
		check_projection( bf16_repo, exl3_repo, arena, p, std::string( p ) + ".weight", H, INTER, "mlp" );
	}

	std::printf( "\n[*] unquantized tensors must survive the export unchanged\n" );
	{
		std::vector<std::string> shared = {
			"hidden_norm.weight", "norm.weight",
			"candidate_selector.hidden_projection.weight",
			"candidate_selector.predecessor_codebook",
			"candidate_selector.successor_codebook",
		};
		for ( int l = 0; l < 5; ++l ) {
			char p[128];
			for ( const char* tail : { "input_layernorm.weight", "post_attention_layernorm.weight",
					"self_attn.q_norm.weight", "self_attn.k_norm.weight" } ) {
				std::snprintf( p, sizeof( p ), "layers.%d.%s", l, tail );
				shared.push_back( p );
			}
			for ( const char* tail : { "attention_conv.base_kernel", "mlp_conv.base_kernel",
					"attention_conv.kernel_projection.weight", "mlp_conv.kernel_projection.weight" } ) {
				std::snprintf( p, sizeof( p ), "layers.%d.%s", l, tail );
				shared.push_back( p );
			}
		}
		check_shared_tensors( bf16_repo, exl3_repo, arena, shared );
	}

	engine::core::arena_destroy( arena );
	bf16_repo.close();
	exl3_repo.close();

	std::printf( "========================================================\n" );
	std::printf( "[+] DFLASH2 TRELLIS/BF16 WEIGHT PARITY PASSED\n" );
	std::printf( "========================================================\n" );
	return 0;
}