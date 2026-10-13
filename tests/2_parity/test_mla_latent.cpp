// Absorbed MLA parity: latent-cache attention (absorb_q -> attend -> expand_out) against an FP64 reference that expands K and V
// per head from the same latents, on the real GLM-5.3-Flash layer-3 kv_b_proj slice (rank 0, 16 heads).
#include "engine/runtime/test_paths.h"
#include "engine/config.h"
#include "engine/core/arena.h"
#include "engine/core/batch.h"
#include "engine/loader/safetensors.h"
#include "engine/modules/mla.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	const char* dir = engine::runtime::test::model_path( "GLM-5.3-Flash-exl3-3.05bpw" );
	constexpr int H = 16, LORA = 512, HD = 256;
	CUDA_CHECK( cudaSetDevice( 0 ) );
	cudaStream_t stream;
	CUDA_CHECK( cudaStreamCreate( &stream ) );
	pod::MemoryArena arena;
	ENGINE_ASSERT( engine::core::arena_create( arena, 0, 1ULL * 1024 * 1024 * 1024 ), "arena" );
	engine::loader::SafeTensorsRepo repo;
	ENGINE_ASSERT( repo.open( dir ), "open model dir" );
	pod::TensorView kv_b = repo.load_slice( "model.language_model.layers.3.self_attn.kv_b_proj.weight", arena, 0, 0, 4, stream, pod::DType::FP16 );
	ENGINE_ASSERT( kv_b.data && kv_b.shape[0] == H * 2 * HD && kv_b.shape[1] == LORA, "kv_b slice shape" );
	std::vector<half> h_w( static_cast<size_t>( kv_b.shape[0] ) * LORA );
	CUDA_CHECK( cudaMemcpy( h_w.data(), kv_b.data, h_w.size() * sizeof( half ), cudaMemcpyDeviceToHost ) );

	bool all_ok = true;
	// Cache formats: scenario s runs format s / 4 (FP16 and the three quantized ones); the reference always uses the unquantized latent
	const pod::KVQ fmts[4] = { pod::KVQ::F16, pod::KVQ::E5M2, pod::KVQ::I8, pod::KVQ::FP4 };
	const double min_cos[4] = { 0.9999, 0.997, 0.9995, 0.98 };
	for ( int scenario_all = 0; scenario_all < ( std::getenv( "MLA_TIME" ) ? 0 : 16 ); ++scenario_all ) {
		const int scenario = scenario_all % 4, fi = scenario_all / 4;
		const pod::KVQ fq = fmts[fi];
		const int max_seq = ( scenario & 1 ) ? 3072 : 1024;
		const int N = max_seq - 40; // tokens in the cache
		uint32_t seed = 4242u + max_seq;
		auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return ( ( seed >> 8 ) / 16777216.0f ) * 2.0f - 1.0f; };
		std::vector<half> h_c( static_cast<size_t>( N ) * LORA );
		for ( auto& v : h_c ) v = __float2half( rnd() * 1.7f );
		// Scenarios 0, 1: three decode-style rows (SIMT kernel); 2, 3: a 40-row prefill chunk (tensor-core kernel, key splits)
		std::vector<int> positions = { N - 1, N / 2, 37 };
		if ( scenario >= 2 ) {
			positions.clear();
			for ( int r = 0; r < 40; ++r ) positions.push_back( N - 40 + r );
		}
		const int R = static_cast<int>( positions.size() );
		std::vector<half> h_q( static_cast<size_t>( R ) * H * HD );
		for ( auto& v : h_q ) v = __float2half( rnd() * 1.5f );

		// Device buffers: cache [slot 0][layer 0..1][max_seq][512], layer 1 used
		const size_t slot_stride = 2ull * max_seq * LORA;
		void* d_cache; void* d_cache_s = nullptr; half* d_c; half* d_q; half* d_qlat; half* d_olat; half* d_out; float* d_ws; pod::BatchRows* d_rows;
		CUDA_CHECK( cudaMalloc( &d_cache, pod::kv_data_bytes( fq, slot_stride ) ) );
		CUDA_CHECK( cudaMemset( d_cache, 0, pod::kv_data_bytes( fq, slot_stride ) ) );
		if ( pod::kv_scale_bytes( fq, slot_stride ) ) {
			CUDA_CHECK( cudaMalloc( &d_cache_s, pod::kv_scale_bytes( fq, slot_stride ) ) );
			CUDA_CHECK( cudaMemset( d_cache_s, 0, pod::kv_scale_bytes( fq, slot_stride ) ) );
		}
		pod::KVRef kvr = pod::kv_ref_f16( d_cache, d_cache );
		kvr.ks = kvr.vs = d_cache_s;
		kvr.kq = kvr.vq = fq;
		CUDA_CHECK( cudaMalloc( &d_c, h_c.size() * sizeof( half ) ) );
		CUDA_CHECK( cudaMalloc( &d_q, h_q.size() * sizeof( half ) ) );
		CUDA_CHECK( cudaMalloc( &d_qlat, static_cast<size_t>( R ) * H * LORA * sizeof( half ) ) );
		CUDA_CHECK( cudaMalloc( &d_olat, static_cast<size_t>( R ) * H * LORA * sizeof( half ) ) );
		CUDA_CHECK( cudaMalloc( &d_out, static_cast<size_t>( R ) * H * HD * sizeof( half ) ) );
		CUDA_CHECK( cudaMalloc( &d_ws, engine::modules::MLALatent::workspace_floats( max_seq, H ) * sizeof( float ) ) );
		CUDA_CHECK( cudaMalloc( &d_rows, sizeof( pod::BatchRows ) ) );
		CUDA_CHECK( cudaMemcpy( d_c, h_c.data(), h_c.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
		CUDA_CHECK( cudaMemcpy( d_q, h_q.data(), h_q.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
		// Fill the cache through the device store: one BatchRows per group of 256 positions
		for ( int p0 = 0; p0 < N; p0 += 256 ) {
			const int n = std::min( 256, N - p0 );
			pod::BatchRows br = pod::BatchRows::causal( n, p0, 0 );
			CUDA_CHECK( cudaMemcpy( d_rows, &br, sizeof( br ), cudaMemcpyHostToDevice ) );
			pod::SlotRef sr; sr.rows = d_rows; sr.row = 0; sr.stride = slot_stride;
			pod::TensorView ct = pod::make_tensor_2d( d_c + static_cast<size_t>( p0 ) * LORA, pod::DType::FP16, n, LORA );
			engine::modules::MLALatent::store( kvr, ct, 1, max_seq, n, sr, stream );
		}
		std::vector<int> slots( R, 0 );
		pod::BatchRows br = pod::BatchRows::batched( R, slots.data(), positions.data() );
		CUDA_CHECK( cudaMemcpy( d_rows, &br, sizeof( br ), cudaMemcpyHostToDevice ) );
		pod::SlotRef sr; sr.rows = d_rows; sr.row = 0; sr.stride = slot_stride;
		pod::TensorView qt = pod::make_tensor_2d( d_q, pod::DType::FP16, R, H * HD );
		pod::TensorView qlat = pod::make_tensor_2d( d_qlat, pod::DType::FP16, R, H * LORA );
		pod::TensorView olat = pod::make_tensor_2d( d_olat, pod::DType::FP16, R, H * LORA );
		pod::TensorView outt = pod::make_tensor_2d( d_out, pod::DType::FP16, R, H * HD );
		engine::modules::MLALatent::absorb_q( qt, kv_b, qlat, H, R, stream );
		engine::modules::MLALatent::attend( kvr, qlat, olat, 1, max_seq, R, H, sr, d_ws, 0, stream );
		engine::modules::MLALatent::expand_out( olat, kv_b, outt, H, R, stream );
		CUDA_CHECK( cudaStreamSynchronize( stream ) );
		std::vector<half> h_out( static_cast<size_t>( R ) * H * HD );
		CUDA_CHECK( cudaMemcpy( h_out.data(), d_out, h_out.size() * sizeof( half ), cudaMemcpyDeviceToHost ) );

		// FP64 reference with explicit per-head K / V
		double dot = 0, na = 0, nb = 0, maxabs = 0, maxref = 0;
		for ( int h = 0; h < H; ++h ) {
			const int total_max = N;
			std::vector<double> K( static_cast<size_t>( total_max ) * HD ), V( static_cast<size_t>( total_max ) * HD );
			for ( int t = 0; t < total_max; ++t ) {
				for ( int d = 0; d < HD; ++d ) {
					double ks = 0, vs = 0;
					const half* wk = h_w.data() + ( static_cast<size_t>( h ) * 2 * HD + d ) * LORA;
					const half* wv = h_w.data() + ( static_cast<size_t>( h ) * 2 * HD + HD + d ) * LORA;
					const half* c = h_c.data() + static_cast<size_t>( t ) * LORA;
					for ( int i = 0; i < LORA; ++i ) {
						const double cv = __half2float( c[i] );
						ks += __half2float( wk[i] ) * cv;
						vs += __half2float( wv[i] ) * cv;
					}
					K[static_cast<size_t>( t ) * HD + d] = ks;
					V[static_cast<size_t>( t ) * HD + d] = vs;
				}
			}
			for ( int r = 0; r < R; ++r ) {
				const int tot = positions[r] + 1;
				std::vector<double> s( tot );
				double mx = -1e300;
				for ( int t = 0; t < tot; ++t ) {
					double a = 0;
					for ( int d = 0; d < HD; ++d ) a += __half2float( h_q[( static_cast<size_t>( r ) * H + h ) * HD + d] ) * K[static_cast<size_t>( t ) * HD + d];
					s[t] = a / 16.0;
					mx = std::max( mx, s[t] );
				}
				double l = 0;
				for ( int t = 0; t < tot; ++t ) { s[t] = std::exp( s[t] - mx ); l += s[t]; }
				for ( int d = 0; d < HD; ++d ) {
					double o = 0;
					for ( int t = 0; t < tot; ++t ) o += s[t] * V[static_cast<size_t>( t ) * HD + d];
					o /= l;
					const double got = __half2float( h_out[( static_cast<size_t>( r ) * H + h ) * HD + d] );
					dot += o * got; na += o * o; nb += got * got;
					maxabs = std::max( maxabs, std::fabs( o - got ) );
					maxref = std::max( maxref, std::fabs( o ) );
				}
			}
		}
		const double cosine = dot / std::sqrt( na * nb + 1e-30 );
		const bool ok = cosine >= min_cos[fi] && ( fq != pod::KVQ::F16 || maxabs <= 0.01 * maxref + 1e-3 );
		all_ok = all_ok && ok;
		std::printf( "  %s %s max_seq %d (%d cached tokens, %d chunks): cos=%.6f max_abs=%.4f (|ref|max=%.3f) %s\n", ( const char* const[] ){ "fp16", "fp8 ", "int8", "fp4 " }[fi], scenario >= 2 ? "prefill" : "decode ", max_seq, N,
			engine::modules::MLALatent::workspace_chunks( max_seq ), cosine, maxabs, maxref, ok ? "ok" : "FAIL" );
		(void) cudaFree( d_cache ); if ( d_cache_s ) (void) cudaFree( d_cache_s ); (void) cudaFree( d_c ); (void) cudaFree( d_q ); (void) cudaFree( d_qlat ); (void) cudaFree( d_olat ); (void) cudaFree( d_out ); (void) cudaFree( d_ws ); (void) cudaFree( d_rows );
	}
	if ( std::getenv( "MLA_TIME" ) ) {
		half* d_a; half* d_b; half* d_c2;
		CUDA_CHECK( cudaMalloc( &d_a, 512ull * H * LORA * sizeof( half ) ) );
		CUDA_CHECK( cudaMalloc( &d_b, 512ull * H * LORA * sizeof( half ) ) );
		CUDA_CHECK( cudaMalloc( &d_c2, 512ull * H * LORA * sizeof( half ) ) );
		CUDA_CHECK( cudaMemset( d_a, 0, 512ull * H * LORA * sizeof( half ) ) );
		cudaEvent_t e0, e1;
		(void) cudaEventCreate( &e0 ); (void) cudaEventCreate( &e1 );
		for ( int R : { 1, 8, 64, 512 } ) {
			pod::TensorView qt = pod::make_tensor_2d( d_a, pod::DType::FP16, R, H * HD );
			pod::TensorView ql = pod::make_tensor_2d( d_b, pod::DType::FP16, R, H * LORA );
			pod::TensorView ot = pod::make_tensor_2d( d_c2, pod::DType::FP16, R, H * HD );
			float ms[2];
			for ( int k = 0; k < 2; ++k ) {
				for ( int it = 0; it < 22; ++it ) {
					if ( it == 2 ) (void) cudaEventRecord( e0, stream );
					if ( k == 0 ) engine::modules::MLALatent::absorb_q( qt, kv_b, ql, H, R, stream );
					else engine::modules::MLALatent::expand_out( ql, kv_b, ot, H, R, stream );
				}
				(void) cudaEventRecord( e1, stream ); (void) cudaEventSynchronize( e1 );
				(void) cudaEventElapsedTime( &ms[k], e0, e1 );
			}
			std::printf( "  rows %3d: absorb_q %.1f us  expand_out %.1f us\n", R, ms[0] * 50.0f, ms[1] * 50.0f );
		}
	}
	std::printf( "%s absorbed MLA parity\n", all_ok ? "[+] PASS" : "[-] FAIL" );
	return all_ok ? 0 : 1;
}
