// The Vulkan backend's tiled attention (fa.slang) against its portable reference kernel (ref_attention.slang) on the same device, over the
// cases the models give it: head_dim 128, 256 and 512; 1 to 8 query heads per KV head; decode rows in distinct slots, decode rows of one slot,
// and prefill chunks; full attention, and sliding windows on ring caches; image rows that see past their position; every KV format; with
// and without a gate. Each case: the relative RMS of the two outputs (the tiled kernel's FP16 dot products: well under 5e-3); also on the
// tensor-core attention where the device launches CUDA kernels (its cases: head_dim 256 and 512, FP16 caches where fa_decode does not take them).
//   test_vulkan_attention
#include "../../src/backend/vulkan/mod_attention.h"
#include "engine/backend/runtime.h"
#include "engine/backend/numeric.h"
#include "engine/core/batch.h"
#include "engine/core/kv_format.h"
#include "engine/core/tensor.h"
#include "engine/modules/attention.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace engine;
using backend::half_t;

namespace {

struct Rng {
	uint64_t s;
	explicit Rng( uint64_t seed ) : s( seed * 2654435761u + 1 ) {}
	float next( float scale ) {
		s = s * 6364136223846793005ull + 1442695040888963407ull;
		return ( static_cast<float>( ( s >> 40 ) & 0xFFFFFF ) / 16777216.0f * 2.0f - 1.0f ) * scale;
	}
};

std::vector<void*> g_allocs;

void* dev( size_t bytes ) {
	void* p = backend::device_malloc( std::max<size_t>( bytes, 16 ) );
	backend::memset_sync( p, 0, std::max<size_t>( bytes, 16 ) );
	g_allocs.push_back( p );
	return p;
}

half_t* up_half( Rng& r, size_t n, float scale ) {
	std::vector<half_t> h( n );
	for ( auto& x : h ) x = backend::from_float_half( r.next( scale ) );
	half_t* d = static_cast<half_t*>( dev( n * sizeof( half_t ) ) );
	backend::memcpy_sync( d, h.data(), n * sizeof( half_t ), backend::Copy::H2D );
	return d;
}

pod::BatchRows* up_rows( const pod::BatchRows& br ) {
	auto* d = static_cast<pod::BatchRows*>( dev( sizeof( pod::BatchRows ) ) );
	backend::memcpy_sync( d, &br, sizeof( br ), backend::Copy::H2D );
	return d;
}

std::vector<float> down( const half_t* d, size_t n ) {
	std::vector<half_t> h( n );
	backend::memcpy_sync( h.data(), d, n * sizeof( half_t ), backend::Copy::D2H );
	std::vector<float> f( n );
	for ( size_t i = 0; i < n; ++i ) f[i] = backend::to_float( h[i] );
	return f;
}

enum class Rows { Slots, OneSlot, Prefill };

struct Case {
	const char* name;
	int hd, hq, hkv;
	int max_seq;         // rows per layer (== the ring for a ring cache)
	bool ring;
	int window;
	pod::KVFormat fmt;
	Rows rows;
	int count;
	int p0;              // first position (OneSlot / Prefill)
	bool gate;
	bool image;          // rows [count / 4, count / 2) see the keys up to count / 2 (needs a window)
};

constexpr int kLayers = 2, kLayer = 1, kSlots = 3;

bool run( const Case& c ) {
	Rng rng( static_cast<uint64_t>( c.hd * 131 + c.hq * 17 + c.count * 7 + c.p0 + static_cast<int>( c.fmt ) * 1000003 + static_cast<int>( c.rows ) * 7919 ) );
	const int row_elems = c.hkv * c.hd;
	const size_t slot_stride = static_cast<size_t>( kLayers ) * c.max_seq * row_elems;
	const size_t elems = slot_stride * kSlots;
	const pod::KVQ kq = pod::kv_k_quant( c.fmt ), vq = pod::kv_v_quant( c.fmt );
	pod::KVRef kv;
	kv.k = dev( pod::kv_data_bytes( kq, elems ) );
	kv.v = dev( pod::kv_data_bytes( vq, elems ) );
	kv.ks = pod::kv_scale_bytes( kq, elems ) ? dev( pod::kv_scale_bytes( kq, elems ) ) : nullptr;
	kv.vs = pod::kv_scale_bytes( vq, elems ) ? dev( pod::kv_scale_bytes( vq, elems ) ) : nullptr;
	kv.kq = kq;
	kv.vq = vq;
	kv.ring = c.ring ? c.max_seq : 0;
	// every row of the layer in every slot, through the store
	for ( int slot = 0; slot < kSlots; ++slot ) {
		for ( int p = 0; p < c.max_seq; p += pod::MAX_ROWS ) {
			const int n = std::min( pod::MAX_ROWS, c.max_seq - p );
			half_t* k = up_half( rng, static_cast<size_t>( n ) * row_elems, 1.5f );
			half_t* v = up_half( rng, static_cast<size_t>( n ) * row_elems, 1.5f );
			pod::BatchRows* rows = up_rows( pod::BatchRows::causal( n, p, slot ) );
			modules::Attention::store_kv_cache( kv, pod::make_tensor_2d( k, pod::DType::FP16, n, row_elems ), pod::make_tensor_2d( v, pod::DType::FP16, n, row_elems ),
				slot_stride, kLayer, rows, n, c.hkv, c.hd, c.max_seq );
		}
	}
	backend::device_sync();

	pod::BatchRows br;
	if ( c.rows == Rows::Slots ) {
		const int span = c.ring ? 4 * c.max_seq : c.max_seq;
		std::vector<int> slots( c.count ), pos( c.count );
		for ( int r = 0; r < c.count; ++r ) {
			slots[r] = ( r * 2 + 1 ) % kSlots;
			pos[r] = ( r == 0 ? span - 1 : ( 37 + r * 977 ) % span );
		}
		br = pod::BatchRows::batched( c.count, slots.data(), pos.data() );
	}
	else br = pod::BatchRows::causal( c.count, c.p0, 1 );
	if ( c.image ) {
		br.bidir = 1;
		for ( int r = 0; r < c.count; ++r ) br.key_end[r] = br.pos[r] + 1;
		for ( int r = c.count / 4; r < c.count / 2; ++r ) br.key_end[r] = br.pos[c.count / 2 - 1] + 1;
	}
	pod::BatchRows* drows = up_rows( br );
	const size_t qn = static_cast<size_t>( c.count ) * c.hq * c.hd;
	half_t* q = up_half( rng, qn, 1.0f );
	half_t* g = c.gate ? up_half( rng, qn, 2.0f ) : nullptr;
	half_t* o_ref = static_cast<half_t*>( dev( qn * sizeof( half_t ) ) );
	half_t* o_fa = static_cast<half_t*>( dev( qn * sizeof( half_t ) ) );
	pod::TensorView tq = pod::make_tensor_2d( q, pod::DType::FP16, c.count, c.hq * c.hd );
	pod::TensorView tg = c.gate ? pod::make_tensor_2d( g, pod::DType::FP16, c.count, c.hq * c.hd ) : pod::TensorView{};
	pod::TensorView tr = pod::make_tensor_2d( o_ref, pod::DType::FP16, c.count, c.hq * c.hd );
	pod::TensorView tf = pod::make_tensor_2d( o_fa, pod::DType::FP16, c.count, c.hq * c.hd );
	const bool one_slot = c.rows != Rows::Slots;
	const int key_hint = c.rows == Rows::Prefill ? br.pos[c.count - 1] + 1 : 0;
	const float scale = 0.0f;
	// each once (pipelines made), then timed over 10 calls
	auto ref = [&]() { modules::vk_attn::reference( kv, tq, tg, tr, slot_stride, kLayer, drows, c.count, c.hq, c.hkv, c.hd, c.max_seq, nullptr, scale, c.window ); };
	auto fa = [&]() { modules::vk_attn::tiled( kv, tq, tg, tf, slot_stride, kLayer, drows, c.count, c.hq, c.hkv, c.hd, c.max_seq, nullptr, scale, c.window, one_slot, key_hint ); };
	auto timed = [&]( auto&& f ) {
		f();
		backend::device_sync();
		const auto t0 = std::chrono::steady_clock::now();
		for ( int i = 0; i < 10; ++i ) f();
		backend::device_sync();
		return std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - t0 ).count() / 10.0;
	};
	const double ms_ref = timed( ref ), ms = timed( fa );
	const std::vector<float> a = down( o_ref, qn );
	auto check = [&]( const half_t* o, double ms_o, const char* name ) {
		const std::vector<float> b = down( o, qn );
		double e = 0.0, r = 0.0, mx = 0.0;
		bool finite = true;
		for ( size_t i = 0; i < qn; ++i ) {
			finite = finite && std::isfinite( b[i] );
			const double d = static_cast<double>( b[i] ) - a[i];
			e += d * d;
			r += static_cast<double>( a[i] ) * a[i];
			mx = std::max( mx, std::fabs( d ) );
		}
		const double rel = std::sqrt( e / std::max( r, 1e-30 ) );
		const bool ok = finite && rel < 5e-3;
		std::printf( "  %-44s rel %.2e max|d| %.3e  %8.3f ms (reference %8.3f)  %s\n", name, rel, mx, ms_o, ms_ref, ok ? "PASS" : "FAIL" );
		return ok;
	};
	bool ok = check( o_fa, ms, c.name );
	// the tensor-core attention where the device launches CUDA kernels (rows of distinct slots a block each)
	{
		half_t* o_tc = static_cast<half_t*>( dev( qn * sizeof( half_t ) ) );
		pod::TensorView tt = pod::make_tensor_2d( o_tc, pod::DType::FP16, c.count, c.hq * c.hd );
		auto tc = [&]() { return modules::vk_attn::tensor_core( kv, tq, tg, tt, slot_stride, kLayer, drows, c.count, c.hq, c.hkv, c.hd, c.max_seq, nullptr, scale, c.window, key_hint,
			!one_slot ); };
		if ( tc() ) {
			const double ms_tc = timed( tc );
			const std::string name = std::string( c.name ) + ", tensor cores";
			ok = check( o_tc, ms_tc, name.c_str() ) && ok;
		}
	}
	for ( void* p : g_allocs ) backend::device_free( p );
	g_allocs.clear();
	return ok;
}

}   // namespace

int main() {
	backend::set_device( 0 );
	using F = pod::KVFormat;
	const Case cases[] = {
		{ "decode 3 slots, 8/2 x 256, 4096", 256, 8, 2, 4096, false, 0, F::FP16, Rows::Slots, 3, 0, false, false },
		{ "decode 3 slots, 8/2 x 256, gate", 256, 8, 2, 4096, false, 0, F::FP16, Rows::Slots, 3, 0, true, false },
		{ "decode 16 slots, 8/2 x 512", 512, 8, 2, 4096, false, 0, F::FP16, Rows::Slots, 16, 0, false, false },
		{ "decode 4 rows one slot, 8/2 x 512", 512, 8, 2, 4096, false, 0, F::FP16, Rows::OneSlot, 4, 3000, false, false },
		{ "decode 3 slots, window 512 ring 1024", 256, 8, 2, 1024, true, 512, F::FP16, Rows::Slots, 3, 0, false, false },
		{ "prefill 512, window 512 ring 1024", 256, 8, 2, 1024, true, 512, F::FP16, Rows::Prefill, 512, 1000, false, false },
		{ "prefill 512, 8/2 x 512 at 2048", 512, 8, 2, 4096, false, 0, F::FP16, Rows::Prefill, 512, 2048, false, false },
		{ "prefill 64, 16/16 x 128 from 0", 128, 16, 16, 1024, false, 0, F::FP16, Rows::Prefill, 64, 0, true, false },
		{ "decode 4 slots, 24/4 x 256", 256, 24, 4, 1024, false, 0, F::FP16, Rows::Slots, 4, 0, true, false },
		{ "prefill 40, 24/4 x 256", 256, 24, 4, 1024, false, 0, F::FP16, Rows::Prefill, 40, 20, true, false },
		{ "decode 2 slots, 32/4 x 512", 512, 32, 4, 2048, false, 0, F::FP16, Rows::Slots, 2, 0, false, false },
		{ "prefill 100, 32/4 x 512", 512, 32, 4, 2048, false, 0, F::FP16, Rows::Prefill, 100, 900, false, false },
		{ "prefill 500, 6/1 x 256 from 0, gate", 256, 6, 1, 4096, false, 0, F::FP16, Rows::Prefill, 500, 0, true, false },
		{ "prefill 500, 6/1 x 256 at 3000, gate", 256, 6, 1, 4096, false, 0, F::FP16, Rows::Prefill, 500, 3000, true, false },
		{ "prefill 64 image rows, window 512 ring 1024", 256, 8, 2, 1024, true, 512, F::FP16, Rows::Prefill, 64, 2000, false, true },
		{ "decode 3 slots, fp8", 256, 8, 2, 2048, false, 0, F::FP8, Rows::Slots, 3, 0, true, false },
		{ "decode 3 slots, k8v4", 256, 8, 2, 2048, false, 0, F::K8V4, Rows::Slots, 3, 0, true, false },
		{ "decode 3 slots, nvfp4", 256, 8, 2, 2048, false, 0, F::NVFP4, Rows::Slots, 3, 0, true, false },
		{ "prefill 256, k8v4", 256, 8, 2, 2048, false, 0, F::K8V4, Rows::Prefill, 256, 512, true, false },
		{ "prefill 256, k8v4 x 512", 512, 8, 2, 2048, false, 0, F::K8V4, Rows::Prefill, 256, 512, false, false },
		{ "prefill 512, 24/4 x 256 k8v4, gate", 256, 24, 4, 2048, false, 0, F::K8V4, Rows::Prefill, 512, 0, true, false },
		{ "prefill 512, fp8 window 512 ring 1024", 256, 8, 2, 1024, true, 512, F::FP8, Rows::Prefill, 512, 1000, false, false },
		{ "prefill 256, fp8 x 512", 512, 8, 2, 2048, false, 0, F::FP8, Rows::Prefill, 256, 512, true, false },
		{ "prefill 512, 24/4 x 256 nvfp4, gate", 256, 24, 4, 2048, false, 0, F::NVFP4, Rows::Prefill, 512, 300, true, false },
		{ "prefill 256, nvfp4 x 512", 512, 8, 2, 2048, false, 0, F::NVFP4, Rows::Prefill, 256, 512, false, false },
		// Gemma 4 E4B's two layer kinds at a few context lengths (one decode row, a 512-row chunk)
		{ "E4B sliding: decode at 600", 256, 8, 2, 1024, true, 512, F::FP16, Rows::OneSlot, 1, 600, false, false },
		{ "E4B sliding: decode at 7000", 256, 8, 2, 1024, true, 512, F::FP16, Rows::OneSlot, 1, 7000, false, false },
		{ "E4B sliding: prefill 512 at 6500", 256, 8, 2, 1024, true, 512, F::FP16, Rows::Prefill, 512, 6500, false, false },
		{ "E4B full: decode at 600", 512, 8, 2, 16384, false, 0, F::FP16, Rows::OneSlot, 1, 600, false, false },
		{ "E4B full: decode at 2000", 512, 8, 2, 16384, false, 0, F::FP16, Rows::OneSlot, 1, 2000, false, false },
		{ "E4B full: decode at 7000", 512, 8, 2, 16384, false, 0, F::FP16, Rows::OneSlot, 1, 7000, false, false },
		{ "E4B full: prefill 512 at 6500", 512, 8, 2, 16384, false, 0, F::FP16, Rows::Prefill, 512, 6500, false, false },
	};
	int fails = 0;
	std::printf( "=== Vulkan attention: tiled against reference ===\n" );
	for ( const Case& c : cases ) fails += run( c ) ? 0 : 1;
	std::printf( "[%s] test_vulkan_attention (%d of %zu cases failed)\n", fails ? "FAIL" : "PASS", fails, sizeof( cases ) / sizeof( cases[0] ) );
	return fails ? 1 : 0;
}
