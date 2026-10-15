// NVFP4 checkpoint loading: compressed-tensors "nvfp4-pack-quantized" weights, prepacked into the fragment order of
// the skinny NVFP4 kernels (src/kernels/cuda/nvfp4/skinny_kernels.cu).
//
// Kernel layout, per 32-column tile and 16-wide K group g, per lane (8 bytes of codes, 1 byte of scale):
//   column  n = tile * 32 + qp * 8 + r      with qp = ( lane >> 2 ) & 3, r = ( lane & 3 ) + ( ( lane & 16 ) ? 4 : 0 )
//   codes   word 0 holds K elements 0..7 of the group, word 1 elements 8..15; within a word the even elements sit in
//           nibbles 0..3 and the odd ones in nibbles 4..7 (k = 2i -> nibble i, k = 2i + 1 -> nibble 4 + i)
//   scale   the e4m3 byte of (n, g), copied raw
// Streams are [tile][group][lane].
#include "engine/loader/safetensors.h"
#include "engine/quant/fp4_fp8.h"
#include "engine/backend/runtime.h"
#include "engine/backend/numeric.h"
#include <algorithm>
#include <cmath>
#include <thread>
#include <cstring>
#include <vector>

namespace engine::loader {

namespace {

using quant::e2m1_code;
using quant::e2m1_to_float;
using quant::e4m3_to_float;

struct Nvfp4Source {
	const uint8_t* packed = nullptr;  // [N, K/2]
	const uint8_t* scales = nullptr;  // [N, K/16]
	float inv_global = 1.0f;
	int N = 0;
	int K = 0;
};

bool open_nvfp4( const SafeTensorsRepo& repo, const char* base, Nvfp4Source& src ) {
	char name[300];
	int shard = -1;
	// compressed-tensors: weight_packed / weight_scale / weight_global_scale (a divisor); modelopt: weight / weight_scale / weight_scale_2 (a multiplier)
	std::snprintf( name, sizeof( name ), "%s.weight_packed", base );
	const bool modelopt = repo.find( name, &shard ) == nullptr;
	std::snprintf( name, sizeof( name ), modelopt ? "%s.weight" : "%s.weight_packed", base );
	const auto* mp = repo.find( name, &shard );
	if ( !mp ) return false;
	src.packed = static_cast<const uint8_t*>( repo.cpu_data( name ) );
	src.N = mp->shape[0];
	src.K = mp->shape[1] * 2;
	std::snprintf( name, sizeof( name ), "%s.weight_scale", base );
	src.scales = static_cast<const uint8_t*>( repo.cpu_data( name ) );
	std::snprintf( name, sizeof( name ), modelopt ? "%s.weight_scale_2" : "%s.weight_global_scale", base );
	const float* g = static_cast<const float*>( repo.cpu_data( name ) );
	if ( !src.packed || !src.scales || !g ) return false;
	src.inv_global = modelopt ? g[0] : 1.0f / g[0];
	return true;
}

}  // namespace

const void* SafeTensorsRepo::cpu_data( const char* name ) const {
	int shard = -1;
	const auto* meta = find( name, &shard );
	return ( meta && shard >= 0 ) ? safetensors_get_data_cpu( shards_[shard], meta ) : nullptr;
}

namespace {

// Prepacks rows / K elements of a host-side [N, K/2] + [N, K/16] source into the skinny kernel's fragment order (N zero-padded to a multiple
// of 32, or to pad_n when larger)
pod::LinearWeight repack_nvfp4(
	const Nvfp4Source& src,
	pod::MemoryArena& arena,
	const int* n_begin,
	const int* n_count,
	int ranges,
	int k0,
	int kcount,
	int pad_n,
	backend::stream_t stream,
	pod::MemoryArena* codes_arena = nullptr
) {
	ENGINE_ASSERT( k0 % 16 == 0 && kcount % 64 == 0 && k0 + kcount <= src.K, "load_nvfp4: bad K range" );
	int n_total = 0;
	for ( int i = 0; i < ranges; ++i ) {
		ENGINE_ASSERT( n_begin[i] >= 0 && n_begin[i] + n_count[i] <= src.N, "load_nvfp4: bad N range" );
		n_total += n_count[i];
	}
	const int n_pad = MAX( ( n_total + 31 ) & ~31, pad_n );
	ENGINE_ASSERT( n_pad % 32 == 0, "load_nvfp4: padded N must be a multiple of 32" );
	const int tiles = n_pad / 32, G = kcount / 16;

	// Output row (in concatenated order) -> source row, -1 for padding
	std::vector<int> row_of( n_pad, -1 );
	for ( int i = 0, o = 0; i < ranges; ++i ) {
		for ( int j = 0; j < n_count[i]; ++j ) row_of[o++] = n_begin[i] + j;
	}

	std::vector<uint8_t> codes( static_cast<size_t>( tiles ) * G * 32 * 8, 0 );
	std::vector<uint8_t> scales( static_cast<size_t>( tiles ) * G * 32, 0 );
	for ( int t = 0; t < tiles; ++t ) {
		for ( int lane = 0; lane < 32; ++lane ) {
			const int qp = ( lane >> 2 ) & 3;
			const int r = ( lane & 3 ) + ( ( lane & 16 ) ? 4 : 0 );
			const int row = row_of[t * 32 + qp * 8 + r];
			if ( row < 0 ) continue;
			const uint8_t* pr = src.packed + static_cast<size_t>( row ) * ( src.K / 2 ) + k0 / 2;
			const uint8_t* sr = src.scales + static_cast<size_t>( row ) * ( src.K / 16 ) + k0 / 16;
			for ( int g = 0; g < G; ++g ) {
				uint32_t w[2] = { 0, 0 };
				for ( int h = 0; h < 2; ++h ) {
					for ( int i = 0; i < 4; ++i ) {
						const uint32_t b = pr[g * 8 + h * 4 + i];
						w[h] |= ( b & 15u ) << ( 4 * i );
						w[h] |= ( b >> 4 ) << ( 16 + 4 * i );
					}
				}
				const size_t at = ( static_cast<size_t>( t ) * G + g ) * 32 + lane;
				std::memcpy( &codes[at * 8], w, 8 );
				scales[at] = sr[g];
			}
		}
	}

	pod::LinearWeight lw;
	lw.type = pod::QuantType::NVFP4;
	lw.bits = 4;
	lw.alpha = src.inv_global;
	const int cshape[2] = { n_pad, kcount / 2 };   // [N, K/2] bytes in the prepacked tile order: the shape carries N and K for the kernels
	const int sshape[1] = { static_cast<int>( scales.size() ) };
	lw.w0 = engine::core::arena_alloc_tensor( codes_arena ? *codes_arena : arena, pod::DType::UINT8, cshape, 2 );
	lw.w1 = engine::core::arena_alloc_tensor( arena, pod::DType::UINT8, sshape, 1 );
	if ( lw.w0.data == nullptr || lw.w1.data == nullptr ) return pod::LinearWeight{};
	backend::memcpy_async( lw.w0.data, codes.data(), codes.size(), codes_arena ? backend::Copy::Auto : backend::Copy::H2D, stream );
	backend::memcpy_async( lw.w1.data, scales.data(), scales.size(), backend::Copy::H2D, stream );
	backend::stream_sync( stream );  // the host staging buffers go out of scope
	return lw;
}

// E4M3 positive codes 0..126 (127 / 0x7F is NaN) decoded
const std::vector<float>& e4m3_table() {
	static const std::vector<float> t = []() {
		std::vector<float> v( 127 );
		for ( int c = 0; c < 127; ++c ) v[c] = e4m3_to_float( static_cast<uint8_t>( c ) );
		return v;
	}();
	return t;
}

float bf16_or_fp16_to_float( const uint8_t* p, bool bf16 ) {
	uint16_t h;
	std::memcpy( &h, p, 2 );
	if ( bf16 ) {
		const uint32_t u = static_cast<uint32_t>( h ) << 16;
		float f;
		std::memcpy( &f, &u, 4 );
		return f;
	}
	const int e = ( h >> 10 ) & 31, m = h & 1023;
	const float v = e == 0 ? std::ldexp( m / 1024.0f, -14 ) : e == 31 ? 65504.0f : std::ldexp( 1.0f + m / 1024.0f, e - 15 );
	return ( h & 0x8000 ) ? -v : v;
}

}  // namespace

pod::LinearWeight SafeTensorsRepo::load_nvfp4(
	const char* base_name,
	pod::MemoryArena& arena,
	const int* n_begin,
	const int* n_count,
	int ranges,
	int k0,
	int kcount,
	backend::stream_t stream,
	int pad_n,
	pod::MemoryArena* codes_arena
) {
	Nvfp4Source src;
	if ( !open_nvfp4( *this, base_name, src ) ) return pod::LinearWeight{};
	return repack_nvfp4( src, arena, n_begin, n_count, ranges, k0, kcount, pad_n, stream, codes_arena );
}

pod::LinearWeight SafeTensorsRepo::load_nvfp4_concat(
	const char* const* base_names,
	const int* n_begin,
	const int* n_count,
	int parts,
	int k0,
	int kcount,
	int pad_n,
	pod::MemoryArena& arena,
	backend::stream_t stream
) {
	// Rows of several prepared tensors (same width, same global scale) stacked into one source
	std::vector<Nvfp4Source> srcs( parts );
	int n_total = 0;
	for ( int i = 0; i < parts; ++i ) {
		if ( !open_nvfp4( *this, base_names[i], srcs[i] ) ) return pod::LinearWeight{};
		ENGINE_ASSERT( srcs[i].K == srcs[0].K && srcs[i].inv_global == srcs[0].inv_global, "load_nvfp4_concat: parts must share K and the global scale" );
		ENGINE_ASSERT( n_begin[i] >= 0 && n_begin[i] + n_count[i] <= srcs[i].N, "load_nvfp4_concat: bad N range" );
		n_total += n_count[i];
	}
	const int K = srcs[0].K;
	std::vector<uint8_t> packed( static_cast<size_t>( n_total ) * ( K / 2 ) ), bscales( static_cast<size_t>( n_total ) * ( K / 16 ) );
	std::vector<int> new_begin( parts ), counts( n_count, n_count + parts );
	for ( int i = 0, o = 0; i < parts; ++i ) {
		new_begin[i] = o;
		std::memcpy( &packed[static_cast<size_t>( o ) * ( K / 2 )], srcs[i].packed + static_cast<size_t>( n_begin[i] ) * ( K / 2 ), static_cast<size_t>( n_count[i] ) * ( K / 2 ) );
		std::memcpy( &bscales[static_cast<size_t>( o ) * ( K / 16 )], srcs[i].scales + static_cast<size_t>( n_begin[i] ) * ( K / 16 ), static_cast<size_t>( n_count[i] ) * ( K / 16 ) );
		o += n_count[i];
	}
	Nvfp4Source src;
	src.packed = packed.data();
	src.scales = bscales.data();
	src.inv_global = srcs[0].inv_global;
	src.N = n_total;
	src.K = K;
	return repack_nvfp4( src, arena, new_begin.data(), counts.data(), parts, k0, kcount, pad_n, stream );
}

namespace {

// Element (r, c) of a quantization source, in float: BF16 / FP16 as stored, FP8 E4M3 times the block scale of its 128 x 128 block
template <typename Part>
float part_value( const Part& p, int r, int c ) {
	if ( !p.fp8 ) return bf16_or_fp16_to_float( p.w + ( static_cast<size_t>( r ) * p.K + c ) * 2, p.bf16 );
	return e4m3_to_float( p.w[static_cast<size_t>( r ) * p.K + c] ) * bf16_or_fp16_to_float( p.sc + ( static_cast<size_t>( r / 128 ) * p.sK + c / 128 ) * 2, true );
}

}  // namespace

pod::LinearWeight SafeTensorsRepo::quantize_nvfp4_concat(
	const char* const* base_names,
	const int* n_begin,
	const int* n_count,
	int parts,
	int k0,
	int kcount,
	int pad_n,
	pod::MemoryArena& arena,
	backend::stream_t stream
) {
	// A source is BF16 / FP16, or FP8 E4M3 with 128 x 128 block scales (<base>.weight_scale_inv, BF16, the multiplier of the dequantized block)
	struct Part { const uint8_t* w; bool bf16; int N, K; bool fp8; const uint8_t* sc; int sK; };
	std::vector<Part> src_parts( parts );
	char name[300];
	float amax = 0.0f;
	for ( int i = 0; i < parts; ++i ) {
		std::snprintf( name, sizeof( name ), "%s.weight", base_names[i] );
		const auto* mp = find( name );
		if ( !mp || mp->ndim != 2 ) return pod::LinearWeight{};
		std::snprintf( name, sizeof( name ), "%s.weight_scale_inv", base_names[i] );
		const bool fp8 = mp->dtype == pod::DType::UINT8 && find( name ) != nullptr;
		if ( !fp8 && mp->dtype != pod::DType::BF16 && mp->dtype != pod::DType::FP16 ) return pod::LinearWeight{};
		const uint8_t* sc = fp8 ? static_cast<const uint8_t*>( cpu_data( name ) ) : nullptr;
		std::snprintf( name, sizeof( name ), "%s.weight", base_names[i] );
		const uint8_t* w = static_cast<const uint8_t*>( cpu_data( name ) );
		if ( !w || ( fp8 && !sc ) ) return pod::LinearWeight{};
		ENGINE_ASSERT( !fp8 || ( mp->shape[0] % 128 == 0 && mp->shape[1] % 128 == 0 ), "quantize_nvfp4: FP8 sources need 128-multiple dimensions" );
		src_parts[i] = { w, mp->dtype == pod::DType::BF16, mp->shape[0], mp->shape[1], fp8, sc, mp->shape[1] / 128 };
		ENGINE_ASSERT( src_parts[i].K == src_parts[0].K && k0 % 16 == 0 && kcount % 64 == 0 && k0 + kcount <= src_parts[i].K, "quantize_nvfp4: bad K range" );
		ENGINE_ASSERT( n_begin[i] >= 0 && n_begin[i] + n_count[i] <= src_parts[i].N, "quantize_nvfp4: bad N range" );
		// Global scale from the whole tensors (so every rank's slice of one matrix shares it): the largest block scale maps to the E4M3 maximum
		for ( int r = 0; r < src_parts[i].N; ++r ) for ( int c = 0; c < src_parts[i].K; ++c ) amax = std::fmax( amax, std::fabs( part_value( src_parts[i], r, c ) ) );
	}
	const float gs = amax > 0.0f ? amax / ( 6.0f * 448.0f ) : 1.0f;

	// Compact source: the requested rows (concatenated over the parts) and K range, quantized block by block
	int n_total = 0;
	std::vector<int> new_begin( parts );
	for ( int i = 0; i < parts; ++i ) { new_begin[i] = n_total; n_total += n_count[i]; }
	struct RowRef { int part, row; };
	std::vector<RowRef> row_src( n_total );
	for ( int i = 0, o = 0; i < parts; ++i ) for ( int j = 0; j < n_count[i]; ++j ) row_src[o++] = { i, n_begin[i] + j };
	std::vector<uint8_t> packed( static_cast<size_t>( n_total ) * ( kcount / 2 ) ), bscales( static_cast<size_t>( n_total ) * ( kcount / 16 ) );
	const std::vector<float>& et = e4m3_table();
	static const float kGrid[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };

	auto quantize_rows = [&]( int r0, int r1 ) {
		float v[16];
		for ( int r = r0; r < r1; ++r ) {
			const Part& pt = src_parts[row_src[r].part];
			const int src_row = row_src[r].row;
			for ( int g = 0; g < kcount / 16; ++g ) {
				float bmax = 0.0f;
				for ( int i = 0; i < 16; ++i ) { v[i] = part_value( pt, src_row, k0 + g * 16 + i ); bmax = std::fmax( bmax, std::fabs( v[i] ) ); }
				// Start from the E4M3 scale nearest to amax / 6, then keep the neighbour (+-2 codes) with the smallest squared error
				const float ideal = bmax / ( 6.0f * gs );
				int c0 = static_cast<int>( std::lower_bound( et.begin(), et.end(), ideal ) - et.begin() );
				if ( c0 >= 127 ) c0 = 126;
				else if ( c0 > 0 && ideal - et[c0 - 1] < et[c0] - ideal ) --c0;
				c0 = MIN( MAX( c0, 1 ), 126 );
				int best = c0;
				float best_err = 1e30f;
				for ( int c = MAX( 1, c0 - 2 ); c <= MIN( 126, c0 + 2 ); ++c ) {
					const float sc = et[c] * gs;
					float err = 0.0f;
					for ( int i = 0; i < 16; ++i ) {
						const float d = kGrid[e2m1_code( v[i] / sc ) & 7] * sc * ( v[i] < 0.0f ? -1.0f : 1.0f ) - v[i];
						err += d * d;
					}
					if ( err < best_err ) { best_err = err; best = c; }
				}
				const float sc = et[best] * gs;
				uint8_t* pk = &packed[( static_cast<size_t>( r ) * ( kcount / 16 ) + g ) * 8];
				for ( int i = 0; i < 16; i += 2 ) pk[i / 2] = static_cast<uint8_t>( e2m1_code( v[i] / sc ) | ( e2m1_code( v[i + 1] / sc ) << 4 ) );
				bscales[static_cast<size_t>( r ) * ( kcount / 16 ) + g] = static_cast<uint8_t>( best );
			}
		}
	};
	const int workers = 4;
	std::vector<std::thread> pool;
	for ( int t = 0; t < workers; ++t ) pool.emplace_back( quantize_rows, static_cast<int>( static_cast<int64_t>( n_total ) * t / workers ), static_cast<int>( static_cast<int64_t>( n_total ) * ( t + 1 ) / workers ) );
	for ( auto& t : pool ) t.join();

	Nvfp4Source src;
	src.packed = packed.data();
	src.scales = bscales.data();
	src.inv_global = gs;
	src.N = n_total;
	src.K = kcount;
	std::vector<int> counts( n_count, n_count + parts );
	return repack_nvfp4( src, arena, new_begin.data(), counts.data(), parts, 0, kcount, pad_n, stream );
}

pod::LinearWeight SafeTensorsRepo::quantize_nvfp4(
	const char* base_name,
	pod::MemoryArena& arena,
	const int* n_begin,
	const int* n_count,
	int ranges,
	int k0,
	int kcount,
	int pad_n,
	backend::stream_t stream
) {
	std::vector<const char*> names( ranges, base_name );
	return quantize_nvfp4_concat( names.data(), n_begin, n_count, ranges, k0, kcount, pad_n, arena, stream );
}

pod::LinearWeight SafeTensorsRepo::quantize_nvfp4_slice(
	const char* base_name,
	pod::MemoryArena& arena,
	bool split_n,
	int rank,
	int tp_size,
	int pad_n,
	backend::stream_t stream
) {
	char name[300];
	std::snprintf( name, sizeof( name ), "%s.weight", base_name );
	const auto* mp = find( name );
	if ( !mp || mp->ndim != 2 ) return pod::LinearWeight{};
	const int N = mp->shape[0], K = mp->shape[1];
	if ( split_n ) {
		const int cnt = N / tp_size, beg = rank * cnt;
		return quantize_nvfp4( base_name, arena, &beg, &cnt, 1, 0, K, pad_n, stream );
	}
	const int cnt = K / tp_size, beg = 0;
	return quantize_nvfp4( base_name, arena, &beg, &N, 1, rank * cnt, cnt, pad_n, stream );
}

pod::LinearWeight SafeTensorsRepo::load_nvfp4_slice(
	const char* base_name,
	pod::MemoryArena& arena,
	bool split_n,
	int rank,
	int tp_size,
	backend::stream_t stream,
	int pad_n,
	pod::MemoryArena* codes_arena
) {
	Nvfp4Source src;
	if ( !open_nvfp4( *this, base_name, src ) ) return pod::LinearWeight{};
	if ( split_n ) {
		const int cnt = src.N / tp_size, beg = rank * cnt;
		return load_nvfp4( base_name, arena, &beg, &cnt, 1, 0, src.K, stream, pad_n, codes_arena );
	}
	const int cnt = src.K / tp_size;
	const int beg = 0;
	return load_nvfp4( base_name, arena, &beg, &src.N, 1, rank * cnt, cnt, stream, pad_n, codes_arena );
}

pod::TensorView SafeTensorsRepo::load_nvfp4_dense(
	const char* base_name,
	pod::MemoryArena& arena,
	bool split_n,
	int rank,
	int tp_size,
	backend::stream_t stream
) {
	Nvfp4Source src;
	if ( !open_nvfp4( *this, base_name, src ) ) return pod::TensorView{};
	const int n_cnt = split_n ? src.N / tp_size : src.N, n0 = split_n ? rank * n_cnt : 0;
	const int k_cnt = split_n ? src.K : src.K / tp_size, k0 = split_n ? 0 : rank * k_cnt;
	std::vector<backend::half_t> out( static_cast<size_t>( n_cnt ) * k_cnt );
	for ( int n = 0; n < n_cnt; ++n ) {
		for ( int k = 0; k < k_cnt; ++k ) {
			const int kk = k0 + k;
			const uint8_t b = src.packed[static_cast<size_t>( n0 + n ) * ( src.K / 2 ) + kk / 2];
			const float sc = e4m3_to_float( src.scales[static_cast<size_t>( n0 + n ) * ( src.K / 16 ) + kk / 16] );
			out[static_cast<size_t>( n ) * k_cnt + k] = backend::from_float_half( e2m1_to_float( ( kk & 1 ) ? ( b >> 4 ) : ( b & 15 ) ) * sc * src.inv_global );
		}
	}
	const int shape[2] = { n_cnt, k_cnt };
	pod::TensorView tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, shape, 2 );
	if ( tv.data == nullptr ) return tv;
	backend::memcpy_async( tv.data, out.data(), out.size() * sizeof( backend::half_t ), backend::Copy::H2D, stream );
	backend::stream_sync( stream );
	return tv;
}

pod::LinearWeight SafeTensorsRepo::load_dense_rows_padded(
	const char* weight_name,
	pod::MemoryArena& arena,
	int row_begin,
	int row_count,
	int pad_rows,
	backend::stream_t stream
) {
	int shard = -1;
	const auto* meta = find( weight_name, &shard );
	if ( !meta || meta->ndim != 2 ) return pod::LinearWeight{};
	ENGINE_ASSERT( row_begin + row_count <= meta->shape[0], "load_dense_rows_padded: row range" );
	ENGINE_ASSERT( meta->dtype == pod::DType::BF16 || meta->dtype == pod::DType::FP16, "load_dense_rows_padded: BF16/FP16 only" );
	const size_t K = meta->shape[1];
	const size_t rows = static_cast<size_t>( row_count ) + pad_rows;
	const uint16_t* src = static_cast<const uint16_t*>( cpu_data( weight_name ) ) + static_cast<size_t>( row_begin ) * K;
	std::vector<backend::half_t> out( rows * K );
	for ( size_t i = 0; i < static_cast<size_t>( row_count ) * K; ++i ) {
		if ( meta->dtype == pod::DType::FP16 ) {
			std::memcpy( &out[i], &src[i], 2 );
		}
		else {
			const uint32_t bits = static_cast<uint32_t>( src[i] ) << 16;
			float f;
			std::memcpy( &f, &bits, 4 );
			out[i] = backend::from_float_half( f );
		}
	}
	std::memset( &out[static_cast<size_t>( row_count ) * K], 0, static_cast<size_t>( pad_rows ) * K * sizeof( backend::half_t ) );
	const int shape[2] = { static_cast<int>( rows ), static_cast<int>( K ) };
	pod::TensorView tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, shape, 2 );
	if ( tv.data == nullptr ) return pod::LinearWeight{};
	backend::memcpy_async( tv.data, out.data(), out.size() * sizeof( backend::half_t ), backend::Copy::H2D, stream );
	backend::stream_sync( stream );
	return pod::LinearWeight::make_fp16( pod::make_tensor_2d( tv.data, pod::DType::FP16, static_cast<int>( K ), static_cast<int>( rows ) ) );
}

}  // namespace engine::loader
