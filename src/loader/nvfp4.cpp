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
#include "engine/backend/runtime.h"
#include "engine/backend/numeric.h"
#include <cmath>
#include <cstring>
#include <vector>

namespace engine::loader {

namespace {

float e4m3_to_float( uint8_t b ) {
	const int e = ( b >> 3 ) & 15, m = b & 7;
	const float v = ( e == 0 ) ? std::ldexp( m / 8.0f, -6 ) : std::ldexp( 1.0f + m / 8.0f, e - 7 );
	return ( b & 0x80 ) ? -v : v;
}

float e2m1_to_float( int c ) {
	static const float lut[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
	return ( c & 8 ) ? -lut[c & 7] : lut[c & 7];
}

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
	std::snprintf( name, sizeof( name ), "%s.weight_packed", base );
	const auto* mp = repo.find( name, &shard );
	if ( !mp ) return false;
	src.packed = static_cast<const uint8_t*>( repo.cpu_data( name ) );
	src.N = mp->shape[0];
	src.K = mp->shape[1] * 2;
	std::snprintf( name, sizeof( name ), "%s.weight_scale", base );
	src.scales = static_cast<const uint8_t*>( repo.cpu_data( name ) );
	std::snprintf( name, sizeof( name ), "%s.weight_global_scale", base );
	const float* g = static_cast<const float*>( repo.cpu_data( name ) );
	if ( !src.packed || !src.scales || !g ) return false;
	src.inv_global = 1.0f / g[0];
	return true;
}

}  // namespace

const void* SafeTensorsRepo::cpu_data( const char* name ) const {
	int shard = -1;
	const auto* meta = find( name, &shard );
	return ( meta && shard >= 0 ) ? safetensors_get_data_cpu( shards_[shard], meta ) : nullptr;
}

pod::LinearWeight SafeTensorsRepo::load_nvfp4(
	const char* base_name,
	pod::MemoryArena& arena,
	const int* n_begin,
	const int* n_count,
	int ranges,
	int k0,
	int kcount,
	backend::stream_t stream
) {
	Nvfp4Source src;
	if ( !open_nvfp4( *this, base_name, src ) ) return pod::LinearWeight{};
	ENGINE_ASSERT( k0 % 16 == 0 && kcount % 64 == 0 && k0 + kcount <= src.K, "load_nvfp4: bad K range" );
	int n_total = 0;
	for ( int i = 0; i < ranges; ++i ) {
		ENGINE_ASSERT( n_begin[i] >= 0 && n_begin[i] + n_count[i] <= src.N, "load_nvfp4: bad N range" );
		n_total += n_count[i];
	}
	const int n_pad = ( n_total + 31 ) & ~31;
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
	const int cshape[1] = { static_cast<int>( codes.size() ) };
	const int sshape[1] = { static_cast<int>( scales.size() ) };
	lw.w0 = engine::core::arena_alloc_tensor( arena, pod::DType::UINT8, cshape, 1 );
	lw.w1 = engine::core::arena_alloc_tensor( arena, pod::DType::UINT8, sshape, 1 );
	if ( lw.w0.data == nullptr || lw.w1.data == nullptr ) return pod::LinearWeight{};
	backend::memcpy_async( lw.w0.data, codes.data(), codes.size(), backend::Copy::H2D, stream );
	backend::memcpy_async( lw.w1.data, scales.data(), scales.size(), backend::Copy::H2D, stream );
	backend::stream_sync( stream );  // the host staging buffers go out of scope
	return lw;
}

pod::LinearWeight SafeTensorsRepo::load_nvfp4_slice(
	const char* base_name,
	pod::MemoryArena& arena,
	bool split_n,
	int rank,
	int tp_size,
	backend::stream_t stream
) {
	Nvfp4Source src;
	if ( !open_nvfp4( *this, base_name, src ) ) return pod::LinearWeight{};
	if ( split_n ) {
		const int cnt = src.N / tp_size, beg = rank * cnt;
		return load_nvfp4( base_name, arena, &beg, &cnt, 1, 0, src.K, stream );
	}
	const int cnt = src.K / tp_size;
	const int beg = 0;
	return load_nvfp4( base_name, arena, &beg, &src.N, 1, rank * cnt, cnt, stream );
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
