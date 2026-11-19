#include "source.h"
#include "engine/quant/fp4_fp8.h"
#include "engine/config.h"
#include "engine/core/log.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <regex>

namespace engine::tools::quant {

namespace {

enum SrcType { T_BF16, T_F16, T_F32, T_E4M3, T_NONE };

SrcType src_type( const std::string& dt ) {
	if ( dt == "BF16" ) return T_BF16;
	if ( dt == "F16" ) return T_F16;
	if ( dt == "F32" ) return T_F32;
	if ( dt == "F8_E4M3" ) return T_E4M3;
	return T_NONE;
}

size_t type_bytes( SrcType t ) { return t == T_F32 ? 4 : t == T_E4M3 ? 1 : 2; }

__device__ __forceinline__ float load_elem( const void* p, size_t i, SrcType t ) {
	switch ( t ) {
		case T_BF16: { return __bfloat162float( static_cast<const __nv_bfloat16*>( p )[i] ); } break;
		case T_F16: { return __half2float( static_cast<const half*>( p )[i] ); } break;
		case T_F32: { return static_cast<const float*>( p )[i]; } break;
		case T_E4M3: { return engine::quant::e4m3_to_float( static_cast<const uint8_t*>( p )[i] ); } break;
		default: break;
	}
	return 0.0f;
}

// dst[c * dst_ld + dst_col0 + r] = src[r, c] (rows x cols), times its block scale for FP8
__global__ void k_load_t( const void* src, SrcType t, int rows, int cols, float* dst, int dst_ld, int dst_col0, const void* scales, SrcType st,
	int s_cols, int bo, int bi ) {
	const size_t n = static_cast<size_t>( rows ) * cols;
	for ( size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x ) {
		const int r = static_cast<int>( i / cols ), c = static_cast<int>( i % cols );
		float v = load_elem( src, i, t );
		if ( scales ) v *= load_elem( scales, static_cast<size_t>( r / bo ) * s_cols + c / bi, st );
		dst[static_cast<size_t>( c ) * dst_ld + dst_col0 + r] = v;
	}
}

bool grow( void*& p, size_t& have, size_t want ) {
	if ( want <= have ) return true;
	if ( p ) CUDA_CHECK( cudaFree( p ) );
	p = nullptr;
	have = 0;
	if ( cudaMalloc( &p, want ) != cudaSuccess ) {
		(void) cudaGetLastError();
		p = nullptr;
		return false;
	}
	have = want;
	return true;
}

} // namespace

bool SourceWeights::resolve( const std::string& base, Ref& r ) {
	if ( repo_.has( base + ".weight" ) ) {
		r.kind = Ref::PLAIN;
		r.name = base + ".weight";
		return true;
	}
	static const std::regex fused( R"(^(.*\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)$)" );
	std::smatch m;
	if ( std::regex_match( base, m, fused ) && repo_.has( m[1].str() + ".gate_up_proj" ) && repo_.has( m[1].str() + ".down_proj" ) ) {
		r.expert = std::stoi( m[2].str() );
		const std::string proj = m[3].str();
		r.kind = proj == "gate_proj" ? Ref::FUSED_GATE : proj == "up_proj" ? Ref::FUSED_UP : Ref::FUSED_DOWN;
		r.name = m[1].str() + ( r.kind == Ref::FUSED_DOWN ? ".down_proj" : ".gate_up_proj" );
		return true;
	}
	const size_t dot = base.rfind( '.' );
	if ( dot != std::string::npos && base.compare( dot + 1, std::string::npos, "qkv_proj" ) == 0 ) {
		const std::string head = base.substr( 0, dot );
		std::vector<std::string> parts = { head + ".q_proj", head + ".k_proj", head + ".v_proj" };
		for ( const auto& p : parts ) {
			if ( !repo_.has( p + ".weight" ) ) return false;
		}
		r.kind = Ref::PARTS;
		r.parts = parts;
		return true;
	}
	return false;
}

bool SourceWeights::has( const std::string& base ) {
	Ref r;
	return resolve( base, r );
}

bool SourceWeights::shape( const std::string& base, int* out, int* in ) {
	Ref r;
	if ( !resolve( base, r ) ) return false;
	if ( r.kind == Ref::PARTS ) {
		int total = 0, k = 0;
		for ( const auto& p : r.parts ) {
			int o = 0;
			if ( !shape( p, &o, &k ) ) return false;
			total += o;
		}
		*out = total;
		*in = k;
		return true;
	}
	const StEntry* e = repo_.find( r.name );
	if ( !e ) return false;
	if ( r.kind == Ref::PLAIN ) {
		if ( e->shape.size() != 2 ) return false;
		*out = static_cast<int>( e->shape[0] );
		*in = static_cast<int>( e->shape[1] );
		return true;
	}
	if ( e->shape.size() != 3 ) return false;
	// gate_up [E, 2 I, H] -> [I, H] per projection; down [E, H, I]
	*out = static_cast<int>( r.kind == Ref::FUSED_DOWN ? e->shape[1] : e->shape[1] / 2 );
	*in = static_cast<int>( e->shape[2] );
	return true;
}

bool SourceWeights::load_plain( const std::string& name, const StEntry* e, int64_t row0, int rows, int cols, float* dst, int dst_ld,
	int dst_col0, backend::stream_t stream ) {
	auto st = static_cast<cudaStream_t>( stream );
	const SrcType t = src_type( e->dtype );
	if ( t == T_NONE ) {
		LOG_ERROR( "[volta_quantize] %s: dtype %s is not read (BF16, F16, F32, F8_E4M3)", name.c_str(), e->dtype.c_str() );
		return false;
	}
	const size_t eb = type_bytes( t );
	const size_t bytes = static_cast<size_t>( rows ) * cols * eb;
	if ( !grow( staging_, staging_bytes_, bytes ) ) return false;
	// The mmapped rows straight to the device (the page cache serves repeated reads of a fused tensor)
	CUDA_CHECK( cudaMemcpyAsync( staging_, e->data + static_cast<size_t>( row0 ) * cols * eb, bytes, cudaMemcpyHostToDevice, st ) );
	const void* sc = nullptr;
	SrcType sct = T_NONE;
	int s_cols = 0, bo = 1, bi = 1;
	if ( t == T_E4M3 ) {
		const std::string sname = name.substr( 0, name.size() - 7 ) + ".weight_scale_inv";      // <base>.weight -> <base>.weight_scale_inv
		const StEntry* s = repo_.find( sname );
		if ( !s || s->shape.size() != 2 || row0 != 0 ) {
			LOG_ERROR( "[volta_quantize] %s: FP8 without its block scales (%s)", name.c_str(), sname.c_str() );
			return false;
		}
		sct = src_type( s->dtype );
		const int s_rows = static_cast<int>( s->shape[0] );
		s_cols = static_cast<int>( s->shape[1] );
		bo = rows % s_rows == 0 ? rows / s_rows : 128;
		bi = cols % s_cols == 0 ? cols / s_cols : 128;
		if ( !grow( scales_, scales_bytes_, s->bytes ) ) return false;
		CUDA_CHECK( cudaMemcpyAsync( scales_, s->data, s->bytes, cudaMemcpyHostToDevice, st ) );
		sc = scales_;
	}
	const size_t n = static_cast<size_t>( rows ) * cols;
	const unsigned blocks = static_cast<unsigned>( std::min<size_t>( ( n + 255 ) / 256, 65535u * 8u ) );
	k_load_t<<<blocks, 256, 0, st>>>( staging_, t, rows, cols, dst, dst_ld, dst_col0, sc, sct, s_cols, bo, bi );
	CUDA_CHECK( cudaPeekAtLastError() );
	// The staging buffers are reused by the next load
	CUDA_CHECK( cudaStreamSynchronize( st ) );
	return true;
}

bool SourceWeights::load( const std::string& base, float* dst, backend::stream_t stream ) {
	Ref r;
	if ( !resolve( base, r ) ) return false;
	if ( r.kind == Ref::PARTS ) {
		int total = 0, k = 0;
		if ( !shape( base, &total, &k ) ) return false;
		int col0 = 0;
		for ( const auto& p : r.parts ) {
			int o = 0, i = 0;
			if ( !shape( p, &o, &i ) ) return false;
			const StEntry* e = repo_.find( p + ".weight" );
			if ( !e || !load_plain( p + ".weight", e, 0, o, i, dst, total, col0, stream ) ) return false;
			col0 += o;
		}
		return true;
	}
	const StEntry* e = repo_.find( r.name );
	if ( !e ) return false;
	int out = 0, in = 0;
	if ( !shape( base, &out, &in ) ) return false;
	switch ( r.kind ) {
		case Ref::PLAIN: { return load_plain( r.name, e, 0, out, in, dst, out, 0, stream ); } break;
		case Ref::FUSED_GATE: { return load_plain( r.name, e, static_cast<int64_t>( r.expert ) * 2 * out, out, in, dst, out, 0, stream ); } break;
		case Ref::FUSED_UP: { return load_plain( r.name, e, static_cast<int64_t>( r.expert ) * 2 * out + out, out, in, dst, out, 0, stream ); } break;
		case Ref::FUSED_DOWN: { return load_plain( r.name, e, static_cast<int64_t>( r.expert ) * out, out, in, dst, out, 0, stream ); } break;
		default: break;
	}
	return false;
}

} // namespace engine::tools::quant
