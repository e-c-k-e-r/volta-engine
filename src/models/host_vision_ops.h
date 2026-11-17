// The operators of the host-CPU vision towers (host_vision.h): a worker pool, the dense product over decoded EXL3 panels, and the row-wise
// pieces of a ViT block. FP16 activations between operators like the GPU towers (the arithmetic of src/backend/cpu/vision.cpp), FP32 inside.
#pragma once

#include "engine/kernels/cpu/pool.h"
#include "engine/kernels/cpu/trellis_linear.h"
#include "engine/loader/safetensors.h"
#include "engine/models/vision_tower.h"
#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::hv {

inline float h2f( uint16_t h ) {
	_Float16 v;
	std::memcpy( &v, &h, 2 );
	return static_cast<float>( v );
}

inline uint16_t f2h( float f ) {
	const _Float16 v = static_cast<_Float16>( f );
	uint16_t h;
	std::memcpy( &h, &v, 2 );
	return h;
}

// Runs loops on the worker pool; one caller at a time
class Par {
public:
	// cpus: the logical CPUs of the threads, the first being the calling thread's; empty: just the caller
	explicit Par( const std::vector<int>& cpus );
	~Par();
	int threads() const { return pool_.size(); }
	// fn( begin, end ) over [0, total) in pieces of `chunk`, concurrently
	void run( int64_t total, int64_t chunk, const std::function<void( int64_t, int64_t )>& fn );
	// The calling thread's CPU affinity moves to the first CPU for the guard's lifetime
	class Pin {
	public:
		explicit Pin( const Par& par );
		~Pin();
	private:
		bool pinned_ = false;
		alignas( 8 ) unsigned char saved_[128] = {};
	};

private:
	struct ForTask;
	std::vector<int> cpus_;
	engine::kernels::cpu::Pool pool_;
	std::vector<std::unique_ptr<ForTask>> ring_;
	size_t next_ = 0;
};

// A weight [k, n] as FP16 panels [n / 16][k][16]
struct Dense {
	int k = 0, n = 0;
	std::vector<uint16_t> w;
};

// The panels of a row-major [n, k] FP16 matrix (n a multiple of 16)
Dense pack_nk( const uint16_t* w, int n, int k );

// out[m][n] (FP32) = a[m][k] (FP32, rows contiguous) . b
void gemm( Par& par, const float* a, int m, const Dense& b, float* out );

// A projection: the (decoded) panels, and for an EXL3 matrix the rotations that go around the dense product; a matrix that was dense in the
// checkpoint (a GGUF vision projector) has none
struct Lin {
	Dense d;
	std::unique_ptr<engine::kernels::cpu::TrellisLinear> rot;
};

// What a Lin needs from the checkpoint, resolved on the loading thread; decode() then runs on the pool
struct LinSource {
	std::string name;
	// dense: a [n, k] FP16 / BF16 / FP32 matrix, row-major (when there are no codes)
	const void* dense = nullptr;
	pod::DType dtype = pod::DType::FP16;
	int n = 0, k = 0;
	const uint16_t* codes = nullptr;
	const uint16_t* suh = nullptr;
	const uint16_t* svh = nullptr;
	int kt = 0, nt = 0, bits = 0;
	bool volta = false;   // the checkpoint's tile order (tile_layout volta)
};
// `base`.trellis / .suh / .svh / .mul1 with a supported bit width, else the dense `base`.weight [n, k]; false (and a message) for neither
bool find_lin( const engine::loader::SafeTensorsRepo& repo, const std::string& base, LinSource* src, std::string* error );
// Decodes the sources into the Lins (same order), on the pool
void decode_all( Par& par, const std::vector<LinSource>& src, const std::vector<Lin*>& dst );

// Scratch buffers the operators share (grown, never shrunk)
struct Scratch {
	std::vector<float> xin, raw, qf, kf, vf;
};

// out[m][n] FP16 = x[m][k] FP16 through the projection
void forward( Par& par, Scratch& s, const Lin& lin, const uint16_t* x, int m, uint16_t* out );

// A checkpoint vector as FP32 (FP16, BF16 or FP32 on disk); empty with *error set when absent
std::vector<float> load_f32( const engine::loader::SafeTensorsRepo& repo, const std::string& name, std::string* error );
// The same as FP16 bits
std::vector<uint16_t> load_f16( const engine::loader::SafeTensorsRepo& repo, const std::string& name, std::string* error );

// ---- row-wise operators ----
void rms_norm( Par& par, const float* x, const float* w, uint16_t* out, int m, int n, float eps );
// out = LayerNorm( x ) * w + b, x FP16
void layer_norm( Par& par, const uint16_t* x, const float* w, const float* b, uint16_t* out, int m, int n, float eps );
// hidden += y + bias
void bias_residual( Par& par, float* hidden, const uint16_t* y, const float* bias, int m, int n );
// x = round( x + bias ), bias null: x unchanged
void add_bias( Par& par, uint16_t* x, const float* bias, int m, int n );
void gelu_erf( Par& par, uint16_t* x, int m, int n );
// out = silu( min( gate, limit ) ) * clamp( up, -limit, limit ) (limit <= 0: plain silu( gate ) * up)
void swiglu( Par& par, const uint16_t* gate, const uint16_t* up, uint16_t* out, int64_t count, float limit );
// Bidirectional attention over all m rows: q, k, v, out [m][heads * 64] FP16; scratch floats of the three inputs are kept in `s`
void attention( Par& par, Scratch& s, const uint16_t* q, const uint16_t* k, const uint16_t* v, uint16_t* out, int m, int heads, float scale );

} // namespace engine::models::hv

namespace engine::models {

// The host tower of one architecture (host_vision.cpp picks it); nullptr with *error set when it cannot be loaded
std::unique_ptr<VisionTower> make_host_glm5_tower( const char* model_dir, const char* mmproj_dir, int max_tokens, const std::vector<int>& cpus, std::string* error );

} // namespace engine::models
