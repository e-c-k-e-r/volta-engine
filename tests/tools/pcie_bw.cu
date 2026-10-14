// Host <-> device bandwidth of every GPU, alone and all together (standalone; no engine headers):
//   nvcc -O3 tests/tools/pcie_bw.cu -o pcie_bw && ./pcie_bw          (hipcc --offload-arch=gfx1100 builds it for ROCm: the cuda* names are mapped below)
// Per GPU: pinned host to device and device to host copies (1, 2 and 4 streams at once) and a kernel reading the pinned buffer in place
// (what the expert tier's cold experts do). Then every GPU copying at once, which shows a shared uplink (a PLX switch, one root port).
// A Gen3 x16 link reads about 12 to 13 GB/s here, Gen4 x16 about 25 to 28, Gen4 x8 about 14.
#if defined(__HIPCC__)
	#include <hip/hip_runtime.h>
	#define cudaError_t hipError_t
	#define cudaGetDeviceCount hipGetDeviceCount
	#define cudaSetDevice hipSetDevice
	#define cudaMalloc hipMalloc
	#define cudaMemcpyAsync hipMemcpyAsync
	#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
	#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
	#define cudaHostAlloc(p, s, f) hipHostMalloc(p, s, f)
	#define cudaHostAllocMapped hipHostMallocMapped
	#define cudaHostGetDevicePointer hipHostGetDevicePointer
	#define cudaStream_t hipStream_t
	#define cudaStreamCreate hipStreamCreate
	#define cudaStreamSynchronize hipStreamSynchronize
	#define cudaDeviceSynchronize hipDeviceSynchronize
	#define cudaEvent_t hipEvent_t
	#define cudaEventCreate hipEventCreate
	#define cudaEventRecord hipEventRecord
	#define cudaEventSynchronize hipEventSynchronize
	#define cudaEventElapsedTime hipEventElapsedTime
	#define cudaDeviceProp hipDeviceProp_t
	#define cudaGetDeviceProperties hipGetDeviceProperties
#else
	#include <cuda_runtime.h>
#endif
#include <cstdio>
#include <cstdint>
#include <chrono>
#include <vector>

__global__ void read_kernel( const uint4* p, size_t n, unsigned* out ) {
	size_t i = blockIdx.x * static_cast<size_t>( blockDim.x ) + threadIdx.x, stride = static_cast<size_t>( gridDim.x ) * blockDim.x;
	unsigned acc = 0;
	for ( ; i < n; i += stride ) {
		const uint4 v = p[i];
		acc ^= v.x ^ v.y ^ v.z ^ v.w;
	}
	if ( acc == 0x12345 ) out[0] = 1;
}

static const size_t kBytes = 256ull << 20;
static const int kReps = 8;

struct Gpu {
	int id = 0;
	void* host = nullptr;
	void* host_mapped = nullptr;
	void* dev = nullptr;
	unsigned* sink = nullptr;
	cudaStream_t s[4];
};

static double now() { return std::chrono::duration<double>( std::chrono::steady_clock::now().time_since_epoch() ).count(); }

static void sync_all( std::vector<Gpu>& g ) {
	for ( auto& x : g ) { cudaSetDevice( x.id ); cudaDeviceSynchronize(); }
}

// Seconds for `reps` copies per stream on `streams` streams of every GPU in `g` at once
static double copies( std::vector<Gpu>& g, int streams, bool h2d ) {
	sync_all( g );
	const double t0 = now();
	for ( int r = 0; r < kReps; ++r ) {
		for ( auto& x : g ) {
			cudaSetDevice( x.id );
			for ( int i = 0; i < streams; ++i ) {
				if ( h2d ) cudaMemcpyAsync( x.dev, x.host, kBytes, cudaMemcpyHostToDevice, x.s[i] );
				else cudaMemcpyAsync( x.host, x.dev, kBytes, cudaMemcpyDeviceToHost, x.s[i] );
			}
		}
	}
	sync_all( g );
	return now() - t0;
}

static double zero_copy( std::vector<Gpu>& g ) {
	sync_all( g );
	const double t0 = now();
	for ( int r = 0; r < kReps; ++r ) {
		for ( auto& x : g ) {
			cudaSetDevice( x.id );
			void* dp = nullptr;
			cudaHostGetDevicePointer( &dp, x.host_mapped, 0 );
			read_kernel<<<384, 256, 0, x.s[0]>>>( static_cast<const uint4*>( dp ), kBytes / 16, x.sink );
		}
	}
	sync_all( g );
	return now() - t0;
}

int main() {
	int n = 0;
	cudaGetDeviceCount( &n );
	std::vector<Gpu> gpus( n );
	for ( int d = 0; d < n; ++d ) {
		Gpu& x = gpus[d];
		x.id = d;
		cudaSetDevice( d );
		cudaHostAlloc( &x.host, kBytes, 0 );
		cudaHostAlloc( &x.host_mapped, kBytes, cudaHostAllocMapped );
		cudaMalloc( &x.dev, kBytes );
		cudaMalloc( reinterpret_cast<void**>( &x.sink ), 4 );
		for ( int i = 0; i < 4; ++i ) cudaStreamCreate( &x.s[i] );
		for ( size_t k = 0; k < kBytes; k += 4096 ) { static_cast<char*>( x.host )[k] = 1; static_cast<char*>( x.host_mapped )[k] = 1; }
		cudaDeviceProp p;
		cudaGetDeviceProperties( &p, d );
		std::printf( "GPU %d: %s\n", d, p.name );
	}
	std::printf( "\nOne GPU at a time (GB/s):\n  gpu | H2D 1 stream | H2D 2 | H2D 4 | D2H 1 | in-place kernel read\n" );
	for ( int d = 0; d < n; ++d ) {
		std::vector<Gpu> one = { gpus[d] };
		copies( one, 1, true );
		const double h1 = kReps * 1.0 * kBytes / copies( one, 1, true ) / 1e9, h2 = kReps * 2.0 * kBytes / copies( one, 2, true ) / 1e9;
		const double h4 = kReps * 4.0 * kBytes / copies( one, 4, true ) / 1e9, d1 = kReps * 1.0 * kBytes / copies( one, 1, false ) / 1e9;
		zero_copy( one );
		const double zc = kReps * 1.0 * kBytes / zero_copy( one ) / 1e9;
		std::printf( "  %3d | %12.1f | %5.1f | %5.1f | %5.1f | %8.1f\n", d, h1, h2, h4, d1, zc );
	}
	if ( n > 1 ) {
		std::printf( "\nAll %d GPUs at once, host to device (per GPU, and the total):\n", n );
		copies( gpus, 1, true );
		const double t = copies( gpus, 1, true );
		std::printf( "  %.1f GB/s per GPU, %.1f GB/s total\n", kReps * 1.0 * kBytes / t / 1e9, n * kReps * 1.0 * kBytes / t / 1e9 );
		const double tz = zero_copy( gpus );
		std::printf( "All %d GPUs at once, in-place kernel reads: %.1f GB/s per GPU, %.1f GB/s total\n", n, kReps * 1.0 * kBytes / tz / 1e9, n * kReps * 1.0 * kBytes / tz / 1e9 );
	}
	return 0;
}
