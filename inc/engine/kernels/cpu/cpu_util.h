#pragma once

#include <cstddef>
#include <cstdint>

// Host CPU helpers shared by the CPU kernels (src/kernels/cpu): ISA detection, FP16 conversion, the 128-point Hadamard transform
namespace engine::kernels::cpu {

constexpr int kBlock = 128;                       // columns the Hadamard transform and the expert pipeline work in
constexpr float kRScale = 0.08838834764831845f;   // 1 / sqrt( 128 )

bool avx2_available();                            // AVX2 + FMA + F16C
float half_bits_to_float( uint16_t h );
void half_to_float( const uint16_t* src, float* dst, size_t n );
void float_to_half( const float* src, uint16_t* dst, size_t n );
void round_to_half( float* v, int n );            // v[i] = float( half( v[i] ) ), n <= 128
void hadamard128( float* v );                     // in place, unnormalised

}   // namespace engine::kernels::cpu
