#pragma once

#include <cstdint>

namespace engine::kernels::cpu {

// One quantized matrix [in -> out] on the host CPU, whatever its storage format. A format provides the matrix-vector product over groups of
// 16 output columns (`units`, the work items the pool hands out) and the transforms the format wraps around it (EXL3: sign flip and Hadamard
// rotation of the input, Hadamard rotation and scale of the output; NVFP4: the scales), both in blocks of kBlock (128) columns so the
// expert pipeline can run them as soon as a block's units are done. in() and out() are multiples of kBlock.
class Linear {
public:
	virtual ~Linear() {}
	virtual int in() const = 0;
	virtual int out() const = 0;
	int units() const { return out() / 16; }
	// Whether run() reads row_sum (the sum of a prepared input row); the pipeline then computes it
	virtual bool needs_row_sum() const { return false; }
	// Block b (kBlock values) of an input row, FP32, to the kernel's input form (kBlock floats of storage, whatever the form)
	virtual void prepare_block( const float* x, int b, float* xin ) const = 0;
	// Units [u0, u1) for `rows` rows. xin: [rows][in()] prepared input; row_sum[r] = sum of row r of xin (formats that need it); raw: [rows][out()]
	virtual void run( int rows, const float* xin, const float* row_sum, int u0, int u1, float* raw ) const = 0;
	// Block b (kBlock values) of an output row, in place: raw product to the linear's FP32 output (before the caller's FP16 rounding)
	virtual void finish_block( float* v, int b ) const = 0;
};

}   // namespace engine::kernels::cpu
