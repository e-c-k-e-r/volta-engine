#pragma once

#include "engine/config.h"
#include <cstddef>
#include <cstdint>

namespace pod {

enum class DType : uint8_t {
	FP16,
	BF16,
	FP32,
	INT32,
	INT8,
	UINT16,
	UINT32,
	UINT8,
	TRELLIS_2,
	TRELLIS_3,
	TRELLIS_4,
	TRELLIS_5,
	TRELLIS_6,
	NVFP4,
	AUTO = 255
};

inline size_t dtype_bytes( DType dtype ) {
	switch ( dtype ) {
		case DType::FP16:
		case DType::BF16:
		case DType::UINT16: {
			return 2;
		} break;
		case DType::FP32:
		case DType::INT32:
		case DType::UINT32: {
			return 4;
		} break;
		case DType::INT8:
		case DType::UINT8:
		case DType::NVFP4: {
			return 1;
		} break;
		case DType::TRELLIS_2:
		case DType::TRELLIS_3:
		case DType::TRELLIS_4:
		case DType::TRELLIS_5:
		case DType::TRELLIS_6: {
			return 2;
		} break;
		default: {
			return 1;
		} break;
	}
}

struct TensorView {
	void*  data;            // Raw GPU device pointer (cudaMalloc)
	DType  dtype;           // Data type identifier
	int    ndim;            // Dimensions count (1 to 4)
	int    shape[4];        // Shape array
	size_t strides[4];      // Strides in elements

	template<typename T>
	T* ptr() {
		return static_cast<T*>( data );
	}

	template<typename T>
	const T* ptr() const {
		return static_cast<const T*>( data );
	}

	inline size_t numel() const {
		size_t n = 1;
		for ( int i = 0; i < ndim; ++i ) {
			n *= static_cast<size_t>( shape[i] );
		}
		return n;
	}

	inline size_t size_bytes() const {
		return numel() * dtype_bytes( dtype );
	}
};

inline TensorView make_tensor_1d( void* data, DType dtype, int d0 ) {
	TensorView v;
	v.data = data;
	v.dtype = dtype;
	v.ndim = 1;
	v.shape[0] = d0;
	v.shape[1] = 0;
	v.shape[2] = 0;
	v.shape[3] = 0;
	v.strides[0] = 1;
	v.strides[1] = 0;
	v.strides[2] = 0;
	v.strides[3] = 0;
	return v;
}

inline TensorView make_tensor_2d( void* data, DType dtype, int d0, int d1 ) {
	TensorView v;
	v.data = data;
	v.dtype = dtype;
	v.ndim = 2;
	v.shape[0] = d0;
	v.shape[1] = d1;
	v.shape[2] = 0;
	v.shape[3] = 0;
	v.strides[0] = static_cast<size_t>( d1 );
	v.strides[1] = 1;
	v.strides[2] = 0;
	v.strides[3] = 0;
	return v;
}

inline TensorView make_tensor_3d( void* data, DType dtype, int d0, int d1, int d2 ) {
	TensorView v;
	v.data = data;
	v.dtype = dtype;
	v.ndim = 3;
	v.shape[0] = d0;
	v.shape[1] = d1;
	v.shape[2] = d2;
	v.shape[3] = 0;
	v.strides[1] = static_cast<size_t>( d2 );
	v.strides[0] = static_cast<size_t>( d1 ) * v.strides[1];
	v.strides[2] = 1;
	v.strides[3] = 0;
	return v;
}

inline TensorView make_tensor_4d( void* data, DType dtype, int d0, int d1, int d2, int d3 ) {
	TensorView v;
	v.data = data;
	v.dtype = dtype;
	v.ndim = 4;
	v.shape[0] = d0;
	v.shape[1] = d1;
	v.shape[2] = d2;
	v.shape[3] = d3;
	v.strides[2] = static_cast<size_t>( d3 );
	v.strides[1] = static_cast<size_t>( d2 ) * v.strides[2];
	v.strides[0] = static_cast<size_t>( d1 ) * v.strides[1];
	v.strides[3] = 1;
	return v;
}

} // namespace pod
