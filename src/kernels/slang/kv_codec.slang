// KV cache element codecs (inc/engine/core/kv_format.h), the portable twin of src/backend/cuda/modules/kv_quant.cuh:
//   0 F16   2 bytes per element
//   1 E5M2  the high byte of the FP16 value
//   2 I8    int8, one FP16 scale per 64 elements (amax / 127)
//   3 Q4    4 bits (offset 8), one FP16 scale per 64 elements (amax / 7); values are Walsh-Hadamard rotated per 64 before rounding
//   4 FP4   E2M1, one E4M3 scale byte per 16 elements (amax / 6)
// Element indices are flat within the pool (slot offset included), so the scale index is the element index >> 6 (or >> 4).
// Data is addressed as bytes (uint8_t*) and scales as halves or bytes, the way the CUDA side does.

float kv_e2m1_value(uint nib) {
	float mag = 0.0f;
	switch (nib & 7u) {
		case 1: mag = 0.5f; break;
		case 2: mag = 1.0f; break;
		case 3: mag = 1.5f; break;
		case 4: mag = 2.0f; break;
		case 5: mag = 3.0f; break;
		case 6: mag = 4.0f; break;
		case 7: mag = 6.0f; break;
		default: break;
	}
	return (nib & 8u) != 0u ? -mag : mag;
}

uint kv_e2m1_encode(float x) {
	let a = abs(x);
	// ties to the even code, as engine::quant::e2m1_index (inc/engine/quant/fp4_fp8.h): 0.25, 1.25, 2.5 and 5.0 round down, 0.75, 1.75 and 3.5 up
	uint idx = a <= 0.25f ? 0u : a < 0.75f ? 1u : a <= 1.25f ? 2u : a < 1.75f ? 3u : a <= 2.5f ? 4u : a < 3.5f ? 5u : a <= 5.0f ? 6u : 7u;
	return idx | ((x < 0.0f && idx != 0u) ? 8u : 0u);
}

// Exact: sub-normals are m / 512, normals (1 + m / 8) * 2^(e - 7), built from the float bit pattern
float kv_e4m3_value(uint b) {
	let e = (b >> 3) & 15u;
	let m = b & 7u;
	float v = 0.0f;
	if (e == 0u) v = float(m) * 0.001953125f;
	else v = asfloat(((e + 120u) << 23) | (m << 20));
	return (b & 0x80u) != 0u ? -v : v;
}

// Round to nearest even, saturating at 448 (the largest finite E4M3); the scales are never negative
uint kv_e4m3_encode(float x) {
	float a = min(abs(x), 448.0f);
	if (!(a > 0.0f)) return 0u;
	int e = max(int((asuint(a) >> 23) & 255u) - 127, -6);
	let step = asfloat(uint(e - 3 + 127) << 23);
	float qf = a / step;  // step is a power of two: exact
	float fl = floor(qf);
	float d = qf - fl;
	int q = int(fl);
	if (d > 0.5f || (d == 0.5f && (q & 1) != 0)) q += 1;
	if (e == -6 && q < 8) return uint(q);
	if (q >= 16) { q = 8; e += 1; }
	uint code = (uint(e + 7) << 3) | uint(q - 8);
	return min(code, 0x7Eu);
}

// FP16 -> E5M2: keep the top 8 bits with round to nearest even; saturate instead of overflowing to infinity
uint kv_e5m2_encode(float x) {
	let bits = f32tof16(x) & 0xFFFFu;
	let sign = bits & 0x8000u;
	let mag = bits & 0x7FFFu;
	if (mag >= 0x7C00u) return (sign >> 8) | (mag > 0x7C00u ? 0x7Eu : 0x7Bu);
	uint r = (mag + 0x7Fu + ((mag >> 8) & 1u)) >> 8;
	if (r > 0x7Bu) r = 0x7Bu;
	return (sign >> 8) | r;
}

int kv_rne(float x) {
	let fl = floor(x);
	let d = x - fl;
	int q = int(fl);
	if (d > 0.5f || (d == 0.5f && (q & 1) != 0)) q += 1;
	return q;
}

// a / b rounded like an IEEE division: GPU divisions may be a few ULP off (Vulkan allows 2.5), which flips quantization codes
// on rounding ties; one residual correction step makes the result the correctly rounded quotient
float kv_div(float a, float b) {
	let q = a / b;
	let r = fma(-q, b, a);
	return fma(r, 1.0f / b, q);
}

// float -> FP16 bits, round to nearest even (the scales are never negative or non-finite). Done with integer arithmetic on purpose:
// the NVIDIA Vulkan driver folds a float -> half -> float round trip into a no-op, which would leave the in-register scale unrounded.
uint kv_f16_bits(float x) {
	let u = asuint(x);
	let e = int((u >> 23) & 255u) - 127 + 15;
	let m = u & 0x7FFFFFu;
	if (e >= 31) return 0x7C00u;
	if (e <= 0) {
		if (e < -10) return 0u;
		let mm = m | 0x800000u;
		let shift = uint(14 - e);
		var h = mm >> shift;
		let rem = mm & ((1u << shift) - 1u);
		let half_way = 1u << (shift - 1u);
		if (rem > half_way || (rem == half_way && (h & 1u) != 0u)) h += 1u;
		return h;
	}
	var h = (uint(e) << 10) | (m >> 13);
	let rem = m & 0x1FFFu;
	if (rem > 0x1000u || (rem == 0x1000u && (h & 1u) != 0u)) h += 1u;
	return min(h, 0x7C00u);
}

float kv_load(uint8_t* data, uint8_t* sc, int q, int64_t i) {
	if (q == 0) return float(((half*)data)[i]);
	if (q == 1) return f16tof32(uint(data[i]) << 8);
	if (q == 2) {
		int s8 = int(data[i]);
		if (s8 > 127) s8 -= 256;
		return float(s8) * float(((half*)sc)[i >> 6]);
	}
	let nib = (uint(data[i >> 1]) >> (uint(i & 1) * 4u)) & 15u;
	if (q == 3) return float(int(nib) - 8) * float(((half*)sc)[i >> 6]);
	return kv_e2m1_value(nib) * kv_e4m3_value(uint(sc[i >> 4]));
}

// In-place orthonormal 64-point Walsh-Hadamard transform (its own inverse)
void kv_fwht64(inout float v[64]) {
	for (int h = 1; h < 64; h <<= 1) {
		for (int i = 0; i < 64; ++i) {
			if ((i & h) == 0) {
				let a = v[i];
				let b = v[i | h];
				v[i] = a + b;
				v[i | h] = a - b;
			}
		}
	}
	for (int i = 0; i < 64; ++i) v[i] *= 0.125f;
}

// Quantize one unit of `kv_unit(q)` source elements starting at flat element idx0 (unit aligned)
int kv_unit(int q) {
	return (q == 2 || q == 3) ? 64 : (q == 4 ? 16 : 1);
}

void kv_store_unit(uint8_t* data, uint8_t* sc, int q, int64_t idx0, half* src) {
	if (q == 0) {
		((half*)data)[idx0] = src[0];
	}
	else if (q == 1) {
		data[idx0] = uint8_t(kv_e5m2_encode(float(src[0])));
	}
	else if (q == 2 || q == 3) {
		float v[64];
		float amax = 0.0f;
		for (int i = 0; i < 64; ++i) v[i] = float(src[i]);
		if (q == 3) kv_fwht64(v);
		for (int i = 0; i < 64; ++i) amax = max(amax, abs(v[i]));
		let sbits = kv_f16_bits(kv_div(amax, q == 2 ? 127.0f : 7.0f));
		let s = f16tof32(sbits);
		let inv = s > 0.0f ? kv_div(1.0f, s) : 0.0f;
		((half*)sc)[idx0 >> 6] = asfloat16(uint16_t(sbits));
		if (q == 2) {
			for (int i = 0; i < 64; ++i) data[idx0 + i] = uint8_t(max(-127, min(127, kv_rne(v[i] * inv))) & 255);
		}
		else {
			for (int i = 0; i < 64; i += 2) {
				let lo = max(-8, min(7, kv_rne(v[i] * inv))) + 8;
				let hi = max(-8, min(7, kv_rne(v[i + 1] * inv))) + 8;
				data[(idx0 >> 1) + (i >> 1)] = uint8_t(uint(lo) | (uint(hi) << 4));
			}
		}
	}
	else {
		float v[16];
		float amax = 0.0f;
		for (int i = 0; i < 16; ++i) {
			v[i] = float(src[i]);
			amax = max(amax, abs(v[i]));
		}
		let sb = kv_e4m3_encode(kv_div(amax, 6.0f));
		let s = kv_e4m3_value(sb);
		let inv = s > 0.0f ? kv_div(1.0f, s) : 0.0f;
		sc[idx0 >> 4] = uint8_t(sb);
		for (int i = 0; i < 16; i += 2) {
			data[(idx0 >> 1) + (i >> 1)] = uint8_t(kv_e2m1_encode(v[i] * inv) | (kv_e2m1_encode(v[i + 1] * inv) << 4));
		}
	}
}
