// Reference dense FP16 GEMM: out[m][n] = sum_k x[m][k] * W[k][n], FP32 accumulation, FP16 output.
// x is [m, k] row-major, W is [k, n] row-major (the plain Linear layout, pod::LinearWeight::make_fp16), out is [m, n].
// Slow-but-correct portable reference. GPU entry: 16x16 groups with shared-memory tiles, grid (ceil(n/16), ceil(m/16)).
// Host entry: one invocation per output element, grid (n, m).

struct RefGemmParams {
	half* x;
	half* w;
	half* out_;
	int m;
	int k;
	int n;
};

#if !ENGINE_TARGET_CPU
groupshared float s_x[256];
groupshared float s_w[256];

[shader("compute")]
[numthreads(16, 16, 1)]
void ref_gemm_f16(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform RefGemmParams p) {
	let tx = int(tid.x);
	let ty = int(tid.y);
	let row = int(gid.y) * 16 + ty;
	let col = int(gid.x) * 16 + tx;
	float acc = 0.0f;
	for (int k0 = 0; k0 < p.k; k0 += 16) {
		// every thread takes part in the barriers; out-of-range loads are zero
		s_x[ty * 16 + tx] = (row < p.m && k0 + tx < p.k) ? float(p.x[int64_t(row) * p.k + k0 + tx]) : 0.0f;
		s_w[ty * 16 + tx] = (col < p.n && k0 + ty < p.k) ? float(p.w[int64_t(k0 + ty) * p.n + col]) : 0.0f;
		GroupMemoryBarrierWithGroupSync();
		for (int i = 0; i < 16; ++i)
			acc = fma(s_x[ty * 16 + i], s_w[i * 16 + tx], acc);
		GroupMemoryBarrierWithGroupSync();
	}
	if (row < p.m && col < p.n)
		p.out_[int64_t(row) * p.n + col] = half(acc);
}
// NT form: W is [n, k] row-major (LinearPlainGEMV layout: out[m][n] = sum_k x[m][k] * w[n][k])
[shader("compute")]
[numthreads(16, 16, 1)]
void ref_gemm_nt_f16(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform RefGemmParams p) {
	let tx = int(tid.x);
	let ty = int(tid.y);
	let row = int(gid.y) * 16 + ty;
	let col = int(gid.x) * 16 + tx;
	float acc = 0.0f;
	for (int k0 = 0; k0 < p.k; k0 += 16) {
		s_x[ty * 16 + tx] = (row < p.m && k0 + tx < p.k) ? float(p.x[int64_t(row) * p.k + k0 + tx]) : 0.0f;
		// thread (tx, ty) stages w[gid.x * 16 + ty][k0 + tx]; the product reads it as s_w[i * 16 + col_in_tile]
		let wn = int(gid.x) * 16 + ty;
		s_w[tx * 16 + ty] = (wn < p.n && k0 + tx < p.k) ? float(p.w[int64_t(wn) * p.k + k0 + tx]) : 0.0f;
		GroupMemoryBarrierWithGroupSync();
		for (int i = 0; i < 16; ++i)
			acc = fma(s_x[ty * 16 + i], s_w[i * 16 + tx], acc);
		GroupMemoryBarrierWithGroupSync();
	}
	if (row < p.m && col < p.n)
		p.out_[int64_t(row) * p.n + col] = half(acc);
}
#else
[shader("compute")]
[numthreads(1, 1, 1)]
void ref_gemm_f16(uint3 gid : SV_GroupID, uniform RefGemmParams p) {
	let col = int(gid.x);
	let row = int(gid.y);
	float acc = 0.0f;
	for (int i = 0; i < p.k; ++i)
		acc = fma(float(p.x[int64_t(row) * p.k + i]), float(p.w[int64_t(i) * p.n + col]), acc);
	p.out_[int64_t(row) * p.n + col] = half(acc);
}
[shader("compute")]
[numthreads(1, 1, 1)]
void ref_gemm_nt_f16(uint3 gid : SV_GroupID, uniform RefGemmParams p) {
	let col = int(gid.x);
	let row = int(gid.y);
	float acc = 0.0f;
	for (int i = 0; i < p.k; ++i)
		acc = fma(float(p.x[int64_t(row) * p.k + i]), float(p.w[int64_t(col) * p.k + i]), acc);
	p.out_[int64_t(row) * p.n + col] = half(acc);
}
#endif
