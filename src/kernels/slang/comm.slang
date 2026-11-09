// The multi-device collective's sum on the device (Vulkan, src/backend/vulkan/mod_comm.cpp): the ranks' partials, read from the host block
// every device imports, summed in rank order in FP32, plus the residual when there is one, then one rounding to FP16 (the host path's
// arithmetic). out may be the residual (each element is read before it is written).
struct ArSumParams { half* parts; half* res; half* out_; int64_t rank_stride; int64_t slot_off; int n; int nranks; int has_res; };

[shader("compute")]
[numthreads(256, 1, 1)]
void ar_sum(uint3 gid : SV_DispatchThreadID, uniform ArSumParams p) {
	let i = int64_t(gid.x) + int64_t(gid.y) * 65535 * 256;
	if (i >= p.n) return;
	float s = 0.0f;
	for (int r = 0; r < p.nranks; ++r) s += float(p.parts[int64_t(r) * p.rank_stride + p.slot_off + i]);
	if (p.has_res != 0) s += float(p.res[i]);
	p.out_[i] = half(s);
}
