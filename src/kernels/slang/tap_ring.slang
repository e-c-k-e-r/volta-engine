// Speculative-decode tap ring kernels (Tier B): publish a hidden-state row into the 5-tap ring at pos % cap, and reduce the
// four mHC streams to one FP16 row (mean, or a single stream). Plain per-element kernels: one source for every target.

struct TapStoreParams {
	half* taps;       // [cap, 5, H]
	half* hidden;     // [m, H]
	int* slots;       // BatchRows::slot
	int* positions;   // BatchRows::pos
	int* owner;       // device scalar: the slot whose rows are published
	int m;
	int tap;
	int H;
	int cap;
};

// Group y selects the row, x tiles the H elements
[shader("compute")]
[numthreads(256, 1, 1)]
void dflash_store_taps(uint3 tid : SV_DispatchThreadID, uniform TapStoreParams p) {
	let r = int(tid.y);
	let i = int(tid.x);
	if (r >= p.m || i >= p.H) return;
	if (p.slots[r] != p.owner[0]) return;
	let pos = p.positions[r];
	if (pos < 0) return;
	p.taps[int64_t(pos % p.cap) * 5 * p.H + int64_t(p.tap) * p.H + i] = p.hidden[int64_t(r) * p.H + i];
}

struct ReduceStreamsParams {
	float* streams;   // [m, 4 * H]
	half* out_;       // [m, H]
	int H;
	int mode;         // 0: mean of the four streams, k > 0: stream k - 1
};

[shader("compute")]
[numthreads(256, 1, 1)]
void reduce_streams(uint3 tid : SV_DispatchThreadID, uniform ReduceStreamsParams p) {
	let i = int(tid.x);
	if (i >= p.H) return;
	let r = int64_t(tid.y);
	float* in_ = p.streams + r * 4 * p.H;
	float v;
	if (p.mode == 0) v = (in_[i] + in_[p.H + i] + in_[2 * p.H + i] + in_[3 * p.H + i]) * 0.25f;
	else v = in_[(p.mode - 1) * p.H + i];
	p.out_[r * p.H + i] = half(v);
}
