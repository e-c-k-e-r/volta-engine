// Paged KV store (Tier B): scatter one K or V row per batch row into
// data[slot * slot_stride + (layer * max_seq + pos) * total_elems]. kv_store_f16 is the FP16 cache; kv_store_q quantizes
// into any of the formats of kv_codec.slang (one invocation per quantization unit: 1, 16 or 64 elements).
#include "kv_codec.slang"

struct KvStoreParams {
	half* data;
	half* src;        // [count, total_elems]
	int* slots;       // [count] (BatchRows::slot)
	int* positions;   // [count] (BatchRows::pos)
	int64_t slot_stride;
	int layer_idx;
	int count;
	int total_elems;
	int max_seq;
};

[shader("compute")]
[numthreads(128, 1, 1)]
void kv_store_f16(uint3 tid : SV_DispatchThreadID, uniform KvStoreParams p) {
	let i = int64_t(tid.x);
	if (i >= int64_t(p.count) * p.total_elems) return;
	let m = int(i / p.total_elems);
	let e = i - int64_t(m) * p.total_elems;
	let dst = int64_t(p.slots[m]) * p.slot_stride + (int64_t(p.layer_idx) * p.max_seq + p.positions[m]) * p.total_elems + e;
	p.data[dst] = p.src[i];
}

struct KvStoreQParams {
	uint8_t* data;
	uint8_t* scales;  // null for FP16 / E5M2
	half* src;        // [count, total_elems]
	int* slots;
	int* positions;
	int64_t slot_stride;
	int layer_idx;
	int count;
	int total_elems;
	int max_seq;
	int q;            // 0 F16, 1 E5M2, 2 I8, 3 Q4, 4 FP4
};

[shader("compute")]
[numthreads(128, 1, 1)]
void kv_store_q(uint3 tid : SV_DispatchThreadID, uniform KvStoreQParams p) {
	let unit = kv_unit(p.q);
	let per_row = p.total_elems / unit;
	let u = int64_t(tid.x);
	if (u >= int64_t(p.count) * per_row) return;
	let m = int(u / per_row);
	let e0 = (u - int64_t(m) * per_row) * unit;
	let dst = int64_t(p.slots[m]) * p.slot_stride + (int64_t(p.layer_idx) * p.max_seq + p.positions[m]) * p.total_elems + e0;
	kv_store_unit(p.data, p.scales, p.q, dst, p.src + int64_t(m) * p.total_elems + e0);
}
