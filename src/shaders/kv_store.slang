// Paged KV store (Tier B), FP16 cache only: scatter one K or V row per batch row into
// data[slot * slot_stride + (layer * max_seq + pos) * total_elems]. Quantized cache formats are CUDA-only.

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
