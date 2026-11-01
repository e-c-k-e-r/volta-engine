#pragma once

#include "engine/backend/types.h"
#include "engine/modules/linear.h"
#include <cstdint>

namespace engine::modules {

// Sparse drafter head: an FP16 copy of the LM-head columns of a subset of the vocabulary, so a draft step reads a few MB instead of the whole
// (quantized) head. Each rank keeps the columns of its own vocabulary shard; the draft logits of the subset are scattered into the shard's
// logits row with every other column at the lowest FP16 value, so the vocabulary-parallel argmax / top-k that follows is unchanged.
// Verification always runs the full head: the subset only decides which tokens the drafter can propose.
struct SparseHead {
	// Columns are read from the head by running it on blocks of identity rows (any head format: EXL3 in either tile order, NVFP4, FP16):
	// ident = rows [k0, k0 + rows) of the hidden-size identity, chunk = ident x head^T = the head's weights at those inputs for every column
	static void identity_rows( backend::half_t* ident, int rows, int hidden, int k0, backend::stream_t stream );
	// w_sub[j, k0 + i] = chunk[i, cols[j]] for the count selected columns (w_sub row-major [count, hidden])
	static void gather( const backend::half_t* chunk, int chunk_ld, int rows, int k0, const int* cols, int count, backend::half_t* w_sub, int hidden,
		backend::stream_t stream );
	// The FP16 columns cols[0 .. count) (device, shard-local) of a head shard [hidden -> shard_cols] into w_sub [count, hidden]: the head runs
	// on identity blocks of ident_rows rows (ident [ident_rows, hidden], chunk [ident_rows, shard_cols] scratch). Synchronizes the stream
	static void extract( const pod::LinearWeight& head, int hidden, int shard_cols, const int* cols, int count, backend::half_t* ident, backend::half_t* chunk,
		int ident_rows, backend::half_t* w_sub, backend::stream_t stream );
	// The draft logits of rows x [rows <= 8, hidden] through the subset: sub [rows, count] = x w_sub^T, then scattered into the shard's logits rows
	static void logits( const backend::half_t* w_sub, int count, const int* cols, const backend::half_t* x, int rows, int hidden, backend::half_t* sub,
		backend::half_t* logits, int logits_ld, backend::stream_t stream );
	// logits[r, c] = -65504 for every column of the shard, then logits[r, cols[j]] = sub[r, j]
	static void scatter( const backend::half_t* sub, int count, const int* cols, backend::half_t* logits, int logits_ld, int rows, backend::stream_t stream );
};

} // namespace engine::modules
