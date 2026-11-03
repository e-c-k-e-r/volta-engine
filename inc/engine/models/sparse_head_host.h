#pragma once

#include "engine/core/tensor.h"
#include "engine/modules/linear.h"
#include "engine/core/arena.h"
#include "engine/backend/types.h"

namespace engine::models {

// The sparse drafter head of one rank (option sparse_head, MTP draft steps; modules/sparse_head): up to cap() columns of this rank's LM-head
// shard copied to an FP16 matrix, and the draft logits computed from them instead of the full head while in use. A model owns one, calls
// init() while it allocates (nothing is allocated, and cap() is 0, while the option is off) and logits() in its draft head.
class SparseHeadHost {
public:
	static constexpr int kIdentRows = 128;   // identity rows per head pass when the columns are read out of the head

	// rows: the most draft rows one logits() call takes
	void init( pod::MemoryArena& arena, int hidden, int shard_real_cols, int shard_cols, int rows );
	int cap() const { return cap_; }
	bool in_use() const { return use_; }
	void use( bool on ) { use_ = on && count_ > 0; }
	// Reads columns cols[0 .. count) (shard-local) out of the head; false without a sparse head
	bool set( const pod::LinearWeight& head, const int* cols, int count, int rank, backend::stream_t stream );
	// logits [rows, shard_cols]: the subset's columns from x [rows, hidden], every other column at the lowest value
	void logits( const backend::half_t* x, int rows, backend::half_t* logits, backend::stream_t stream ) const;

private:
	int hidden_ = 0, shard_cols_ = 0, rows_ = 0, cap_ = 0, count_ = 0;
	bool use_ = false;
	backend::half_t* w_ = nullptr;       // [cap, hidden]
	int* cols_ = nullptr;                // [cap] shard-local column of each row of w_
	backend::half_t* sub_ = nullptr;     // [rows, cap] subset logits
	backend::half_t* ident_ = nullptr;   // [kIdentRows, hidden]
	backend::half_t* chunk_ = nullptr;   // [kIdentRows, shard_cols]
};

// The Model overrides of a pipeline that owns a SparseHeadHost named `host`, an LM head `head` and its rank `rank`
#define ENGINE_SPARSE_HEAD_HOOKS( host, head, rank ) \
	int sparse_head_cap() const override { return host.cap(); } \
	bool sparse_head_set( const int* cols, int count, backend::stream_t stream = nullptr ) override { return host.set( head, cols, count, rank, stream ); } \
	void sparse_head_use( bool on ) override { host.use( on ); }

} // namespace engine::models
