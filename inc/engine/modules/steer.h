#pragma once

#include "engine/backend/types.h"
#include "engine/core/batch.h"
#include <cstdint>
#include <string>
#include <vector>

namespace engine::modules {

// Runtime steering: per-slot edits of the residual stream (control vectors, direction clamps) and a per-slot logit penalty on reflection markers.
// Everything the kernels read is device-resident state indexed by the slot of each row (BatchRows), so the decode graphs stay valid and a
// slot without steering is untouched (its rows return before reading or writing the residual).
constexpr int kSteerEntries = 2;         // edits per steered layer, applied in order (the second sees the first's result)
constexpr int kSteerPenaltyIds = 256;    // token ids of a slot's logit penalty

enum SteerKind : int32_t { STEER_OFF = 0, STEER_ADD = 1, STEER_CLAMP = 2 };
// Rows a slot's residual edit applies to. A row is `prompt` below prompt_len, `think` from there to think_end, `answer` after it.
enum SteerScope : int32_t { STEER_PROMPT = 1, STEER_THINK = 2, STEER_ANSWER = 4, STEER_GENERATED = 6, STEER_ALL = 7 };

// One edit as the host sees it: `layer` is a model layer index, par the strength (add) or the reference level (clamp), vec the direction (hidden floats)
struct SteerEntry {
	int layer = 0;
	int kind = STEER_ADD;
	float par = 0.0f;
	std::vector<float> vec;
};

// "first-last" of the steer_layers option: each side an index, a fraction of the layer count (0.33, 1/3) or "last"; empty = 1/3-2/3. False when malformed.
bool parse_steer_layers( const std::string& spec, int num_layers, int* first, int* last );

struct SteerSlotCtl {
	int32_t scope;        // SteerScope bits of the residual edit; 0 = the slot is not steered
	int32_t prompt_len;   // tokens of the request's prompt (rows below it are prompt rows)
	int32_t think_end;    // position after the reasoning block (INT32_MAX while it is open or can open, 0 = no reasoning block)
	int32_t penalty_n;    // marker ids in use (0 = no penalty)
	float penalty;        // logit subtracted from them while the predicted token lies inside the reasoning block
	int32_t pad[3];
};

// Device state of one rank. vec [slots][layers][kSteerEntries][hidden] fp32, par and kind [slots][layers][kSteerEntries]: the strength of an
// add entry, the reference level of a clamp entry.
struct SteerTables {
	int hidden = 0;
	int layers = 0;       // steered layers
	int slots = 0;
	float* vec = nullptr;
	float* par = nullptr;
	int32_t* kind = nullptr;
	SteerSlotCtl* ctl = nullptr;
	int32_t* penalty_ids = nullptr;   // [slots][kSteerPenaltyIds] global token ids
};

struct ResidualSteer {
	// hidden [rows, hidden] fp16 after steered layer `layer_index`: per row, the entries of the row's slot
	//   add    x += strength * v
	//   clamp  x -= max( x . v - ref, 0 ) * v
	// in fp32, written back to fp16 (saturated). Rows of a slot whose scope does not cover them are not touched.
	static void apply( const SteerTables& tables, backend::half_t* hidden, const pod::BatchRows* d_rows, int rows, int layer_index, backend::stream_t stream );

	// logits [*, logit_cols] fp16 of this rank's vocabulary slice [shard_lo, shard_lo + shard_cols): for the `count` logit rows j (the forward's row
	// row0 + j), subtract the slot's penalty from its marker ids in the slice while the row's predicted token lies inside the reasoning block
	static void penalize( const SteerTables& tables, backend::half_t* logits, int logit_cols, const pod::BatchRows* d_rows, int row0, int count,
		int shard_lo, int shard_cols, backend::stream_t stream );
};

} // namespace engine::modules
