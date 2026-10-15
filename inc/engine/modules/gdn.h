#pragma once

#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/core/batch.h"
#include "engine/backend/types.h"

namespace engine::modules {

struct GDN {
	static void project_gates(
		const pod::TensorView& norm_hidden,
		const pod::TensorView& in_proj_a,
		const pod::TensorView& in_proj_b,
		const pod::TensorView& A_log,
		const pod::TensorView& dt_bias,
		pod::TensorView& gate_out,
		pod::TensorView& beta_out,
		int num_v_heads,
		int hidden_size,
		backend::stream_t stream = nullptr,
		int rows = 1                    // independent rows: norm_hidden [rows, hidden], gate/beta [rows, heads]
	);

	static void causal_conv1d_silu(
		pod::TensorView& qkv,
		pod::TensorView& conv_history,
		const pod::TensorView& conv_weights,
		int channels,
		backend::stream_t stream = nullptr,
		const pod::SlotRef& slot = {},
		float* snap = nullptr,          // per-row post-state copy, written only while *snap_on != 0
		const int* snap_on = nullptr
	);

	static void l2_norm_heads(
		pod::TensorView& qkv,
		int num_k_heads,
		int head_dim,
		backend::stream_t stream = nullptr,
		bool scale_q = false,
		int rows = 1,
		size_t row_stride = 0           // elements between rows of qkv
	);

	static void recurrent_step(
		float* gdn_state,
		const pod::TensorView& qkv,
		const pod::TensorView& gate,
		const pod::TensorView& beta,
		pod::TensorView& attn_out,
		int num_k_heads,
		int num_v_heads,
		int head_dim,
		backend::stream_t stream = nullptr,
		const pod::SlotRef& slot = {},
		float* snap = nullptr,          // per-row post-state copy, written only while *snap_on != 0
		const int* snap_on = nullptr
	);

	// Multi-row forms for a causal block or a batch: rows are walked in order per channel / head, the
	// recurrent state stays in registers across consecutive rows of one slot, and the slot of each row is
	// read on device from d_rows. snap (optional) receives the state after each of the first snap_rows rows
	// at snap + row * snap_stride while *snap_on != 0.
	static void causal_conv1d_silu_rows(
		pod::TensorView& qkv,             // [rows, channels]
		float* history,                   // layer base; slot s at + s * slot_stride
		const pod::TensorView& conv_weights,
		int channels,
		int rows,
		const pod::BatchRows* d_rows,
		size_t slot_stride,
		float* snap,
		size_t snap_stride,
		int snap_rows,
		const int* snap_on,
		backend::stream_t stream
	);

	// Prefill form (rows above the decode sizes, one slot): rows run in parallel tiles, reading src and writing a
	// separate dst; the history is then advanced to the last three inputs. Bit-identical to the sequential kernel.
	static void causal_conv1d_silu_tiled(
		const pod::TensorView& src,
		pod::TensorView& dst,
		float* history,
		const pod::TensorView& conv_weights,
		int channels,
		int rows,
		const pod::BatchRows* d_rows,
		size_t slot_stride,
		backend::stream_t stream
	);

	static void recurrent_step_rows(
		float* gdn_state,                 // layer base; slot s at + s * slot_stride
		const pod::TensorView& qkv,       // [rows, (2 * k_heads + v_heads) * head_dim]
		const pod::TensorView& gate,      // [rows, v_heads]
		const pod::TensorView& beta,
		pod::TensorView& attn_out,        // [rows, v_heads * head_dim]
		int num_k_heads,
		int num_v_heads,
		int head_dim,
		int rows,
		const pod::BatchRows* d_rows,
		size_t slot_stride,
		float* snap,
		size_t snap_stride,
		int snap_rows,
		const int* snap_on,
		backend::stream_t stream
	);

	static void project_kda_gates(
		const pod::TensorView& norm_hidden,
		const pod::TensorView& b_proj,
		const pod::TensorView& f_a_proj,
		const pod::TensorView& f_b_proj,
		const pod::TensorView& g_a_proj,
		const pod::TensorView& g_b_proj,
		const pod::TensorView& A_log,
		const pod::TensorView& dt_bias,
		pod::TensorView& beta_out,
		pod::TensorView& g_decay_out,
		pod::TensorView& z_out,
		pod::TensorView& f_a_buf,
		pod::TensorView& g_a_buf,
		int num_heads = 16,
		int head_dim = 128,
		int hidden_size = 4096,
		backend::stream_t stream = nullptr,
		int rows = 1                    // norm_hidden [rows, hidden]; outputs are [rows, ...]
	);

	// Multi-row KDA step with the per-head state held in registers across consecutive rows of one slot;
	// bit-identical to stepping recurrent_step_kda_fused row by row. Optional per-row post-state snapshots.
	static void recurrent_step_kda_rows(
		float* gdn_state,                 // layer base; slot s at + s * slot_stride
		const pod::TensorView& qkv,       // [rows, 3 * heads * 128]
		const pod::TensorView& g_decay,   // [rows, heads * 128]
		const pod::TensorView& beta,      // [rows, heads]
		const pod::TensorView& z,         // [rows, heads * 128]
		const pod::TensorView& o_norm,
		pod::TensorView& attn_out,        // [rows, heads * 128]
		int num_heads,
		int head_dim,
		float eps,
		int rows,
		const pod::BatchRows* d_rows,
		size_t slot_stride,
		float* snap,
		size_t snap_stride,
		int snap_rows,
		const int* snap_on,
		backend::stream_t stream
	);

	// Speculative verify without a state snapshot per row (36 MiB each): while *journal_on is set the rows step leaves the slot state as it was before
	// the block and records each row's k, v, decay and beta (kda_journal_row_bytes( heads ) bytes at journal + row * journal_row_stride, rows
	// below journal_rows); kda_replay then applies rows [first_row, first_row + nrows) of the journal to a slot's state of every linear layer
	// (layer l at gdn_state + l * layer_stride, its journal at journal + l * journal_layer_stride), bit-identical to having stepped them.
	// Where kda_journal_supported() is false (CPU, Vulkan) the models keep the snapshots.
	static bool kda_journal_supported();
	static size_t kda_journal_row_bytes( int num_heads );
	static void recurrent_step_kda_rows_journal(
		float* gdn_state,
		const pod::TensorView& qkv,
		const pod::TensorView& g_decay,
		const pod::TensorView& beta,
		const pod::TensorView& z,
		const pod::TensorView& o_norm,
		pod::TensorView& attn_out,
		int num_heads,
		int head_dim,
		float eps,
		int rows,
		const pod::BatchRows* d_rows,
		size_t slot_stride,
		unsigned char* journal,
		size_t journal_row_stride,
		int journal_rows,
		const int* journal_on,
		backend::stream_t stream
	);
	static void kda_replay(
		float* gdn_state,
		size_t slot_stride,
		size_t layer_stride,
		int layers,
		int slot,
		const unsigned char* journal,
		size_t journal_layer_stride,
		size_t journal_row_stride,
		int first_row,
		int nrows,
		int num_heads,
		backend::stream_t stream
	);

	static void recurrent_step_kda(
		float* gdn_state,
		const pod::TensorView& qkv,
		const pod::TensorView& g_decay,
		const pod::TensorView& beta,
		pod::TensorView& attn_out,
		int num_heads = 16,
		int head_dim = 128,
		backend::stream_t stream = nullptr
	);

	static void recurrent_step_kda_fused(
		float* gdn_state,
		const pod::TensorView& qkv,
		const pod::TensorView& g_decay,
		const pod::TensorView& beta,
		const pod::TensorView& z,
		const pod::TensorView& o_norm,
		pod::TensorView& attn_out,
		int num_heads = 16,
		int head_dim = 128,
		float eps = 1e-5f,
		backend::stream_t stream = nullptr,
		const pod::SlotRef& slot = {}
	);

	static void gated_rmsnorm(
		pod::TensorView& attn_out,
		const pod::TensorView& z,
		const pod::TensorView& norm_weight,
		int num_v_heads,
		int head_dim,
		backend::stream_t stream = nullptr,
		bool use_sigmoid = false,
		int rows = 1                    // packed [rows, num_v_heads * head_dim] attn_out and z
	);
};

} // namespace engine::modules
