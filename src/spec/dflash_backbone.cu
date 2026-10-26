#include "engine/spec/dflash.h"
#include "engine/modules/norm.h"
#include "engine/modules/swiglu.h"
#include "engine/modules/embedding.h"
#include "engine/modules/linear.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <chrono>
#include <cstring>

namespace engine::spec {

void DFlashSpeculator::forward_backbone(
	const pod::TensorView& embed_table,
	const int* d_input_ids,
	int rows,
	const half* tapped_states,
	int n_ctx,
	int ctx_pos0,
	int block_pos0,
	pod::TensorView& hidden_out,
	cudaStream_t stream,
	int taps_ring_cap
) {
	CUDA_CHECK( cudaSetDevice( device_ ) );
	ENGINE_ASSERT( weights_loaded_, "DFlashSpeculator::forward_backbone: weights not loaded" );
	ENGINE_ASSERT( rows >= 1 && rows <= config_.block_size, "DFlashSpeculator::forward_backbone: rows" );

	const int H = config_.hidden_size;
	const int L = config_.num_hidden_layers;
	const int taps = config_.conv_kernel_size;
	const int groups = config_.conv_groups();
	const int proj_out = config_.conv_proj_out();
	const int hd = config_.head_dim;
	const int qh_local = config_.num_q_heads / tp_size_;
	const int kvh_local = config_.num_kv_heads / tp_size_;
	const int rope_pairs = hd / 2;
	const int kv_total = ring_len_ + config_.block_size;
	const int stride_slot = drafter_k_.shape[1] * drafter_k_.shape[2] * hd;
	const int stride_layer = drafter_k_.shape[2] * hd;

	// The cache ring's own K/V live in a separate tail, so passing kv_total as the "ring" keeps the
	// single conditional subtract in the attention kernel correct for the context part while the own
	// rows sit above it.
	const int ctx_ring = ring_len_;
	const int ctx_strides_slot = stride_slot;
	const int ctx_strides_layer = stride_layer;

	// ---- Stage 1: project the target context into the drafter's own KV cache ----
	// fc is column-parallel on the output. At tp > 1 each rank computes its H/tp columns, writes them into
	// a zeroed full-width buffer and a sum across ranks gathers them (x + 0 is exact), so the norm and
	// every later stage see the full-width row on every rank.
	ENGINE_ASSERT( tp_size_ == 1 || ring_ != nullptr, "DFlashSpeculator::forward_backbone: TP > 1 needs set_ring" );
	if ( n_ctx > 0 && fc_.is_valid() ) {
		for ( int base = 0; base < n_ctx; base += ctx_chunk_ ) {
			const int cnt = MIN( ctx_chunk_, n_ctx - base );
			// taps_ring_cap > 0 means `tapped_states` is a position-indexed ring rather than a packed
			// [n_ctx, 5*H] block, so the chunk's first row sits at (ctx_pos0 + base) % cap.
			const size_t tap_off = ( taps_ring_cap > 0 )
				? static_cast<size_t>( ( ctx_pos0 + base ) % taps_ring_cap ) * 5 * H
				: static_cast<size_t>( base ) * 5 * H;
			const pod::TensorView taps_in = pod::make_tensor_2d(
				const_cast<half*>( tapped_states ) + tap_off,
				pod::DType::FP16, cnt, 5 * H );
			pod::TensorView fc_out = pod::make_tensor_2d( bb_ctx_fc_.ptr<half>(), pod::DType::FP16, cnt, H );
			hess_.accumulate( L, 0, taps_in.data, 5 * H, cnt, stream );
			if ( tp_size_ == 1 ) {
				engine::modules::Linear::forward( taps_in, fc_, fc_out, stream );
			}
			else {
				const int Hs = H / tp_size_;
				// bb_ctx_resid_ is FP32 [chunk, H]: ample room for the FP16 [cnt, H / tp] local slice, and it is
				// only written again by the norm below, after the gather has consumed the slice.
				pod::TensorView fc_local = pod::make_tensor_2d( bb_ctx_resid_.ptr<half>(), pod::DType::FP16, cnt, Hs );
				engine::modules::Linear::forward( taps_in, fc_, fc_local, stream );
				CUDA_CHECK( cudaMemsetAsync( fc_out.data, 0, static_cast<size_t>( cnt ) * H * sizeof( half ), stream ) );
				dflash_place_cols( fc_out.ptr<half>(), fc_local.ptr<const half>(), cnt, H, Hs, rank_ * Hs, stream );
				// The ring all-reduce handles at most 16 rows per launch (one block per row).
				for ( int r0 = 0; r0 < cnt; r0 += 16 ) {
					pod::TensorView part = pod::make_tensor_2d( fc_out.ptr<half>() + static_cast<size_t>( r0 ) * H,
						pod::DType::FP16, MIN( 16, cnt - r0 ), H );
					ring_->all_reduce( part, stream );
				}
			}
			pod::TensorView ctx_n = pod::make_tensor_2d( bb_ctx_normed_.ptr<half>(), pod::DType::FP16, cnt, H );
			pod::TensorView ctx_r = pod::make_tensor_2d( bb_ctx_resid_.ptr<float>(), pod::DType::FP32, cnt, H );
			dflash_rmsnorm_residual( fc_out, pod::TensorView{}, fc_norm_, ctx_n, ctx_r, cnt,
				config_.rms_norm_eps, 0.0f, stream );

			for ( int l = 0; l < L; ++l ) {
				DFlashLayer& ly = layers_[l];
				pod::TensorView kv_stage = pod::make_tensor_2d(
					bb_kv_stage_.ptr<half>(), pod::DType::FP16, cnt, kvh_local * hd );
				const int slot0 = ( kv_len_ + base ) % ctx_ring;
				hess_.accumulate( l, 1, ctx_n.data, H, cnt, stream );
				engine::modules::Linear::forward( ctx_n, ly.k_proj, kv_stage, stream );
				dflash_qk_norm_rope( kv_stage.ptr<half>(),
					( ly.k_norm.data != nullptr ) ? ly.k_norm.ptr<const half>() : nullptr,
					config_.rope_theta, cnt, kvh_local, hd, rope_pairs, ctx_pos0 + base,
					config_.rms_norm_eps, 0.0f, stream );
				dflash_cache_store( drafter_k_.ptr<half>(), kv_stage.ptr<const half>(), cnt, kvh_local, hd,
					slot0, ctx_ring, l, ctx_strides_slot, ctx_strides_layer, stream );
				engine::modules::Linear::forward( ctx_n, ly.v_proj, kv_stage, stream );
				dflash_cache_store( drafter_v_.ptr<half>(), kv_stage.ptr<const half>(), cnt, kvh_local, hd,
					slot0, ctx_ring, l, ctx_strides_slot, ctx_strides_layer, stream );
			}
		}
		kv_len_ += n_ctx;
		if ( kv_len_ > ring_len_ ) kv_pos0_ = kv_len_ - ring_len_;
	}

	// Visible context window: at most sliding_window entries ending at the newest.
	const int ring_n = MIN( kv_len_, config_.sliding_window );
	const int ctx_slot0 = ( kv_len_ > ctx_ring ? kv_pos0_ : 0 )
		+ ( kv_len_ - ring_n ) % ctx_ring;

	// ---- Stage 2: the block of `rows` query positions ----
	engine::modules::Embedding::forward( embed_table, d_input_ids, rows, bb_normed_, stream );

	// The first layer's input_layernorm takes no incoming residual: the reference sets
	// residual = hidden and normalizes the embedding alone (qwen3_dflash2.py:295-297). After that the
	// FP32 residual carries across every layer.
	const pod::TensorView empty{};
	for ( int l = 0; l < L; ++l ) {
		DFlashLayer& ly = layers_[l];

		// (a) input_layernorm + residual split -> normed, residual
		{
// Layer 0 normalizes the embedding. Every later layer normalizes the previous layer's
			// FP32 conv output, which is what the range-preserving transport hands forward.
			pod::TensorView x = ( l == 0 )
				? pod::make_tensor_2d( bb_normed_.ptr<half>(), pod::DType::FP16, rows, H )
				: pod::make_tensor_2d( bb_stage_.ptr<float>(), pod::DType::FP32, rows, H );
			pod::TensorView n = pod::make_tensor_2d( bb_normed_.ptr<half>(), pod::DType::FP16, rows, H );
			pod::TensorView r = pod::make_tensor_2d( bb_residual_.ptr<float>(), pod::DType::FP32, rows, H );
			pod::TensorView rin = ( l == 0 ) ? empty
				: pod::make_tensor_2d( bb_residual_.ptr<float>(), pod::DType::FP32, rows, H );
			dflash_rmsnorm_residual( x, rin, ly.input_norm, n, r, rows, config_.rms_norm_eps, 0.0f, stream );
		}
		const pod::TensorView normed = pod::make_tensor_2d( bb_normed_.ptr<half>(), pod::DType::FP16, rows, H );

		// (b) grouped dynamic conv, pre-attention side. kernel_projection emits both sides of the
		//     sandwich as one [rows, 2*taps*groups] output, side 0 first.
		if ( ly.attn_conv_proj.data != nullptr ) {
			dflash_f16_gemv_nt( ly.attn_conv_proj.ptr<const half>(), normed.ptr<const half>(),
				bb_dyn_a_.ptr<half>(), rows, proj_out, H, stream );
			dflash_conv_side_split( bb_coef_a0_.ptr<float>(), bb_dyn_a_.ptr<const half>(), rows, taps, groups, 0, stream );
			dflash_conv_side_split( bb_coef_a1_.ptr<float>(), bb_dyn_a_.ptr<const half>(), rows, taps, groups, 1, stream );
		}
		{
			const pod::TensorView c0 = pod::make_tensor_3d(
				bb_coef_a0_.ptr<float>(), pod::DType::FP32, rows, taps, groups );
			const pod::TensorView base = pod::make_tensor_2d(
				const_cast<half*>( ly.attn_conv_base.ptr<const half>() ) + hd * 0,
				pod::DType::FP16, taps, H );
			dflash2_dynconv( normed, c0, base, bb_stage_h_, config_.conv_group_size, false, stream );
		}
		const pod::TensorView stage = pod::make_tensor_2d( bb_stage_h_.ptr<half>(), pod::DType::FP16, rows, H );

		// (c) attention: projections, per-head norm + RoPE, then attention over
		//     [context window | this block's own rows]
		{
			pod::TensorView q_out = pod::make_tensor_2d(
				bb_q_.ptr<half>(), pod::DType::FP16, rows, qh_local * hd );
			hess_.accumulate( l, 0, stage.data, H, rows, stream );
			hess_.accumulate( l, 1, stage.data, H, rows, stream );
			engine::modules::Linear::forward( stage, ly.q_proj, q_out, stream );
			dflash_qk_norm_rope( bb_q_.ptr<half>(),
				( ly.q_norm.data != nullptr ) ? ly.q_norm.ptr<const half>() : nullptr,
				config_.rope_theta, rows, qh_local, hd, rope_pairs, block_pos0,
				config_.rms_norm_eps, 0.0f, stream );

			pod::TensorView kv_stage = pod::make_tensor_2d(
				bb_kv_stage_.ptr<half>(), pod::DType::FP16, rows, kvh_local * hd );
			engine::modules::Linear::forward( stage, ly.k_proj, kv_stage, stream );
			dflash_qk_norm_rope( kv_stage.ptr<half>(),
				( ly.k_norm.data != nullptr ) ? ly.k_norm.ptr<const half>() : nullptr,
				config_.rope_theta, rows, kvh_local, hd, rope_pairs, block_pos0,
				config_.rms_norm_eps, 0.0f, stream );
			dflash_cache_store( drafter_k_.ptr<half>(), kv_stage.ptr<const half>(), rows, kvh_local, hd,
				own_slot_, kv_total, l, stride_slot, stride_layer, stream );
			engine::modules::Linear::forward( stage, ly.v_proj, kv_stage, stream );
			dflash_cache_store( drafter_v_.ptr<half>(), kv_stage.ptr<const half>(), rows, kvh_local, hd,
				own_slot_, kv_total, l, stride_slot, stride_layer, stream );
		}
		{
			const pod::TensorView attn_out = pod::make_tensor_3d(
				bb_attn_out_.ptr<half>(), pod::DType::FP16, rows, qh_local, hd );
			dflash_attn( bb_q_.ptr<const half>(), drafter_k_.ptr<const half>(), drafter_v_.ptr<const half>(),
				bb_attn_out_.ptr<half>(), rows, qh_local, kvh_local, hd,
				ring_n, ctx_slot0, ctx_ring, l, ctx_strides_slot, ctx_strides_layer, hd, own_slot_,
				own_first_row_, stream );
			// o_proj is row-parallel; at TP 1 it is the whole projection and needs no reduce.
			// Linear::forward reads x.shape[1] as K, so this must be the flat [rows, q_heads*head_dim]
			// view. The 3D head view reports shape[1] == q_heads, which the Trellis path ignored.
			const pod::TensorView attn_flat = pod::make_tensor_2d(
				bb_attn_out_.ptr<half>(), pod::DType::FP16, rows, qh_local * hd );
			hess_.accumulate( l, 2, attn_flat.data, qh_local * hd, rows, stream );
			engine::modules::Linear::forward( attn_flat, ly.o_proj, bb_o_proj_, stream );
			if ( tp_size_ > 1 ) {
				pod::TensorView o_full = pod::make_tensor_2d( bb_o_proj_.ptr<half>(), pod::DType::FP16, rows, H );
				ring_->all_reduce( o_full, stream );
			}
		}

		// (d) grouped dynamic conv, post-attention side (side 1 of the same projection)
		{
			// The reference consumes self_attn's output here, which is o_proj's result, not the
			// concatenated head outputs. At tp_size 1 both are [rows, H] so the shape alone does not
			// catch a mistake; the buffers differ, and this used to read bb_attn_out_.
			const pod::TensorView c1 = pod::make_tensor_3d(
				bb_coef_a1_.ptr<float>(), pod::DType::FP32, rows, taps, groups );
			const pod::TensorView base = pod::make_tensor_2d(
				const_cast<half*>( ly.attn_conv_base.ptr<const half>() ) + taps * H,
				pod::DType::FP16, taps, H );
			// Widen o_proj before the conv: it peaks past 15000, and the conv over it overflowed
			// FP16 to inf, which the next dot product then spread into NaN across a whole row.
			pod::TensorView o_in = pod::make_tensor_2d(
				bb_stage_f32_.ptr<float>(), pod::DType::FP32, rows, H );
			dflash_h2f( bb_o_proj_.ptr<const half>(), o_in.ptr<float>(), rows * H, stream );
			pod::TensorView conv_out = pod::make_tensor_2d(
				bb_stage_.ptr<float>(), pod::DType::FP32, rows, H );
			dflash2_dynconv( o_in, c1, base, conv_out, config_.conv_group_size, false, stream );
		}

		// (e) post_attention_layernorm + residual split
		{
			pod::TensorView x = pod::make_tensor_2d( bb_stage_.ptr<float>(), pod::DType::FP32, rows, H );
			pod::TensorView n = pod::make_tensor_2d( bb_hidden_.ptr<half>(), pod::DType::FP16, rows, H );
			pod::TensorView r = pod::make_tensor_2d( bb_residual_.ptr<float>(), pod::DType::FP32, rows, H );
			pod::TensorView rin = pod::make_tensor_2d( bb_residual_.ptr<float>(), pod::DType::FP32, rows, H );
			dflash_rmsnorm_residual( x, rin, ly.post_attn_norm, n, r, rows, config_.rms_norm_eps, 0.0f, stream );
		}
		pod::TensorView normed_m = pod::make_tensor_2d( bb_hidden_.ptr<half>(), pod::DType::FP16, rows, H );

		// (f) grouped dynamic conv, pre-MLP side
		if ( ly.mlp_conv_proj.data != nullptr ) {
			dflash_f16_gemv_nt( ly.mlp_conv_proj.ptr<const half>(), normed_m.ptr<const half>(),
				bb_dyn_m_.ptr<half>(), rows, proj_out, H, stream );
			dflash_conv_side_split( bb_coef_m0_.ptr<float>(), bb_dyn_m_.ptr<const half>(), rows, taps, groups, 0, stream );
			dflash_conv_side_split( bb_coef_m1_.ptr<float>(), bb_dyn_m_.ptr<const half>(), rows, taps, groups, 1, stream );
		}
		{
			const pod::TensorView c0 = pod::make_tensor_3d(
				bb_coef_m0_.ptr<float>(), pod::DType::FP32, rows, taps, groups );
			const pod::TensorView base = pod::make_tensor_2d(
				const_cast<half*>( ly.mlp_conv_base.ptr<const half>() ),
				pod::DType::FP16, taps, H );
			dflash2_dynconv( normed_m, c0, base, bb_stage_h_, config_.conv_group_size, false, stream );
		}

		// (g) SwiGLU MLP: gate/up are column-parallel, down is row-parallel.
		{
			pod::TensorView s = pod::make_tensor_2d( bb_stage_h_.ptr<half>(), pod::DType::FP16, rows, H );
			hess_.accumulate( l, 3, s.data, H, rows, stream );
			engine::modules::Linear::forward( s, ly.gate_proj, bb_ffn_gate_, stream );
			engine::modules::Linear::forward( s, ly.up_proj, bb_ffn_up_, stream );
			engine::modules::SwiGLU::forward( bb_ffn_gate_, bb_ffn_up_, bb_ffn_mlp_, stream );
			hess_.accumulate( l, 4, bb_ffn_mlp_.data, config_.intermediate_size / tp_size_, rows, stream );
			engine::modules::Linear::forward( bb_ffn_mlp_, ly.down_proj, bb_hidden_, stream );
			if ( tp_size_ > 1 ) {
				pod::TensorView down_full = pod::make_tensor_2d( bb_hidden_.ptr<half>(), pod::DType::FP16, rows, H );
				ring_->all_reduce( down_full, stream );
			}
		}

		// (h) grouped dynamic conv, post-MLP side
		{
			pod::TensorView m_in = pod::make_tensor_2d(
				bb_stage_f32_.ptr<float>(), pod::DType::FP32, rows, H );
			dflash_h2f( bb_hidden_.ptr<const half>(), m_in.ptr<float>(), rows * H, stream );
			pod::TensorView c1 = pod::make_tensor_3d(
				bb_coef_m1_.ptr<float>(), pod::DType::FP32, rows, taps, groups );
			pod::TensorView base = pod::make_tensor_2d(
				const_cast<half*>( ly.mlp_conv_base.ptr<const half>() ) + taps * H,
				pod::DType::FP16, taps, H );
			pod::TensorView conv_out = pod::make_tensor_2d(
				bb_stage_.ptr<float>(), pod::DType::FP32, rows, H );
			dflash2_dynconv( m_in, c1, base, conv_out, config_.conv_group_size, false, stream );
		}
	}

	// Final norm on the block output.
	{
		// The residual IS part of the input here. The reference finishes the stack with
		// `hidden_states, _ = self.norm(hidden_states, residual)`, and with residual non-None its norm
		// computes bf16(x * 256 + residual) before normalising. Passing an empty residual here norms
		// only the last sublayer's output and throws the whole accumulated stream away -- which is why
		// the drafter's hidden was uncorrelated with the target's while every intermediate stage looked
		// healthy. Ours carries the stream at scale 1 throughout (the reference's 256s cancel against
		// the /256 on each sublayer output, which this engine does not apply), so the residual goes in
		// unscaled.
		pod::TensorView x = ( config_.num_hidden_layers > 0 )
			? pod::make_tensor_2d( bb_stage_.ptr<float>(), pod::DType::FP32, rows, H )
			: pod::make_tensor_2d( bb_normed_.ptr<half>(), pod::DType::FP16, rows, H );
		pod::TensorView o = pod::make_tensor_2d( hidden_out.ptr<half>(), pod::DType::FP16, rows, H );
		pod::TensorView r = pod::make_tensor_2d( bb_residual_.ptr<float>(), pod::DType::FP32, rows, H );
		const pod::TensorView rin = ( config_.num_hidden_layers > 0 )
			? pod::make_tensor_2d( bb_residual_.ptr<float>(), pod::DType::FP32, rows, H )
			: pod::TensorView{};
		dflash_rmsnorm_residual( x, rin, final_norm_, o, r, rows, config_.rms_norm_eps, 0.0f, stream );
	}
}

void DFlashSpeculator::draft_backbone(
	const pod::TensorView& embed_table,
	int anchor_token,
	const half* tapped_states,
	int block_pos0,
	int taps_cap,
	cudaStream_t stream
) {
	CUDA_CHECK( cudaSetDevice( device_ ) );
	const int M = config_.block_size;

	ENGINE_ASSERT( embed_table.data != nullptr, "draft_backbone: no embedding table" );
	ENGINE_ASSERT( weights_loaded_, "draft_backbone: weights not loaded" );

	// The query block is forward-looking, not a window of history. Slot 0 is the anchor -- the last
	// confirmed token -- and slot i>0 holds the token the drafter already proposed for position
	// anchor+i, or mask_token_id where it has nothing yet ("Masked query slots are fed to the draft as
	// mask_token_id"). The reference pairs lattice step l with row l and anchor_token_ids at row 0, so
	// row l predicts position anchor+l+1 and the walk reads row l conditioned on step l-1's choice.
	//
	// Building the block from the last M real tokens instead made rows 1..M-1 predict tokens the
	// sequence already contains, so only step 0 could ever be accepted and everything past it was
	// structurally incapable of helping.
	int ids[M];
	ids[0] = anchor_token;
	// Every remaining slot is a mask token, every round. Upstream states the layout directly:
	// "Block input = [anchor, mask x (block_size-1)]; each remaining row predicts its own position".
	// Those slots mean "position not yet known" to a model trained that way, so feeding real tokens
	// there -- previously accepted drafts -- is off-distribution even though the information is true.
	for ( int i = 0; i < M - 1; ++i ) ids[1 + i] = config_.mask_token_id;

	CUDA_CHECK( cudaMemcpyAsync( d_draft_tokens_, &anchor_token, sizeof( int ), cudaMemcpyHostToDevice, stream ) );
	CUDA_CHECK( cudaMemcpyAsync( d_input_ids_, ids, M * sizeof( int ), cudaMemcpyHostToDevice, stream ) );

	// Context stage. The drafter's queries attend to K/V projected from the target's tapped states,
	// not just to their own block: with the context suppressed its hidden state is essentially
	// uncorrelated with the target's (cos ~ 0.05) and its top-16 candidates never contain the token
	// the target actually picks. Ingest only the positions this round adds, since the ring already
	// holds everything before ctx_ingested_.
	// The block_pos0 argument is the anchor token's absolute position (the caller passes the same pos it
	// gave the target's forward for that token), and the target produced tapped hidden states for
	// positions 0..=pos inclusive. Reading M-1 positions past that ingests 7 rows the target never wrote
	// -- zeros on a fresh ring, stale rows after a wrap -- as if they were context. anchor_pos was
	// therefore block_pos0 + M - 1, and every draft round attended over 7 garbage keys.
	const int anchor_pos = block_pos0;

	// Who owns the anchor position's K/V: the block, not the taps. Context covers 0..pos-1 and the
	// block contributes rows 0..M-1, so row 0 supplies the anchor's K/V from the drafter's own
	// embedding of it, matching update_kv_from_target, which fills 0..pos-1 and leaves the next
	// anchor's position to the block. Measured on the four coherence prompts: the drafter's top-1 is
	// the token the target picks 14/15 times this way, against 5/17 with the anchor taken from the taps.
	own_first_row_ = 0;
	int n_ctx = anchor_pos - ctx_ingested_;
	if ( n_ctx < 0 ) n_ctx = 0;
	if ( taps_cap > 0 && n_ctx > taps_cap ) {
		// More positions pending than the tap ring holds (a prompt longer than the ring): only the newest
		// taps_cap positions still exist (older slots were overwritten by the wrap) and the drafter's
		// window is no wider, so skip ahead and ingest [anchor_pos - taps_cap, anchor_pos). Clamping n_ctx
		// alone would read ring slots that now hold later positions and label them with the old ones.
		ctx_ingested_ = anchor_pos - taps_cap;
		n_ctx = taps_cap;
	}
	if ( n_ctx > 0 ) ctx_ingested_ += n_ctx;
	else n_ctx = 0;

	forward_backbone(
		embed_table, d_input_ids_, M,
		tapped_states, n_ctx, ctx_ingested_ - n_ctx, block_pos0,
		draft_hidden_, stream, taps_cap
	);
}

void DFlashSpeculator::project_gate( int rows, cudaStream_t stream ) {
	CUDA_CHECK( cudaSetDevice( device_ ) );
	const int H = config_.hidden_size;
	const int sel_rank = config_.selector_rank;

	// gate = hidden_projection(hidden): a plain [selector_rank, hidden] FP16 matrix, so one row of the
	// drafter output gives one row of gate. This is the bilinear factor the walk needs; leaving it at
	// whatever the arena held was the last unwritten input on this path.
	if ( hidden_proj_w_.data == nullptr ) return;
	ENGINE_ASSERT( hidden_proj_w_.shape[0] == sel_rank && hidden_proj_w_.shape[1] == H,
		"project_gate: hidden_projection shape does not match selector_rank / hidden_size" );

	// The walk reads gate rows 1..block_size - 1 (step i scores candidate row i + 1 against gate row i + 1), so the
	// projection has to cover every block row; stopping at block_size - 1 left the last step on a stale row.
	rows = MIN( rows, config_.block_size );
	pod::TensorView h = pod::make_tensor_2d(
		draft_hidden_.ptr<half>(), pod::DType::FP16, rows, H );
	pod::TensorView g = pod::make_tensor_2d(
		const_cast<half*>( d_gate_ ), pod::DType::FP16, rows, sel_rank );
	dflash_f16_gemv_nt(
		hidden_proj_w_.ptr<const half>(), h.ptr<const half>(),
		g.ptr<half>(), rows, sel_rank, H, stream );
}

void DFlashSpeculator::draft_select(
	const float* vals,
	const int* ix,
	int rows,
	cudaStream_t stream
) {
	CUDA_CHECK( cudaSetDevice( device_ ) );
	rows = MIN( rows, config_.block_size - 1 );
	if ( vals == nullptr || ix == nullptr ) return;
	ENGINE_ASSERT( pred_codebook_.data != nullptr && succ_codebook_.data != nullptr,
		"draft_select: selector codebooks not loaded" );

	// Candidates come from block rows 1..M, not 0..M-1. Row 0 is the anchor and is never scored:
	// exllamav3's sample_from_state calls walk_block with state[:, 1:] and logits[:, 1:] ("The
	// generator passes state[:, 1:], a strided view for more than one row"), so walk_block's `hidden`
	// is already sliced and `gate[:, i]` is block row i+1. Step i therefore reads candidate row i+1
	// *and* gate row i+1, and both need the same one-row shift. Walking rows 0..M-1 puts every step
	// one position early.
	const int kk = config_.selector_top_k;
	dflash2_selector_walk(
		vals + kk, ix + kk, d_gate_ + config_.selector_rank,
		pred_codebook_.ptr<const half>(), succ_codebook_.ptr<const half>(),
		d_anchor_, d_draft_tokens_, d_conf_,
		1, rows, kk, config_.selector_rank, stream
	);
}

void DFlashSpeculator::set_walk_anchor( int anchor, cudaStream_t stream ) {
	CUDA_CHECK( cudaSetDevice( device_ ) );
	CUDA_CHECK( cudaMemcpyAsync( d_anchor_, &anchor, sizeof( int ), cudaMemcpyHostToDevice, stream ) );
}

} // namespace engine::spec
