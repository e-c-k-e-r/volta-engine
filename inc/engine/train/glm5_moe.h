#pragma once

// Training forms of GLM-5.3-Flash (glm5_next) operations: the MoE routing and the routed experts' SwiGLU (FP32 on the caller's stream; see
// .agents/GLM_STAGEC_PLAN.md). The rest of the MoE layer comes from qwen4_exp.h: router_input_backward takes the [rows, k] logit gradient
// of glm_route_backward as it is (R: the router [experts, hidden], the checkpoint's mlp.gate.weight layout; the engine's loaded copy is
// transposed), and the grouped routes (route_gather, route_scale, route_combine, route_dots_pad, seg_colsum_prod) carry over unchanged. The
// shared expert and the dense MLP clamp before the SiLU: grad.h's swiglu_forward / swiglu_backward with limit 10.
#include "engine/backend/types.h"
#include <cstddef>
#include <cstdint>

namespace engine::train::glm5_next {

// Sigmoid routing over the router's logits [rows, experts] (the product is the caller's): s_e = sigmoid( logit_e ); the selection is the k
// largest s_e + bias_e (bias [experts], e_score_correction_bias; null: 0; ties to the lower expert index), ids [rows, k] best first; the
// weights w [rows, k] = scale s_{ids_j} / ( sum_j' s_{ids_j'} + 1e-9 ) (the bias does not enter them). experts <= 1024, k <= 16
void glm_route_forward( const float* logits, const float* bias, int rows, int experts, int k, float scale, int* ids, float* w, backend::stream_t stream );
// The same weights with the selection given (ids [rows, k], held from an earlier call), w overwritten
void glm_route_given( const float* logits, const int* ids, int rows, int experts, int k, float scale, float* w, backend::stream_t stream );
// From dw [rows, k] to dlogit [rows, k], the gradient at the selected logits (overwrites; the selection is held fixed, every other logit's
// gradient is 0): dlogit_j = s_j ( 1 - s_j ) ( scale dw_j - sum_j' w_j' dw_j' ) / ( sum_j' s_j' + 1e-9 )
void glm_route_backward( const float* logits, const int* ids, int rows, int experts, int k, float scale, const float* dw, float* dlogit,
	backend::stream_t stream );

// The routed experts' SwiGLU with the clamp after the SiLU, elementwise: a = min( silu( g ), limit ) clamp( u, -limit, limit ) (limit <= 0:
// silu( g ) u). Not grad.h's swiglu_forward form, silu( min( g, limit ) ) clamp( u, -limit, limit )
void swiglu_after_forward( const float* g, const float* u, float* a, size_t n, float limit, backend::stream_t stream );
// dg, du from da at the same g, u (overwrite): dg through silu where silu( g ) < limit (0 at or above it), du inside ( -limit, limit ) (0 at
// or outside its ends)
void swiglu_after_backward( const float* g, const float* u, const float* da, float* dg, float* du, size_t n, float limit, backend::stream_t stream );

// The same in qwen4_exp's grouped-route layout (as route_swiglu: P padded rows, m per expert, the expert of padded row p gexp[p / m];
// per-expert vectors [cols] from device tables of pointers indexed by expert; cols a multiple of 4): G = Zg sv_gate, U = Zu sv_up,
// A = min( silu( G ), limit ) clamp( U, -limit, limit ) (overwrite; padding rows, zero in Zg and Zu, give zeros)
void glm_route_swiglu( const float* Zg, const float* Zu, const int* gexp, int m, const float* const* svg, const float* const* svu, int P, int cols, float limit,
	float* G, float* U, float* A, backend::stream_t stream );
// From dAk (the down projection's input gradient before su_down: dA = dAk su_down, sud the su_down table): dG, dU through the SwiGLU
// (overwrite; zeros on padding rows)
void glm_route_swiglu_backward( const float* dAk, const float* G, const float* U, const int* gexp, int m, const float* const* sud, int P, int cols, float limit,
	float* dG, float* dU, backend::stream_t stream );

// bytes (whole 4-byte words) from mapped pinned host memory to the device by a kernel's reads: a small upload that does not queue on the copy
// engine behind a large transfer of another stream (the routed experts' stage copies)
void upload_words( const void* mapped, void* dst, size_t bytes, backend::stream_t stream );

} // namespace engine::train::glm5_next
