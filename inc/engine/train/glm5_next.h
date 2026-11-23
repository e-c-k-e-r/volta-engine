#pragma once

// The GLM-5.3-Flash (glm5_next) training model over its EXL3 checkpoint, for the end-to-end tuning of the quantized tensors' scales (as
// engine/train/qwen4_exp.h's Model for Flash-Next): a contiguous range of decoder layers on one device (and the head when the range ends
// the model), each layer's forward and its backward from the layer's input streams, the forward recomputed. Every EXL3 tensor's input and
// output scales are trainable as log deltas on the checkpoint's own (su = s0u e^au, sv = s0v e^av), and the norm weights as log deltas too (GLM's
// are plain weights of very different sizes);
// everything else is frozen. Streams are [rows, 4, hidden] FP32, sequences from position 0.
//
// The layer: the mHC site before each sublayer (engine/train/glm5_mhc.h), KDA (engine/train/glm5_kda.h) or MLA (dense causal attention: the
// DSA selection keeps every key of a sequence up to its 2048 selected tokens) as the attention, then the dense MLP (the first layers) or
// the MoE (sigmoid routing, the routed experts' clamp after the SiLU, engine/train/glm5_moe.h; the shared expert). The head: the streams'
// mean, the final RMSNorm, lm_head and the KL to the teacher.
//
// Tensor parallel (ranks > 1, one Model per device and host thread, an NCCL communicator between them, the ranks calling in step): every
// rank holds its block of rows of the streams (rows / ranks of them, in rank order) and runs the per-position work on them (the mHC sites,
// the dense MLP, the shared expert, the head's norm); the sublayers see all rows through all-gathers and give partial sums back through
// reduce-scatters: KDA and MLA by heads, the routed experts by expert (a contiguous block per rank), the head by vocabulary. The scales of
// the tensors every rank holds stay identical (their gradients are summed before Adam); a rank's experts are its own.
//
// The routed experts' codes (the bulk of the checkpoint) can stay in pinned host memory (set_expert_host): each MoE layer's codes of this
// rank's experts are then copied to one of two device stages ahead of their use, the next layer's while the current one runs.
#include "engine/train/glm5_kda.h"
#include "engine/train/glm5_mhc.h"
#include "engine/train/glm5_moe.h"
#include "engine/train/qwen4_exp.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace engine::train::glm5_next {

using qwen4_exp::Dense;
using qwen4_exp::Teacher;
using qwen4_exp::Tuned;

struct Config {
	int hidden = 4096, hc = kHc, layers = 45, dense_layers = 3;
	int kda_heads = 64;                                 // KDA heads of kKdaDim (keys and values)
	float gate_lower = -5.0f;
	int heads = 64, head_dim = 256, q_rank = 1536, kv_rank = 512;   // MLA (no rotary part)
	int experts = 288, top_k = 8, inter = 2048, dense_inter = 12288, shared_inter = 2048;
	float route_scale = 2.5f, swiglu_limit = 10.0f;
	int vocab = 154880;
	float eps = 1e-5f, hc_eps = 1e-6f;
	int sinkhorn = 20;
	std::vector<int> full_layers;                       // the MLA layers
	bool read( const std::string& model_dir );          // config.json (text_config)
	bool full_attention( int layer ) const;
	bool moe( int layer ) const { return layer >= dense_layers; }
};

class Model {
public:
	Model();
	~Model();
	Model( const Model& ) = delete;
	Model& operator=( const Model& ) = delete;

	// Layers [first, last) of the checkpoint in dir on device, with workspace for up to max_rows rows; the head when last is the layer count.
	// ranks > 1: rank of a tensor-parallel group (comm: its ncclComm_t)
	bool load( const std::string& dir, int first, int last, int device, int max_rows, int rank = 0, int ranks = 1, void* comm = nullptr );
	const Config& config() const { return cfg_; }
	int first() const { return first_; }
	int last() const { return last_; }
	bool has_head() const { return last_ == cfg_.layers; }
	int device() const { return device_; }
	int rank() const { return rank_; }
	int ranks() const { return ranks_; }
	backend::stream_t stream() const;

	// Layer L forward: x_out = layer( x_in ) over rows of seqs sequences of rows / seqs, each from position 0 (x_in, x_out: the rank's rows).
	// keep: the sublayer outputs are kept for the layer's next backward, which then skips their recompute
	void forward( int L, const float* x_in, float* x_out, int rows, bool keep = false, int seqs = 1 );
	// Its backward: dx_in (overwritten) from dx_out, the forward recomputed from x_in; scale gradients accumulate in every tensor's gu / gv
	void backward( int L, const float* x_in, const float* dx_out, float* dx_in, int rows, int seqs = 1 );
	// The head on the final streams x (the rank's rows): KL( teacher || model ) summed over all rows (returned; the teacher's rows: all);
	// with dx, its gradient (overwritten) and the lm_head scale gradients
	double head( const float* x, int rows, const Teacher& t, float* dx );

	// Routing held fixed at the selection of each layer's last call (for gradient checks across a perturbation; the weights still follow
	// the logits); off: every call selects
	void freeze_routing( bool on );
	// Layer L's routed experts of its last call: rows x top_k ids (host)
	const std::vector<int>& selection( int L ) const;

	// The trainable tensors' products on the tensor cores (the default: FP16 cores, activations and gradients rounded to FP16 per row at a
	// power-of-two scale, FP32 accumulation; everything else FP32), or off: FP32 products
	void set_fp16( bool on );
	// The routed experts' codes in pinned host memory (before load; on by default), staged per layer as above; off: on the device. With
	// device_gib > 0, whole MoE layers' codes of this rank stay on the device up to that size (spread evenly over the layers, so that the
	// staged layers' copies overlap with the resident ones' work)
	void set_expert_host( bool on, double device_gib = 0.0 );

	// Phase timing (synchronizes the stream at every phase boundary): on, the seconds per phase accumulate until reported and reset
	void set_timing( bool on );
	std::string timing_report( bool reset = true );
	// The device memory by component
	std::string memory_report() const;

	// The tensors this rank holds (with ranks > 1: the shared ones and its own experts), and the trainable dense tensors (the norm weights;
	// every rank holds them all)
	std::vector<Tuned*> tensors();
	std::vector<Dense*> dense_tensors();
	// The norm weights trained along with the scales (on by default); off: their gradients are computed but not applied
	void set_train_norms( bool on );
	// Room for every layer's kept sublayer outputs (before load; on by default): off, a kept forward keeps nothing and the backward
	// recomputes the sublayers
	void set_keep_outputs( bool on );
	// Where Adam's state lives (before load): on the device (the default) or with host, in pinned host memory (qwen4_exp::Model::set_optimizer)
	void set_optimizer( bool host, int moment_bits );
	void zero_grads();
	// Adam on every tensor's log deltas and the norm weights' deltas from the accumulated gradients (times grad_scale; the shared tensors' and
	// the norms' summed over the ranks first), then the current values
	void adam( const AdamOptions& o, int step, float grad_scale );
	// The current values kept as the best, and the kept ones at the end as FP16: per tensor name its (input, output) scales, per norm weight
	// name its values
	void keep_best();
	void best_scales( std::vector<std::pair<std::string, std::pair<std::vector<uint16_t>, std::vector<uint16_t>>>>& out );
	void best_dense( std::vector<std::pair<std::string, std::vector<uint16_t>>>& out );

private:
	struct Impl;
	std::unique_ptr<Impl> p_;
	Config cfg_;
	int first_ = 0, last_ = 0, device_ = 0, rank_ = 0, ranks_ = 1;
};

// The token embeddings of a checkpoint (its BF16 table in pinned host memory, one copy per process): rows as FP32
class Embedding {
public:
	Embedding();
	~Embedding();
	bool open( const std::string& model_dir );
	int hidden() const;
	// out [n, hidden] host FP32 rows of the tokens ids
	void rows( const int* ids, int n, float* out ) const;

private:
	struct Impl;
	std::unique_ptr<Impl> p_;
};

} // namespace engine::train::glm5_next
