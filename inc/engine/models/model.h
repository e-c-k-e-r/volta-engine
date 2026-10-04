#pragma once

#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/core/arena.h"
#include "engine/core/comm.h"
#include <cuda_runtime.h>
#include <string>
#include <memory>

namespace engine::runtime {
struct KVCachePool;
}

namespace engine::models {

struct ModelMetadata {
	std::string model_name;
	int hidden_size;
	int vocab_size;
	int num_layers;
	int num_heads;
	int num_kv_heads;
	int head_dim;
	int max_seq_len;
	int tp_size;
};

class ModelPipeline {
public:
	virtual ~ModelPipeline() = default;

	// Initialization & Weight Loading
	virtual bool init( int device_id, pod::MemoryArena& arena, engine::comm::CollectiveRing* ring ) = 0;
	virtual bool load_weights( const char* model_dir, cudaStream_t stream = nullptr ) = 0;

	// Execution & Forward Passes
	virtual void embed_token( int token_id, int pos, cudaStream_t stream = nullptr ) = 0;
	virtual void forward_step( int token_id, int pos, int slot_id, cudaStream_t stream = nullptr ) = 0;
	virtual void forward_step( int token_id, int pos, cudaStream_t stream = nullptr ) {
		forward_step( token_id, pos, 0, stream );
	}
	virtual void forward_batch( const int* token_ids, int batch_size, int pos, cudaStream_t stream = nullptr ) = 0;
	virtual int sample_next_token( cudaStream_t stream = nullptr ) = 0;

	// CUDA Graph Management
	virtual bool capture_graph( int batch_size, cudaStream_t stream = nullptr ) = 0;
	virtual void launch_graph( int batch_size, cudaStream_t stream = nullptr ) = 0;

	// Descriptors & Buffers
	virtual pod::TensorView get_logits() = 0;
	virtual const ModelMetadata& get_metadata() const = 0;

	// KV Cache Management
	virtual void set_kv_pool( engine::runtime::KVCachePool* pool ) {
		(void) pool;
	}
	virtual void reset_kv_cache( int slot_id = 0 ) = 0;
};

// Factory helper to instantiate models dynamically
std::unique_ptr<ModelPipeline> create_model_pipeline( const std::string& model_type );

} // namespace engine::models
