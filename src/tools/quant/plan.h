#pragma once

// What volta_quantize does with a template checkpoint: its quantized tensors (a <base>.trellis with its width and codebook) and the Hessian
// that serves each, from an engine capture directory (test_ppl --calibrate with hessian_dir: L<layer>_s<site>_r<rank>.f32). The sites per
// model are those of the model pipelines (qwen4_exp, glm5_next, qwen3_5); a routed expert's down projection takes the Hessian of its own
// activation, computed from the route dump (expert_hessian.h).
#include <cstdint>
#include <string>
#include <vector>

namespace engine::tools::quant {

struct Model {
	std::string type;          // config.json model_type: qwen4_exp, glm5_next, qwen3_5, ...
	int layers = 0;            // decoder layers
	int hidden = 0;
	int experts = 0;
	float swiglu_limit = 0.0f; // 0: none
	bool read( const std::string& template_dir );
};

struct PlanTensor {
	std::string base;          // template name without the .trellis suffix
	int code = 0;              // width code
	int cb = 0;                // 0 3INST, 1 mcg, 2 mul1
	int k = 0, n = 0;          // input and output size
	std::vector<std::string> hess;   // capture files (one, or one per rank of a row-parallel input: block diagonal); empty: none
	int down_layer = -1;       // routed expert down projection: the route-dump layer and expert whose activation gives the Hessian
	int down_expert = -1;
	std::string src_layer_prefix;    // for a down projection: the "<...>.mlp.experts" prefix of the expert's gate / up projections
	int layer = -1;            // decoder layer (ordering), -1 none
};

// The capture files of a tensor of a model (empty when the capture has none); a routed expert down projection fills down_layer / expert
void hessian_sources( const Model& m, const std::string& hess_dir, int ranks, PlanTensor& t );

} // namespace engine::tools::quant
