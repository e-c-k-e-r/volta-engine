#pragma once

#include "engine/models/vision_tower.h"
#include <memory>
#include <string>
#include <vector>

namespace engine::models {

// The vision tower of the checkpoint in `model_dir` on the host CPU (the vision_cpu option), so the GPUs keep no tower weights and no scratch. Every
// quantized projection of the tower is decoded once into FP16 panels in host memory (the rotations of the EXL3 format stay around the dense
// product; a vision projector GGUF in `mmproj_dir`, when given, holds dense BF16 weights instead), and the operators run on the logical CPUs `cpus` (the first is the calling thread's while it encodes). Images of up to max_tokens
// language-model tokens. GLM-5 and DeepSeek-V4.1; nullptr with *error set for another architecture or a checkpoint without the quantized tower.
std::unique_ptr<VisionTower> make_host_vision_tower( const char* model_dir, const char* mmproj_dir, int max_tokens, const std::vector<int>& cpus, std::string* error );

} // namespace engine::models
