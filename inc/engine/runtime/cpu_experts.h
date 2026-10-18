#pragma once

#include "engine/kernels/cpu/linear.h"
#include "engine/kernels/cpu/pool.h"
#include "engine/runtime/cpu_expert_mailbox.h"
#include <cstdint>
#include <memory>
#include <vector>

// A routed expert evaluated on the host CPU: gate and up [hidden -> intermediate], SwiGLU, down [intermediate -> hidden], each linear in
// whatever quantized format the kernels in src/kernels/cpu provide (kernels::cpu::Linear)
namespace engine::cpu_moe {

constexpr int kMaxRows = kMailboxRows;      // rows (tokens) one expert call takes

struct Expert {
	const kernels::cpu::Linear *gate = nullptr, *up = nullptr, *down = nullptr;   // not owned
	int hidden() const { return gate->in(); }
	int inter() const { return gate->out(); }
};

struct ExpertJob;
struct ManyTask;

// Work memory of expert_forward: a few jobs used in turn (a job stays untouchable while a pool thread that was preempted may still hold it)
struct Scratch {
	Scratch();
	~Scratch();
	std::vector<std::unique_ptr<ExpertJob>> jobs;
	std::vector<std::unique_ptr<ManyTask>> many;
	size_t next = 0, next_many = 0;
};

// out[m][h] += wts[m] * expert( x[m] )[h] for m < rows (FP32 accumulate; the three linears round to FP16 between stages like the device
// kernels). x is FP16 [rows, hidden]; act_limit is the SwiGLU clamp. The pool's threads share the work in units of 16 output columns; a unit or a
// block's finishing step that a thread has held for too long (the scheduler took its core) is redone by another thread, so the call does not wait
// for a descheduled thread
void expert_forward( kernels::cpu::Pool& pool, const Expert& e, int rows, const uint16_t* x, const float* wts, float* out, Scratch& s, float act_limit = 10.0f );

// Several experts at once (the experts of one request), each as expert_forward; their pieces share the pool's threads, so one expert's tail and
// phase barrier are filled with another's work
struct ExpertCall {
	Expert e;
	int rows;
	const uint16_t* x;
	const float* wts;
	float* out;
};
void expert_forward_many( kernels::cpu::Pool& pool, const ExpertCall* calls, int count, Scratch& s, float act_limit = 10.0f );

}   // namespace engine::cpu_moe
