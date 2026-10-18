#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace engine::kernels::cpu {

// A job the pool's threads work on together. work() does one piece of the job if one is available and returns whether it did; finished() says the
// job's results are complete. Any thread may call work() at any time, also after finished() and also while another thread is stuck in the middle of a
// piece (a preempted thread), so a job must tolerate a piece being done twice. refs counts the threads that may still touch the job: it must stay
// alive until it drops to zero
struct Task {
	virtual ~Task() {}
	virtual bool work() = 0;
	virtual bool finished() const = 0;
	std::atomic<int> refs{ 0 };
};

// Worker threads for the CPU kernels. run( task ) lets the workers and the caller work on the task until it is finished and returns then, without waiting
// for workers that have not come back (a worker the scheduler took the core from does not hold the caller up). Workers poll for `spin_us` after a task
// before they sleep, so the back-to-back tasks of a token never pay a wake-up
class Pool {
public:
	// cpus: the logical CPU each of the threads 1.. is pinned to (empty: not pinned); `threads` includes the caller
	Pool( int threads, const std::vector<int>& cpus, int spin_us = 2000 );
	~Pool();
	Pool( const Pool& ) = delete;
	Pool& operator=( const Pool& ) = delete;
	int size() const { return n_; }
	void run( Task& task );

private:
	void worker( int tid );
	int n_ = 1, spin_us_ = 2000;
	std::vector<std::thread> th_;
	std::vector<int> cpus_;
	std::atomic<uint32_t> gen_{ 0 };
	std::atomic<Task*> cur_{ nullptr };
	std::atomic<int> parked_{ 0 };
	std::atomic<bool> stop_{ false };
	std::mutex mu_;
	std::condition_variable cv_;
};

}   // namespace engine::kernels::cpu
