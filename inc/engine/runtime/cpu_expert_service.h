#pragma once

#include "engine/runtime/cpu_experts.h"
#include "engine/runtime/cpu_expert_mailbox.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace engine::cpu_moe {

// The process-wide CPU expert service: owns the worker pool, the experts that live on the host and one mailbox per rank. A dispatcher thread
// watches the mailboxes; for each request (the routes a rank's device handed over, see engine/modules/cpu_experts.h) it evaluates the
// experts on the pool and publishes the weighted sums
class Service {
public:
	static Service& get();
	// Starts the pool once: one thread per entry of `cpus`, thread i pinned to cpus[i] (empty: 4 unpinned threads); later calls are ignored
	void start( const std::vector<int>& cpus, int spin_us = 2000 );
	bool started() const { return started_; }
	// The rank's mailbox (pinned, mapped host memory), created on first use
	Mailbox* mailbox( int rank, int hidden, float act_limit );
	// A host expert of (rank, layer) under its global expert id; the service owns the linears
	void add_expert( int rank, int layer, int expert, std::unique_ptr<kernels::cpu::Linear> gate, std::unique_ptr<kernels::cpu::Linear> up, std::unique_ptr<kernels::cpu::Linear> down );
	// The linears of a host expert (gate, up, down), or null
	bool find_expert( int rank, int layer, int expert, kernels::cpu::Linear** lin3 ) const;
	// Bytes of host-side data the attached experts added (scales, repacked codes), over all ranks
	void add_host_bytes( size_t n ) { host_bytes_ += n; }
	size_t host_bytes() const { return host_bytes_; }
	// Memory for a CPU-layout expert matrix, bound to NUMA node `node` when >= 0 (released with the service)
	void* numa_alloc( size_t bytes, int node );
	// "10-19,22" -> {10..19, 22}
	static std::vector<int> parse_cpus( const std::string& list );
	// Stops the dispatcher and releases the mailboxes, host experts and pool, so a later start() begins afresh. Must run before the device contexts go: the
	// mailboxes are mapped pinned memory, which the dispatcher would otherwise read after cudaDeviceReset unmapped it
	void quiesce();
	~Service();

private:
	struct Owned { std::unique_ptr<kernels::cpu::Linear> gate, up, down; };
	struct Rank {
		Mailbox* mb = nullptr;
		int hidden = 0;
		float act_limit = 10.0f;
		uint64_t seen = 0;
		bool err_reported = false;
		std::vector<std::unordered_map<int, Owned>> layers;
	};
	void loop();
	void handle( Rank& r );
	void log_stats();
	bool started_ = false;
	std::unique_ptr<kernels::cpu::Pool> pool_;
	std::thread th_;
	std::atomic<bool> stop_{ false };
	mutable std::mutex mu_;
	std::vector<std::unique_ptr<Rank>> ranks_;
	std::vector<std::pair<void*, size_t>> maps_;
	std::atomic<size_t> host_bytes_{ 0 };
	std::atomic<uint32_t> ranks_version_{ 0 };   // bumped when a rank's mailbox appears, so the dispatcher refreshes its list only then
	struct Group { const Owned* o = nullptr; int first = 0, m = 0; std::vector<uint16_t> xg; std::vector<float> wts, tmp; };
	std::vector<Group> groups_;       // the experts of the request being served
	std::vector<ExpertCall> calls_;
	Scratch scratch_;
	std::atomic<uint64_t> requests_{ 0 }, experts_{ 0 }, rows_{ 0 };
	std::atomic<uint64_t> busy_ns_{ 0 };
	struct Mark { uint64_t requests = 0, experts = 0, busy_ns = 0, wait_ns = 0, waits = 0; } last_;   // totals at the last log line
};

}   // namespace engine::cpu_moe
