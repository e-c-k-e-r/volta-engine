#pragma once
// Host and GPU readings for the monitor page. NVML is loaded at run time (libnvidia-ml.so.1), so a machine without it just reports no GPUs.
#include <string>
#include <vector>
#include <cstdint>

namespace engine::server {

struct GpuReading {
	std::string name;
	double power_w = 0.0, power_limit_w = 0.0, energy_kj = 0.0;
	int temp_c = 0, util_gpu = 0, util_mem = 0;
	int sm_mhz = 0, mem_mhz = 0;
	uint64_t mem_used = 0, mem_total = 0;
	uint64_t throttle = 0;       // NVML clocks-throttle-reason bits
};

struct HostReading {
	double cpu_pct = 0.0;        // all cores, since the previous reading
	uint64_t mem_used = 0, mem_total = 0;
	uint64_t rss = 0;            // this process
	double load1 = 0.0;
};

class SystemStats {
public:
	SystemStats();
	~SystemStats();
	// Cached for a short while, so several viewers do not multiply the NVML calls
	void read( std::vector<GpuReading>* gpus, HostReading* host );
	bool has_nvml() const { return lib_ != nullptr; }

private:
	void* lib_ = nullptr;
	std::vector<void*> dev_;
	std::vector<GpuReading> gpus_;
	HostReading host_;
	double read_at_ = -1.0;
	uint64_t cpu_total_ = 0, cpu_idle_ = 0;
	struct Api;
	Api* api_ = nullptr;
};

// Names of the NVML throttle-reason bits that are set
std::vector<std::string> throttle_names( uint64_t bits );

} // namespace engine::server
