#include "engine/server/system_stats.h"
#include <dlfcn.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <unistd.h>

namespace engine::server {

namespace {

using nvmlRet = int;
struct nvmlMemory { unsigned long long total, free, used; };
struct nvmlUtil { unsigned gpu, memory; };

double now_s() {
	return std::chrono::duration<double>( std::chrono::steady_clock::now().time_since_epoch() ).count();
}

} // namespace

struct SystemStats::Api {
	nvmlRet ( *init )() = nullptr;
	nvmlRet ( *count )( unsigned* ) = nullptr;
	nvmlRet ( *handle )( unsigned, void** ) = nullptr;
	nvmlRet ( *name )( void*, char*, unsigned ) = nullptr;
	nvmlRet ( *power )( void*, unsigned* ) = nullptr;
	nvmlRet ( *limit )( void*, unsigned* ) = nullptr;
	nvmlRet ( *temp )( void*, int, unsigned* ) = nullptr;
	nvmlRet ( *util )( void*, nvmlUtil* ) = nullptr;
	nvmlRet ( *clock )( void*, int, unsigned* ) = nullptr;
	nvmlRet ( *mem )( void*, nvmlMemory* ) = nullptr;
	nvmlRet ( *throttle )( void*, unsigned long long* ) = nullptr;
	nvmlRet ( *energy )( void*, unsigned long long* ) = nullptr;
};

SystemStats::SystemStats() {
	lib_ = dlopen( "libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL );
	if ( !lib_ ) return;
	api_ = new Api();
#define NVML_SYM( field, sym ) api_->field = reinterpret_cast<decltype( api_->field )>( dlsym( lib_, sym ) )
	NVML_SYM( init, "nvmlInit_v2" ); NVML_SYM( count, "nvmlDeviceGetCount_v2" ); NVML_SYM( handle, "nvmlDeviceGetHandleByIndex_v2" );
	NVML_SYM( name, "nvmlDeviceGetName" ); NVML_SYM( power, "nvmlDeviceGetPowerUsage" ); NVML_SYM( limit, "nvmlDeviceGetEnforcedPowerLimit" );
	NVML_SYM( temp, "nvmlDeviceGetTemperature" ); NVML_SYM( util, "nvmlDeviceGetUtilizationRates" ); NVML_SYM( clock, "nvmlDeviceGetClockInfo" );
	NVML_SYM( mem, "nvmlDeviceGetMemoryInfo" ); NVML_SYM( throttle, "nvmlDeviceGetCurrentClocksThrottleReasons" );
	NVML_SYM( energy, "nvmlDeviceGetTotalEnergyConsumption" );
#undef NVML_SYM
	unsigned n = 0;
	if ( !api_->init || !api_->count || !api_->handle || api_->init() != 0 || api_->count( &n ) != 0 ) {
		dlclose( lib_ );
		lib_ = nullptr;
		return;
	}
	for ( unsigned i = 0; i < n; ++i ) {
		void* h = nullptr;
		if ( api_->handle( i, &h ) == 0 ) dev_.push_back( h );
	}
	gpus_.resize( dev_.size() );
}

SystemStats::~SystemStats() {
	delete api_;
}

void SystemStats::read( std::vector<GpuReading>* gpus, HostReading* host ) {
	const double t = now_s();
	if ( read_at_ < 0.0 || t - read_at_ > 0.4 ) {
		read_at_ = t;
		for ( size_t i = 0; i < dev_.size(); ++i ) {
			GpuReading& g = gpus_[i];
			void* h = dev_[i];
			char nm[96] = {};
			unsigned v = 0;
			nvmlUtil u = {};
			nvmlMemory m = {};
			unsigned long long ll = 0;
			if ( api_->name && api_->name( h, nm, sizeof( nm ) ) == 0 ) g.name = nm;
			if ( api_->power && api_->power( h, &v ) == 0 ) g.power_w = v / 1000.0;
			if ( api_->limit && api_->limit( h, &v ) == 0 ) g.power_limit_w = v / 1000.0;
			if ( api_->temp && api_->temp( h, 0, &v ) == 0 ) g.temp_c = static_cast<int>( v );
			if ( api_->util && api_->util( h, &u ) == 0 ) { g.util_gpu = static_cast<int>( u.gpu ); g.util_mem = static_cast<int>( u.memory ); }
			if ( api_->clock && api_->clock( h, 1, &v ) == 0 ) g.sm_mhz = static_cast<int>( v );
			if ( api_->clock && api_->clock( h, 2, &v ) == 0 ) g.mem_mhz = static_cast<int>( v );
			if ( api_->mem && api_->mem( h, &m ) == 0 ) { g.mem_used = m.used; g.mem_total = m.total; }
			if ( api_->throttle && api_->throttle( h, &ll ) == 0 ) g.throttle = ll;
			if ( api_->energy && api_->energy( h, &ll ) == 0 ) g.energy_kj = ll / 1.0e6;
		}
		// host: CPU busy since the previous reading, memory, this process
		{
			std::ifstream f( "/proc/stat" );
			std::string tag;
			uint64_t user = 0, nice = 0, sys = 0, idle = 0, iow = 0, irq = 0, sirq = 0, steal = 0;
			f >> tag >> user >> nice >> sys >> idle >> iow >> irq >> sirq >> steal;
			const uint64_t total = user + nice + sys + idle + iow + irq + sirq + steal, idl = idle + iow;
			if ( cpu_total_ > 0 && total > cpu_total_ ) host_.cpu_pct = 100.0 * ( 1.0 - static_cast<double>( idl - cpu_idle_ ) / static_cast<double>( total - cpu_total_ ) );
			cpu_total_ = total;
			cpu_idle_ = idl;
		}
		{
			std::ifstream f( "/proc/meminfo" );
			std::string key, unit;
			uint64_t kb = 0, mt = 0, ma = 0;
			while ( f >> key >> kb >> unit ) {
				if ( key == "MemTotal:" ) mt = kb;
				if ( key == "MemAvailable:" ) ma = kb;
			}
			host_.mem_total = mt * 1024;
			host_.mem_used = ( mt - ma ) * 1024;
		}
		{
			std::ifstream f( "/proc/self/statm" );
			uint64_t pages = 0, rss = 0;
			f >> pages >> rss;
			host_.rss = rss * static_cast<uint64_t>( sysconf( _SC_PAGESIZE ) );
		}
		{
			std::ifstream f( "/proc/loadavg" );
			f >> host_.load1;
		}
	}
	*gpus = gpus_;
	*host = host_;
}

std::vector<std::string> throttle_names( uint64_t bits ) {
	static const struct { uint64_t bit; const char* name; } kReasons[] = {
		{ 0x1, "gpu idle" }, { 0x2, "app clocks" }, { 0x4, "sw power cap" }, { 0x8, "hw slowdown" }, { 0x10, "sync boost" },
		{ 0x20, "sw thermal" }, { 0x40, "hw thermal" }, { 0x80, "hw power brake" }, { 0x100, "display clocks" },
	};
	std::vector<std::string> out;
	for ( const auto& r : kReasons ) {
		if ( bits & r.bit ) out.push_back( r.name );
	}
	return out;
}

} // namespace engine::server
