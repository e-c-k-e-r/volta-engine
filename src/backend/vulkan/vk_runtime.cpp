// Vulkan backend implementation: headless compute device, BDA allocations, streams as deferred command buffers, graphs as
// pre-recorded command buffers. One logical device per process (selected by ENGINE_VK_DEVICE or set_device), single-threaded.
#include "engine/backend/vulkan/device.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace engine::backend {

struct vk_stream;

namespace {

struct Alloc {
	VkBuffer buffer;
	VkDeviceMemory memory;
	uint64_t size;
};

struct Staging {
	VkBuffer buffer = VK_NULL_HANDLE;
	VkDeviceMemory memory = VK_NULL_HANDLE;
	void* map = nullptr;
	void* host_dst = nullptr; // D2H: copied here after the wait
	size_t bytes = 0;
};

struct Ctx {
	bool tried = false, ok = false;
	int device_index = 0;   // physical device index
	VkPhysicalDevice phys = VK_NULL_HANDLE;
	VkDevice device = VK_NULL_HANDLE;
	VkQueue queue = VK_NULL_HANDLE;
	uint32_t family = 0;
	VkCommandPool pool = VK_NULL_HANDLE;
	VkPhysicalDeviceMemoryProperties mem{};
	VkPhysicalDeviceProperties props{};
	bool rte16 = false;   // shaderRoundingModeRTEFloat16: float -> half conversions can be made round-to-nearest-even
	std::map<uint64_t, Alloc> allocs; // keyed by device address
	uint64_t allocated = 0;           // bytes in allocs
	vk_stream* default_stream = nullptr;
	PFN_vkGetBufferDeviceAddress get_address = nullptr;
};

// One logical device per rank (set_device(d) selects physical device base + d, base = ENGINE_VK_DEVICE or 0). The instance is shared, every
// device has its own Ctx, and the current device is per thread (like CUDA's): the rank threads drive their devices concurrently
struct Shared {
	std::mutex mu;
	bool tried = false;
	VkInstance instance = VK_NULL_HANDLE;
	std::vector<VkPhysicalDevice> phys;
	int base = 0;
};

constexpr int MAX_DEV = ENGINE_NUM_GPUS;
Shared sh;
Ctx gs[MAX_DEV];
thread_local int t_dev = 0;
inline Ctx& cur() { return gs[t_dev]; }

} // namespace

struct vk_stream {
	VkSemaphore timeline = VK_NULL_HANDLE;
	uint64_t value = 0;
	VkCommandBuffer cb = VK_NULL_HANDLE; // currently recording (or null)
	bool capture = false;
	size_t capture_nodes = 0;
	std::vector<VkCommandBuffer> inflight;
	std::vector<Staging> staging;
	size_t staging_bytes = 0;
};

struct vk_graph {
	VkCommandBuffer cb = VK_NULL_HANDLE;
	size_t nodes = 0;
};

namespace vk {
struct Pipeline {
	VkShaderModule module = VK_NULL_HANDLE;
	VkPipelineLayout layout = VK_NULL_HANDLE;
	VkPipeline pipeline = VK_NULL_HANDLE;
	uint32_t push_bytes = 0;
};
} // namespace vk

namespace {

// Instance and physical device list, once
bool init_instance() {
	if ( sh.tried ) return sh.instance != VK_NULL_HANDLE && !sh.phys.empty();
	sh.tried = true;
	if ( const char* e = std::getenv( "ENGINE_VK_DEVICE" ) ) sh.base = std::atoi( e );
	VkApplicationInfo app{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
	app.pApplicationName = "volta-engine";
	app.apiVersion = VK_API_VERSION_1_2;
	VkInstanceCreateInfo ici{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
	ici.pApplicationInfo = &app;
	if ( vkCreateInstance( &ici, nullptr, &sh.instance ) != VK_SUCCESS ) return false;
	uint32_t n = 0;
	vkEnumeratePhysicalDevices( sh.instance, &n, nullptr );
	sh.phys.resize( n );
	vkEnumeratePhysicalDevices( sh.instance, &n, sh.phys.data() );
	return n > 0;
}

// Device `dev` (logical): physical device base + dev
bool init_context( int dev ) {
	std::lock_guard<std::mutex> lock( sh.mu );
	Ctx& g = gs[dev];
	if ( g.tried ) return g.ok;
	g.tried = true;
	if ( !init_instance() ) return false;
	g.device_index = sh.base + dev;
	if ( g.device_index < 0 || size_t( g.device_index ) >= sh.phys.size() ) return false;
	g.phys = sh.phys[g.device_index];
	vkGetPhysicalDeviceProperties( g.phys, &g.props );
	vkGetPhysicalDeviceMemoryProperties( g.phys, &g.mem );
	{
		VkPhysicalDeviceFloatControlsProperties fc{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES };
		VkPhysicalDeviceProperties2 p2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &fc };
		vkGetPhysicalDeviceProperties2( g.phys, &p2 );
		g.rte16 = fc.shaderRoundingModeRTEFloat16 == VK_TRUE;
	}

	// Feature requirements of the Slang SPIR-V: PhysicalStorageBufferAddresses, Float16, Int64 (+ Int16 for the BF16 table)
	VkPhysicalDeviceVulkan11Features f11{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
	VkPhysicalDeviceVulkan12Features f12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f11 };
	VkPhysicalDeviceFeatures2 f2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f12 };
	vkGetPhysicalDeviceFeatures2( g.phys, &f2 );
	if ( !f12.bufferDeviceAddress || !f12.shaderFloat16 || !f12.timelineSemaphore || !f2.features.shaderInt64 ) return false;

	VkPhysicalDeviceVulkan11Features want11{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
	want11.storageBuffer16BitAccess = f11.storageBuffer16BitAccess;
	want11.uniformAndStorageBuffer16BitAccess = f11.uniformAndStorageBuffer16BitAccess;
	VkPhysicalDeviceVulkan12Features want12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &want11 };
	want12.bufferDeviceAddress = VK_TRUE;
	want12.shaderFloat16 = VK_TRUE;
	want12.timelineSemaphore = VK_TRUE;
	want12.shaderInt8 = f12.shaderInt8;
	want12.storageBuffer8BitAccess = f12.storageBuffer8BitAccess;
	VkPhysicalDeviceFeatures2 want2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &want12 };
	want2.features.shaderInt64 = VK_TRUE;
	want2.features.shaderInt16 = f2.features.shaderInt16;
	want2.features.shaderFloat64 = f2.features.shaderFloat64;

	uint32_t qn = 0;
	vkGetPhysicalDeviceQueueFamilyProperties( g.phys, &qn, nullptr );
	std::vector<VkQueueFamilyProperties> qf( qn );
	vkGetPhysicalDeviceQueueFamilyProperties( g.phys, &qn, qf.data() );
	bool found = false;
	for ( uint32_t i = 0; i < qn && !found; ++i ) {
		if ( qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT ) { g.family = i; found = true; }
	}
	if ( !found ) return false;

	const float prio = 1.0f;
	VkDeviceQueueCreateInfo qci{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
	qci.queueFamilyIndex = g.family;
	qci.queueCount = 1;
	qci.pQueuePriorities = &prio;
	VkDeviceCreateInfo dci{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &want2 };
	dci.queueCreateInfoCount = 1;
	dci.pQueueCreateInfos = &qci;
	if ( vkCreateDevice( g.phys, &dci, nullptr, &g.device ) != VK_SUCCESS ) return false;
	vkGetDeviceQueue( g.device, g.family, 0, &g.queue );

	VkCommandPoolCreateInfo pci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
	pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	pci.queueFamilyIndex = g.family;
	if ( vkCreateCommandPool( g.device, &pci, nullptr, &g.pool ) != VK_SUCCESS ) return false;
	g.get_address = reinterpret_cast<PFN_vkGetBufferDeviceAddress>( vkGetDeviceProcAddr( g.device, "vkGetBufferDeviceAddress" ) );
	g.ok = g.get_address != nullptr;
	return g.ok;
}

Ctx& ctx() {
	Ctx& c = gs[t_dev];
	if ( !c.ok ) {
		ENGINE_ASSERT( init_context( t_dev ), "Vulkan backend: no usable Vulkan device (check ENGINE_VK_DEVICE)" );
	}
	return c;
}

uint32_t find_memory( uint32_t type_bits, VkMemoryPropertyFlags want ) {
	for ( uint32_t i = 0; i < cur().mem.memoryTypeCount; ++i ) {
		if ( ( type_bits & ( 1u << i ) ) && ( cur().mem.memoryTypes[i].propertyFlags & want ) == want ) return i;
	}
	return UINT32_MAX;
}

// Buffer + memory; device-local with BDA, or host-visible/coherent and mapped when host is true. False on failure
bool create_buffer( size_t bytes, bool host, VkBuffer& buf, VkDeviceMemory& mem, void** map ) {
	VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
	bci.size = MAX( bytes, size_t( 4 ) );
	bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	if ( !host ) bci.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
	if ( vkCreateBuffer( cur().device, &bci, nullptr, &buf ) != VK_SUCCESS ) return false;
	VkMemoryRequirements req;
	vkGetBufferMemoryRequirements( cur().device, buf, &req );
	const uint32_t type = find_memory( req.memoryTypeBits, host ? ( VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT ) : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT );
	if ( type == UINT32_MAX ) { vkDestroyBuffer( cur().device, buf, nullptr ); return false; }
	VkMemoryAllocateFlagsInfo flags{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO };
	flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
	VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, host ? nullptr : &flags };
	mai.allocationSize = req.size;
	mai.memoryTypeIndex = type;
	if ( vkAllocateMemory( cur().device, &mai, nullptr, &mem ) != VK_SUCCESS ) { vkDestroyBuffer( cur().device, buf, nullptr ); return false; }
	vkBindBufferMemory( cur().device, buf, mem, 0 );
	if ( host && map ) vkMapMemory( cur().device, mem, 0, VK_WHOLE_SIZE, 0, map );
	return true;
}

// Device pointer -> (buffer, offset)
bool resolve( const void* p, VkBuffer& buf, VkDeviceSize& off ) {
	const uint64_t a = reinterpret_cast<uint64_t>( p );
	auto it = cur().allocs.upper_bound( a );
	if ( it == cur().allocs.begin() ) return false;
	--it;
	if ( a >= it->first + it->second.size ) return false;
	buf = it->second.buffer;
	off = a - it->first;
	return true;
}

bool is_device_ptr( const void* p ) {
	VkBuffer b;
	VkDeviceSize o;
	return resolve( p, b, o );
}

vk_stream* resolve_stream( stream_t s ) { return s ? s : cur().default_stream; }

VkCommandBuffer begin_cb( bool reusable ) {
	VkCommandBufferAllocateInfo ai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	ai.commandPool = cur().pool;
	ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	ai.commandBufferCount = 1;
	VkCommandBuffer cb;
	ENGINE_CHECK( vkAllocateCommandBuffers( cur().device, &ai, &cb ) );
	VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	bi.flags = reusable ? VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT : VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	ENGINE_CHECK( vkBeginCommandBuffer( cb, &bi ) );
	return cb;
}

VkCommandBuffer recording( vk_stream* s ) {
	if ( !s->cb ) s->cb = begin_cb( s->capture );
	return s->cb;
}

// Everything recorded so far is visible to everything recorded next (and to the host at the end of the buffer)
void barrier( VkCommandBuffer cb, VkPipelineStageFlags dst = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VkAccessFlags dst_access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT ) {
	VkMemoryBarrier mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
	mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
	mb.dstAccessMask = dst_access;
	vkCmdPipelineBarrier( cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, dst, 0, 1, &mb, 0, nullptr, 0, nullptr );
}

void submit( vk_stream* s, VkCommandBuffer cb ) {
	const uint64_t wait = s->value, signal = ++s->value;
	const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
	VkTimelineSemaphoreSubmitInfo tsi{ VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO };
	tsi.waitSemaphoreValueCount = 1;
	tsi.pWaitSemaphoreValues = &wait;
	tsi.signalSemaphoreValueCount = 1;
	tsi.pSignalSemaphoreValues = &signal;
	VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO, &tsi };
	si.waitSemaphoreCount = 1;
	si.pWaitSemaphores = &s->timeline;
	si.pWaitDstStageMask = &stage;
	si.commandBufferCount = 1;
	si.pCommandBuffers = &cb;
	si.signalSemaphoreCount = 1;
	si.pSignalSemaphores = &s->timeline;
	ENGINE_CHECK( vkQueueSubmit( cur().queue, 1, &si, VK_NULL_HANDLE ) );
}

void flush( vk_stream* s ) {
	ENGINE_ASSERT( !s->capture, "Vulkan backend: stream flush during graph capture" );
	if ( !s->cb ) return;
	barrier( s->cb, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT );
	ENGINE_CHECK( vkEndCommandBuffer( s->cb ) );
	submit( s, s->cb );
	s->inflight.push_back( s->cb );
	s->cb = VK_NULL_HANDLE;
}

void sync( vk_stream* s ) {
	flush( s );
	VkSemaphoreWaitInfo wi{ VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
	wi.semaphoreCount = 1;
	wi.pSemaphores = &s->timeline;
	wi.pValues = &s->value;
	ENGINE_CHECK( vkWaitSemaphores( cur().device, &wi, UINT64_MAX ) );
	for ( auto& st : s->staging ) {
		if ( st.host_dst ) std::memcpy( st.host_dst, st.map, st.bytes );
		vkUnmapMemory( cur().device, st.memory );
		vkDestroyBuffer( cur().device, st.buffer, nullptr );
		vkFreeMemory( cur().device, st.memory, nullptr );
	}
	s->staging.clear();
	s->staging_bytes = 0;
	if ( !s->inflight.empty() ) vkFreeCommandBuffers( cur().device, cur().pool, uint32_t( s->inflight.size() ), s->inflight.data() );
	s->inflight.clear();
}

vk_stream* make_stream() {
	auto* s = new vk_stream();
	VkSemaphoreTypeCreateInfo ti{ VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
	ti.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
	VkSemaphoreCreateInfo ci{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &ti };
	ENGINE_CHECK( vkCreateSemaphore( cur().device, &ci, nullptr, &s->timeline ) );
	return s;
}


// SPIR-V float -> half conversions without a rounding-mode decoration are round-toward-zero on NVIDIA's Vulkan driver, while the CUDA
// kernels and the host reference round to nearest even (a 50% mismatch in the FP16 results, compounding through a model). Slang has no
// switch for it, so the module gets OpCapability RoundingModeRTE, OpExtension "SPV_KHR_float_controls" and
// OpExecutionMode <entry> RoundingModeRTE 16 inserted
std::vector<uint32_t> with_rte16( const std::vector<uint32_t>& in ) {
	constexpr uint32_t kOpExtension = 10, kOpExtInstImport = 11, kOpMemoryModel = 14, kOpEntryPoint = 15, kOpCapability = 17, kOpExecutionMode = 16;
	std::vector<uint32_t> out( in.begin(), in.begin() + 5 );
	size_t i = 5;
	bool cap_done = false, ext_done = false, mode_done = false;
	std::vector<uint32_t> entries;
	auto emit_ext = [&]() {
		const char name[] = "SPV_KHR_float_controls";
		const uint32_t words = uint32_t( ( sizeof( name ) + 3 ) / 4 );
		out.push_back( ( ( 1 + words ) << 16 ) | kOpExtension );
		const size_t at = out.size();
		out.resize( at + words, 0 );
		std::memcpy( &out[at], name, sizeof( name ) );
		ext_done = true;
	};
	while ( i < in.size() ) {
		const uint32_t op = in[i] & 0xffffu, len = in[i] >> 16;
		if ( len == 0 || i + len > in.size() ) return in;
		if ( !cap_done && op != kOpCapability ) {
			out.insert( out.end(), { ( 2u << 16 ) | kOpCapability, 4467u } ); // RoundingModeRTE
			cap_done = true;
		}
		if ( !ext_done && cap_done && ( op == kOpExtInstImport || op == kOpMemoryModel ) ) emit_ext();
		if ( !mode_done && op != kOpEntryPoint && !entries.empty() ) {
			for ( uint32_t id : entries ) out.insert( out.end(), { ( 4u << 16 ) | kOpExecutionMode, id, 4462u, 16u } );
			mode_done = true;
		}
		if ( op == kOpEntryPoint ) entries.push_back( in[i + 2] );
		out.insert( out.end(), in.begin() + i, in.begin() + i + len );
		i += len;
	}
	return mode_done ? out : in;
}

} // namespace

void check_result( int result, const char* expr, const char* file, int line ) {
	if ( ENGINE_UNLIKELY( result != 0 ) ) {
		std::fprintf( stderr, "[VULKAN ERROR] %s (%s:%d): VkResult %d\n", expr, file, line, result );
		std::_Exit( 1 );
	}
}

// Makes `device` (logical, 0..) the calling thread's current device, creating its context on first use
void set_device( int device ) {
	ENGINE_ASSERT( device >= 0 && device < MAX_DEV, "Vulkan backend: device index out of range" );
	t_dev = device;
	ctx();
	if ( !cur().default_stream ) cur().default_stream = make_stream();
}
int get_device() { return t_dev; }
int device_count() {
	std::lock_guard<std::mutex> lock( sh.mu );
	if ( !init_instance() ) return 0;
	return std::max( 0, int( sh.phys.size() ) - sh.base );
}

stream_t stream_create() {
	ctx();
	if ( !cur().default_stream ) cur().default_stream = make_stream();
	return make_stream();
}
void stream_destroy( stream_t s ) {
	sync( s );
	vkDestroySemaphore( cur().device, s->timeline, nullptr );
	delete s;
}
void stream_sync( stream_t s ) {
	ctx();
	if ( !cur().default_stream ) cur().default_stream = make_stream();
	sync( resolve_stream( s ) );
}
void device_sync() {
	ctx();
	if ( !cur().default_stream ) cur().default_stream = make_stream();
	sync( cur().default_stream );
	vkQueueWaitIdle( cur().queue );
}

bool memcpy_async_try( void* dst, const void* src, size_t bytes, Copy dir, stream_t st ) {
	ctx();
	if ( !cur().default_stream ) cur().default_stream = make_stream();
	if ( bytes == 0 ) return true;
	vk_stream* s = resolve_stream( st );
	if ( dir == Copy::Auto ) {
		const bool d = is_device_ptr( dst ), sr = is_device_ptr( src );
		dir = d && sr ? Copy::D2D : d ? Copy::H2D : Copy::D2H;
	}
	VkBuffer db = VK_NULL_HANDLE, sb = VK_NULL_HANDLE;
	VkDeviceSize doff = 0, soff = 0;
	if ( dir == Copy::D2D ) {
		if ( !resolve( dst, db, doff ) || !resolve( src, sb, soff ) ) return false;
		VkBufferCopy r{ soff, doff, bytes };
		vkCmdCopyBuffer( recording( s ), sb, db, 1, &r );
	} else {
		ENGINE_ASSERT( !s->capture, "Vulkan backend: host transfers inside a graph capture not implemented" );
		// Bound the host memory held by staging buffers of in-flight transfers (a weight load copies tens of GiB)
		if ( s->staging_bytes + bytes > ( size_t( 1 ) << 29 ) && s->staging_bytes > 0 ) sync( s );
		Staging st_buf;
		st_buf.bytes = bytes;
		if ( !create_buffer( bytes, true, st_buf.buffer, st_buf.memory, &st_buf.map ) ) return false;
		if ( dir == Copy::H2D ) {
			if ( !resolve( dst, db, doff ) ) return false;
			std::memcpy( st_buf.map, src, bytes );
			VkBufferCopy r{ 0, doff, bytes };
			vkCmdCopyBuffer( recording( s ), st_buf.buffer, db, 1, &r );
		} else {
			if ( !resolve( src, sb, soff ) ) return false;
			st_buf.host_dst = dst;
			VkBufferCopy r{ soff, 0, bytes };
			vkCmdCopyBuffer( recording( s ), sb, st_buf.buffer, 1, &r );
		}
		s->staging.push_back( st_buf );
		s->staging_bytes += bytes;
	}
	barrier( s->cb );
	if ( s->capture ) ++s->capture_nodes;
	return true;
}
void memcpy_async( void* dst, const void* src, size_t bytes, Copy dir, stream_t s ) {
	ENGINE_ASSERT( memcpy_async_try( dst, src, bytes, dir, s ), "Vulkan backend: memcpy_async failed (unknown device pointer or out of memory)" );
}
void memcpy2d_async( void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, Copy dir, stream_t s ) {
	for ( size_t r = 0; r < height; ++r ) {
		memcpy_async( static_cast<char*>( dst ) + r * dpitch, static_cast<const char*>( src ) + r * spitch, width, dir, s );
	}
}
void memcpy_sync( void* dst, const void* src, size_t bytes, Copy dir ) {
	memcpy_async( dst, src, bytes, dir, nullptr );
	stream_sync( nullptr );
}
// Device to device through the host: the source device is read back on its default stream (its queue must be idle, as it is when the ranks
// are parked between forwards), then the bytes are queued onto `s` of the destination device (the caller's current device)
void peer_copy_async( void* dst, int dst_device, const void* src, int src_device, size_t bytes, stream_t s ) {
	if ( bytes == 0 ) return;
	if ( dst_device == src_device ) {
		memcpy_async( dst, src, bytes, Copy::D2D, s );
		return;
	}
	std::vector<uint8_t> tmp( bytes );
	const int prev = t_dev;
	t_dev = src_device;
	memcpy_async( tmp.data(), src, bytes, Copy::D2H, nullptr );
	stream_sync( nullptr );
	t_dev = dst_device;
	memcpy_async( dst, tmp.data(), bytes, Copy::H2D, s );   // the data is in a staging buffer once this returns
	t_dev = prev;
}
void memset_async( void* dst, int value, size_t bytes, stream_t st ) {
	ctx();
	if ( !cur().default_stream ) cur().default_stream = make_stream();
	if ( bytes == 0 ) return;
	vk_stream* s = resolve_stream( st );
	VkBuffer b;
	VkDeviceSize off;
	ENGINE_ASSERT( resolve( dst, b, off ), "Vulkan backend: memset on an unknown device pointer" );
	ENGINE_ASSERT( off % 4 == 0 && bytes % 4 == 0, "Vulkan backend: memset with unaligned address/size not implemented" );
	const uint32_t byte = uint32_t( value ) & 0xffu;
	vkCmdFillBuffer( recording( s ), b, off, bytes, byte * 0x01010101u );
	barrier( s->cb );
	if ( s->capture ) ++s->capture_nodes;
}
void memset_sync( void* dst, int value, size_t bytes ) {
	memset_async( dst, value, bytes, nullptr );
	stream_sync( nullptr );
}

void* host_alloc_pinned( size_t bytes, Pin mode ) {
	if ( mode != Pin::Default ) return nullptr; // device-visible mapped host memory not implemented
	return std::malloc( bytes );
}
void host_free_pinned( void* p ) { std::free( p ); }

bool last_error_ok() { return true; }
void check_last_error() {}
void clear_last_error() {}
void device_reset() {}

bool set_device_try( int device ) {
	if ( device < 0 || device >= MAX_DEV ) return false;
	t_dev = device;
	if ( !gs[device].ok && !init_context( device ) ) return false;
	if ( !cur().default_stream ) cur().default_stream = make_stream();
	return true;
}

// Device-local heap size less what this process allocated (no VK_EXT_memory_budget: other processes are not accounted for)
size_t device_free_bytes( int device ) {
	if ( device < 0 || device >= MAX_DEV ) device = t_dev;
	if ( !gs[device].ok && !init_context( device ) ) return 0;
	const Ctx& c = gs[device];
	uint64_t heap = 0;
	for ( uint32_t i = 0; i < c.mem.memoryHeapCount; ++i ) {
		if ( c.mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT ) heap = MAX( heap, c.mem.memoryHeaps[i].size );
	}
	return heap > c.allocated ? size_t( heap - c.allocated ) : 0;
}

void* device_malloc_try( size_t bytes ) {
	ctx();
	if ( !cur().default_stream ) cur().default_stream = make_stream();
	Alloc a{};
	a.size = MAX( bytes, size_t( 4 ) );
	if ( !create_buffer( bytes, false, a.buffer, a.memory, nullptr ) ) return nullptr;
	VkBufferDeviceAddressInfo ai{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
	ai.buffer = a.buffer;
	const uint64_t addr = cur().get_address( cur().device, &ai );
	cur().allocs[addr] = a;
	cur().allocated += a.size;
	return reinterpret_cast<void*>( addr );
}
void* device_malloc( size_t bytes ) {
	void* p = device_malloc_try( bytes );
	ENGINE_ASSERT( p != nullptr, "Vulkan backend: device_malloc failed" );
	return p;
}
void device_free( void* p ) {
	if ( !p ) return;
	auto it = cur().allocs.find( reinterpret_cast<uint64_t>( p ) );
	if ( it == cur().allocs.end() ) return;
	vkQueueWaitIdle( cur().queue ); // cudaFree synchronizes implicitly
	vkDestroyBuffer( cur().device, it->second.buffer, nullptr );
	vkFreeMemory( cur().device, it->second.memory, nullptr );
	cur().allocated -= it->second.size;
	cur().allocs.erase( it );
}

void graph_capture_begin( stream_t st ) {
	ctx();
	if ( !cur().default_stream ) cur().default_stream = make_stream();
	vk_stream* s = resolve_stream( st );
	ENGINE_ASSERT( !s->capture, "Vulkan backend: nested graph capture" );
	flush( s );
	s->capture = true;
	s->capture_nodes = 0;
}
graph_t graph_capture_end( stream_t st ) {
	vk_stream* s = resolve_stream( st );
	ENGINE_ASSERT( s->capture, "Vulkan backend: graph_capture_end without begin" );
	auto* gr = new vk_graph();
	if ( !s->cb ) recording( s ); // empty capture
	ENGINE_CHECK( vkEndCommandBuffer( s->cb ) );
	gr->cb = s->cb;
	gr->nodes = s->capture_nodes;
	s->cb = VK_NULL_HANDLE;
	s->capture = false;
	return gr;
}
// The recorded command buffer is the executable: it moves to the executable object, so destroying the graph afterwards (as GraphCache does right
// after instantiating) does not free what the executable still replays
graph_exec_t graph_instantiate( graph_t gr ) {
	auto* e = new vk_graph();
	e->cb = gr->cb;
	e->nodes = gr->nodes;
	gr->cb = VK_NULL_HANDLE;
	return e;
}
size_t graph_node_count( graph_t gr ) { return gr->nodes; }
void graph_launch( graph_exec_t e, stream_t st ) {
	vk_stream* s = resolve_stream( st );
	flush( s );
	submit( s, e->cb );
}
void graph_destroy( graph_t gr ) {
	if ( !gr ) return;
	vkQueueWaitIdle( cur().queue );
	if ( gr->cb ) vkFreeCommandBuffers( cur().device, cur().pool, 1, &gr->cb );
	delete gr;
}
void graph_exec_destroy( graph_exec_t e ) { graph_destroy( e ); }

namespace vk {

bool available() {
	if ( !cur().ok && !init_context( t_dev ) ) return false;
	if ( !cur().default_stream ) cur().default_stream = make_stream();
	return true;
}
const char* device_name() { return ctx().props.deviceName; }
uint32_t max_push_constants_bytes() { return ctx().props.limits.maxPushConstantsSize; }

Pipeline* pipeline_create( const char* spv_path, uint32_t push_bytes ) {
	ctx();
	if ( push_bytes > cur().props.limits.maxPushConstantsSize || push_bytes % 4 ) return nullptr;
	FILE* f = std::fopen( spv_path, "rb" );
	if ( !f ) return nullptr;
	std::fseek( f, 0, SEEK_END );
	const long n = std::ftell( f );
	std::fseek( f, 0, SEEK_SET );
	std::vector<uint32_t> code( ( size_t( n ) + 3 ) / 4 );
	const bool read_ok = n > 0 && std::fread( code.data(), 1, size_t( n ), f ) == size_t( n );
	std::fclose( f );
	if ( !read_ok ) return nullptr;
	if ( cur().rte16 && code.size() > 5 && code[0] == 0x07230203u ) code = with_rte16( code );

	auto* p = new Pipeline();
	p->push_bytes = push_bytes;
	VkShaderModuleCreateInfo smi{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
	smi.codeSize = code.size() * 4;
	smi.pCode = code.data();
	VkPushConstantRange range{ VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes };
	VkPipelineLayoutCreateInfo pli{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
	pli.pushConstantRangeCount = push_bytes ? 1 : 0;
	pli.pPushConstantRanges = &range;
	VkComputePipelineCreateInfo cpi{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
	cpi.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
	cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	cpi.stage.pName = "main";
	if ( vkCreateShaderModule( cur().device, &smi, nullptr, &p->module ) != VK_SUCCESS
		|| vkCreatePipelineLayout( cur().device, &pli, nullptr, &p->layout ) != VK_SUCCESS ) {
		pipeline_destroy( p );
		return nullptr;
	}
	cpi.stage.module = p->module;
	cpi.layout = p->layout;
	if ( vkCreateComputePipelines( cur().device, VK_NULL_HANDLE, 1, &cpi, nullptr, &p->pipeline ) != VK_SUCCESS ) {
		pipeline_destroy( p );
		return nullptr;
	}
	return p;
}

void pipeline_destroy( Pipeline* p ) {
	if ( !p ) return;
	vkQueueWaitIdle( cur().queue );
	if ( p->pipeline ) vkDestroyPipeline( cur().device, p->pipeline, nullptr );
	if ( p->layout ) vkDestroyPipelineLayout( cur().device, p->layout, nullptr );
	if ( p->module ) vkDestroyShaderModule( cur().device, p->module, nullptr );
	delete p;
}

void dispatch( stream_t st, Pipeline* p, const void* push, uint32_t gx, uint32_t gy, uint32_t gz ) {
	ctx();
	if ( !cur().default_stream ) cur().default_stream = make_stream();
	vk_stream* s = resolve_stream( st );
	VkCommandBuffer cb = recording( s );
	vkCmdBindPipeline( cb, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipeline );
	if ( p->push_bytes ) vkCmdPushConstants( cb, p->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, p->push_bytes, push );
	vkCmdDispatch( cb, gx, gy, gz );
	barrier( cb );
	if ( s->capture ) ++s->capture_nodes;
}

} // namespace vk
} // namespace engine::backend
