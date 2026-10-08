// Vulkan backend implementation: headless compute device, BDA allocations, streams as deferred command buffers, graphs as
// pre-recorded command buffers. One logical device per process (selected by VOLTA_VK_DEVICE or set_device), single-threaded.
#include "engine/backend/vulkan/device.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
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
	int device_index = 0;
	VkInstance instance = VK_NULL_HANDLE;
	VkPhysicalDevice phys = VK_NULL_HANDLE;
	VkDevice device = VK_NULL_HANDLE;
	VkQueue queue = VK_NULL_HANDLE;
	uint32_t family = 0;
	VkCommandPool pool = VK_NULL_HANDLE;
	VkPhysicalDeviceMemoryProperties mem{};
	VkPhysicalDeviceProperties props{};
	std::map<uint64_t, Alloc> allocs; // keyed by device address
	vk_stream* default_stream = nullptr;
	PFN_vkGetBufferDeviceAddress get_address = nullptr;
};

Ctx g;

} // namespace

struct vk_stream {
	VkSemaphore timeline = VK_NULL_HANDLE;
	uint64_t value = 0;
	VkCommandBuffer cb = VK_NULL_HANDLE; // currently recording (or null)
	bool capture = false;
	size_t capture_nodes = 0;
	std::vector<VkCommandBuffer> inflight;
	std::vector<Staging> staging;
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

bool init_context() {
	if ( g.tried ) return g.ok;
	g.tried = true;
	if ( const char* e = std::getenv( "VOLTA_VK_DEVICE" ) ) g.device_index = std::atoi( e );

	VkApplicationInfo app{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
	app.pApplicationName = "volta-engine";
	app.apiVersion = VK_API_VERSION_1_2;
	VkInstanceCreateInfo ici{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
	ici.pApplicationInfo = &app;
	if ( vkCreateInstance( &ici, nullptr, &g.instance ) != VK_SUCCESS ) return false;

	uint32_t n = 0;
	vkEnumeratePhysicalDevices( g.instance, &n, nullptr );
	if ( n == 0 || g.device_index < 0 || uint32_t( g.device_index ) >= n ) return false;
	std::vector<VkPhysicalDevice> devs( n );
	vkEnumeratePhysicalDevices( g.instance, &n, devs.data() );
	g.phys = devs[g.device_index];
	vkGetPhysicalDeviceProperties( g.phys, &g.props );
	vkGetPhysicalDeviceMemoryProperties( g.phys, &g.mem );

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
	ENGINE_ASSERT( init_context(), "Vulkan backend: no usable Vulkan device (check VOLTA_VK_DEVICE)" );
	return g;
}

uint32_t find_memory( uint32_t type_bits, VkMemoryPropertyFlags want ) {
	for ( uint32_t i = 0; i < g.mem.memoryTypeCount; ++i ) {
		if ( ( type_bits & ( 1u << i ) ) && ( g.mem.memoryTypes[i].propertyFlags & want ) == want ) return i;
	}
	return UINT32_MAX;
}

// Buffer + memory; device-local with BDA, or host-visible/coherent and mapped when host is true. False on failure
bool create_buffer( size_t bytes, bool host, VkBuffer& buf, VkDeviceMemory& mem, void** map ) {
	VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
	bci.size = MAX( bytes, size_t( 4 ) );
	bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	if ( !host ) bci.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
	if ( vkCreateBuffer( g.device, &bci, nullptr, &buf ) != VK_SUCCESS ) return false;
	VkMemoryRequirements req;
	vkGetBufferMemoryRequirements( g.device, buf, &req );
	const uint32_t type = find_memory( req.memoryTypeBits, host ? ( VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT ) : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT );
	if ( type == UINT32_MAX ) { vkDestroyBuffer( g.device, buf, nullptr ); return false; }
	VkMemoryAllocateFlagsInfo flags{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO };
	flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
	VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, host ? nullptr : &flags };
	mai.allocationSize = req.size;
	mai.memoryTypeIndex = type;
	if ( vkAllocateMemory( g.device, &mai, nullptr, &mem ) != VK_SUCCESS ) { vkDestroyBuffer( g.device, buf, nullptr ); return false; }
	vkBindBufferMemory( g.device, buf, mem, 0 );
	if ( host && map ) vkMapMemory( g.device, mem, 0, VK_WHOLE_SIZE, 0, map );
	return true;
}

// Device pointer -> (buffer, offset)
bool resolve( const void* p, VkBuffer& buf, VkDeviceSize& off ) {
	const uint64_t a = reinterpret_cast<uint64_t>( p );
	auto it = g.allocs.upper_bound( a );
	if ( it == g.allocs.begin() ) return false;
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

vk_stream* resolve_stream( stream_t s ) { return s ? s : g.default_stream; }

VkCommandBuffer begin_cb( bool reusable ) {
	VkCommandBufferAllocateInfo ai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	ai.commandPool = g.pool;
	ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	ai.commandBufferCount = 1;
	VkCommandBuffer cb;
	ENGINE_CHECK( vkAllocateCommandBuffers( g.device, &ai, &cb ) );
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
	ENGINE_CHECK( vkQueueSubmit( g.queue, 1, &si, VK_NULL_HANDLE ) );
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
	ENGINE_CHECK( vkWaitSemaphores( g.device, &wi, UINT64_MAX ) );
	for ( auto& st : s->staging ) {
		if ( st.host_dst ) std::memcpy( st.host_dst, st.map, st.bytes );
		vkUnmapMemory( g.device, st.memory );
		vkDestroyBuffer( g.device, st.buffer, nullptr );
		vkFreeMemory( g.device, st.memory, nullptr );
	}
	s->staging.clear();
	if ( !s->inflight.empty() ) vkFreeCommandBuffers( g.device, g.pool, uint32_t( s->inflight.size() ), s->inflight.data() );
	s->inflight.clear();
}

vk_stream* make_stream() {
	auto* s = new vk_stream();
	VkSemaphoreTypeCreateInfo ti{ VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
	ti.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
	VkSemaphoreCreateInfo ci{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &ti };
	ENGINE_CHECK( vkCreateSemaphore( g.device, &ci, nullptr, &s->timeline ) );
	return s;
}

} // namespace

void check_result( int result, const char* expr, const char* file, int line ) {
	if ( ENGINE_UNLIKELY( result != 0 ) ) {
		std::fprintf( stderr, "[VULKAN ERROR] %s (%s:%d): VkResult %d\n", expr, file, line, result );
		std::_Exit( 1 );
	}
}

void set_device( int device ) {
	if ( !g.tried ) {
		g.device_index = device;
		std::string v = std::to_string( device );
		setenv( "VOLTA_VK_DEVICE", v.c_str(), 1 );
		ctx();
		g.default_stream = make_stream();
		return;
	}
	ENGINE_ASSERT( device == g.device_index, "Vulkan backend: set_device switching the device after initialization not implemented" );
}
int get_device() { return g.device_index; }
int device_count() {
	if ( g.instance == VK_NULL_HANDLE ) ctx();
	uint32_t n = 0;
	vkEnumeratePhysicalDevices( g.instance, &n, nullptr );
	return int( n );
}

stream_t stream_create() {
	ctx();
	if ( !g.default_stream ) g.default_stream = make_stream();
	return make_stream();
}
void stream_destroy( stream_t s ) {
	sync( s );
	vkDestroySemaphore( g.device, s->timeline, nullptr );
	delete s;
}
void stream_sync( stream_t s ) {
	ctx();
	if ( !g.default_stream ) g.default_stream = make_stream();
	sync( resolve_stream( s ) );
}
void device_sync() {
	ctx();
	if ( !g.default_stream ) g.default_stream = make_stream();
	sync( g.default_stream );
	vkQueueWaitIdle( g.queue );
}

bool memcpy_async_try( void* dst, const void* src, size_t bytes, Copy dir, stream_t st ) {
	ctx();
	if ( !g.default_stream ) g.default_stream = make_stream();
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
void peer_copy_async( void*, int, const void*, int, size_t, stream_t ) {
	ENGINE_ASSERT( false, "Vulkan backend: peer_copy_async (multi-device) not implemented" );
}
void memset_async( void* dst, int value, size_t bytes, stream_t st ) {
	ctx();
	if ( !g.default_stream ) g.default_stream = make_stream();
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
	if ( !g.tried ) {
		if ( device < 0 ) return false;
		set_device( device );
		return g.ok;
	}
	return device == g.device_index;
}

void* device_malloc_try( size_t bytes ) {
	ctx();
	if ( !g.default_stream ) g.default_stream = make_stream();
	Alloc a{};
	a.size = MAX( bytes, size_t( 4 ) );
	if ( !create_buffer( bytes, false, a.buffer, a.memory, nullptr ) ) return nullptr;
	VkBufferDeviceAddressInfo ai{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
	ai.buffer = a.buffer;
	const uint64_t addr = g.get_address( g.device, &ai );
	g.allocs[addr] = a;
	return reinterpret_cast<void*>( addr );
}
void* device_malloc( size_t bytes ) {
	void* p = device_malloc_try( bytes );
	ENGINE_ASSERT( p != nullptr, "Vulkan backend: device_malloc failed" );
	return p;
}
void device_free( void* p ) {
	if ( !p ) return;
	auto it = g.allocs.find( reinterpret_cast<uint64_t>( p ) );
	if ( it == g.allocs.end() ) return;
	vkQueueWaitIdle( g.queue ); // cudaFree synchronizes implicitly
	vkDestroyBuffer( g.device, it->second.buffer, nullptr );
	vkFreeMemory( g.device, it->second.memory, nullptr );
	g.allocs.erase( it );
}

void graph_capture_begin( stream_t st ) {
	ctx();
	if ( !g.default_stream ) g.default_stream = make_stream();
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
graph_exec_t graph_instantiate( graph_t gr ) { return gr; } // the recorded command buffer is the executable
size_t graph_node_count( graph_t gr ) { return gr->nodes; }
void graph_launch( graph_exec_t e, stream_t st ) {
	vk_stream* s = resolve_stream( st );
	flush( s );
	submit( s, e->cb );
}
void graph_destroy( graph_t gr ) {
	if ( !gr ) return;
	vkQueueWaitIdle( g.queue );
	vkFreeCommandBuffers( g.device, g.pool, 1, &gr->cb );
	delete gr;
}
void graph_exec_destroy( graph_exec_t ) {} // owned by the graph

namespace vk {

bool available() {
	if ( !init_context() ) return false;
	if ( !g.default_stream ) g.default_stream = make_stream();
	return true;
}
const char* device_name() { return ctx().props.deviceName; }
uint32_t max_push_constants_bytes() { return ctx().props.limits.maxPushConstantsSize; }

Pipeline* pipeline_create( const char* spv_path, uint32_t push_bytes ) {
	ctx();
	if ( push_bytes > g.props.limits.maxPushConstantsSize || push_bytes % 4 ) return nullptr;
	FILE* f = std::fopen( spv_path, "rb" );
	if ( !f ) return nullptr;
	std::fseek( f, 0, SEEK_END );
	const long n = std::ftell( f );
	std::fseek( f, 0, SEEK_SET );
	std::vector<uint32_t> code( ( size_t( n ) + 3 ) / 4 );
	const bool read_ok = n > 0 && std::fread( code.data(), 1, size_t( n ), f ) == size_t( n );
	std::fclose( f );
	if ( !read_ok ) return nullptr;

	auto* p = new Pipeline();
	p->push_bytes = push_bytes;
	VkShaderModuleCreateInfo smi{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
	smi.codeSize = size_t( n );
	smi.pCode = code.data();
	VkPushConstantRange range{ VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes };
	VkPipelineLayoutCreateInfo pli{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
	pli.pushConstantRangeCount = push_bytes ? 1 : 0;
	pli.pPushConstantRanges = &range;
	VkComputePipelineCreateInfo cpi{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
	cpi.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
	cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	cpi.stage.pName = "main";
	if ( vkCreateShaderModule( g.device, &smi, nullptr, &p->module ) != VK_SUCCESS
		|| vkCreatePipelineLayout( g.device, &pli, nullptr, &p->layout ) != VK_SUCCESS ) {
		pipeline_destroy( p );
		return nullptr;
	}
	cpi.stage.module = p->module;
	cpi.layout = p->layout;
	if ( vkCreateComputePipelines( g.device, VK_NULL_HANDLE, 1, &cpi, nullptr, &p->pipeline ) != VK_SUCCESS ) {
		pipeline_destroy( p );
		return nullptr;
	}
	return p;
}

void pipeline_destroy( Pipeline* p ) {
	if ( !p ) return;
	vkQueueWaitIdle( g.queue );
	if ( p->pipeline ) vkDestroyPipeline( g.device, p->pipeline, nullptr );
	if ( p->layout ) vkDestroyPipelineLayout( g.device, p->layout, nullptr );
	if ( p->module ) vkDestroyShaderModule( g.device, p->module, nullptr );
	delete p;
}

void dispatch( stream_t st, Pipeline* p, const void* push, uint32_t gx, uint32_t gy, uint32_t gz ) {
	ctx();
	if ( !g.default_stream ) g.default_stream = make_stream();
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
