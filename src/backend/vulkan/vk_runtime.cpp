// Vulkan backend implementation: headless compute device, BDA allocations, streams as deferred command buffers, graphs as
// pre-recorded command buffers. A logical device per rank (selected by ENGINE_VK_DEVICE and set_device); several threads may share a device as
// long as each stream is used by one thread at a time.
#include "engine/backend/vulkan/device.h"

// VK_NV_cuda_kernel_launch is a provisional extension (vulkan_beta.h)
#define VK_ENABLE_BETA_EXTENSIONS
#include <vulkan/vulkan.h>

#include "engine/runtime/options.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// The option table lives in the host library, which the module-only smoke binaries do not link: bound weakly (no option = defaults)
namespace engine::runtime {
__attribute__(( weak )) Options& options();
}

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
	size_t capacity = 0;      // the buffer's size (a pooled one: a power of two)
};

// Host transfers stage through mapped buffers. Those of up to kPoolMax bytes come from a per-device pool of power-of-two sizes and go back to
// it at the stream's sync (an allocation per eager transfer cost milliseconds); larger ones (weight loads) are made and freed per transfer
constexpr size_t kPoolMin = size_t( 64 ) << 10, kPoolMax = size_t( 64 ) << 20, kPoolLimit = size_t( 256 ) << 20;

struct Ctx {
	bool tried = false, ok = false;
	int device_index = 0;   // physical device index
	VkPhysicalDevice phys = VK_NULL_HANDLE;
	VkDevice device = VK_NULL_HANDLE;
	VkQueue queue = VK_NULL_HANDLE;
	uint32_t family = 0;
	VkPhysicalDeviceMemoryProperties mem{};
	VkPhysicalDeviceProperties props{};
	PFN_vkCmdPushDescriptorSetKHR push_descriptor = nullptr;
	std::mutex queue_mu;        // vkQueueSubmit / vkQueueWaitIdle need external synchronization: a device's queue is shared by the threads that use it
	std::mutex* host_queue_mu = nullptr;   // an adopted device's: the host application's mutex for the queue (instead of queue_mu)
	bool host_import = false;   // VK_EXT_external_memory_host: a host allocation imported as device-addressable memory
	bool sem_fd = false;        // VK_KHR_external_semaphore_fd: timeline semaphores shared with the other devices
	size_t host_import_align = 0;   // VK_KHR_push_descriptor: storage buffers bound inside the command buffer
	bool int_dot = false;   // shaderIntegerDotProduct (VK_KHR_shader_integer_dot_product) enabled: packed 4 x int8 dot products
	bool budget = false;    // VK_EXT_memory_budget: the heaps' budget and usage (the driver's and other processes' allocations included)
	bool sg32 = false;      // VK_EXT_subgroup_size_control: pipelines created with a required subgroup size of 32 (the device's default is another)
	// vulkan_profile: a timestamp before and after every dispatch (pairs in qpool, the pipeline of each in qwho), read at the next sync
	VkQueryPool qpool = VK_NULL_HANDLE;
	uint32_t qused = 0;
	bool qreset = true;
	std::vector<const vk::Pipeline*> qwho;
	bool rte16 = false;   // shaderRoundingModeRTEFloat16: float -> half conversions can be made round-to-nearest-even
	// VK_NV_cuda_kernel_launch: CUDA modules by name (a failed one null), their functions by "module:name" (a missing one null)
	bool cuda = false;
	uint32_t sm_count = 0;   // VK_NV_shader_sm_builtins' shaderSMCount (0: unknown)
	PFN_vkCreateCudaModuleNV cuda_create_module = nullptr;
	PFN_vkCreateCudaFunctionNV cuda_create_function = nullptr;
	PFN_vkCmdCudaLaunchKernelNV cuda_cmd_launch = nullptr;
	std::map<std::string, VkCudaModuleNV> cuda_modules;
	std::map<std::string, vk::CudaFunction*> cuda_functions;
	// Shared by the threads that use the device (a language model's and the TTS's, say): the allocation table and the staging pool are taken
	// under mu; every stream records into command buffers of its own pool, so threads on different streams never share a pool
	std::mutex mu;
	std::map<uint64_t, Alloc> allocs; // keyed by device address
	std::multimap<size_t, Staging> staging_pool;   // idle pooled staging buffers by capacity
	size_t staging_pool_bytes = 0;
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
	int select_index = -1;       // select_device: the base by index (-1: ENGINE_VK_DEVICE) or by PCI device ID
	uint32_t select_pci = 0;
	bool adopted = false;        // adopt_device: the host's instance and device
	vk::HostDevice host;
	std::string adopt_error;     // why the adopted device can't be used
};

constexpr int MAX_DEV = ENGINE_NUM_GPUS;
Shared sh;
Ctx gs[MAX_DEV];
thread_local int t_dev = 0;
inline Ctx& cur() { return gs[t_dev]; }
inline std::mutex& queue_mutex( Ctx& c ) { return c.host_queue_mu ? *c.host_queue_mu : c.queue_mu; }

} // namespace

// A cross-device sync point of a capture: after its segment the signals (at launch each counter advanced, its semaphore signalled with the new
// value; the capture-time values for a capture that falls back to running), before the next segment the waits (at their counters' values)
struct GraphSync {
	std::vector<VkSemaphore> sig;
	std::vector<uint64_t*> sig_ctr;
	std::vector<uint64_t> sig_val;
	std::vector<VkSemaphore> wait;
	std::vector<const uint64_t*> wait_ctr;
	std::vector<uint64_t> wait_val;
};

// A stream's command pool, shared with the graphs captured on it (a graph may outlive its stream)
struct CmdPool {
	VkDevice device = VK_NULL_HANDLE;
	VkCommandPool pool = VK_NULL_HANDLE;
	~CmdPool() { if ( pool ) vkDestroyCommandPool( device, pool, nullptr ); }
};

struct vk_stream {
	VkSemaphore timeline = VK_NULL_HANDLE;
	uint64_t value = 0;
	std::shared_ptr<CmdPool> pool;
	VkCommandBuffer cb = VK_NULL_HANDLE; // currently recording (or null)
	bool capture = false;
	size_t capture_nodes = 0;
	std::vector<VkCommandBuffer> cap_segs;   // capture: the closed segments, segment i followed by cap_syncs[i]
	std::vector<GraphSync> cap_syncs;
	bool capture_failed = false;             // a command the capture cannot hold ran it (capture_abort): graph_capture_end returns no graph
	bool concurrent = false;   // inside concurrent_begin / concurrent_end: a command's barrier is owed (held) until the next command or the end
	bool held = false;
	bool run_scratch = false;  // an op of the run took a scratch buffer since its last full barrier
	std::vector<VkCommandBuffer> inflight;
	std::vector<Staging> staging;
	size_t staging_bytes = 0;
	std::vector<VkSemaphore> extra_wait, extra_signal;   // the next submission's waits / signals on shared semaphores (stream_wait / stream_signal)
	std::vector<uint64_t> extra_wait_values, extra_signal_values;
};

// A captured sequence: command buffers split where a cross-device sync was recorded (segs.size() == syncs.size() + 1). parity_ctr: the first
// counted signal's counter, parity the parity of its value at capture (the collective's double-buffered slots alternate from it)
struct vk_graph {
	std::shared_ptr<CmdPool> pool;   // the capturing stream's: the segments' pool
	std::vector<VkCommandBuffer> segs;
	std::vector<GraphSync> syncs;
	uint64_t* parity_ctr = nullptr;
	uint64_t parity = 0;
	VkSemaphore parity_sem = VK_NULL_HANDLE;
	size_t nodes = 0;
};

namespace vk {
struct Pipeline {
	VkShaderModule module = VK_NULL_HANDLE;
	VkPipelineLayout layout = VK_NULL_HANDLE;
	VkPipeline pipeline = VK_NULL_HANDLE;
	VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;   // pipelines with storage-buffer bindings (pipeline_create_bound)
	uint32_t push_bytes = 0, buffers = 0;
	std::string name;   // its .spv file's name (vulkan_profile's rows)
};
// A CUDA kernel of a loaded module; prof names its vulkan_profile rows
struct CudaFunction {
	VkCudaFunctionNV fn = VK_NULL_HANDLE;
	Pipeline prof;
};
} // namespace vk

namespace {

// Instance and physical device list, once
bool init_instance() {
	if ( sh.tried ) return sh.instance != VK_NULL_HANDLE && !sh.phys.empty();
	sh.tried = true;
	if ( sh.adopted ) {
		sh.instance = static_cast<VkInstance>( sh.host.instance );
		sh.phys = { static_cast<VkPhysicalDevice>( sh.host.physical_device ) };
		sh.base = 0;
		return true;
	}
	if ( sh.select_index >= 0 ) sh.base = sh.select_index;
	else if ( const char* e = std::getenv( "ENGINE_VK_DEVICE" ) ) sh.base = std::atoi( e );
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
	if ( sh.select_pci != 0 ) {
		sh.base = -1;
		for ( uint32_t i = 0; i < n && sh.base < 0; ++i ) {
			VkPhysicalDeviceProperties p;
			vkGetPhysicalDeviceProperties( sh.phys[i], &p );
			if ( p.deviceID == sh.select_pci ) sh.base = static_cast<int>( i );
		}
		if ( sh.base < 0 ) {
			std::fprintf( stderr, "[-] Vulkan backend: no physical device with PCI device ID 0x%04x\n", sh.select_pci );
			return false;
		}
	}
	return n > 0;
}

// What an adopted device was created with: the features of the VkPhysicalDeviceFeatures2 chain (the Vulkan 1.1 / 1.2 / 1.3 structs or the
// single-feature ones) the backend looks at, and its extensions
struct Enabled {
	bool int64 = false, int16 = false, float64 = false, sb16 = false, usb16 = false, bda = false, fp16 = false, timeline = false, int8 = false, sb8 = false;
	bool dot = false, sg_ctl = false, cuda = false;
	std::vector<std::string> exts;
	bool has( const char* e ) const {
		for ( const auto& x : exts ) {
			if ( x == e ) return true;
		}
		return false;
	}
};
Enabled enabled_of( const vk::HostDevice& h ) {
	Enabled e;
	e.exts = h.enabled_extensions;
	for ( auto* b = static_cast<const VkBaseInStructure*>( h.enabled_features ); b != nullptr; b = b->pNext ) {
		switch ( b->sType ) {
			case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2: {
				const auto* f = reinterpret_cast<const VkPhysicalDeviceFeatures2*>( b );
				e.int64 = f->features.shaderInt64;
				e.int16 = f->features.shaderInt16;
				e.float64 = f->features.shaderFloat64;
			} break;
			case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES: {
				const auto* f = reinterpret_cast<const VkPhysicalDeviceVulkan11Features*>( b );
				e.sb16 = e.sb16 || f->storageBuffer16BitAccess;
				e.usb16 = e.usb16 || f->uniformAndStorageBuffer16BitAccess;
			} break;
			case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES: {
				const auto* f = reinterpret_cast<const VkPhysicalDevice16BitStorageFeatures*>( b );
				e.sb16 = e.sb16 || f->storageBuffer16BitAccess;
				e.usb16 = e.usb16 || f->uniformAndStorageBuffer16BitAccess;
			} break;
			case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES: {
				const auto* f = reinterpret_cast<const VkPhysicalDeviceVulkan12Features*>( b );
				e.bda = e.bda || f->bufferDeviceAddress;
				e.fp16 = e.fp16 || f->shaderFloat16;
				e.timeline = e.timeline || f->timelineSemaphore;
				e.int8 = e.int8 || f->shaderInt8;
				e.sb8 = e.sb8 || f->storageBuffer8BitAccess;
			} break;
			case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES: {
				e.bda = e.bda || reinterpret_cast<const VkPhysicalDeviceBufferDeviceAddressFeatures*>( b )->bufferDeviceAddress;
			} break;
			case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES: {
				const auto* f = reinterpret_cast<const VkPhysicalDeviceShaderFloat16Int8Features*>( b );
				e.fp16 = e.fp16 || f->shaderFloat16;
				e.int8 = e.int8 || f->shaderInt8;
			} break;
			case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES: {
				e.timeline = e.timeline || reinterpret_cast<const VkPhysicalDeviceTimelineSemaphoreFeatures*>( b )->timelineSemaphore;
			} break;
			case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES: {
				e.sb8 = e.sb8 || reinterpret_cast<const VkPhysicalDevice8BitStorageFeatures*>( b )->storageBuffer8BitAccess;
			} break;
			case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES: {
				const auto* f = reinterpret_cast<const VkPhysicalDeviceVulkan13Features*>( b );
				e.dot = e.dot || f->shaderIntegerDotProduct;
				e.sg_ctl = e.sg_ctl || f->subgroupSizeControl;
			} break;
			case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES: {
				e.dot = e.dot || reinterpret_cast<const VkPhysicalDeviceShaderIntegerDotProductFeatures*>( b )->shaderIntegerDotProduct;
			} break;
			case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES: {
				e.sg_ctl = e.sg_ctl || reinterpret_cast<const VkPhysicalDeviceSubgroupSizeControlFeatures*>( b )->subgroupSizeControl;
			} break;
			case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUDA_KERNEL_LAUNCH_FEATURES_NV: {
				e.cuda = e.cuda || reinterpret_cast<const VkPhysicalDeviceCudaKernelLaunchFeaturesNV*>( b )->cudaKernelLaunchFeatures;
			} break;
			default: break;
		}
	}
	return e;
}

vk_stream* make_stream( Ctx& g );

// Device `dev` (logical): physical device base + dev
bool init_context( int dev ) {
	std::lock_guard<std::mutex> lock( sh.mu );
	Ctx& g = gs[dev];
	if ( g.tried ) return g.ok;
	g.tried = true;
	if ( !init_instance() ) return false;
	if ( sh.adopted && dev != 0 ) return false;   // the host's device is the only one
	const Enabled en = sh.adopted ? enabled_of( sh.host ) : Enabled();
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
	bool has_dot_ext = false, has_push_desc = false, has_host_import = false, has_sem_fd = false, has_mem_fd = false, has_budget = false, has_sg_ctl = false;
	bool has_cuda = false, has_sm_builtins = false;
	{
		uint32_t en = 0;
		vkEnumerateDeviceExtensionProperties( g.phys, nullptr, &en, nullptr );
		std::vector<VkExtensionProperties> ext( en );
		vkEnumerateDeviceExtensionProperties( g.phys, nullptr, &en, ext.data() );
		for ( const auto& e : ext ) {
			has_dot_ext = has_dot_ext || std::strcmp( e.extensionName, VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME ) == 0;
			has_push_desc = has_push_desc || std::strcmp( e.extensionName, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME ) == 0;
			has_host_import = has_host_import || std::strcmp( e.extensionName, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME ) == 0;
			has_sem_fd = has_sem_fd || std::strcmp( e.extensionName, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME ) == 0;
			has_mem_fd = has_mem_fd || std::strcmp( e.extensionName, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME ) == 0;
			has_budget = has_budget || std::strcmp( e.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME ) == 0;
			has_sg_ctl = has_sg_ctl || std::strcmp( e.extensionName, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME ) == 0;
			has_cuda = has_cuda || std::strcmp( e.extensionName, VK_NV_CUDA_KERNEL_LAUNCH_EXTENSION_NAME ) == 0;
			has_sm_builtins = has_sm_builtins || std::strcmp( e.extensionName, VK_NV_SHADER_SM_BUILTINS_EXTENSION_NAME ) == 0;
		}
	}
	// the CUDA launches unless vk_cuda_launch says no (the shader kernels then run everything)
	if ( engine::runtime::options != nullptr && !engine::runtime::options().vk_cuda_launch ) has_cuda = false;
	if ( sh.adopted ) {
		// the extensions the host enabled (subgroup size control and integer dot products are core in Vulkan 1.3)
		const bool v13 = g.props.apiVersion >= VK_API_VERSION_1_3;
		has_dot_ext = en.has( VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME ) || ( v13 && en.dot );
		has_push_desc = en.has( VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME );
		has_host_import = en.has( VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME );
		has_sem_fd = en.has( VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME );
		has_mem_fd = en.has( VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME );
		has_budget = en.has( VK_EXT_MEMORY_BUDGET_EXTENSION_NAME );
		has_sg_ctl = en.has( VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME ) || ( v13 && en.sg_ctl );
		has_cuda = has_cuda && en.has( VK_NV_CUDA_KERNEL_LAUNCH_EXTENSION_NAME ) && en.cuda;   // optional: used where the host enabled it
	}
	if ( has_sm_builtins ) {
		VkPhysicalDeviceShaderSMBuiltinsPropertiesNV smp{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_PROPERTIES_NV };
		VkPhysicalDeviceProperties2 p2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &smp };
		vkGetPhysicalDeviceProperties2( g.phys, &p2 );
		g.sm_count = smp.shaderSMCount;
	}
	// The kernels that reduce over a warp (subgroup operations, lane = thread % 32) take subgroups of 32: a device whose compute subgroups are
	// another size by default (AMD's wave64, Intel's SIMD8 / 16) gets every pipeline created with a required subgroup size of 32 where it allows it
	VkPhysicalDeviceSubgroupSizeControlPropertiesEXT sgp{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT };
	VkPhysicalDeviceSubgroupProperties sg{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES, has_sg_ctl ? &sgp : nullptr };
	{
		VkPhysicalDeviceProperties2 p2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &sg };
		vkGetPhysicalDeviceProperties2( g.phys, &p2 );
	}
	VkPhysicalDeviceSubgroupSizeControlFeaturesEXT fsg{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT };
	VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR fdot{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR };
	VkPhysicalDeviceCudaKernelLaunchFeaturesNV fcuda{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUDA_KERNEL_LAUNCH_FEATURES_NV };
	void* query_chain = nullptr;
	if ( has_sg_ctl ) { fsg.pNext = query_chain; query_chain = &fsg; }
	if ( has_dot_ext ) { fdot.pNext = query_chain; query_chain = &fdot; }
	if ( has_cuda ) { fcuda.pNext = query_chain; query_chain = &fcuda; }
	VkPhysicalDeviceVulkan11Features f11{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, query_chain };
	VkPhysicalDeviceVulkan12Features f12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f11 };
	VkPhysicalDeviceFeatures2 f2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f12 };
	vkGetPhysicalDeviceFeatures2( g.phys, &f2 );
	if ( !f12.bufferDeviceAddress || !f12.shaderFloat16 || !f12.timelineSemaphore || !f2.features.shaderInt64 ) {
		sh.adopt_error = "the device lacks bufferDeviceAddress, shaderFloat16, timelineSemaphore or shaderInt64";
		return false;
	}
	if ( sh.adopted ) {
		// every feature the backend enables on a device of its own (those this one supports) must be enabled on the host's
		std::string missing;
		auto need = [&]( bool supported, bool on, const char* name ) {
			if ( supported && !on ) missing += std::string( missing.empty() ? "" : ", " ) + name;
		};
		need( true, en.bda, "bufferDeviceAddress" );
		need( true, en.fp16, "shaderFloat16" );
		need( true, en.timeline, "timelineSemaphore" );
		need( true, en.int64, "shaderInt64" );
		need( f11.storageBuffer16BitAccess, en.sb16, "storageBuffer16BitAccess" );
		need( f11.uniformAndStorageBuffer16BitAccess, en.usb16, "uniformAndStorageBuffer16BitAccess" );
		need( f12.shaderInt8, en.int8, "shaderInt8" );
		need( f12.storageBuffer8BitAccess, en.sb8, "storageBuffer8BitAccess" );
		need( f2.features.shaderInt16, en.int16, "shaderInt16" );
		need( f2.features.shaderFloat64, en.float64, "shaderFloat64" );
		if ( !missing.empty() ) {
			sh.adopt_error = "the host's device was created without " + missing;
			return false;
		}
		fdot.shaderIntegerDotProduct = en.dot ? fdot.shaderIntegerDotProduct : VK_FALSE;
		fsg.subgroupSizeControl = en.sg_ctl ? fsg.subgroupSizeControl : VK_FALSE;
	}

	g.int_dot = has_dot_ext && fdot.shaderIntegerDotProduct == VK_TRUE;
	g.cuda = has_cuda && fcuda.cudaKernelLaunchFeatures == VK_TRUE;
	if ( sg.subgroupSize != 32 ) {
		g.sg32 = has_sg_ctl && fsg.subgroupSizeControl == VK_TRUE && sgp.minSubgroupSize <= 32 && sgp.maxSubgroupSize >= 32 &&
			( sgp.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT ) != 0;
		if ( !g.sg32 ) std::fprintf( stderr, "[-] Vulkan backend: %s runs subgroups of %u and can't be made to run 32 (the warp-wide kernels assume 32)\n", g.props.deviceName, sg.subgroupSize );
		if ( !g.sg32 && sh.adopted ) {
			sh.adopt_error = "the device runs subgroups of " + std::to_string( sg.subgroupSize ) + ": the host's device needs subgroupSizeControl (VK_EXT_subgroup_size_control or Vulkan 1.3)";
			return false;
		}
	}
	VkPhysicalDeviceSubgroupSizeControlFeaturesEXT want_sg{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT };
	want_sg.subgroupSizeControl = VK_TRUE;
	VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR want_dot{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR };
	want_dot.shaderIntegerDotProduct = VK_TRUE;
	VkPhysicalDeviceCudaKernelLaunchFeaturesNV want_cuda{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUDA_KERNEL_LAUNCH_FEATURES_NV };
	want_cuda.cudaKernelLaunchFeatures = VK_TRUE;
	void* want_chain = nullptr;
	if ( g.sg32 ) { want_sg.pNext = want_chain; want_chain = &want_sg; }
	if ( g.int_dot ) { want_dot.pNext = want_chain; want_chain = &want_dot; }
	if ( g.cuda ) { want_cuda.pNext = want_chain; want_chain = &want_cuda; }
	VkPhysicalDeviceVulkan11Features want11{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, want_chain };
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
	if ( sh.adopted ) {
		found = sh.host.queue_family < qn && ( qf[sh.host.queue_family].queueFlags & VK_QUEUE_COMPUTE_BIT ) != 0;
		if ( !found ) sh.adopt_error = "the host's queue family has no compute";
		g.family = sh.host.queue_family;
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
	std::vector<const char*> exts;
	if ( g.int_dot ) exts.push_back( VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME );
	if ( has_push_desc ) exts.push_back( VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME );
	if ( has_host_import ) exts.push_back( VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME );
	if ( has_sem_fd ) exts.push_back( VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME );
	if ( has_budget ) exts.push_back( VK_EXT_MEMORY_BUDGET_EXTENSION_NAME );
	if ( g.sg32 ) exts.push_back( VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME );
	if ( g.cuda ) exts.push_back( VK_NV_CUDA_KERNEL_LAUNCH_EXTENSION_NAME );
	dci.enabledExtensionCount = static_cast<uint32_t>( exts.size() );
	dci.ppEnabledExtensionNames = exts.data();
	if ( sh.adopted ) {
		g.device = static_cast<VkDevice>( sh.host.device );
		g.host_queue_mu = sh.host.queue_mutex;
	} else if ( vkCreateDevice( g.phys, &dci, nullptr, &g.device ) != VK_SUCCESS ) {
		return false;
	}
	g.budget = has_budget;
	if ( has_push_desc ) g.push_descriptor = reinterpret_cast<PFN_vkCmdPushDescriptorSetKHR>( vkGetDeviceProcAddr( g.device, "vkCmdPushDescriptorSetKHR" ) );
	g.host_import = has_host_import;
	g.sem_fd = has_sem_fd;
	if ( has_host_import ) {
		VkPhysicalDeviceExternalMemoryHostPropertiesEXT hp{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT };
		VkPhysicalDeviceProperties2 pp{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &hp };
		vkGetPhysicalDeviceProperties2( g.phys, &pp );
		g.host_import_align = hp.minImportedHostPointerAlignment;
	}
	if ( sh.adopted ) g.queue = static_cast<VkQueue>( sh.host.queue );
	else vkGetDeviceQueue( g.device, g.family, 0, &g.queue );

	g.get_address = reinterpret_cast<PFN_vkGetBufferDeviceAddress>( vkGetDeviceProcAddr( g.device, "vkGetBufferDeviceAddress" ) );
	if ( g.get_address == nullptr ) return false;
	if ( g.cuda ) {
		g.cuda_create_module = reinterpret_cast<PFN_vkCreateCudaModuleNV>( vkGetDeviceProcAddr( g.device, "vkCreateCudaModuleNV" ) );
		g.cuda_create_function = reinterpret_cast<PFN_vkCreateCudaFunctionNV>( vkGetDeviceProcAddr( g.device, "vkCreateCudaFunctionNV" ) );
		g.cuda_cmd_launch = reinterpret_cast<PFN_vkCmdCudaLaunchKernelNV>( vkGetDeviceProcAddr( g.device, "vkCmdCudaLaunchKernelNV" ) );
		g.cuda = g.cuda_create_module && g.cuda_create_function && g.cuda_cmd_launch;
	}
	g.default_stream = make_stream( g );   // here, once: the threads sharing the device all see it
	g.ok = true;
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
	// device buffers as dedicated allocations: NVIDIA's driver maps those with large pages, the others take about twice as long in kernels
	// whose reads stride far apart (a GEMM over long rows), like an address-translation miss per access
	VkMemoryDedicatedAllocateInfo dedicated{ VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO };
	dedicated.buffer = buf;
	VkMemoryAllocateFlagsInfo flags{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, &dedicated };
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
	std::lock_guard<std::mutex> lk( cur().mu );
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

VkCommandBuffer begin_cb( VkCommandPool pool, bool reusable ) {
	VkCommandBufferAllocateInfo ai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	ai.commandPool = pool;
	ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	ai.commandBufferCount = 1;
	VkCommandBuffer cb;
	ENGINE_CHECK( vkAllocateCommandBuffers( cur().device, &ai, &cb ) );
	VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	bi.flags = reusable ? VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT : VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	ENGINE_CHECK( vkBeginCommandBuffer( cb, &bi ) );
	return cb;
}

// vulkan_profile (eager forwards; captured graphs are not timed): every dispatch's GPU time between a timestamp before it and one after its
// barrier, summed per pipeline at each sync and printed at exit
constexpr uint32_t kProfQueries = 65536;
std::mutex g_prof_mu;
std::map<std::string, std::pair<double, uint64_t>> g_prof;   // pipeline name -> (ms, dispatches)
double g_prof_span = 0.0;                                     // ms from the first to the last timestamp of each sync's batch, summed

void prof_report() {
	std::lock_guard<std::mutex> lk( g_prof_mu );
	if ( g_prof.empty() ) return;
	std::vector<std::pair<std::string, std::pair<double, uint64_t>>> rows( g_prof.begin(), g_prof.end() );
	std::sort( rows.begin(), rows.end(), []( const auto& a, const auto& b ) { return a.second.first > b.second.first; } );
	double total = 0.0;
	for ( const auto& r : rows ) total += r.second.first;
	std::fprintf( stderr, "[vulkan_profile] %.2f ms in %zu kernels' dispatches (each with its barrier), %.2f ms from first to last timestamp\n", total, rows.size(),
		g_prof_span );
	for ( const auto& r : rows ) {
		std::fprintf( stderr, "  %-34s %9llu calls %10.3f ms %9.2f us/call %5.1f%%\n", r.first.c_str(), static_cast<unsigned long long>( r.second.second ),
			r.second.first, 1000.0 * r.second.first / static_cast<double>( r.second.second ), 100.0 * r.second.first / total );
	}
}

bool profiling() {
	static const bool on = [] {
		const bool o = engine::runtime::options != nullptr && engine::runtime::options().vulkan_profile;
		if ( o ) std::atexit( prof_report );
		return o;
	}();
	return on;
}

// The pairs written since the last harvest, once the work that wrote them is done (sync)
void prof_harvest() {
	Ctx& c = cur();
	if ( c.qused == 0 ) return;
	std::vector<uint64_t> ts( c.qused );
	ENGINE_CHECK( vkGetQueryPoolResults( c.device, c.qpool, 0, c.qused, ts.size() * sizeof( uint64_t ), ts.data(), sizeof( uint64_t ),
		VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT ) );
	const double ms_per_tick = c.props.limits.timestampPeriod * 1e-6;
	{
		std::lock_guard<std::mutex> lk( g_prof_mu );
		for ( size_t i = 0; i < c.qwho.size(); ++i ) {
			auto& e = g_prof[c.qwho[i]->name];
			e.first += static_cast<double>( ts[2 * i + 1] - ts[2 * i] ) * ms_per_tick;
			++e.second;
		}
		g_prof_span += static_cast<double>( ts[c.qused - 1] - ts[0] ) * ms_per_tick;
	}
	c.qused = 0;
	c.qwho.clear();
	c.qreset = true;
}

VkCommandBuffer recording( vk_stream* s ) {
	if ( !s->cb ) {
		s->cb = begin_cb( s->pool->pool, s->capture );
		if ( profiling() && !s->capture ) {
			Ctx& c = cur();
			if ( c.qpool == VK_NULL_HANDLE ) {
				VkQueryPoolCreateInfo qi{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
				qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
				qi.queryCount = kProfQueries;
				ENGINE_CHECK( vkCreateQueryPool( c.device, &qi, nullptr, &c.qpool ) );
			}
			if ( c.qreset ) {
				vkCmdResetQueryPool( s->cb, c.qpool, 0, kProfQueries );
				c.qreset = false;
			}
		}
	}
	return s->cb;
}

// A timed dispatch's opening timestamp (true: close it with prof_end after the dispatch and its barrier)
bool prof_begin( vk_stream* s, VkCommandBuffer cb ) {
	Ctx& c = cur();
	if ( !profiling() || s->capture || c.qpool == VK_NULL_HANDLE || c.qreset || c.qused + 2 > kProfQueries ) return false;
	vkCmdWriteTimestamp( cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, c.qpool, c.qused );
	return true;
}

void prof_end( VkCommandBuffer cb, const vk::Pipeline* p ) {
	Ctx& c = cur();
	vkCmdWriteTimestamp( cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, c.qpool, c.qused + 1 );
	c.qwho.push_back( p );
	c.qused += 2;
}

// Everything recorded so far is visible to everything recorded next (and to the host at the end of the buffer)
void barrier( VkCommandBuffer cb, VkPipelineStageFlags dst = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VkAccessFlags dst_access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT ) {
	VkMemoryBarrier mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
	mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
	mb.dstAccessMask = dst_access;
	vkCmdPipelineBarrier( cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, dst, 0, 1, &mb, 0, nullptr, 0, nullptr );
}
// Before a command: the barrier its predecessor owes (in a run of independent ops, unless concurrent_next waived it)
void pay_held( vk_stream* s, VkCommandBuffer cb ) {
	if ( !s->held ) return;
	barrier( cb );
	s->held = false;
}
// After a command: its barrier, now or (in a run) before the next command
void after_command( vk_stream* s, VkCommandBuffer cb ) {
	if ( s->concurrent ) s->held = true;
	else barrier( cb );
}
// The command buffer a command is recorded into, its predecessor's barrier paid
VkCommandBuffer command_cb( vk_stream* s ) {
	VkCommandBuffer cb = recording( s );
	pay_held( s, cb );
	return cb;
}

// The stream's own timeline orders its submissions; the shared semaphores of stream_wait / stream_signal ride along on the next one
void submit( vk_stream* s, VkCommandBuffer cb ) {
	std::vector<VkSemaphore> waits{ s->timeline }, signals{ s->timeline };
	std::vector<uint64_t> wait_values{ s->value }, signal_values{ ++s->value };
	waits.insert( waits.end(), s->extra_wait.begin(), s->extra_wait.end() );
	wait_values.insert( wait_values.end(), s->extra_wait_values.begin(), s->extra_wait_values.end() );
	signals.insert( signals.end(), s->extra_signal.begin(), s->extra_signal.end() );
	signal_values.insert( signal_values.end(), s->extra_signal_values.begin(), s->extra_signal_values.end() );
	s->extra_wait.clear();
	s->extra_wait_values.clear();
	s->extra_signal.clear();
	s->extra_signal_values.clear();
	const std::vector<VkPipelineStageFlags> stages( waits.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT );
	VkTimelineSemaphoreSubmitInfo tsi{ VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO };
	tsi.waitSemaphoreValueCount = static_cast<uint32_t>( wait_values.size() );
	tsi.pWaitSemaphoreValues = wait_values.data();
	tsi.signalSemaphoreValueCount = static_cast<uint32_t>( signal_values.size() );
	tsi.pSignalSemaphoreValues = signal_values.data();
	VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO, &tsi };
	si.waitSemaphoreCount = static_cast<uint32_t>( waits.size() );
	si.pWaitSemaphores = waits.data();
	si.pWaitDstStageMask = stages.data();
	si.commandBufferCount = cb ? 1 : 0;
	si.pCommandBuffers = cb ? &cb : nullptr;
	si.signalSemaphoreCount = static_cast<uint32_t>( signals.size() );
	si.pSignalSemaphores = signals.data();
	std::lock_guard<std::mutex> lk( queue_mutex( cur() ) );
	ENGINE_CHECK( vkQueueSubmit( cur().queue, 1, &si, VK_NULL_HANDLE ) );
}

// The next submission's signals of sync g (values ++*ctr at a launch, the recorded ones when a capture falls back) and waits of sync g (values
// *ctr at a launch, the recorded ones otherwise)
void add_signals( vk_stream* s, const GraphSync& g, bool launch ) {
	for ( size_t i = 0; i < g.sig.size(); ++i ) {
		s->extra_signal.push_back( g.sig[i] );
		s->extra_signal_values.push_back( launch ? ++*g.sig_ctr[i] : g.sig_val[i] );
	}
}
void add_waits( vk_stream* s, const GraphSync& g, bool launch ) {
	for ( size_t i = 0; i < g.wait.size(); ++i ) {
		s->extra_wait.push_back( g.wait[i] );
		s->extra_wait_values.push_back( launch ? *g.wait_ctr[i] : g.wait_val[i] );
	}
}

// Capture: the commands recorded so far as a closed segment (an empty one when there are none), followed by a new sync point
void close_segment( vk_stream* s ) {
	VkCommandBuffer cb = recording( s );
	pay_held( s, cb );
	ENGINE_CHECK( vkEndCommandBuffer( cb ) );
	s->cap_segs.push_back( cb );
	s->cb = VK_NULL_HANDLE;
	s->cap_syncs.emplace_back();
}

// A command the capture cannot hold (a flush, a host transfer): the segments recorded so far run now with their recorded sync values, the open
// one becomes the stream's eager buffer (the last sync point's waits ahead of it), and the rest of the body records eagerly; graph_capture_end
// then returns no graph
void capture_abort( vk_stream* s ) {
	s->capture = false;
	s->capture_failed = true;
	for ( size_t i = 0; i < s->cap_segs.size(); ++i ) {
		if ( i > 0 ) add_waits( s, s->cap_syncs[i - 1], false );
		add_signals( s, s->cap_syncs[i], false );
		submit( s, s->cap_segs[i] );
		s->inflight.push_back( s->cap_segs[i] );
	}
	if ( !s->cap_syncs.empty() ) add_waits( s, s->cap_syncs.back(), false );
	s->cap_segs.clear();
	s->cap_syncs.clear();
}

void flush( vk_stream* s ) {
	if ( s->capture ) capture_abort( s );
	if ( !s->cb ) return;
	pay_held( s, s->cb );
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
	if ( profiling() ) prof_harvest();
	for ( auto& st : s->staging ) {
		if ( st.host_dst ) std::memcpy( st.host_dst, st.map, st.bytes );
		{
			std::lock_guard<std::mutex> lk( cur().mu );
			if ( st.capacity <= kPoolMax && cur().staging_pool_bytes + st.capacity <= kPoolLimit ) {
				st.host_dst = nullptr;
				cur().staging_pool_bytes += st.capacity;
				cur().staging_pool.emplace( st.capacity, st );
				continue;
			}
		}
		vkUnmapMemory( cur().device, st.memory );
		vkDestroyBuffer( cur().device, st.buffer, nullptr );
		vkFreeMemory( cur().device, st.memory, nullptr );
	}
	s->staging.clear();
	s->staging_bytes = 0;
	if ( !s->inflight.empty() ) vkFreeCommandBuffers( cur().device, s->pool->pool, uint32_t( s->inflight.size() ), s->inflight.data() );
	s->inflight.clear();
}

// A stream of device g: its timeline and its own command pool
vk_stream* make_stream( Ctx& g ) {
	auto* s = new vk_stream();
	VkSemaphoreTypeCreateInfo ti{ VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
	ti.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
	VkSemaphoreCreateInfo ci{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &ti };
	ENGINE_CHECK( vkCreateSemaphore( g.device, &ci, nullptr, &s->timeline ) );
	VkCommandPoolCreateInfo pci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
	pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	pci.queueFamilyIndex = g.family;
	s->pool = std::make_shared<CmdPool>();
	s->pool->device = g.device;
	ENGINE_CHECK( vkCreateCommandPool( g.device, &pci, nullptr, &s->pool->pool ) );
	return s;
}
vk_stream* make_stream() { return make_stream( cur() ); }


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
	{ std::lock_guard<std::mutex> qlk( queue_mutex( cur() ) ); vkQueueWaitIdle( cur().queue ); }
}

// A run of independent ops: off under vulkan_profile, which times each dispatch with its barrier
void concurrent_begin( stream_t st ) {
	ctx();
	if ( !cur().default_stream ) cur().default_stream = make_stream();
	vk_stream* s = resolve_stream( st );
	ENGINE_ASSERT( !s->concurrent, "Vulkan backend: nested concurrent_begin" );
	s->concurrent = !profiling();
}
void concurrent_next( stream_t st ) {
	vk_stream* s = resolve_stream( st );
	s->held = false;
}
void concurrent_end( stream_t st ) {
	vk_stream* s = resolve_stream( st );
	if ( s->held && s->cb ) barrier( s->cb );
	s->held = false;
	s->concurrent = false;
	s->run_scratch = false;
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
		vkCmdCopyBuffer( command_cb( s ), sb, db, 1, &r );
	} else {
		if ( s->capture ) capture_abort( s );   // a host transfer cannot be replayed: the capture falls back to running
		// Bound the host memory held by staging buffers of in-flight transfers (a weight load copies tens of GiB)
		if ( s->staging_bytes + bytes > ( size_t( 1 ) << 29 ) && s->staging_bytes > 0 ) sync( s );
		Staging st_buf;
		size_t cap = bytes;
		if ( bytes <= kPoolMax ) {
			cap = kPoolMin;
			while ( cap < bytes ) cap <<= 1;
		}
		bool pooled = false;
		{
			std::lock_guard<std::mutex> lk( cur().mu );
			if ( auto it = cur().staging_pool.find( cap ); it != cur().staging_pool.end() ) {
				st_buf = it->second;
				cur().staging_pool.erase( it );
				cur().staging_pool_bytes -= cap;
				pooled = true;
			}
		}
		if ( !pooled && !create_buffer( cap, true, st_buf.buffer, st_buf.memory, &st_buf.map ) ) return false;
		st_buf.bytes = bytes;
		st_buf.capacity = cap;
		if ( dir == Copy::H2D ) {
			if ( !resolve( dst, db, doff ) ) return false;
			std::memcpy( st_buf.map, src, bytes );
			VkBufferCopy r{ 0, doff, bytes };
			vkCmdCopyBuffer( command_cb( s ), st_buf.buffer, db, 1, &r );
		} else {
			if ( !resolve( src, sb, soff ) ) return false;
			st_buf.host_dst = dst;
			VkBufferCopy r{ soff, 0, bytes };
			vkCmdCopyBuffer( command_cb( s ), sb, st_buf.buffer, 1, &r );
		}
		s->staging.push_back( st_buf );
		s->staging_bytes += bytes;
	}
	after_command( s, s->cb );
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
	vkCmdFillBuffer( command_cb( s ), b, off, bytes, byte * 0x01010101u );
	after_command( s, s->cb );
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

// The largest device-local heap's budget less its usage (VK_EXT_memory_budget: what the driver, this process and the others hold), else its
// size less what this process allocated
size_t device_free_bytes( int device ) {
	if ( device < 0 || device >= MAX_DEV ) device = t_dev;
	if ( !gs[device].ok && !init_context( device ) ) return 0;
	const Ctx& c = gs[device];
	uint32_t hi = 0;
	uint64_t heap = 0;
	for ( uint32_t i = 0; i < c.mem.memoryHeapCount; ++i ) {
		if ( ( c.mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT ) && c.mem.memoryHeaps[i].size > heap ) {
			heap = c.mem.memoryHeaps[i].size;
			hi = i;
		}
	}
	if ( c.budget ) {
		VkPhysicalDeviceMemoryBudgetPropertiesEXT b{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT };
		VkPhysicalDeviceMemoryProperties2 p2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, &b };
		vkGetPhysicalDeviceMemoryProperties2( c.phys, &p2 );
		return b.heapBudget[hi] > b.heapUsage[hi] ? size_t( b.heapBudget[hi] - b.heapUsage[hi] ) : 0;
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
	std::lock_guard<std::mutex> lk( cur().mu );
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
	Alloc a{};
	{
		std::lock_guard<std::mutex> lk( cur().mu );
		auto it = cur().allocs.find( reinterpret_cast<uint64_t>( p ) );
		if ( it == cur().allocs.end() ) return;
		a = it->second;
		cur().allocated -= a.size;
		cur().allocs.erase( it );
	}
	{ std::lock_guard<std::mutex> qlk( queue_mutex( cur() ) ); vkQueueWaitIdle( cur().queue ); } // cudaFree synchronizes implicitly
	vkDestroyBuffer( cur().device, a.buffer, nullptr );
	vkFreeMemory( cur().device, a.memory, nullptr );
}

void graph_capture_begin( stream_t st ) {
	ctx();
	if ( !cur().default_stream ) cur().default_stream = make_stream();
	vk_stream* s = resolve_stream( st );
	ENGINE_ASSERT( !s->capture, "Vulkan backend: nested graph capture" );
	flush( s );
	s->capture = true;
	s->capture_failed = false;
	s->capture_nodes = 0;
}
// nullptr when the capture fell back to running (capture_abort)
graph_t graph_capture_end( stream_t st ) {
	vk_stream* s = resolve_stream( st );
	ENGINE_ASSERT( s->capture || s->capture_failed, "Vulkan backend: graph_capture_end without begin" );
	if ( s->capture_failed ) {
		s->capture_failed = false;
		return nullptr;
	}
	auto* gr = new vk_graph();
	gr->pool = s->pool;
	if ( !s->cb ) recording( s ); // the last segment (empty, or the whole capture)
	pay_held( s, s->cb );
	ENGINE_CHECK( vkEndCommandBuffer( s->cb ) );
	gr->segs = std::move( s->cap_segs );
	gr->segs.push_back( s->cb );
	gr->syncs = std::move( s->cap_syncs );
	for ( const GraphSync& g : gr->syncs ) {
		if ( g.sig.empty() ) continue;
		gr->parity_ctr = g.sig_ctr[0];
		gr->parity = g.sig_val[0] & 1;
		gr->parity_sem = g.sig[0];
		break;
	}
	gr->nodes = s->capture_nodes;
	s->cap_segs.clear();
	s->cap_syncs.clear();
	s->cb = VK_NULL_HANDLE;
	s->capture = false;
	return gr;
}
// The recorded command buffers are the executable: they move to the executable object, so destroying the graph afterwards (as GraphCache does
// right after instantiating) does not free what the executable still replays
graph_exec_t graph_instantiate( graph_t gr ) {
	auto* e = new vk_graph( std::move( *gr ) );
	gr->segs.clear();
	return e;
}
size_t graph_node_count( graph_t gr ) { return gr->nodes; }
// The segments in order, each sync point's signals with fresh counter values and its waits ahead of the next segment. A replay whose first
// counted signal would take the other parity than at capture first signals once more (every rank of the collective in step), so the slots the
// segments use alternate as they did
void graph_launch( graph_exec_t e, stream_t st ) {
	vk_stream* s = resolve_stream( st );
	flush( s );
	if ( e->parity_ctr != nullptr && ( ( *e->parity_ctr + 1 ) & 1 ) != e->parity ) {
		s->extra_signal.push_back( e->parity_sem );
		s->extra_signal_values.push_back( ++*e->parity_ctr );
		submit( s, VK_NULL_HANDLE );
	}
	for ( size_t i = 0; i < e->segs.size(); ++i ) {
		if ( i > 0 ) add_waits( s, e->syncs[i - 1], true );
		if ( i < e->syncs.size() ) add_signals( s, e->syncs[i], true );
		submit( s, e->segs[i] );
	}
}
void graph_destroy( graph_t gr ) {
	if ( !gr ) return;
	{ std::lock_guard<std::mutex> qlk( queue_mutex( cur() ) ); vkQueueWaitIdle( cur().queue ); }
	if ( !gr->segs.empty() ) vkFreeCommandBuffers( gr->pool->device, gr->pool->pool, static_cast<uint32_t>( gr->segs.size() ), gr->segs.data() );
	delete gr;
}
void graph_exec_destroy( graph_exec_t e ) { graph_destroy( e ); }

namespace vk {

void select_device( int index, uint32_t pci_device_id ) {
	std::lock_guard<std::mutex> lock( sh.mu );
	sh.select_index = index;
	sh.select_pci = pci_device_id;
}

bool adopt_device( const HostDevice& host, std::string* error ) {
	{
		std::lock_guard<std::mutex> lock( sh.mu );
		if ( sh.tried ) {
			*error = "the Vulkan backend is already initialized";
			return false;
		}
		if ( !host.instance || !host.physical_device || !host.device || !host.queue ) {
			*error = "an adopted device needs its instance, physical device, device and queue";
			return false;
		}
		sh.adopted = true;
		sh.host = host;
	}
	if ( !init_context( 0 ) ) {
		*error = sh.adopt_error.empty() ? "the host's Vulkan device can't be used" : sh.adopt_error;
		return false;
	}
	return true;
}

bool available() {
	if ( !cur().ok && !init_context( t_dev ) ) return false;
	if ( !cur().default_stream ) cur().default_stream = make_stream();
	return true;
}
const char* device_name() { return ctx().props.deviceName; }
uint32_t max_push_constants_bytes() { return ctx().props.limits.maxPushConstantsSize; }
bool int_dot() { return ctx().int_dot; }

// The library build links the kernels in (a generated table): the SPIR-V of a kernel by name, or null (weak: the other builds read the files)
__attribute__(( weak )) const uint8_t* embedded_spv( const char* name, size_t* bytes );

namespace {
Pipeline* create_pipeline( const char* spv_path, uint32_t push_bytes, uint32_t buffers, const uint32_t* spec, uint32_t spec_count ) {
	ctx();
	if ( buffers > 0 && !cur().push_descriptor ) return nullptr;
	if ( push_bytes > cur().props.limits.maxPushConstantsSize || push_bytes % 4 ) return nullptr;
	std::string name( spv_path );
	name = name.substr( name.find_last_of( '/' ) + 1 );
	if ( name.size() > 4 && name.compare( name.size() - 4, 4, ".spv" ) == 0 ) name.resize( name.size() - 4 );
	std::vector<uint32_t> code;
	size_t embedded_bytes = 0;
	const uint8_t* embedded = embedded_spv != nullptr && std::getenv( "ENGINE_SPV_DIR" ) == nullptr ? embedded_spv( name.c_str(), &embedded_bytes ) : nullptr;
	if ( embedded != nullptr ) {
		code.resize( ( embedded_bytes + 3 ) / 4 );
		std::memcpy( code.data(), embedded, embedded_bytes );
	} else {
		FILE* f = std::fopen( spv_path, "rb" );
		if ( !f ) return nullptr;
		std::fseek( f, 0, SEEK_END );
		const long n = std::ftell( f );
		std::fseek( f, 0, SEEK_SET );
		code.resize( ( size_t( n ) + 3 ) / 4 );
		const bool read_ok = n > 0 && std::fread( code.data(), 1, size_t( n ), f ) == size_t( n );
		std::fclose( f );
		if ( !read_ok ) return nullptr;
	}
	if ( cur().rte16 && code.size() > 5 && code[0] == 0x07230203u ) code = with_rte16( code );

	auto* p = new Pipeline();
	p->push_bytes = push_bytes;
	p->name = name;
	VkShaderModuleCreateInfo smi{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
	smi.codeSize = code.size() * 4;
	smi.pCode = code.data();
	VkPushConstantRange range{ VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes };
	VkPipelineLayoutCreateInfo pli{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
	pli.pushConstantRangeCount = push_bytes ? 1 : 0;
	pli.pPushConstantRanges = &range;
	p->buffers = buffers;
	if ( buffers > 0 ) {
		std::vector<VkDescriptorSetLayoutBinding> b( buffers );
		for ( uint32_t i = 0; i < buffers; ++i ) b[i] = { i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
		VkDescriptorSetLayoutCreateInfo dli{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
		dli.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
		dli.bindingCount = buffers;
		dli.pBindings = b.data();
		if ( vkCreateDescriptorSetLayout( cur().device, &dli, nullptr, &p->set_layout ) != VK_SUCCESS ) {
			pipeline_destroy( p );
			return nullptr;
		}
		pli.setLayoutCount = 1;
		pli.pSetLayouts = &p->set_layout;
	}
	std::vector<VkSpecializationMapEntry> entries( spec_count );
	for ( uint32_t i = 0; i < spec_count; ++i ) entries[i] = { i, i * 4, 4 };
	VkSpecializationInfo si{ spec_count, entries.data(), spec_count * 4u, spec };
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
	VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT rsg{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT };
	rsg.requiredSubgroupSize = 32;
	if ( cur().sg32 ) cpi.stage.pNext = &rsg;
	if ( spec_count > 0 ) cpi.stage.pSpecializationInfo = &si;
	cpi.layout = p->layout;
	if ( vkCreateComputePipelines( cur().device, VK_NULL_HANDLE, 1, &cpi, nullptr, &p->pipeline ) != VK_SUCCESS ) {
		pipeline_destroy( p );
		return nullptr;
	}
	return p;
}
} // namespace

Pipeline* pipeline_create( const char* spv_path, uint32_t push_bytes ) { return create_pipeline( spv_path, push_bytes, 0, nullptr, 0 ); }
Pipeline* pipeline_create_bound( const char* spv_path, uint32_t push_bytes, uint32_t buffers, const uint32_t* spec, uint32_t spec_count ) {
	return create_pipeline( spv_path, push_bytes, buffers, spec, spec_count );
}

void pipeline_destroy( Pipeline* p ) {
	if ( !p ) return;
	{ std::lock_guard<std::mutex> qlk( queue_mutex( cur() ) ); vkQueueWaitIdle( cur().queue ); }
	if ( p->pipeline ) vkDestroyPipeline( cur().device, p->pipeline, nullptr );
	if ( p->layout ) vkDestroyPipelineLayout( cur().device, p->layout, nullptr );
	if ( p->module ) vkDestroyShaderModule( cur().device, p->module, nullptr );
	if ( p->set_layout ) vkDestroyDescriptorSetLayout( cur().device, p->set_layout, nullptr );
	delete p;
}

void dispatch( stream_t st, Pipeline* p, const void* push, uint32_t gx, uint32_t gy, uint32_t gz ) {
	ctx();
	if ( !cur().default_stream ) cur().default_stream = make_stream();
	vk_stream* s = resolve_stream( st );
	VkCommandBuffer cb = command_cb( s );
	const bool timed = prof_begin( s, cb );
	vkCmdBindPipeline( cb, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipeline );
	if ( p->push_bytes ) vkCmdPushConstants( cb, p->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, p->push_bytes, push );
	vkCmdDispatch( cb, gx, gy, gz );
	after_command( s, cb );
	if ( timed ) prof_end( cb, p );
	if ( s->capture ) ++s->capture_nodes;
}

void dispatch_bound( stream_t st, Pipeline* p, const void* push, const void* const* buffers, uint32_t gx, uint32_t gy, uint32_t gz ) {
	ctx();
	if ( !cur().default_stream ) cur().default_stream = make_stream();
	vk_stream* s = resolve_stream( st );
	VkCommandBuffer cb = command_cb( s );
	const bool timed = prof_begin( s, cb );
	vkCmdBindPipeline( cb, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipeline );
	// each binding: from the pointer to the end of its allocation (at most maxStorageBufferRange)
	std::vector<VkDescriptorBufferInfo> info( p->buffers );
	std::vector<VkWriteDescriptorSet> writes( p->buffers );
	for ( uint32_t i = 0; i < p->buffers; ++i ) {
		VkBuffer buf;
		VkDeviceSize off;
		ENGINE_ASSERT( resolve( buffers[i], buf, off ), "Vulkan backend: a bound buffer is not a device allocation" );
		ENGINE_ASSERT( off % cur().props.limits.minStorageBufferOffsetAlignment == 0, "Vulkan backend: a bound buffer is misaligned" );
		const uint64_t a = reinterpret_cast<uint64_t>( buffers[i] );
		VkDeviceSize left = 0;
		{
			std::lock_guard<std::mutex> lk( cur().mu );
			auto it = --cur().allocs.upper_bound( a );
			left = it->second.size - off;
		}
		info[i] = { buf, off, std::min<VkDeviceSize>( left, cur().props.limits.maxStorageBufferRange ) };
		writes[i] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
		writes[i].dstBinding = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writes[i].pBufferInfo = &info[i];
	}
	cur().push_descriptor( cb, VK_PIPELINE_BIND_POINT_COMPUTE, p->layout, 0, p->buffers, writes.data() );
	if ( p->push_bytes ) vkCmdPushConstants( cb, p->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, p->push_bytes, push );
	vkCmdDispatch( cb, gx, gy, gz );
	after_command( s, cb );
	if ( timed ) prof_end( cb, p );
	if ( s->capture ) ++s->capture_nodes;
}

bool cuda_launch_available() { return ctx().cuda; }
uint32_t sm_count() { return ctx().sm_count; }

CudaFunction* cuda_function( const char* ptx_path, const char* name ) {
	Ctx& c = ctx();
	if ( !c.cuda ) return nullptr;
	std::string mod( ptx_path );
	mod = mod.substr( mod.find_last_of( '/' ) + 1 );
	if ( mod.size() > 4 && mod.compare( mod.size() - 4, 4, ".ptx" ) == 0 ) mod.resize( mod.size() - 4 );
	const std::string key = mod + ":" + name;
	std::lock_guard<std::mutex> lk( c.mu );
	if ( auto it = c.cuda_functions.find( key ); it != c.cuda_functions.end() ) return it->second;
	auto mit = c.cuda_modules.find( mod );
	if ( mit == c.cuda_modules.end() ) {
		// the PTX text, NUL-terminated: the library's embedded copy, else the file
		std::vector<char> text;
		size_t embedded_bytes = 0;
		const uint8_t* embedded = embedded_spv != nullptr && std::getenv( "ENGINE_SPV_DIR" ) == nullptr ? embedded_spv( mod.c_str(), &embedded_bytes ) : nullptr;
		if ( embedded != nullptr ) text.assign( embedded, embedded + embedded_bytes );
		else if ( FILE* f = std::fopen( ptx_path, "rb" ) ) {
			std::fseek( f, 0, SEEK_END );
			const long n = std::ftell( f );
			std::fseek( f, 0, SEEK_SET );
			text.resize( n > 0 ? size_t( n ) : 0 );
			if ( n <= 0 || std::fread( text.data(), 1, size_t( n ), f ) != size_t( n ) ) text.clear();
			std::fclose( f );
		}
		VkCudaModuleNV m = VK_NULL_HANDLE;
		const auto t0 = std::chrono::steady_clock::now();
		if ( !text.empty() ) {
			text.push_back( '\0' );
			VkCudaModuleCreateInfoNV ci{ VK_STRUCTURE_TYPE_CUDA_MODULE_CREATE_INFO_NV };
			ci.dataSize = text.size();
			ci.pData = text.data();
			if ( c.cuda_create_module( c.device, &ci, nullptr, &m ) != VK_SUCCESS ) m = VK_NULL_HANDLE;
		}
		if ( m == VK_NULL_HANDLE ) std::fprintf( stderr, "[!] Vulkan backend: CUDA module %s could not be loaded; its kernels run on the shader cores\n", mod.c_str() );
		else std::fprintf( stderr, "[*] Vulkan backend: CUDA module %s loaded (%.0f KB) in %.0f ms\n", mod.c_str(), text.size() / 1024.0,
			std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - t0 ).count() );
		mit = c.cuda_modules.emplace( mod, m ).first;
	}
	CudaFunction* cf = nullptr;
	if ( mit->second != VK_NULL_HANDLE ) {
		VkCudaFunctionCreateInfoNV fi{ VK_STRUCTURE_TYPE_CUDA_FUNCTION_CREATE_INFO_NV };
		fi.module = mit->second;
		fi.pName = name;
		VkCudaFunctionNV f = VK_NULL_HANDLE;
		if ( c.cuda_create_function( c.device, &fi, nullptr, &f ) == VK_SUCCESS ) {
			cf = new CudaFunction();
			cf->fn = f;
			cf->prof.name = name;
		}
	}
	c.cuda_functions[key] = cf;
	return cf;
}

void cuda_launch( stream_t st, CudaFunction* fn, uint32_t gx, uint32_t gy, uint32_t gz, uint32_t bx, uint32_t by, uint32_t bz, uint32_t smem,
	const void* const* params, uint32_t nparams ) {
	ctx();
	if ( !cur().default_stream ) cur().default_stream = make_stream();
	vk_stream* s = resolve_stream( st );
	VkCommandBuffer cb = command_cb( s );
	const bool timed = prof_begin( s, cb );
	VkCudaLaunchInfoNV li{ VK_STRUCTURE_TYPE_CUDA_LAUNCH_INFO_NV };
	li.function = fn->fn;
	li.gridDimX = gx;
	li.gridDimY = gy;
	li.gridDimZ = gz;
	li.blockDimX = bx;
	li.blockDimY = by;
	li.blockDimZ = bz;
	li.sharedMemBytes = smem;
	li.paramCount = nparams;
	li.pParams = params;
	cur().cuda_cmd_launch( cb, &li );
	after_command( s, cb );
	if ( timed ) prof_end( cb, &fn->prof );
	if ( s->capture ) ++s->capture_nodes;
}

void scratch_use( stream_t st ) {
	ctx();
	if ( !cur().default_stream ) cur().default_stream = make_stream();
	vk_stream* s = resolve_stream( st );
	if ( !s->concurrent ) return;
	if ( s->run_scratch && s->cb ) {
		barrier( s->cb );
		s->held = false;
	}
	s->run_scratch = true;
}

bool bound_pipelines() { return ctx().push_descriptor != nullptr; }

bool shared_sync() { return ctx().host_import && ctx().sem_fd; }
size_t host_import_alignment() { return ctx().host_import_align; }

void* import_host_memory( void* host, size_t bytes ) {
	ctx();
	if ( !cur().host_import || reinterpret_cast<uintptr_t>( host ) % cur().host_import_align != 0 || bytes % cur().host_import_align != 0 ) return nullptr;
	auto props = reinterpret_cast<PFN_vkGetMemoryHostPointerPropertiesEXT>( vkGetDeviceProcAddr( cur().device, "vkGetMemoryHostPointerPropertiesEXT" ) );
	VkMemoryHostPointerPropertiesEXT mp{ VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT };
	if ( props == nullptr || props( cur().device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, host, &mp ) != VK_SUCCESS ) return nullptr;
	VkExternalMemoryBufferCreateInfo eb{ VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO };
	eb.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
	VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &eb };
	bci.size = bytes;
	bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
	Alloc a{};
	a.size = bytes;
	if ( vkCreateBuffer( cur().device, &bci, nullptr, &a.buffer ) != VK_SUCCESS ) return nullptr;
	VkMemoryRequirements req;
	vkGetBufferMemoryRequirements( cur().device, a.buffer, &req );
	uint32_t type = 0;
	const uint32_t bits = mp.memoryTypeBits & req.memoryTypeBits;
	while ( type < 32 && !( bits & ( 1u << type ) ) ) ++type;
	if ( type == 32 ) { vkDestroyBuffer( cur().device, a.buffer, nullptr ); return nullptr; }
	VkImportMemoryHostPointerInfoEXT im{ VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT };
	im.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
	im.pHostPointer = host;
	VkMemoryAllocateFlagsInfo flags{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, &im };
	flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
	VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &flags };
	mai.allocationSize = bytes;
	mai.memoryTypeIndex = type;
	if ( vkAllocateMemory( cur().device, &mai, nullptr, &a.memory ) != VK_SUCCESS ) { vkDestroyBuffer( cur().device, a.buffer, nullptr ); return nullptr; }
	vkBindBufferMemory( cur().device, a.buffer, a.memory, 0 );
	VkBufferDeviceAddressInfo ai{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
	ai.buffer = a.buffer;
	const uint64_t addr = cur().get_address( cur().device, &ai );
	std::lock_guard<std::mutex> lk( cur().mu );
	cur().allocs[addr] = a;   // not counted in `allocated`: host memory
	return reinterpret_cast<void*>( addr );
}

struct SharedSemaphore { VkSemaphore sem = VK_NULL_HANDLE; };

void* shared_timeline_create() {
	ctx();
	if ( !cur().sem_fd ) return nullptr;
	VkExportSemaphoreCreateInfo es{ VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO };
	es.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
	VkSemaphoreTypeCreateInfo st{ VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, &es };
	st.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
	VkSemaphoreCreateInfo sci{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &st };
	auto* h = new SharedSemaphore();
	if ( vkCreateSemaphore( cur().device, &sci, nullptr, &h->sem ) != VK_SUCCESS ) { delete h; return nullptr; }
	return h;
}

int shared_timeline_fd( void* h ) {
	auto get = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>( vkGetDeviceProcAddr( ctx().device, "vkGetSemaphoreFdKHR" ) );
	VkSemaphoreGetFdInfoKHR gi{ VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR };
	gi.semaphore = static_cast<SharedSemaphore*>( h )->sem;
	gi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
	int fd = -1;
	return get != nullptr && get( cur().device, &gi, &fd ) == VK_SUCCESS ? fd : -1;
}

void* shared_timeline_import( int fd ) {
	ctx();
	VkSemaphoreTypeCreateInfo st{ VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
	st.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
	VkSemaphoreCreateInfo sci{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &st };
	auto* h = new SharedSemaphore();
	if ( vkCreateSemaphore( cur().device, &sci, nullptr, &h->sem ) != VK_SUCCESS ) { delete h; return nullptr; }
	auto imp = reinterpret_cast<PFN_vkImportSemaphoreFdKHR>( vkGetDeviceProcAddr( cur().device, "vkImportSemaphoreFdKHR" ) );
	VkImportSemaphoreFdInfoKHR is{ VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR };
	is.semaphore = h->sem;
	is.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
	is.fd = fd;
	if ( imp == nullptr || imp( cur().device, &is ) != VK_SUCCESS ) {
		vkDestroySemaphore( cur().device, h->sem, nullptr );
		delete h;
		return nullptr;
	}
	return h;
}

// Fixed values cannot be replayed: inside a capture the capture falls back to running (flush)
void stream_signal( stream_t st, void* sem, uint64_t value ) {
	ctx();
	if ( !cur().default_stream ) cur().default_stream = make_stream();
	vk_stream* s = resolve_stream( st );
	if ( s->capture ) flush( s );
	s->extra_signal.push_back( static_cast<SharedSemaphore*>( sem )->sem );
	s->extra_signal_values.push_back( value );
	if ( s->cb ) flush( s );
	else submit( s, VK_NULL_HANDLE );
}

void stream_wait( stream_t st, void* const* sems, const uint64_t* values, int n ) {
	ctx();
	if ( !cur().default_stream ) cur().default_stream = make_stream();
	vk_stream* s = resolve_stream( st );
	if ( s->cb || s->capture ) flush( s );   // only what comes after waits
	for ( int i = 0; i < n; ++i ) {
		s->extra_wait.push_back( static_cast<SharedSemaphore*>( sems[i] )->sem );
		s->extra_wait_values.push_back( values[i] );
	}
}

void stream_signal_next( stream_t st, void* sem, uint64_t* counter ) {
	ctx();
	if ( !cur().default_stream ) cur().default_stream = make_stream();
	vk_stream* s = resolve_stream( st );
	const uint64_t v = ++*counter;
	const VkSemaphore vs = static_cast<SharedSemaphore*>( sem )->sem;
	if ( !s->capture ) {
		s->extra_signal.push_back( vs );
		s->extra_signal_values.push_back( v );
		if ( s->cb ) flush( s );
		else submit( s, VK_NULL_HANDLE );
		return;
	}
	close_segment( s );
	GraphSync& g = s->cap_syncs.back();
	g.sig.push_back( vs );
	g.sig_ctr.push_back( counter );
	g.sig_val.push_back( v );
}

void stream_wait_counter( stream_t st, void* const* sems, int n, const uint64_t* counter ) {
	ctx();
	if ( !cur().default_stream ) cur().default_stream = make_stream();
	vk_stream* s = resolve_stream( st );
	if ( !s->capture ) {
		if ( s->cb ) flush( s );   // only what comes after waits
		for ( int i = 0; i < n; ++i ) {
			s->extra_wait.push_back( static_cast<SharedSemaphore*>( sems[i] )->sem );
			s->extra_wait_values.push_back( *counter );
		}
		return;
	}
	if ( s->cb != VK_NULL_HANDLE || s->cap_syncs.empty() ) close_segment( s );
	GraphSync& g = s->cap_syncs.back();
	for ( int i = 0; i < n; ++i ) {
		g.wait.push_back( static_cast<SharedSemaphore*>( sems[i] )->sem );
		g.wait_ctr.push_back( counter );
		g.wait_val.push_back( *counter );
	}
}

} // namespace vk
} // namespace engine::backend
