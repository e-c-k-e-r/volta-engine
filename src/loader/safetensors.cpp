#include "engine/loader/safetensors.h"
#include "engine/core/log.h"
#include "engine/backend/runtime.h"
#include "engine/backend/numeric.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <cstdio>
#include <algorithm>

namespace engine::loader {

static pod::DType parse_dtype_str( const char* str, size_t len ) {
	if ( len == 3 && std::strncmp( str, "F16", 3 ) == 0 ) {
		return pod::DType::FP16;
	}
	if ( len == 4 && std::strncmp( str, "BF16", 4 ) == 0 ) {
		return pod::DType::BF16;
	}
	if ( len == 3 && std::strncmp( str, "F32", 3 ) == 0 ) {
		return pod::DType::FP32;
	}
	if ( len == 3 && std::strncmp( str, "I32", 3 ) == 0 ) {
		return pod::DType::INT32;
	}
	if ( len == 2 && std::strncmp( str, "I8", 2 ) == 0 ) {
		return pod::DType::INT8;
	}
	if ( len == 3 && std::strncmp( str, "U16", 3 ) == 0 ) {
		return pod::DType::UINT16;
	}
	if ( len == 3 && std::strncmp( str, "I16", 3 ) == 0 ) {
		return pod::DType::UINT16;
	}
	if ( len == 3 && std::strncmp( str, "U32", 3 ) == 0 ) {
		return pod::DType::UINT32;
	}
	if ( len == 2 && std::strncmp( str, "U8", 2 ) == 0 ) {
		return pod::DType::UINT8;
	}
	if ( len == 7 && std::strncmp( str, "F8_E4M3", 7 ) == 0 ) {
		return pod::DType::UINT8;  // raw bytes; the NVFP4 loader decodes them
	}
	return pod::DType::FP16;
}

static bool parse_header( pod::SafeTensorsFile& st ) {
	const char* json = static_cast<const char*>( st.mmap_addr ) + 8;
	size_t len = st.header_len;
	size_t pos = 0;

	auto skip_whitespace = [&]() {
		while ( pos < len && ( json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n' || json[pos] == '\r' ) ) {
			++pos;
		}
	};

	skip_whitespace();
	if ( pos >= len || json[pos] != '{' ) {
		return false;
	}
	++pos;

	while ( pos < len ) {
		skip_whitespace();
		if ( pos < len && json[pos] == '}' ) {
			++pos;
			break;
		}

		if ( pos >= len || json[pos] != '"' ) {
			return false;
		}
		++pos;
		size_t key_start = pos;
		while ( pos < len && json[pos] != '"' ) {
			++pos;
		}
		if ( pos >= len ) {
			return false;
		}
		size_t key_len = pos - key_start;
		++pos;

		skip_whitespace();
		if ( pos >= len || json[pos] != ':' ) {
			return false;
		}
		++pos;
		skip_whitespace();

		if ( pos < len && json[pos] == '{' ) {
			++pos;
			bool is_meta = ( key_len == 12 && std::strncmp( json + key_start, "__metadata__", 12 ) == 0 );

			pod::SafeTensorMeta meta;
			std::memset( &meta, 0, sizeof( pod::SafeTensorMeta ) );
			size_t name_copy_len = key_len < sizeof( meta.name ) - 1 ? key_len : sizeof( meta.name ) - 1;
			std::memcpy( meta.name, json + key_start, name_copy_len );
			meta.name[name_copy_len] = '\0';
			meta.ndim = 0;
			size_t offset_begin = 0;
			size_t offset_end = 0;

			while ( pos < len ) {
				skip_whitespace();
				if ( pos < len && json[pos] == '}' ) {
					++pos;
					break;
				}

				if ( pos >= len || json[pos] != '"' ) {
					++pos;
					continue;
				}
				++pos;
				size_t subkey_start = pos;
				while ( pos < len && json[pos] != '"' ) {
					++pos;
				}
				if ( pos >= len ) {
					return false;
				}
				size_t subkey_len = pos - subkey_start;
				++pos;

				skip_whitespace();
				if ( pos < len && json[pos] == ':' ) {
					++pos;
				}
				skip_whitespace();

				if ( subkey_len == 5 && std::strncmp( json + subkey_start, "dtype", 5 ) == 0 ) {
					if ( pos < len && json[pos] == '"' ) {
						++pos;
						size_t dt_start = pos;
						while ( pos < len && json[pos] != '"' ) {
							++pos;
						}
						size_t dt_len = pos - dt_start;
						if ( pos < len ) {
							++pos;
						}
						meta.dtype = parse_dtype_str( json + dt_start, dt_len );
					}
				}
				else if ( subkey_len == 5 && std::strncmp( json + subkey_start, "shape", 5 ) == 0 ) {
					if ( pos < len && json[pos] == '[' ) {
						++pos;
						while ( pos < len && json[pos] != ']' ) {
							skip_whitespace();
							if ( json[pos] == ']' ) {
								break;
							}
							int dim = 0;
							while ( pos < len && std::isdigit( static_cast<unsigned char>( json[pos] ) ) ) {
								dim = dim * 10 + ( json[pos] - '0' );
								++pos;
							}
							if ( meta.ndim < 4 ) {
								meta.shape[meta.ndim++] = dim;
							}
							skip_whitespace();
							if ( pos < len && json[pos] == ',' ) {
								++pos;
							}
						}
						if ( pos < len && json[pos] == ']' ) {
							++pos;
						}
					}
				}
				else if ( subkey_len == 12 && std::strncmp( json + subkey_start, "data_offsets", 12 ) == 0 ) {
					if ( pos < len && json[pos] == '[' ) {
						++pos;
						skip_whitespace();
						while ( pos < len && std::isdigit( static_cast<unsigned char>( json[pos] ) ) ) {
							offset_begin = offset_begin * 10 + ( json[pos] - '0' );
							++pos;
						}
						skip_whitespace();
						if ( pos < len && json[pos] == ',' ) {
							++pos;
						}
						skip_whitespace();
						while ( pos < len && std::isdigit( static_cast<unsigned char>( json[pos] ) ) ) {
							offset_end = offset_end * 10 + ( json[pos] - '0' );
							++pos;
						}
						skip_whitespace();
						if ( pos < len && json[pos] == ']' ) {
							++pos;
						}
					}
				}
				else {
					if ( pos < len && json[pos] == '"' ) {
						++pos;
						while ( pos < len && json[pos] != '"' ) {
							++pos;
						}
						if ( pos < len ) {
							++pos;
						}
					}
					else if ( pos < len && json[pos] == '[' ) {
						++pos;
						while ( pos < len && json[pos] != ']' ) {
							++pos;
						}
						if ( pos < len ) {
							++pos;
						}
					}
					else if ( pos < len && json[pos] == '{' ) {
						++pos;
						while ( pos < len && json[pos] != '}' ) {
							++pos;
						}
						if ( pos < len ) {
							++pos;
						}
					}
					else {
						while ( pos < len && json[pos] != ',' && json[pos] != '}' ) {
							++pos;
						}
					}
				}

				skip_whitespace();
				if ( pos < len && json[pos] == ',' ) {
					++pos;
				}
			}

			if ( !is_meta ) {
				meta.data_offset_bytes = 8 + st.header_len + offset_begin;
				meta.size_bytes = offset_end - offset_begin;

				if ( st.tensor_count >= st.tensor_capacity ) {
					st.tensor_capacity *= 2;
					st.tensors = static_cast<pod::SafeTensorMeta*>( std::realloc( st.tensors, st.tensor_capacity * sizeof( pod::SafeTensorMeta ) ) );
				}
				st.tensors[st.tensor_count++] = meta;
			}
		}
		else {
			while ( pos < len && json[pos] != ',' && json[pos] != '}' ) {
				++pos;
			}
		}

		skip_whitespace();
		if ( pos < len && json[pos] == ',' ) {
			++pos;
		}
	}

	return true;
}

bool safetensors_open( pod::SafeTensorsFile& st, const char* filepath ) {
	safetensors_close( st );

	st.fd = ::open( filepath, O_RDONLY );
	if ( st.fd < 0 ) {
		return false;
	}

	struct stat sb;
	if ( fstat( st.fd, &sb ) < 0 ) {
		::close( st.fd );
		st.fd = -1;
		return false;
	}
	st.file_size = static_cast<size_t>( sb.st_size );
	if ( st.file_size < 8 ) {
		::close( st.fd );
		st.fd = -1;
		return false;
	}

	st.mmap_addr = mmap( nullptr, st.file_size, PROT_READ, MAP_SHARED, st.fd, 0 );
	if ( st.mmap_addr == MAP_FAILED ) {
		st.mmap_addr = nullptr;
		::close( st.fd );
		st.fd = -1;
		return false;
	}

	st.header_len = *reinterpret_cast<const uint64_t*>( st.mmap_addr );
	if ( 8 + st.header_len > st.file_size ) {
		safetensors_close( st );
		return false;
	}

	st.tensor_capacity = 64;
	st.tensors = static_cast<pod::SafeTensorMeta*>( std::malloc( st.tensor_capacity * sizeof( pod::SafeTensorMeta ) ) );
	st.tensor_count = 0;

	if ( !parse_header( st ) ) {
		safetensors_close( st );
		return false;
	}

	return true;
}

void safetensors_close( pod::SafeTensorsFile& st ) {
	if ( st.mmap_addr != nullptr && st.mmap_addr != MAP_FAILED ) {
		munmap( st.mmap_addr, st.file_size );
		st.mmap_addr = nullptr;
	}
	if ( st.fd >= 0 ) {
		::close( st.fd );
		st.fd = -1;
	}
	if ( st.tensors != nullptr ) {
		std::free( st.tensors );
		st.tensors = nullptr;
	}
	st.file_size = 0;
	st.header_len = 0;
	st.tensor_count = 0;
	st.tensor_capacity = 0;
}

const pod::SafeTensorMeta* safetensors_find( const pod::SafeTensorsFile& st, const char* name ) {
	for ( int i = 0; i < st.tensor_count; ++i ) {
		if ( std::strcmp( st.tensors[i].name, name ) == 0 ) {
			return &st.tensors[i];
		}
	}
	return nullptr;
}

const void* safetensors_get_data_cpu( const pod::SafeTensorsFile& st, const pod::SafeTensorMeta* meta ) {
	if ( meta == nullptr || st.mmap_addr == nullptr ) {
		return nullptr;
	}
	if ( meta->data_offset_bytes + meta->size_bytes > st.file_size ) {
		return nullptr;
	}
	return static_cast<const char*>( st.mmap_addr ) + meta->data_offset_bytes;
}

bool safetensors_load_to_device( const pod::SafeTensorsFile& st, const pod::SafeTensorMeta* meta, void* dst_device_ptr, backend::stream_t stream ) {
	if ( meta == nullptr || dst_device_ptr == nullptr ) {
		return false;
	}
	const void* src_cpu = safetensors_get_data_cpu( st, meta );
	if ( src_cpu == nullptr ) {
		return false;
	}

	if ( ENGINE_UNLIKELY( !backend::memcpy_async_try( dst_device_ptr, src_cpu, meta->size_bytes, backend::Copy::Auto, stream ) ) ) {
		return false;
	}
	if ( stream == nullptr ) {
		backend::device_sync();
	}
	return true;
}

pod::TensorView safetensors_alloc_tensor( const pod::SafeTensorsFile& st, const char* name, pod::MemoryArena& arena, backend::stream_t stream ) {
	const pod::SafeTensorMeta* meta = safetensors_find( st, name );
	if ( meta == nullptr ) {
		return pod::TensorView{};
	}

	pod::TensorView tv = engine::core::arena_alloc_tensor( arena, meta->dtype, meta->shape, meta->ndim );
	if ( tv.data != nullptr ) {
		safetensors_load_to_device( st, meta, tv.data, stream );
	}
	return tv;
}

static inline backend::half_t bf16_to_fp16( uint16_t bf ) {
	uint32_t u32 = static_cast<uint32_t>( bf ) << 16;
	float f = *reinterpret_cast<const float*>( &u32 );
	return backend::from_float_half( f );
}

static inline backend::half_t f32_to_fp16( float f ) {
	return backend::from_float_half( f );
}

static size_t get_dtype_size( pod::DType dt ) {
	switch ( dt ) {
		case pod::DType::FP16:
		case pod::DType::BF16:
		case pod::DType::UINT16:
			return 2;
		case pod::DType::FP32:
		case pod::DType::INT32:
		case pod::DType::UINT32:
			return 4;
		case pod::DType::INT8:
		case pod::DType::UINT8:
			return 1;
		default:
			return 2;
	}
}

SafeTensorsRepo::SafeTensorsRepo() {
	num_shards_ = 0;
	dir_path_[0] = '\0';
	for ( int i = 0; i < MAX_SHARDS; ++i ) {
		shard_names_[i][0] = '\0';
	}
}

SafeTensorsRepo::~SafeTensorsRepo() {
	close();
}

void SafeTensorsRepo::close() {
	for ( int i = 0; i < num_shards_; ++i ) {
		safetensors_close( shards_[i] );
		shard_names_[i][0] = '\0';
	}
	num_shards_ = 0;
	dir_path_[0] = '\0';
	tensor_map_.clear();
}

bool SafeTensorsRepo::open( const char* dir_path ) {
	close();
	std::strncpy( dir_path_, dir_path, sizeof( dir_path_ ) - 1 );
	dir_path_[sizeof( dir_path_ ) - 1] = '\0';

	std::vector<std::string> shard_files;

	// 1. Try to parse model.safetensors.index.json
	char index_path[1024];
	std::snprintf( index_path, sizeof( index_path ), "%s/model.safetensors.index.json", dir_path );

	FILE* f_idx = std::fopen( index_path, "rb" );
	if ( f_idx != nullptr ) {
		std::fseek( f_idx, 0, SEEK_END );
		long sz = std::ftell( f_idx );
		std::fseek( f_idx, 0, SEEK_SET );
		if ( sz > 0 && sz < 64 * 1024 * 1024 ) {
			std::string buf( sz, '\0' );
			size_t read_bytes = std::fread( buf.data(), 1, sz, f_idx );
			if ( read_bytes == static_cast<size_t>( sz ) ) {
				size_t pos = 0;
				while ( ( pos = buf.find( ".safetensors\"", pos ) ) != std::string::npos ) {
					size_t start = pos;
					while ( start > 0 && buf[start - 1] != '"' ) {
						--start;
					}
					std::string fname = buf.substr( start, pos + 12 - start );
					bool exists = false;
					for ( const auto& s : shard_files ) {
						if ( s == fname ) {
							exists = true;
							break;
						}
					}
					if ( !exists ) {
						shard_files.push_back( fname );
					}
					pos += 12;
				}
			}
		}
		std::fclose( f_idx );
	}

	// 2. If index.json was not found or empty, scan directory for safetensors
	if ( shard_files.empty() ) {
		DIR* dir = opendir( dir_path );
		if ( dir != nullptr ) {
			struct dirent* entry;
			while ( ( entry = readdir( dir ) ) != nullptr ) {
				size_t len = std::strlen( entry->d_name );
				if ( len > 12 && std::strcmp( entry->d_name + len - 12, ".safetensors" ) == 0 ) {
					if ( std::strncmp( entry->d_name, "model", 5 ) == 0 ) {
						shard_files.push_back( entry->d_name );
					}
				}
			}
			closedir( dir );
			std::sort( shard_files.begin(), shard_files.end() );
		}
	}

	// 3. Fallback: single model.safetensors
	if ( shard_files.empty() ) {
		shard_files.push_back( "model.safetensors" );
	}

	// 4. Open each shard and populate tensor_map_
	for ( const auto& fname : shard_files ) {
		if ( num_shards_ >= MAX_SHARDS ) {
			break;
		}
		char full_path[1024];
		std::snprintf( full_path, sizeof( full_path ), "%s/%s", dir_path, fname.c_str() );

		if ( safetensors_open( shards_[num_shards_], full_path ) ) {
			std::strncpy( shard_names_[num_shards_], fname.c_str(), sizeof( shard_names_[0] ) - 1 );
			int s_idx = num_shards_;
			num_shards_++;

			for ( int t = 0; t < shards_[s_idx].tensor_count; ++t ) {
				tensor_map_[shards_[s_idx].tensors[t].name] = { s_idx, &shards_[s_idx].tensors[t] };
			}
		}
	}

	// Tensors a checkpoint keeps outside its index (GLM-5.3 ships the DSA k-pool gate / APE in kpool_aux.safetensors)
	{
		char aux_path[1024];
		std::snprintf( aux_path, sizeof( aux_path ), "%s/kpool_aux.safetensors", dir_path );
		FILE* fa = ( num_shards_ < MAX_SHARDS ) ? std::fopen( aux_path, "rb" ) : nullptr;
		if ( fa != nullptr ) {
			std::fclose( fa );
			if ( safetensors_open( shards_[num_shards_], aux_path ) ) {
				std::strncpy( shard_names_[num_shards_], "kpool_aux.safetensors", sizeof( shard_names_[0] ) - 1 );
				const int s_idx = num_shards_++;
				for ( int t = 0; t < shards_[s_idx].tensor_count; ++t ) {
					tensor_map_[shards_[s_idx].tensors[t].name] = { s_idx, &shards_[s_idx].tensors[t] };
				}
			}
		}
	}

	LOG_INFO( "[SafeTensorsRepo] Loaded %d shards, mapped %zu tensors\n", num_shards_, tensor_map_.size() );
	std::fflush( stdout );

	return num_shards_ > 0;
}

const pod::SafeTensorMeta* SafeTensorsRepo::find( const char* name, int* out_shard_idx ) const {
	auto it = tensor_map_.find( name );
	if ( it == tensor_map_.end() ) {
		return nullptr;
	}
	if ( out_shard_idx != nullptr ) {
		*out_shard_idx = it->second.first;
	}
	return it->second.second;
}

pod::TensorView SafeTensorsRepo::load(
	const char* name,
	pod::MemoryArena& arena,
	backend::stream_t stream,
	pod::DType target_dtype
) {
	int shard_idx = -1;
	const auto* meta = find( name, &shard_idx );
	if ( !meta || shard_idx < 0 ) {
		return pod::TensorView{};
	}

	pod::DType out_dtype = target_dtype;
	if ( target_dtype == pod::DType::AUTO ) {
		if ( meta->dtype == pod::DType::BF16 ) {
			if ( std::strstr( name, "embed_tokens" ) != nullptr ) {
				out_dtype = pod::DType::BF16;
			}
			else {
				out_dtype = pod::DType::FP16;
			}
		}
		else {
			out_dtype = meta->dtype;
		}
	}

	pod::TensorView tv = engine::core::arena_alloc_tensor( arena, out_dtype, meta->shape, meta->ndim );
	if ( tv.data == nullptr ) {
		return tv;
	}

	const void* src_cpu = safetensors_get_data_cpu( shards_[shard_idx], meta );
	if ( src_cpu == nullptr ) {
		return pod::TensorView{};
	}

	size_t total_elems = 1;
	for ( int i = 0; i < meta->ndim; ++i ) {
		total_elems *= static_cast<size_t>( meta->shape[i] );
	}

	if ( out_dtype == meta->dtype ) {
		backend::memcpy_async( tv.data, src_cpu, meta->size_bytes, backend::Copy::Auto, stream );
		backend::stream_sync( stream );
	}
	else if ( meta->dtype == pod::DType::BF16 && out_dtype == pod::DType::FP16 ) {
		const uint16_t* src_bf16 = static_cast<const uint16_t*>( src_cpu );
		std::vector<backend::half_t> h_fp16( total_elems );
		for ( size_t i = 0; i < total_elems; ++i ) {
			h_fp16[i] = bf16_to_fp16( src_bf16[i] );
		}
		backend::memcpy_async( tv.data, h_fp16.data(), total_elems * sizeof( backend::half_t ), backend::Copy::Auto, stream );
		backend::stream_sync( stream );
	}
	else if ( meta->dtype == pod::DType::FP32 && out_dtype == pod::DType::FP16 ) {
		const float* src_f32 = static_cast<const float*>( src_cpu );
		std::vector<backend::half_t> h_fp16( total_elems );
		for ( size_t i = 0; i < total_elems; ++i ) {
			h_fp16[i] = f32_to_fp16( src_f32[i] );
		}
		backend::memcpy_async( tv.data, h_fp16.data(), total_elems * sizeof( backend::half_t ), backend::Copy::Auto, stream );
		backend::stream_sync( stream );
	}
	else {
		backend::memcpy_async( tv.data, src_cpu, meta->size_bytes, backend::Copy::Auto, stream );
		backend::stream_sync( stream );
	}

	return tv;
}

pod::TensorView SafeTensorsRepo::load_slice(
	const char* name,
	pod::MemoryArena& arena,
	int slice_dim,
	int rank,
	int tp_size,
	backend::stream_t stream,
	pod::DType target_dtype
) {
	if ( tp_size <= 1 ) {
		return load( name, arena, stream, target_dtype );
	}

	int shard_idx = -1;
	const auto* meta = find( name, &shard_idx );
	if ( !meta || shard_idx < 0 ) {
		return pod::TensorView{};
	}

	ENGINE_ASSERT( slice_dim >= 0 && slice_dim < meta->ndim, "Invalid slice_dim" );

	pod::DType out_dtype = target_dtype;
	if ( target_dtype == pod::DType::AUTO ) {
		if ( meta->dtype == pod::DType::BF16 ) {
			if ( std::strstr( name, "embed_tokens" ) != nullptr ) {
				out_dtype = pod::DType::BF16;
			}
			else {
				out_dtype = pod::DType::FP16;
			}
		}
		else {
			out_dtype = meta->dtype;
		}
	}

	int local_shape[4];
	for ( int i = 0; i < meta->ndim; ++i ) {
		local_shape[i] = meta->shape[i];
	}
	local_shape[slice_dim] = meta->shape[slice_dim] / tp_size;

	pod::TensorView tv = engine::core::arena_alloc_tensor( arena, out_dtype, local_shape, meta->ndim );
	if ( tv.data == nullptr ) {
		return tv;
	}

	const void* src_cpu = safetensors_get_data_cpu( shards_[shard_idx], meta );
	if ( src_cpu == nullptr ) {
		return pod::TensorView{};
	}

	size_t elem_size = get_dtype_size( meta->dtype );

	if ( meta->ndim == 1 ) {
		size_t local_elems = static_cast<size_t>( local_shape[0] );
		size_t offset_elems = static_cast<size_t>( rank ) * local_elems;

		if ( out_dtype == meta->dtype ) {
			const uint8_t* src_ptr = static_cast<const uint8_t*>( src_cpu ) + offset_elems * elem_size;
			backend::memcpy_async( tv.data, src_ptr, local_elems * elem_size, backend::Copy::Auto, stream );
			backend::stream_sync( stream );
		}
		else if ( meta->dtype == pod::DType::BF16 && out_dtype == pod::DType::FP16 ) {
			const uint16_t* src_bf16 = static_cast<const uint16_t*>( src_cpu ) + offset_elems;
			std::vector<backend::half_t> h_fp16( local_elems );
			for ( size_t i = 0; i < local_elems; ++i ) {
				h_fp16[i] = bf16_to_fp16( src_bf16[i] );
			}
			backend::memcpy_async( tv.data, h_fp16.data(), local_elems * sizeof( backend::half_t ), backend::Copy::Auto, stream );
			backend::stream_sync( stream );
		}
		else if ( meta->dtype == pod::DType::FP32 && out_dtype == pod::DType::FP16 ) {
			const float* src_f32 = static_cast<const float*>( src_cpu ) + offset_elems;
			std::vector<backend::half_t> h_fp16( local_elems );
			for ( size_t i = 0; i < local_elems; ++i ) {
				h_fp16[i] = f32_to_fp16( src_f32[i] );
			}
			backend::memcpy_async( tv.data, h_fp16.data(), local_elems * sizeof( backend::half_t ), backend::Copy::Auto, stream );
			backend::stream_sync( stream );
		}
	}
	else if ( meta->ndim == 2 ) {
		int rows = meta->shape[0];
		int cols = meta->shape[1];

		if ( slice_dim == 0 ) {
			int local_rows = local_shape[0];
			size_t num_elems = static_cast<size_t>( local_rows ) * cols;
			size_t offset_elems = static_cast<size_t>( rank ) * num_elems;

			if ( out_dtype == meta->dtype ) {
				const uint8_t* src_ptr = static_cast<const uint8_t*>( src_cpu ) + offset_elems * elem_size;
				backend::memcpy_async( tv.data, src_ptr, num_elems * elem_size, backend::Copy::Auto, stream );
				backend::stream_sync( stream );
			}
			else if ( meta->dtype == pod::DType::BF16 && out_dtype == pod::DType::FP16 ) {
				const uint16_t* src_bf16 = static_cast<const uint16_t*>( src_cpu ) + offset_elems;
				std::vector<backend::half_t> h_fp16( num_elems );
				for ( size_t i = 0; i < num_elems; ++i ) {
					h_fp16[i] = bf16_to_fp16( src_bf16[i] );
				}
				backend::memcpy_async( tv.data, h_fp16.data(), num_elems * sizeof( backend::half_t ), backend::Copy::Auto, stream );
				backend::stream_sync( stream );
			}
			else if ( meta->dtype == pod::DType::FP32 && out_dtype == pod::DType::FP16 ) {
				const float* src_f32 = static_cast<const float*>( src_cpu ) + offset_elems;
				std::vector<backend::half_t> h_fp16( num_elems );
				for ( size_t i = 0; i < num_elems; ++i ) {
					h_fp16[i] = f32_to_fp16( src_f32[i] );
				}
				backend::memcpy_async( tv.data, h_fp16.data(), num_elems * sizeof( backend::half_t ), backend::Copy::Auto, stream );
				backend::stream_sync( stream );
			}
		}
		else if ( slice_dim == 1 ) {
			int local_cols = local_shape[1];
			size_t total_local = static_cast<size_t>( rows ) * local_cols;

			if ( out_dtype == meta->dtype ) {
				std::vector<uint8_t> h_buf( total_local * elem_size );
				size_t chunk_bytes = static_cast<size_t>( local_cols) * elem_size;
				for ( int r = 0; r < rows; ++r ) {
					const uint8_t* src_r = static_cast<const uint8_t*>( src_cpu ) + ( static_cast<size_t>( r ) * cols + rank * local_cols ) * elem_size;
					uint8_t* dst_r = h_buf.data() + static_cast<size_t>( r ) * chunk_bytes;
					std::memcpy( dst_r, src_r, chunk_bytes );
				}
				backend::memcpy_async( tv.data, h_buf.data(), h_buf.size(), backend::Copy::Auto, stream );
				backend::stream_sync( stream );
			}
			else if ( meta->dtype == pod::DType::BF16 && out_dtype == pod::DType::FP16 ) {
				std::vector<backend::half_t> h_buf( total_local );
				const uint16_t* src_bf16 = static_cast<const uint16_t*>( src_cpu );
				for ( int r = 0; r < rows; ++r ) {
					for ( int c = 0; c < local_cols; ++c ) {
						h_buf[r * local_cols + c] = bf16_to_fp16( src_bf16[r * cols + rank * local_cols + c] );
					}
				}
				backend::memcpy_async( tv.data, h_buf.data(), total_local * sizeof( backend::half_t ), backend::Copy::Auto, stream );
				backend::stream_sync( stream );
			}
		}
	}
	else if ( meta->ndim == 3 ) {
		// Trellis [BK, BN, B]
		int BK = meta->shape[0];
		int BN = meta->shape[1];
		int B = meta->shape[2];

		if ( slice_dim == 0 ) {
			int local_BK = local_shape[0];
			size_t slice_bytes = static_cast<size_t>( local_BK ) * BN * B * sizeof( uint16_t );
			size_t offset_bytes = static_cast<size_t>( rank ) * slice_bytes;
			const uint8_t* src_ptr = static_cast<const uint8_t*>( src_cpu ) + offset_bytes;
			backend::memcpy_async( tv.data, src_ptr, slice_bytes, backend::Copy::Auto, stream );
			backend::stream_sync( stream );
		}
		else if ( slice_dim == 1 ) {
			int local_BN = local_shape[1];
			size_t total_elems = static_cast<size_t>( BK ) * local_BN * B;
			std::vector<uint16_t> h_buf( total_elems );
			const uint16_t* src_u16 = static_cast<const uint16_t*>( src_cpu );
			size_t chunk_elems = static_cast<size_t>( local_BN ) * B;

			for ( int k = 0; k < BK; ++k ) {
				const uint16_t* src_k = src_u16 + ( static_cast<size_t>( k ) * BN + rank * local_BN ) * B;
				uint16_t* dst_k = h_buf.data() + static_cast<size_t>( k ) * chunk_elems;
				std::memcpy( dst_k, src_k, chunk_elems * sizeof( uint16_t ) );
			}
			backend::memcpy_async( tv.data, h_buf.data(), total_elems * sizeof( uint16_t ), backend::Copy::Auto, stream );
			backend::stream_sync( stream );
		}
	}

	return tv;
}

size_t SafeTensorsRepo::trellis_bytes( const char* base_name ) const {
	char name_buf[512];
	std::snprintf( name_buf, sizeof( name_buf ), "%s.trellis", base_name );
	const pod::SafeTensorMeta* meta = find( name_buf );
	return meta != nullptr ? static_cast<size_t>( meta->size_bytes ) : 0;
}

size_t SafeTensorsRepo::bytes_with_prefix( const char* prefix ) const {
	size_t total = 0;
	const size_t n = std::strlen( prefix );
	for ( int i = 0; i < num_shards_; ++i ) {
		const pod::SafeTensorsFile* st = get_shard( i );
		if ( !st ) continue;
		for ( int t = 0; t < st->tensor_count; ++t ) {
			if ( std::strncmp( st->tensors[t].name, prefix, n ) == 0 ) total += st->tensors[t].size_bytes + 256;
		}
	}
	return total;
}

pod::LinearWeight SafeTensorsRepo::load_linear(
	const char* base_name,
	pod::MemoryArena& arena,
	int bits,
	backend::stream_t stream,
	pod::MemoryArena* code_arena
) {
	char name_buf[512];
	std::snprintf( name_buf, sizeof( name_buf ), "%s.trellis", base_name );
	if ( find( name_buf ) == nullptr ) {
		// No quantized tensor: this is an unquantized checkpoint (the DFlash2 drafter's own
		// pre-quantization release, for instance), where the weight is a plain [n, k] matrix. Load it
		// dense and hand it to the same FP16 GEMV the unquantized projections already use, so such a
		// checkpoint can be dropped into cfg.draft_model_dir unchanged and A/B'd against the quantized
		// one. Max |w| there is 20.9 against FP16's 65504, and BF16 -> FP16 is exact for normals.
		std::snprintf( name_buf, sizeof( name_buf ), "%s.weight", base_name );
		int shard = -1;
		const pod::SafeTensorMeta* meta = find( name_buf, &shard );
		if ( meta == nullptr || meta->ndim != 2 ) {
			LOG_ERROR( "load_linear: no tensor group for '%s'\n", base_name );
			ENGINE_ASSERT( false, "load_linear: neither a trellis group nor a 2-D weight for this name" );
		}
		pod::TensorView w = load( name_buf, arena, stream, pod::DType::FP16 );
		const int n = meta->shape[0], k = meta->shape[1];
		// The tensor is stored [n, k]; Linear::forward takes n from shape[1] and k from shape[0].
		return pod::LinearWeight::make_fp16(
			pod::make_tensor_2d( w.data, pod::DType::FP16, k, n ) );
	}
	pod::TensorView tr = load( name_buf, code_arena ? *code_arena : arena, stream );

	std::snprintf( name_buf, sizeof( name_buf ), "%s.suh", base_name );
	pod::TensorView suh = load( name_buf, arena, stream, pod::DType::FP16 );

	std::snprintf( name_buf, sizeof( name_buf ), "%s.svh", base_name );
	pod::TensorView svh = load( name_buf, arena, stream, pod::DType::FP16 );

	std::snprintf( name_buf, sizeof( name_buf ), "%s.mul1", base_name );
	int dummy_shard = -1;
	bool has_mul1 = ( find( name_buf, &dummy_shard ) != nullptr );
	std::snprintf( name_buf, sizeof( name_buf ), "%s.mcg", base_name );
	const bool has_mcg = ( find( name_buf, &dummy_shard ) != nullptr );

	int actual_bits = ( tr.data && tr.ndim >= 3 ) ? ( tr.shape[2] / 16 ) : bits;
	return pod::LinearWeight::make_trellis( tr, suh, svh, actual_bits, has_mul1, has_mcg );
}

pod::LinearWeight SafeTensorsRepo::load_linear_slice(
	const char* base_name,
	pod::MemoryArena& arena,
	int bits,
	int slice_dim,
	int rank,
	int tp_size,
	backend::stream_t stream
) {
	char name_buf[512];
	std::snprintf( name_buf, sizeof( name_buf ), "%s.trellis", base_name );
	if ( find( name_buf ) == nullptr && tp_size > 1 ) {
		// Unquantized checkpoint (the DFlash2 drafter's own release): a dense [n, k] weight. Trellis slice_dim 1 splits
		// the output dimension n (dense dim 0), slice_dim 0 splits the input dimension k (dense dim 1).
		std::snprintf( name_buf, sizeof( name_buf ), "%s.weight", base_name );
		const pod::SafeTensorMeta* meta = find( name_buf );
		ENGINE_ASSERT( meta != nullptr && meta->ndim == 2, "load_linear_slice: neither a trellis group nor a 2-D weight for this name" );
		pod::TensorView w = load_slice( name_buf, arena, slice_dim == 1 ? 0 : 1, rank, tp_size, stream, pod::DType::FP16 );
		return pod::LinearWeight::make_fp16( pod::make_tensor_2d( w.data, pod::DType::FP16, w.shape[1], w.shape[0] ) );
	}
	pod::TensorView tr = load_slice( name_buf, arena, slice_dim, rank, tp_size, stream );

	std::snprintf( name_buf, sizeof( name_buf ), "%s.mul1", base_name );
	int dummy_shard = -1;
	bool has_mul1 = ( find( name_buf, &dummy_shard ) != nullptr );
	std::snprintf( name_buf, sizeof( name_buf ), "%s.mcg", base_name );
	const bool has_mcg = ( find( name_buf, &dummy_shard ) != nullptr );

	int actual_bits = ( tr.data && tr.ndim >= 3 ) ? ( tr.shape[2] / 16 ) : bits;

	if ( slice_dim == 0 ) {
		std::snprintf( name_buf, sizeof( name_buf ), "%s.suh", base_name );
		pod::TensorView suh = load_slice( name_buf, arena, 0, rank, tp_size, stream, pod::DType::FP16 );

		std::snprintf( name_buf, sizeof( name_buf ), "%s.svh", base_name );
		pod::TensorView svh = load( name_buf, arena, stream, pod::DType::FP16 );

		return pod::LinearWeight::make_trellis( tr, suh, svh, actual_bits, has_mul1, has_mcg );
	}
	else {
		std::snprintf( name_buf, sizeof( name_buf ), "%s.suh", base_name );
		pod::TensorView suh = load( name_buf, arena, stream, pod::DType::FP16 );

		std::snprintf( name_buf, sizeof( name_buf ), "%s.svh", base_name );
		pod::TensorView svh = load_slice( name_buf, arena, 0, rank, tp_size, stream, pod::DType::FP16 );

		return pod::LinearWeight::make_trellis( tr, suh, svh, actual_bits, has_mul1, has_mcg );
	}
}

pod::LinearWeight SafeTensorsRepo::load_linear_slice_padded(
	const char* base_name,
	pod::MemoryArena& arena,
	int bits,
	int rank,
	int tp_size,
	int pad_cols,
	backend::stream_t stream
) {
	char name_buf[512];
	std::snprintf( name_buf, sizeof( name_buf ), "%s.trellis", base_name );
	const auto* meta = find( name_buf );
	if ( !meta ) return pod::LinearWeight{};
	ENGINE_ASSERT( meta->ndim == 3 && meta->shape[1] % tp_size == 0, "Trellis column count must divide across ranks" );
	const int local_blocks = meta->shape[1] / tp_size;
	return load_linear_cols_padded( base_name, arena, bits, rank * local_blocks, local_blocks, pad_cols, stream );
}

pod::LinearWeight SafeTensorsRepo::load_linear_cols_padded(
	const char* base_name,
	pod::MemoryArena& arena,
	int bits,
	int col_block_begin,
	int col_blocks,
	int pad_cols,
	backend::stream_t stream
) {
	char name_buf[512];
	int shard = -1;

	// Trellis: rank's column slice, zero-padded to pad_cols. The pad rows are never read for a token
	// decision (the argmax range stops at the real width), but the kernel still walks them, so they
	// have to be finite: zero trellis plus zero svh makes those logits exactly 0.0.
	std::snprintf( name_buf, sizeof( name_buf ), "%s.trellis", base_name );
	const auto* meta_tr = find( name_buf, &shard );
	if ( !meta_tr || shard < 0 ) return pod::LinearWeight{};
	ENGINE_ASSERT( meta_tr->ndim == 3, "Trellis tensor must be 3D" );

	const int BK = meta_tr->shape[0];
	const int BN = meta_tr->shape[1];
	const int B = meta_tr->shape[2];
	const int local_BN = col_blocks;
	ENGINE_ASSERT( col_block_begin >= 0 && col_block_begin + local_BN <= BN, "column range outside the tensor" );
	ENGINE_ASSERT( local_BN % 8 == 0, "column range must be whole 128-column Hadamard spans" );
	const int padded_BN = local_BN + pad_cols / 16;
	ENGINE_ASSERT( pad_cols % 128 == 0, "Pad must be a whole number of 128-column Hadamard spans" );
	ENGINE_ASSERT( padded_BN * 16 % 256 == 0, "Padded shard width must be a multiple of 256 for the int8 GEMV" );

	int tr_shape[3] = { BK, padded_BN, B };
	pod::TensorView tr = engine::core::arena_alloc_tensor( arena, pod::DType::UINT16, tr_shape, 3 );
	if ( !tr.data ) return pod::LinearWeight{};

	std::vector<uint16_t> h_tr( static_cast<size_t>( BK ) * padded_BN * B, 0 );
	const uint16_t* src_tr = static_cast<const uint16_t*>( safetensors_get_data_cpu( shards_[shard], meta_tr ) );
	ENGINE_ASSERT( src_tr != nullptr, "Failed to get CPU data for trellis tensor" );
	size_t chunk_elems = static_cast<size_t>( local_BN ) * B;
	for ( int k = 0; k < BK; ++k ) {
		const uint16_t* src_k = src_tr + ( static_cast<size_t>( k ) * BN + col_block_begin ) * B;
		std::memcpy( h_tr.data() + static_cast<size_t>( k ) * padded_BN * B, src_k, chunk_elems * sizeof( uint16_t ) );
	}
	backend::memcpy_async( tr.data, h_tr.data(), h_tr.size() * sizeof( uint16_t ), backend::Copy::Auto, stream );
	backend::stream_sync( stream );

	// Input Hadamard scales are shared by every rank (they scale the activation, not the output).
	std::snprintf( name_buf, sizeof( name_buf ), "%s.suh", base_name );
	pod::TensorView suh = load( name_buf, arena, stream, pod::DType::FP16 );

	// Output Hadamard scales: this rank's columns, zero-padded by pad_cols.
	std::snprintf( name_buf, sizeof( name_buf ), "%s.svh", base_name );
	shard = -1;
	const auto* meta_svh = find( name_buf, &shard );
	if ( !meta_svh || shard < 0 ) return pod::LinearWeight{};
	ENGINE_ASSERT( meta_svh->ndim == 1 && meta_svh->shape[0] == BN * 16, "svh must be one scale per column" );

	std::vector<backend::half_t> h_svh( static_cast<size_t>( local_BN ) * 16 + pad_cols, backend::from_float_half( 0.0f ) );
	const backend::half_t* src_svh = static_cast<const backend::half_t*>( safetensors_get_data_cpu( shards_[shard], meta_svh ) );
	ENGINE_ASSERT( src_svh != nullptr, "Failed to get CPU data for svh" );
	std::memcpy( h_svh.data(), src_svh + static_cast<size_t>( col_block_begin ) * 16, ( h_svh.size() - pad_cols ) * sizeof( backend::half_t ) );

	int svh_shape[1] = { static_cast<int>( h_svh.size() ) };
	pod::TensorView svh = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, svh_shape, 1 );
	if ( !svh.data ) return pod::LinearWeight{};
	backend::memcpy_async( svh.data, h_svh.data(), h_svh.size() * sizeof( backend::half_t ), backend::Copy::Auto, stream );
	backend::stream_sync( stream );

	std::snprintf( name_buf, sizeof( name_buf ), "%s.mul1", base_name );
	int dummy_shard = -1;
	bool has_mul1 = ( find( name_buf, &dummy_shard ) != nullptr );
	std::snprintf( name_buf, sizeof( name_buf ), "%s.mcg", base_name );
	const bool has_mcg = ( find( name_buf, &dummy_shard ) != nullptr );

	int actual_bits = B / 16;
	return pod::LinearWeight::make_trellis( tr, suh, svh, actual_bits ? actual_bits : bits, has_mul1, has_mcg );
}

pod::LinearWeight SafeTensorsRepo::load_linear_slice_fused(
	const char* base_name_a,
	const char* base_name_b,
	pod::MemoryArena& arena,
	int bits,
	int slice_dim,
	int rank,
	int tp_size,
	backend::stream_t stream
) {
	char name_buf_a[512];
	char name_buf_b[512];
	std::snprintf( name_buf_a, sizeof( name_buf_a ), "%s.trellis", base_name_a );
	std::snprintf( name_buf_b, sizeof( name_buf_b ), "%s.trellis", base_name_b );

	int shard_a = -1, shard_b = -1;
	const auto* meta_a = find( name_buf_a, &shard_a );
	const auto* meta_b = find( name_buf_b, &shard_b );
	if ( !meta_a || !meta_b || shard_a < 0 || shard_b < 0 ) {
		return pod::LinearWeight{};
	}

	ENGINE_ASSERT( meta_a->ndim == 3 && meta_b->ndim == 3, "Trellis tensor must be 3D" );
	ENGINE_ASSERT( meta_a->shape[0] == meta_b->shape[0], "K dimension must match" );
	ENGINE_ASSERT( meta_a->shape[2] == meta_b->shape[2], "B dimension must match" );

	int BK = meta_a->shape[0];
	int BN_a = meta_a->shape[1];
	int BN_b = meta_b->shape[1];
	int B = meta_a->shape[2];

	int local_BN_a = BN_a / tp_size;
	int local_BN_b = BN_b / tp_size;
	int local_BN_fused = local_BN_a + local_BN_b;

	int fused_shape[3] = { BK, local_BN_fused, B };
	pod::TensorView tr = engine::core::arena_alloc_tensor( arena, pod::DType::UINT16, fused_shape, 3 );
	if ( !tr.data ) {
		return pod::LinearWeight{};
	}

	const void* src_cpu_a = safetensors_get_data_cpu( shards_[shard_a], meta_a );
	const void* src_cpu_b = safetensors_get_data_cpu( shards_[shard_b], meta_b );
	ENGINE_ASSERT( src_cpu_a && src_cpu_b, "Failed to get CPU data for trellis tensors" );

	const uint16_t* u16_a = static_cast<const uint16_t*>( src_cpu_a );
	const uint16_t* u16_b = static_cast<const uint16_t*>( src_cpu_b );

	size_t total_fused_elems = static_cast<size_t>( BK ) * local_BN_fused * B;
	std::vector<uint16_t> h_fused( total_fused_elems );

	size_t chunk_a_elems = static_cast<size_t>( local_BN_a ) * B;
	size_t chunk_b_elems = static_cast<size_t>( local_BN_b ) * B;
	size_t chunk_fused_elems = static_cast<size_t>( local_BN_fused ) * B;

	for ( int k = 0; k < BK; ++k ) {
		const uint16_t* src_ak = u16_a + ( static_cast<size_t>( k ) * BN_a + rank * local_BN_a ) * B;
		const uint16_t* src_bk = u16_b + ( static_cast<size_t>( k ) * BN_b + rank * local_BN_b ) * B;
		uint16_t* dst_k = h_fused.data() + static_cast<size_t>( k ) * chunk_fused_elems;

		std::memcpy( dst_k, src_ak, chunk_a_elems * sizeof( uint16_t ) );
		std::memcpy( dst_k + chunk_a_elems, src_bk, chunk_b_elems * sizeof( uint16_t ) );
	}

	backend::memcpy_async( tr.data, h_fused.data(), total_fused_elems * sizeof( uint16_t ), backend::Copy::Auto, stream );
	backend::stream_sync( stream );

	std::snprintf( name_buf_a, sizeof( name_buf_a ), "%s.suh", base_name_a );
	pod::TensorView suh = load( name_buf_a, arena, stream, pod::DType::FP16 );

	std::snprintf( name_buf_a, sizeof( name_buf_a ), "%s.svh", base_name_a );
	std::snprintf( name_buf_b, sizeof( name_buf_b ), "%s.svh", base_name_b );

	int svh_shard_a = -1, svh_shard_b = -1;
	const auto* meta_svh_a = find( name_buf_a, &svh_shard_a );
	const auto* meta_svh_b = find( name_buf_b, &svh_shard_b );
	ENGINE_ASSERT( meta_svh_a && meta_svh_b, "svh metadata missing" );

	int local_N_a = meta_svh_a->shape[0] / tp_size;
	int local_N_b = meta_svh_b->shape[0] / tp_size;
	int local_N_fused = local_N_a + local_N_b;

	int svh_shape[1] = { local_N_fused };
	pod::TensorView svh = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, svh_shape, 1 );

	const void* src_svh_a = safetensors_get_data_cpu( shards_[svh_shard_a], meta_svh_a );
	const void* src_svh_b = safetensors_get_data_cpu( shards_[svh_shard_b], meta_svh_b );
	ENGINE_ASSERT( src_svh_a && src_svh_b, "Failed to get CPU data for svh tensors" );

	const backend::half_t* h_svh_a = static_cast<const backend::half_t*>( src_svh_a ) + rank * local_N_a;
	const backend::half_t* h_svh_b = static_cast<const backend::half_t*>( src_svh_b ) + rank * local_N_b;

	std::vector<backend::half_t> h_svh_fused( local_N_fused );
	std::memcpy( h_svh_fused.data(), h_svh_a, local_N_a * sizeof( backend::half_t ) );
	std::memcpy( h_svh_fused.data() + local_N_a, h_svh_b, local_N_b * sizeof( backend::half_t ) );

	backend::memcpy_async( svh.data, h_svh_fused.data(), local_N_fused * sizeof( backend::half_t ), backend::Copy::Auto, stream );
	backend::stream_sync( stream );

	int actual_bits = ( tr.data && tr.ndim >= 3 ) ? ( tr.shape[2] / 16 ) : bits;
	return pod::LinearWeight::make_trellis( tr, suh, svh, actual_bits );
}

pod::LinearWeight SafeTensorsRepo::load_grouped_trellis_slice(
	const char* base_name,
	pod::MemoryArena& arena,
	const int* group_channel_blocks,
	const int* rank_channel_blocks,
	int num_groups,
	int rank,
	backend::stream_t stream
) {
	char name_buf[512];
	std::snprintf( name_buf, sizeof( name_buf ), "%s.trellis", base_name );
	pod::TensorView tr = load_grouped_slice(
		name_buf, arena, group_channel_blocks, rank_channel_blocks, num_groups, rank, 1, stream, pod::DType::UINT16
	);
	if ( !tr.data ) {
		return pod::LinearWeight{};
	}

	std::snprintf( name_buf, sizeof( name_buf ), "%s.suh", base_name );
	pod::TensorView suh = load( name_buf, arena, stream, pod::DType::FP16 );

	std::vector<int> group_channels( num_groups );
	std::vector<int> rank_channels( num_groups );
	for ( int g = 0; g < num_groups; ++g ) {
		group_channels[g] = group_channel_blocks[g] * 16;
		rank_channels[g] = rank_channel_blocks[g] * 16;
	}

	std::snprintf( name_buf, sizeof( name_buf ), "%s.svh", base_name );
	pod::TensorView svh = load_grouped_slice(
		name_buf, arena, group_channels.data(), rank_channels.data(), num_groups, rank, 0, stream, pod::DType::FP16
	);

	std::snprintf( name_buf, sizeof( name_buf ), "%s.mul1", base_name );
	int dummy_shard = -1;
	bool has_mul1 = ( find( name_buf, &dummy_shard ) != nullptr );
	std::snprintf( name_buf, sizeof( name_buf ), "%s.mcg", base_name );
	const bool has_mcg = ( find( name_buf, &dummy_shard ) != nullptr );

	int actual_bits = ( tr.data && tr.ndim >= 3 ) ? ( tr.shape[2] / 16 ) : 5;
	return pod::LinearWeight::make_trellis( tr, suh, svh, actual_bits, has_mul1, has_mcg );
}

pod::TensorView SafeTensorsRepo::load_grouped_slice(
	const char* name,
	pod::MemoryArena& arena,
	const int* group_sizes,
	const int* rank_sizes,
	int num_groups,
	int rank,
	int slice_dim,
	backend::stream_t stream,
	pod::DType target_dtype
) {
	int shard_idx = -1;
	const auto* meta = find( name, &shard_idx );
	if ( !meta || shard_idx < 0 || num_groups <= 0 ) {
		return pod::TensorView{};
	}

	int total_group_size = 0;
	int total_rank_size = 0;
	for ( int g = 0; g < num_groups; ++g ) {
		total_group_size += group_sizes[g];
		total_rank_size += rank_sizes[g];
	}

	if ( slice_dim < 0 || slice_dim >= meta->ndim || meta->shape[slice_dim] != total_group_size ) {
		return pod::TensorView{};
	}

	int local_shape[4] = { 0, 0, 0, 0 };
	for ( int i = 0; i < meta->ndim && i < 4; ++i ) {
		local_shape[i] = meta->shape[i];
	}
	local_shape[slice_dim] = total_rank_size;

	pod::DType out_dtype = target_dtype;
	if ( out_dtype == pod::DType::FP16 && ( meta->dtype == pod::DType::FP16 || meta->dtype == pod::DType::BF16 || meta->dtype == pod::DType::FP32 ) ) {
		out_dtype = pod::DType::FP16;
	}
	else if ( target_dtype == pod::DType::FP16 && meta->dtype == pod::DType::UINT16 ) {
		out_dtype = pod::DType::UINT16;
	}

	pod::TensorView tv = engine::core::arena_alloc_tensor( arena, out_dtype, local_shape, meta->ndim );
	if ( tv.data == nullptr ) {
		return tv;
	}

	const void* src_cpu = safetensors_get_data_cpu( shards_[shard_idx], meta );
	if ( src_cpu == nullptr ) {
		return pod::TensorView{};
	}

	size_t outer_size = 1;
	for ( int i = 0; i < slice_dim; ++i ) {
		outer_size *= static_cast<size_t>( meta->shape[i] );
	}

	size_t inner_size = 1;
	for ( int i = slice_dim + 1; i < meta->ndim; ++i ) {
		inner_size *= static_cast<size_t>( meta->shape[i] );
	}

	size_t total_local_elems = outer_size * static_cast<size_t>( total_rank_size ) * inner_size;

	if ( out_dtype == pod::DType::FP16 && meta->dtype == pod::DType::BF16 ) {
		const uint16_t* src_bf16 = static_cast<const uint16_t*>( src_cpu );
		std::vector<backend::half_t> h_dst( total_local_elems );

		for ( size_t o = 0; o < outer_size; ++o ) {
			int src_grp_start = 0;
			int dst_grp_start = 0;
			for ( int g = 0; g < num_groups; ++g ) {
				int src_offset = src_grp_start + rank * rank_sizes[g];
				int dst_offset = dst_grp_start;
				int count = rank_sizes[g];

				for ( int c = 0; c < count; ++c ) {
					size_t src_idx = ( o * total_group_size + ( src_offset + c ) ) * inner_size;
					size_t dst_idx = ( o * total_rank_size + ( dst_offset + c ) ) * inner_size;
					for ( size_t k = 0; k < inner_size; ++k ) {
						h_dst[dst_idx + k] = bf16_to_fp16( src_bf16[src_idx + k] );
					}
				}

				src_grp_start += group_sizes[g];
				dst_grp_start += rank_sizes[g];
			}
		}

		backend::memcpy_async( tv.data, h_dst.data(), total_local_elems * sizeof( backend::half_t ), backend::Copy::Auto, stream );
		backend::stream_sync( stream );
	}
	else if ( out_dtype == pod::DType::FP16 && meta->dtype == pod::DType::FP32 ) {
		const float* src_f32 = static_cast<const float*>( src_cpu );
		std::vector<backend::half_t> h_dst( total_local_elems );

		for ( size_t o = 0; o < outer_size; ++o ) {
			int src_grp_start = 0;
			int dst_grp_start = 0;
			for ( int g = 0; g < num_groups; ++g ) {
				int src_offset = src_grp_start + rank * rank_sizes[g];
				int dst_offset = dst_grp_start;
				int count = rank_sizes[g];

				for ( int c = 0; c < count; ++c ) {
					size_t src_idx = ( o * total_group_size + ( src_offset + c ) ) * inner_size;
					size_t dst_idx = ( o * total_rank_size + ( dst_offset + c ) ) * inner_size;
					for ( size_t k = 0; k < inner_size; ++k ) {
						h_dst[dst_idx + k] = f32_to_fp16( src_f32[src_idx + k] );
					}
				}

				src_grp_start += group_sizes[g];
				dst_grp_start += rank_sizes[g];
			}
		}

		backend::memcpy_async( tv.data, h_dst.data(), total_local_elems * sizeof( backend::half_t ), backend::Copy::Auto, stream );
		backend::stream_sync( stream );
	}
	else {
		size_t elem_size = get_dtype_size( meta->dtype );
		std::vector<uint8_t> h_dst( total_local_elems * elem_size );
		const uint8_t* src_bytes = static_cast<const uint8_t*>( src_cpu );

		for ( size_t o = 0; o < outer_size; ++o ) {
			int src_grp_start = 0;
			int dst_grp_start = 0;
			for ( int g = 0; g < num_groups; ++g ) {
				int src_offset = src_grp_start + rank * rank_sizes[g];
				int dst_offset = dst_grp_start;
				int count = rank_sizes[g];

				size_t src_byte_idx = ( ( o * total_group_size + src_offset ) * inner_size ) * elem_size;
				size_t dst_byte_idx = ( ( o * total_rank_size + dst_offset ) * inner_size ) * elem_size;
				size_t copy_bytes = static_cast<size_t>( count ) * inner_size * elem_size;

				std::memcpy( h_dst.data() + dst_byte_idx, src_bytes + src_byte_idx, copy_bytes );

				src_grp_start += group_sizes[g];
				dst_grp_start += rank_sizes[g];
			}
		}

		backend::memcpy_async( tv.data, h_dst.data(), h_dst.size(), backend::Copy::Auto, stream );
		backend::stream_sync( stream );
	}

	return tv;
}

} // namespace engine::loader
