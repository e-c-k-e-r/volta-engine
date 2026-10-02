#include "engine/loader/safetensors.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <cstdlib>
#include <cctype>

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
	if ( len == 3 && std::strncmp( str, "U32", 3 ) == 0 ) {
		return pod::DType::UINT32;
	}
	if ( len == 2 && std::strncmp( str, "U8", 2 ) == 0 ) {
		return pod::DType::UINT8;
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

bool safetensors_load_to_device( const pod::SafeTensorsFile& st, const pod::SafeTensorMeta* meta, void* dst_device_ptr, cudaStream_t stream ) {
	if ( meta == nullptr || dst_device_ptr == nullptr ) {
		return false;
	}
	const void* src_cpu = safetensors_get_data_cpu( st, meta );
	if ( src_cpu == nullptr ) {
		return false;
	}

	cudaError_t err = cudaMemcpyAsync(
		dst_device_ptr,
		src_cpu,
		meta->size_bytes,
		cudaMemcpyHostToDevice,
		stream
	);
	if ( ENGINE_UNLIKELY( err != cudaSuccess ) ) {
		return false;
	}
	if ( stream == nullptr ) {
		cudaDeviceSynchronize();
	}
	return true;
}

pod::TensorView safetensors_alloc_tensor( const pod::SafeTensorsFile& st, const char* name, pod::MemoryArena& arena, cudaStream_t stream ) {
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

} // namespace engine::loader
