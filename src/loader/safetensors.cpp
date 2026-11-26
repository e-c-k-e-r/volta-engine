#include <thread>
#include <chrono>
#include <atomic>
#include "engine/modules/trellis_bits.h"
#include "engine/modules/trellis.h"
#include "engine/loader/safetensors.h"
#include "engine/loader/gguf.h"
#include "engine/modules/gguf.h"
#include "engine/core/log.h"
#include "engine/runtime/options.h"
#include "engine/backend/runtime.h"
#include "engine/backend/numeric.h"
#include "engine/quant/fp4_fp8.h"
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
#include <mutex>
#include <array>
#include <map>
#include <string>
#include <vector>

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
	if ( len == 7 && std::strncmp( str, "F8_E8M0", 7 ) == 0 ) {
		return pod::DType::UINT8;  // raw bytes: power-of-two block scales (DeepSeek-V4.1's FP8 / FP4 tensors)
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

			pod::SafeTensorMeta meta{};
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
							if ( json[pos] == '\\' && pos + 1 < len ) { ++pos; } // escaped character (metadata values carry JSON strings)
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
	if ( st.gguf != nullptr ) gguf_release( st );
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
	if ( meta != nullptr && meta->host != nullptr ) {
		return meta->host;
	}
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

namespace {

// An EXL3 checkpoint names its tile order in quantization_config.json ("tile_layout": "volta"; absent = the stock order), and the weights the
// quantizer skipped (--skip: they keep the template's order) in "tile_layout_stock", a list of substrings of their names. The kernels read either order,
// per weight; the other order would decode to noise without any error. Searches the file's text; nothing is parsed.
struct TileLayout {
	bool volta = false;
	std::vector<std::string> stock;
};

TileLayout exl3_tile_layout( const char* dir ) {
	TileLayout out;
	char path[1200];
	std::snprintf( path, sizeof( path ), "%s/quantization_config.json", dir );
	const int fd = ::open( path, O_RDONLY );
	if ( fd < 0 ) return out;
	struct stat st;
	if ( fstat( fd, &st ) != 0 || st.st_size <= 0 ) { ::close( fd ); return out; }
	void* map = mmap( nullptr, static_cast<size_t>( st.st_size ), PROT_READ, MAP_PRIVATE, fd, 0 );
	::close( fd );
	if ( map == MAP_FAILED ) return out;
	const std::string text( static_cast<const char*>( map ), static_cast<size_t>( st.st_size ) );
	munmap( map, static_cast<size_t>( st.st_size ) );
	if ( text.find( "\"quant_method\"" ) == std::string::npos || text.find( "\"exl3\"" ) == std::string::npos ) return out;
	const size_t at = text.find( "\"tile_layout\"" );
	out.volta = at != std::string::npos && text.compare( text.find( '"', text.find( ':', at ) ), 7, "\"volta\"" ) == 0;
	const size_t sa = out.volta ? text.find( "\"tile_layout_stock\"" ) : std::string::npos;
	if ( sa != std::string::npos ) {
		const size_t open_at = text.find( '[', sa ), close_at = text.find( ']', open_at );
		for ( size_t p = open_at; p != std::string::npos && close_at != std::string::npos; ) {
			p = text.find( '"', p + 1 );
			const size_t q = p == std::string::npos ? p : text.find( '"', p + 1 );
			if ( q == std::string::npos || q > close_at ) break;
			out.stock.push_back( text.substr( p + 1, q - p - 1 ) );
			p = q;
		}
	}
	return out;
}

} // namespace

bool SafeTensorsRepo::open( const char* dir_path, bool mmproj ) {
	close();
	std::strncpy( dir_path_, dir_path, sizeof( dir_path_ ) - 1 );
	dir_path_[sizeof( dir_path_ ) - 1] = '\0';
	const TileLayout layout = mmproj ? TileLayout() : exl3_tile_layout( dir_path );
	volta_tiles_ = layout.volta;
	stock_names_ = layout.stock;
	ENGINE_ASSERT( !volta_tiles_ || engine::modules::trellis_volta_tiles_supported(),
		"this EXL3 checkpoint is in the Volta tile order (tile_layout in quantization_config.json), which this build's GPU kernels do not read (SM_70 CUDA, CPU and Vulkan builds do)" );

	std::vector<std::string> shard_files;

	// 1. Try to parse model.safetensors.index.json (a projector file is a GGUF of its own: straight to step 3)
	char index_path[1024];
	std::snprintf( index_path, sizeof( index_path ), "%s/model.safetensors.index.json", dir_path );

	FILE* f_idx = mmproj ? nullptr : std::fopen( index_path, "rb" );
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
	if ( shard_files.empty() && !mmproj ) {
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

	// 3. No safetensors shards: one GGUF file in the directory (its tensors under the Hugging Face names of its architecture; config.json and
	// the tokenizer still come from the directory)
	if ( shard_files.empty() ) {
		std::vector<std::string> ggufs;
		if ( DIR* dir = opendir( dir_path ) ) {
			while ( struct dirent* entry = readdir( dir ) ) {
				const size_t len = std::strlen( entry->d_name );
				if ( len > 5 && std::strcmp( entry->d_name + len - 5, ".gguf" ) == 0 && ( std::strstr( entry->d_name, "mmproj" ) != nullptr ) == mmproj ) ggufs.push_back( entry->d_name );
			}
			closedir( dir );
		}
		std::sort( ggufs.begin(), ggufs.end() );
		if ( ggufs.size() > 1 ) {
			// a split model beside other GGUF files (DeepSeek-V4.1's shared Engram file, which a split's last part links to): its parts only
			std::vector<std::string> parts;
			for ( const std::string& f : ggufs ) {
				int no = 0, count = 0;
				const size_t at = f.rfind( "-of-" );
				if ( at != std::string::npos && at >= 6 && std::sscanf( f.c_str() + at - 6, "-%5d-of-%5d.gguf", &no, &count ) == 2 && no >= 1 && no <= count ) parts.push_back( f );
			}
			if ( !parts.empty() && parts.size() < ggufs.size() ) ggufs = parts;
		}
		if ( ggufs.size() > 1 ) {
			// the files of one split model (name-00001-of-0000N.gguf ... name-0000N-of-0000N.gguf), in order
			bool split = static_cast<int>( ggufs.size() ) <= MAX_SHARDS;
			for ( size_t i = 0; i < ggufs.size() && split; ++i ) {
				char tail[32];
				std::snprintf( tail, sizeof( tail ), "-%05zu-of-%05zu.gguf", i + 1, ggufs.size() );
				const std::string& f = ggufs[i];
				split = f.size() > std::strlen( tail ) && f.compare( f.size() - std::strlen( tail ), std::string::npos, tail ) == 0 &&
					f.compare( 0, f.size() - std::strlen( tail ), ggufs[0], 0, ggufs[0].size() - std::strlen( tail ) ) == 0;
			}
			if ( !split ) {
				LOG_ERROR( "[SafeTensorsRepo] %s holds %zu GGUF files that are not the parts of one split model; keep one model\n", dir_path, ggufs.size() );
				return false;
			}
		}
		if ( !ggufs.empty() ) {
			for ( size_t i = 0; i < ggufs.size(); ++i ) {
				char full_path[1024];
				std::snprintf( full_path, sizeof( full_path ), "%s/%s", dir_path, ggufs[i].c_str() );
				if ( !engine::loader::gguf_open( shards_[i], full_path, i > 0 ? &shards_[0] : nullptr ) ) return false;
				std::strncpy( shard_names_[i], ggufs[i].c_str(), sizeof( shard_names_[i] ) - 1 );
				num_shards_ = static_cast<int>( i ) + 1;
				for ( int t = 0; t < shards_[i].tensor_count; ++t ) tensor_map_[shards_[i].tensors[t].name] = { static_cast<int>( i ), &shards_[i].tensors[t] };
			}
			LOG_INFO( "[SafeTensorsRepo] GGUF %s (%zu files), mapped %zu tensors\n", ggufs[0].c_str(), ggufs.size(), tensor_map_.size() );
			return true;
		}
	}

	if ( mmproj ) return false;   // no projector file

	// 4. Fallback: single model.safetensors
	if ( shard_files.empty() ) {
		shard_files.push_back( "model.safetensors" );
	}

	// 5. Open each shard and populate tensor_map_
	for ( const auto& fname : shard_files ) {
		if ( num_shards_ >= MAX_SHARDS ) {
			LOG_ERROR( "[SafeTensorsRepo] %s has more than %d shard files; the rest are ignored\n", dir_path, MAX_SHARDS );
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

	// Tensors a checkpoint keeps outside its index: GLM-5.3 ships the DSA k-pool gate / APE in kpool_aux.safetensors, exllamav3's DeepSeek-V4.1
	// conversions the Engram tables in ngram_embedding.safetensors, the published Flash-Next EXL3 branches the vision tower in vision_k6.safetensors (4.05 bpw)
	// and the MTP head's hyper-connection mixer in mtp_hyper_connection_mixer_patch.safetensors (3.05 bpw)
	for ( const char* aux : { "kpool_aux.safetensors", "ngram_embedding.safetensors", "vision_k6.safetensors", "mtp_hyper_connection_mixer_patch.safetensors" } ) {
		if ( std::find( shard_files.begin(), shard_files.end(), std::string( aux ) ) != shard_files.end() ) continue;
		char aux_path[1024];
		std::snprintf( aux_path, sizeof( aux_path ), "%s/%s", dir_path, aux );
		FILE* fa = ( num_shards_ < MAX_SHARDS ) ? std::fopen( aux_path, "rb" ) : nullptr;
		if ( fa != nullptr ) {
			std::fclose( fa );
			if ( safetensors_open( shards_[num_shards_], aux_path ) ) {
				std::strncpy( shard_names_[num_shards_], aux, sizeof( shard_names_[0] ) - 1 );
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
	meta = engine::loader::gguf_float_view( shards_[shard_idx], meta, stream );   // a GGUF block tensor read as floats: dequantized once
	FloatView fp8_hold;
	meta = fp8_float_view( name, meta, fp8_hold );   // an FP8 block-scaled weight read as floats

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
	if ( tv.data == nullptr || arena.stream_skip ) {
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
	else if ( ( meta->dtype == pod::DType::BF16 || meta->dtype == pod::DType::FP16 ) && out_dtype == pod::DType::FP32 ) {
		// the original GLM checkpoint's mHC mixers are BF16 where its conversions hold FP32
		std::vector<float> h_f32( total_elems );
		for ( size_t i = 0; i < total_elems; ++i ) {
			const uint16_t v = static_cast<const uint16_t*>( src_cpu )[i];
			if ( meta->dtype == pod::DType::BF16 ) {
				const uint32_t u = static_cast<uint32_t>( v ) << 16;
				std::memcpy( &h_f32[i], &u, sizeof( float ) );
			}
			else h_f32[i] = backend::to_float( static_cast<const backend::half_t*>( src_cpu )[i] );
		}
		backend::memcpy_async( tv.data, h_f32.data(), total_elems * sizeof( float ), backend::Copy::Auto, stream );
		backend::stream_sync( stream );
	}
	else if ( ( meta->dtype == pod::DType::FP32 || meta->dtype == pod::DType::FP16 ) && out_dtype == pod::DType::BF16 ) {
		// GGUF embeddings stored as F32 / F16, or dequantized from blocks: round to BF16 (nearest even)
		std::vector<uint16_t> h_bf16( total_elems );
		for ( size_t i = 0; i < total_elems; ++i ) {
			float f = meta->dtype == pod::DType::FP32 ? static_cast<const float*>( src_cpu )[i] : backend::to_float( static_cast<const backend::half_t*>( src_cpu )[i] );
			uint32_t u;
			std::memcpy( &u, &f, sizeof( u ) );
			h_bf16[i] = static_cast<uint16_t>( ( u + 0x7FFFu + ( ( u >> 16 ) & 1u ) ) >> 16 );
		}
		backend::memcpy_async( tv.data, h_bf16.data(), total_elems * sizeof( uint16_t ), backend::Copy::Auto, stream );
		backend::stream_sync( stream );
	}
	else {
		// same-width types copied as they are (raw codes); a width change has no conversion here and would copy the wrong bytes
		if ( get_dtype_size( out_dtype ) != get_dtype_size( meta->dtype ) ) {
			LOG_ERROR( "[SafeTensorsRepo] %s: no conversion from its dtype to the one requested\n", name );
			ENGINE_ASSERT( false, "SafeTensorsRepo::load: unsupported dtype conversion" );
		}
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
	meta = engine::loader::gguf_float_view( shards_[shard_idx], meta, stream );
	FloatView fp8_hold;
	meta = fp8_float_view( name, meta, fp8_hold );

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
			else if ( meta->dtype == pod::DType::FP32 && out_dtype == pod::DType::FP16 ) {
				std::vector<backend::half_t> h_buf( total_local );
				const float* src_f32 = static_cast<const float*>( src_cpu );
				for ( int r = 0; r < rows; ++r ) {
					for ( int c = 0; c < local_cols; ++c ) {
						h_buf[r * local_cols + c] = f32_to_fp16( src_f32[static_cast<size_t>( r ) * cols + rank * local_cols + c] );
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

bool SafeTensorsRepo::file_ref( const char* tensor_name, int* fd, size_t* offset, size_t* bytes ) const {
	int shard = -1;
	const pod::SafeTensorMeta* meta = find( tensor_name, &shard );
	if ( meta == nullptr || shard < 0 || shards_[shard].fd < 0 ) return false;
	*fd = shards_[shard].fd;
	*offset = meta->data_offset_bytes;
	*bytes = meta->size_bytes;
	return true;
}

size_t SafeTensorsRepo::bytes_of( const std::string& prefix, std::initializer_list<const char*> skip ) const {
	size_t total = 0;
	for ( const auto& kv : tensor_map_ ) {
		const std::string& name = kv.first;
		if ( name.compare( 0, prefix.size(), prefix ) != 0 ) continue;
		bool skipped = false;
		for ( const char* s : skip ) skipped = skipped || name.find( s ) != std::string::npos;
		if ( !skipped ) total += static_cast<size_t>( kv.second.second->size_bytes );
	}
	return total;
}

bool SafeTensorsRepo::copy_host_bf16( const char* name, uint16_t* dst, size_t elems, backend::stream_t stream ) {
	int shard = -1;
	const auto* meta = find( name, &shard );
	if ( meta == nullptr || shard < 0 || meta->ndim < 1 ) return false;
	meta = engine::loader::gguf_float_view( shards_[shard], meta, stream );
	const void* src = safetensors_get_data_cpu( shards_[shard], meta );
	size_t n = 1;
	for ( int d = 0; d < meta->ndim; ++d ) n *= static_cast<size_t>( meta->shape[d] );
	if ( src == nullptr || n != elems ) return false;
	if ( meta->dtype == pod::DType::BF16 ) std::memcpy( dst, src, n * sizeof( uint16_t ) );
	else {
		ENGINE_ASSERT( meta->dtype == pod::DType::FP16 || meta->dtype == pod::DType::FP32, "copy_host_bf16: BF16, FP16, FP32 or GGUF tensors" );
		for ( size_t i = 0; i < n; ++i ) {
			// round to nearest even
			float f = meta->dtype == pod::DType::FP32 ? static_cast<const float*>( src )[i] : backend::to_float( static_cast<const backend::half_t*>( src )[i] );
			uint32_t u;
			std::memcpy( &u, &f, sizeof( u ) );
			dst[i] = static_cast<uint16_t>( ( u + 0x7FFFu + ( ( u >> 16 ) & 1u ) ) >> 16 );
		}
	}
	return true;
}

namespace {

// A token embedding stored as a trellis (exllamav3 1.6: <key>.trellis [rows, groups * (1 + 16 K)] int16 and <key>.signs [groups, 256] FP16): a row is `groups` rings of
// 256 values, each a ring scale (FP16) and 16 K words of K-bit symbols read through the MUL1 codebook like the Engram rows (a state stacks the symbols of the positions i, i - 1,
// ..., modulo the ring), then an orthonormal Walsh-Hadamard transform over the ring and the sign vector of its group. Decoded to BF16 rows of groups * 256 columns.
bool decode_trellis_embedding( const uint16_t* trellis, int64_t rows, int row_words, const uint16_t* signs, int groups, uint16_t* out ) {
	if ( groups <= 0 || row_words % groups != 0 ) return false;
	const int ring_words = row_words / groups, K = ( ring_words - 1 ) / 16;
	if ( K < 1 || K > 8 || ring_words != 1 + 16 * K ) return false;
	constexpr int kDim = 256;
	const int words = 16 * K, mask = ( 1 << K ) - 1, taps = ( 15 + K ) / K;
	auto half_f = []( uint16_t b ) { _Float16 h; std::memcpy( &h, &b, 2 ); return static_cast<float>( h ); };
	std::vector<float> codebook( 65536 ), sign( static_cast<size_t>( groups ) * kDim );
	{
		const float k_inv = half_f( 0x1eee ), k_bias = half_f( 0xc931 );
		for ( uint32_t st = 0; st < 65536; ++st ) {
			const uint32_t prod = st * 0x83DCD12Du;
			const float h = static_cast<float>( 1024 + ( prod & 255 ) + ( ( prod >> 8 ) & 255 ) + ( ( prod >> 16 ) & 255 ) + ( prod >> 24 ) );
			volatile float t = h * k_inv;
			const _Float16 r = static_cast<_Float16>( t + k_bias );
			codebook[st] = static_cast<float>( r );
		}
		for ( size_t i = 0; i < sign.size(); ++i ) sign[i] = half_f( signs[i] );
	}
	const int cols = groups * kDim;
	std::atomic<int64_t> next{ 0 };
	auto work = [&]() {
		std::vector<float> v( kDim );
		std::vector<int> sym( kDim );
		for ( ;; ) {
			const int64_t r0 = next.fetch_add( 64 );
			if ( r0 >= rows ) break;
			for ( int64_t r = r0; r < std::min<int64_t>( r0 + 64, rows ); ++r ) {
				for ( int g = 0; g < groups; ++g ) {
					const uint16_t* ring = trellis + static_cast<size_t>( r ) * row_words + static_cast<size_t>( g ) * ring_words;
					const float scale = half_f( ring[0] );
					const uint16_t* w = ring + 1;
					for ( int i = 0; i < kDim; ++i ) {
						const int bit = i * K, idx = bit >> 4;
						const uint32_t win = static_cast<uint32_t>( w[idx] ) | ( static_cast<uint32_t>( w[( idx + 1 ) % words] ) << 16 );
						sym[i] = static_cast<int>( ( win >> ( bit & 15 ) ) & mask );
					}
					for ( int i = 0; i < kDim; ++i ) {
						uint32_t st = 0;
						for ( int j = 0; j < taps; ++j ) st |= static_cast<uint32_t>( sym[( i - j + kDim ) % kDim] ) << ( j * K );
						v[i] = codebook[st & 0xFFFF] * scale;
					}
					for ( int h = 1; h < kDim; h *= 2 )
						for ( int b = 0; b < kDim; b += 2 * h )
							for ( int k = 0; k < h; ++k ) {
								const float a = v[b + k], c = v[b + k + h];
								v[b + k] = a + c;
								v[b + k + h] = a - c;
							}
					uint16_t* dst = out + static_cast<size_t>( r ) * cols + static_cast<size_t>( g ) * kDim;
					for ( int i = 0; i < kDim; ++i ) {
						const float f = v[i] * 0.0625f * sign[static_cast<size_t>( g ) * kDim + i];
						uint32_t u;
						std::memcpy( &u, &f, 4 );
						u += 0x7fffu + ( ( u >> 16 ) & 1u );
						dst[i] = static_cast<uint16_t>( u >> 16 );
					}
				}
			}
		}
	};
	const unsigned n = std::max( 1u, std::min( 16u, std::thread::hardware_concurrency() ) );
	std::vector<std::thread> pool;
	for ( unsigned t = 1; t < n; ++t ) pool.emplace_back( work );
	work();
	for ( auto& t : pool ) t.join();
	return true;
}

} // namespace

pod::TensorView SafeTensorsRepo::shared_host_bf16( const char* name, backend::stream_t stream ) {
	static std::mutex mu;
	static std::map<std::string, pod::TensorView> tables;
	std::lock_guard<std::mutex> lk( mu );
	if ( const auto it = tables.find( name ); it != tables.end() ) return it->second;
	const auto* meta = find( name );
	// a table stored as a trellis (<base>.trellis + <base>.signs) is decoded to BF16 here
	std::string base( name );
	if ( base.size() > 7 && base.compare( base.size() - 7, 7, ".weight" ) == 0 ) base.resize( base.size() - 7 );
	const auto* tm = meta == nullptr ? find( ( base + ".trellis" ).c_str() ) : nullptr;
	const auto* sm = tm != nullptr ? find( ( base + ".signs" ).c_str() ) : nullptr;
	if ( meta == nullptr && !( tm != nullptr && sm != nullptr && tm->ndim == 2 && sm->ndim == 2 ) ) return pod::TensorView{};
	if ( meta != nullptr && meta->ndim != 2 ) return pod::TensorView{};
	const int rows = meta != nullptr ? meta->shape[0] : tm->shape[0], cols = meta != nullptr ? meta->shape[1] : sm->shape[0] * 256;
	const size_t n = static_cast<size_t>( rows ) * cols;
	auto* dst = static_cast<uint16_t*>( backend::host_alloc_pinned( n * sizeof( uint16_t ), backend::Pin::MappedPortable ) );
	if ( dst == nullptr ) {
		// a backend without mapped host memory the device reads (Vulkan): the caller keeps its own copy
		LOG_WARN( "[SafeTensorsRepo] %s: no mapped pinned host memory on this backend, kept on the device instead\n", name );
		return pod::TensorView{};
	}
	bool ok;
	if ( meta != nullptr ) ok = copy_host_bf16( name, dst, n, stream );
	else {
		const auto t0 = std::chrono::steady_clock::now();
		ok = decode_trellis_embedding( static_cast<const uint16_t*>( cpu_data( ( base + ".trellis" ).c_str() ) ), rows, static_cast<int>( tm->shape[1] ),
			static_cast<const uint16_t*>( cpu_data( ( base + ".signs" ).c_str() ) ), static_cast<int>( sm->shape[0] ), dst );
		if ( ok ) LOG_INFO( "[SafeTensorsRepo] %s decoded from its trellis: %d rows x %d columns in %.1f s\n", base.c_str(), rows, cols, std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count() );
	}
	if ( !ok ) {
		backend::host_free_pinned( dst );
		return pod::TensorView{};
	}
	const pod::TensorView t = pod::make_tensor_2d( dst, pod::DType::BF16, rows, cols );
	tables[name] = t;
	return t;
}

size_t SafeTensorsRepo::trellis_bytes( const char* base_name ) const {
	char name_buf[512];
	std::snprintf( name_buf, sizeof( name_buf ), "%s.trellis", base_name );
	const pod::SafeTensorMeta* meta = find( name_buf );
	if ( meta == nullptr ) {
		// a GGUF weight: its blocks are the codes; an FP8 one: its bytes
		std::snprintf( name_buf, sizeof( name_buf ), "%s.weight", base_name );
		meta = find( name_buf );
		if ( meta == nullptr || ( meta->dtype != pod::DType::GGUF && !fp8_block_weight( base_name ) ) ) return 0;
	}
	return static_cast<size_t>( meta->size_bytes );
}

namespace {

// Element i of a block-scale tensor (F32 or BF16)
float block_scale( const pod::SafeTensorMeta* m, const void* data, size_t i ) {
	if ( m->dtype == pod::DType::FP32 ) return static_cast<const float*>( data )[i];
	const uint32_t bits = static_cast<uint32_t>( static_cast<const uint16_t*>( data )[i] ) << 16;
	float f;
	std::memcpy( &f, &bits, 4 );
	return f;
}

// The 256 E4M3 values
const float* e4m3_table() {
	static const std::array<float, 256> t = [] {
		std::array<float, 256> v{};
		for ( int b = 0; b < 256; ++b ) v[b] = engine::quant::e4m3_to_float( static_cast<unsigned>( b ) );
		return v;
	}();
	return t.data();
}

} // namespace

bool SafeTensorsRepo::fp8_block_weight( const char* base_name ) const {
	char name_buf[512];
	std::snprintf( name_buf, sizeof( name_buf ), "%s.weight", base_name );
	const pod::SafeTensorMeta* w = find( name_buf );
	if ( w == nullptr || w->dtype != pod::DType::UINT8 || w->ndim != 2 ) return false;
	std::snprintf( name_buf, sizeof( name_buf ), "%s.weight_scale_inv", base_name );
	const pod::SafeTensorMeta* s = find( name_buf );
	return s != nullptr && s->ndim == 2 && ( s->dtype == pod::DType::FP32 || s->dtype == pod::DType::BF16 ) && s->shape[0] == ( w->shape[0] + 127 ) / 128 &&
		s->shape[1] == ( w->shape[1] + 127 ) / 128;
}

const pod::SafeTensorMeta* SafeTensorsRepo::fp8_float_view( const char* name, const pod::SafeTensorMeta* meta, FloatView& hold ) const {
	const size_t len = std::strlen( name );
	if ( meta == nullptr || meta->dtype != pod::DType::UINT8 || len < 7 || std::strcmp( name + len - 7, ".weight" ) != 0 ) return meta;
	const std::string base( name, len - 7 );
	if ( !fp8_block_weight( base.c_str() ) ) return meta;
	const std::string sname = base + ".weight_scale_inv";
	const pod::SafeTensorMeta* sm = find( sname.c_str() );
	const uint8_t* q = static_cast<const uint8_t*>( cpu_data( name ) );
	const void* s = cpu_data( sname.c_str() );
	ENGINE_ASSERT( q != nullptr && s != nullptr, "fp8_float_view: no CPU data for the weight or its scales" );
	const size_t n = static_cast<size_t>( meta->shape[0] ), k = static_cast<size_t>( meta->shape[1] ), sk = static_cast<size_t>( sm->shape[1] );
	const float* lut = e4m3_table();
	hold.values.resize( n * k );
	for ( size_t r = 0; r < n; ++r ) {
		const uint8_t* qr = q + r * k;
		float* out = hold.values.data() + r * k;
		for ( size_t c0 = 0; c0 < k; c0 += 128 ) {
			const float d = block_scale( sm, s, ( r / 128 ) * sk + c0 / 128 );
			for ( size_t c = c0; c < std::min( k, c0 + 128 ); ++c ) out[c] = lut[qr[c]] * d;
		}
	}
	hold.meta = *meta;
	hold.meta.dtype = pod::DType::FP32;
	hold.meta.size_bytes = n * k * sizeof( float );
	hold.meta.host = hold.values.data();
	return &hold.meta;
}

pod::LinearWeight SafeTensorsRepo::load_fp8_slice( const char* base_name, pod::MemoryArena& arena, bool split_n, int rank, int parts, backend::stream_t stream,
	pod::MemoryArena* code_arena ) {
	if ( !fp8_block_weight( base_name ) ) return pod::LinearWeight{};
	char wname[512], sname[512];
	std::snprintf( wname, sizeof( wname ), "%s.weight", base_name );
	std::snprintf( sname, sizeof( sname ), "%s.weight_scale_inv", base_name );
	int shard = -1;
	const pod::SafeTensorMeta* wm = find( wname, &shard );
	const pod::SafeTensorMeta* sm = find( sname );
	const int n = wm->shape[0], k = wm->shape[1];
	ENGINE_ASSERT( n % 128 == 0 && k % 128 == 0, "load_fp8_slice: the weight must be whole 128 x 128 blocks" );
	const int nl = split_n ? n / parts : n, kl = split_n ? k : k / parts;
	ENGINE_ASSERT( ( split_n ? n : k ) % ( parts * 128 ) == 0, "load_fp8_slice: a rank's slice must be whole 128-wide blocks" );
	const int n0 = split_n ? rank * nl : 0, k0 = split_n ? 0 : rank * kl;
	pod::MemoryArena& ca = code_arena ? *code_arena : arena;
	void* codes = engine::core::arena_alloc( ca, static_cast<size_t>( nl ) * kl, 256 );
	ENGINE_ASSERT( codes != nullptr, "load_fp8_slice: arena full" );
	if ( !ca.stream_skip ) {
		const uint8_t* src = static_cast<const uint8_t*>( safetensors_get_data_cpu( shards_[shard], wm ) );
		ENGINE_ASSERT( src != nullptr, "load_fp8_slice: no CPU data for the weight" );
		if ( kl == k ) backend::memcpy_async( codes, src + static_cast<size_t>( n0 ) * k, static_cast<size_t>( nl ) * k, backend::Copy::Auto, stream );
		else {
			std::vector<uint8_t> h( static_cast<size_t>( nl ) * kl );
			for ( int r = 0; r < nl; ++r ) std::memcpy( h.data() + static_cast<size_t>( r ) * kl, src + static_cast<size_t>( r ) * k + k0, kl );
			backend::memcpy_async( codes, h.data(), h.size(), backend::Copy::Auto, stream );
		}
		backend::stream_sync( stream );
	}
	const void* sdata = cpu_data( sname );
	ENGINE_ASSERT( sdata != nullptr, "load_fp8_slice: no CPU data for the scales" );
	const int sn = nl / 128, sk = kl / 128, sk_all = k / 128;
	std::vector<float> hs( static_cast<size_t>( sn ) * sk );
	for ( int i = 0; i < sn; ++i )
		for ( int j = 0; j < sk; ++j ) hs[static_cast<size_t>( i ) * sk + j] = block_scale( sm, sdata, static_cast<size_t>( n0 / 128 + i ) * sk_all + k0 / 128 + j );
	const int sshape[2] = { sn, sk };
	pod::TensorView scales = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, sshape, 2 );
	ENGINE_ASSERT( scales.data != nullptr, "load_fp8_slice: arena full" );
	backend::memcpy_async( scales.data, hs.data(), hs.size() * sizeof( float ), backend::Copy::Auto, stream );
	backend::stream_sync( stream );
	return pod::LinearWeight::make_fp8( pod::make_tensor_2d( codes, pod::DType::UINT8, kl, nl ), scales );
}


namespace {

// The [k/16, n/16, 16 * bits] shape the quantization_config.json of `dir` lists for the trellis of `base_name` (an expert tensor: the entry of expert 0
// of the same layer and projection stands for all of them, they share a width). Searches the file's text; nothing is parsed.
bool synth_trellis_shape( const char* dir, const char* base_name, std::array<int, 3>& shape ) {
	static std::map<std::string, std::array<int, 3>> cache;
	std::string key = base_name;
	const size_t at = key.find( ".experts." );
	if ( at != std::string::npos ) {
		const size_t d0 = at + 9;
		size_t d1 = d0;
		while ( d1 < key.size() && std::isdigit( static_cast<unsigned char>( key[d1] ) ) ) ++d1;
		key.replace( d0, d1 - d0, "0" );
	}
	if ( auto it = cache.find( key ); it != cache.end() ) { shape = it->second; return true; }
	char path[1200];
	std::snprintf( path, sizeof( path ), "%s/quantization_config.json", dir );
	const int fd = ::open( path, O_RDONLY );
	if ( fd < 0 ) return false;
	struct stat st;
	if ( fstat( fd, &st ) != 0 || st.st_size <= 0 ) { ::close( fd ); return false; }
	void* map = mmap( nullptr, static_cast<size_t>( st.st_size ), PROT_READ, MAP_PRIVATE, fd, 0 );
	::close( fd );
	if ( map == MAP_FAILED ) return false;
	const std::string needle = "\"" + key + ".trellis\"";
	const char* text = static_cast<const char*>( map );
	const size_t n = static_cast<size_t>( st.st_size );
	bool ok = false;
	if ( const void* hit = memmem( text, n, needle.data(), needle.size() ) ) {
		const char* p = static_cast<const char*>( hit ) + needle.size();
		const char* end = text + n;
		const char* sh = static_cast<const char*>( memmem( p, static_cast<size_t>( std::min<ptrdiff_t>( end - p, 400 ) ), "\"shape\"", 7 ) );
		if ( sh != nullptr ) {
			const char* q = static_cast<const char*>( memchr( sh, '[', 40 ) );
			if ( q != nullptr && std::sscanf( q, "[%d, %d, %d]", &shape[0], &shape[1], &shape[2] ) == 3 ) { cache[key] = shape; ok = true; }
		}
	}
	munmap( map, n );
	return ok;
}

} // namespace

size_t SafeTensorsRepo::synth_trellis_bytes( const char* base_name ) const {
	std::array<int, 3> sh;
	if ( !synth_trellis_shape( dir_path_, base_name, sh ) ) return 0;
	return static_cast<size_t>( sh[0] ) * sh[1] * sh[2] * 2;
}

pod::LinearWeight SafeTensorsRepo::synth_linear( const char* base_name, pod::MemoryArena& arena, backend::stream_t stream, pod::MemoryArena* code_arena ) const {
	std::array<int, 3> sh;
	if ( !synth_trellis_shape( dir_path_, base_name, sh ) ) return pod::LinearWeight{};
	const int bits = engine::modules::trellis_code_from_u16( sh[2] ), k = sh[0] * 16, n = sh[1] * 16;
	const size_t code_bytes = static_cast<size_t>( sh[0] ) * sh[1] * sh[2] * 2;
	static std::vector<uint16_t> pattern;   // one pseudo-random buffer, copied into every generated expert (the decode cost does not depend on the values)
	if ( pattern.size() * 2 < code_bytes ) {
		pattern.resize( code_bytes / 2 );
		uint64_t x = 0x9E3779B97F4A7C15ull;
		for ( auto& v : pattern ) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; v = static_cast<uint16_t>( x >> 24 ); }
	}
	// Signs for suh (+-1) and a small scale for svh (+-0.02: unit-variance codes with +-1 on both sides would make every expert output as large as
	// the hidden state, and the generated model would overflow to NaN, whose argmax is not a token id)
	static std::vector<uint16_t> signs, small;
	if ( signs.size() < static_cast<size_t>( std::max( k, n ) ) ) {
		signs.resize( std::max( k, n ) );
		small.resize( signs.size() );
		for ( size_t i = 0; i < signs.size(); ++i ) {
			const bool neg = ( ( i * 2654435761u ) >> 7 ) & 1;
			signs[i] = neg ? 0xBC00 : 0x3C00;    // -1 / +1 in FP16
			small[i] = neg ? 0xA51E : 0x251E;    // -0.02 / +0.02
		}
	}
	int tshape[3] = { sh[0], sh[1], sh[2] };
	pod::TensorView tr = engine::core::arena_alloc_tensor( code_arena ? *code_arena : arena, pod::DType::UINT16, tshape, 3 );
	const int kshape[1] = { k }, nshape[1] = { n };
	pod::TensorView suh = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, kshape, 1 );
	pod::TensorView svh = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, nshape, 1 );
	if ( tr.data == nullptr || suh.data == nullptr || svh.data == nullptr ) return pod::LinearWeight{};
	backend::memcpy_async( tr.data, pattern.data(), code_bytes, backend::Copy::H2D, stream );
	backend::memcpy_async( suh.data, signs.data(), static_cast<size_t>( k ) * 2, backend::Copy::H2D, stream );
	backend::memcpy_async( svh.data, small.data(), static_cast<size_t>( n ) * 2, backend::Copy::H2D, stream );
	backend::stream_sync( stream );   // the host buffers are reused by the next call
	return tiles( pod::LinearWeight::make_trellis( tr, suh, svh, bits, true, false ), base_name );
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

pod::LinearWeight SafeTensorsRepo::gguf_linear( int shard, const pod::SafeTensorMeta* meta, pod::MemoryArena& arena,
	const std::vector<std::pair<int, int>>& rows, const std::vector<std::pair<int, int>>& cols, int pad_rows, backend::stream_t stream, pod::MemoryArena* code_arena ) {
	const auto type = static_cast<engine::modules::GgmlType>( meta->gguf_type );
	const engine::modules::GgmlBlock blk = engine::modules::ggml_block( type );
	if ( !engine::modules::gguf_linear_supported( type ) ) {
		LOG_ERROR( "[SafeTensorsRepo] %s: GGUF type %s has no matrix kernel\n", meta->name, engine::modules::ggml_type_name( type ) );
		ENGINE_ASSERT( false, "gguf_linear: unsupported GGUF block type" );
	}
	engine::modules::LinearGguf::prepare();
	const size_t row_bytes = static_cast<size_t>( meta->shape[1] / blk.elems ) * blk.bytes;
	int n_local = pad_rows, k_local = 0;
	for ( const auto& r : rows ) n_local += r.second;
	for ( const auto& c : cols ) {
		ENGINE_ASSERT( c.first % blk.elems == 0 && c.second % blk.elems == 0, "gguf_linear: a column range splits a block" );
		k_local += c.second;
	}
	const size_t local_row_bytes = static_cast<size_t>( k_local / blk.elems ) * blk.bytes;
	const uint8_t* src = static_cast<const uint8_t*>( safetensors_get_data_cpu( shards_[shard], meta ) );
	ENGINE_ASSERT( src != nullptr, "gguf_linear: no CPU data for the tensor" );
	std::vector<uint8_t> h( static_cast<size_t>( n_local ) * local_row_bytes, 0 );
	uint8_t* dst = h.data();
	for ( const auto& r : rows ) {
		for ( int i = r.first; i < r.first + r.second; ++i ) {
			for ( const auto& c : cols ) {
				const size_t bytes = static_cast<size_t>( c.second / blk.elems ) * blk.bytes;
				std::memcpy( dst, src + static_cast<size_t>( i ) * row_bytes + static_cast<size_t>( c.first / blk.elems ) * blk.bytes, bytes );
				dst += bytes;
			}
		}
	}
	// 64 bytes of slack past the last row: kernels that read blocks in aligned stretches may run a few bytes over (expert tier slots, whose
	// size is the blocks', get none: the MoE kernels read whole blocks)
	const size_t slack = code_arena ? 0 : 64;
	void* mem = engine::core::arena_alloc( code_arena ? *code_arena : arena, h.size() + slack, 256 );
	ENGINE_ASSERT( mem != nullptr, "gguf_linear: arena full" );
	pod::TensorView tv = pod::make_tensor_2d( mem, pod::DType::UINT8, n_local, static_cast<int>( local_row_bytes ) );
	backend::memcpy_async( tv.data, h.data(), h.size(), backend::Copy::Auto, stream );
	// dense weights in the backend's faster layout where it has one (routed experts keep the file's: the MoE kernels read it)
	uint32_t held = meta->gguf_type;
	if ( std::strstr( meta->name, ".experts." ) == nullptr ) held = static_cast<uint32_t>( engine::modules::LinearGguf::repack( tv.data, type, n_local, k_local, stream ) );
	backend::stream_sync( stream );
	pod::LinearWeight lw = pod::LinearWeight::make_gguf( tv, held, k_local, n_local );
	if ( meta->gguf_flags & engine::loader::GGUF_COLS_V_TILED ) {
		// the gathered columns hold k_local / (v_per_k * head_dim) key heads per v
		const engine::loader::GgufVHeads h = engine::loader::gguf_v_heads( shards_[shard] );
		lw.gguf_tile_heads = k_local / ( h.v_heads / h.k_heads * h.head_v_dim );
		lw.gguf_tile_dim = h.head_v_dim;
	}
	return lw;
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
		if ( meta->dtype == pod::DType::GGUF ) return gguf_linear( shard, meta, arena, { { 0, meta->shape[0] } }, { { 0, meta->shape[1] } }, 0, stream, code_arena );
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

	int actual_bits = ( tr.data && tr.ndim >= 3 ) ? ( engine::modules::trellis_code_from_u16( tr.shape[2] ) ) : bits;
	pod::LinearWeight lw = tiles( pod::LinearWeight::make_trellis( tr, suh, svh, actual_bits, has_mul1, has_mcg ), base_name );
	// The extended trellis' tile companions, when the tensor has them (the gains unless tile_gains is off)
	std::snprintf( name_buf, sizeof( name_buf ), "%s.tstart", base_name );
	if ( find( name_buf, &dummy_shard ) != nullptr ) lw.w3 = load( name_buf, arena, stream );
	std::snprintf( name_buf, sizeof( name_buf ), "%s.tgain", base_name );
	if ( runtime::options().tile_gains && find( name_buf, &dummy_shard ) != nullptr ) lw.w4 = load( name_buf, arena, stream );
	return lw;
}

void SafeTensorsRepo::refuse_companions( const char* base_name, const char* reader ) const {
	char name_buf[512];
	int shard = -1;
	for ( const char* ext : { ".tstart", ".tgain" } ) {
		std::snprintf( name_buf, sizeof( name_buf ), "%s%s", base_name, ext );
		if ( find( name_buf, &shard ) != nullptr ) {
			LOG_ERROR( "%s: '%s' is an extended trellis (tile starts / gains), which only load_linear reads (whole tensors)\n", reader, base_name );
			ENGINE_ASSERT( false, "an extended trellis needs a whole-tensor load" );
		}
	}
}

pod::LinearWeight SafeTensorsRepo::load_linear_slice(
	const char* base_name,
	pod::MemoryArena& arena,
	int bits,
	int slice_dim,
	int rank,
	int tp_size,
	backend::stream_t stream,
	pod::MemoryArena* code_arena
) {
	char name_buf[512];
	std::snprintf( name_buf, sizeof( name_buf ), "%s.trellis", base_name );
	char weight_buf[512];
	std::snprintf( weight_buf, sizeof( weight_buf ), "%s.weight", base_name );
	if ( find( name_buf ) == nullptr && ( tp_size > 1 || find( weight_buf ) != nullptr ) ) {
		// Unquantized checkpoint (the DFlash2 drafter's own release, an FP8 one dequantized): a dense [n, k] weight. Trellis slice_dim 1 splits
		// the output dimension n (dense dim 0), slice_dim 0 splits the input dimension k (dense dim 1).
		std::snprintf( name_buf, sizeof( name_buf ), "%s.weight", base_name );
		int shard = -1;
		const pod::SafeTensorMeta* meta = find( name_buf, &shard );
		ENGINE_ASSERT( meta != nullptr && meta->ndim == 2, "load_linear_slice: neither a trellis group nor a 2-D weight for this name" );
		if ( meta->dtype == pod::DType::GGUF ) {
			const int n = meta->shape[0], k = meta->shape[1];
			if ( slice_dim == 1 ) return gguf_linear( shard, meta, arena, { { rank * ( n / tp_size ), n / tp_size } }, { { 0, k } }, 0, stream, code_arena );
			std::vector<std::pair<int, int>> cols;
			if ( meta->gguf_flags & engine::loader::GGUF_COLS_V_TILED ) {
				// tiled input columns [v][k_head][head_dim]: this rank's key heads are one run per v
				const engine::loader::GgufVHeads h = engine::loader::gguf_v_heads( shards_[shard] );
				const int run = h.k_heads / tp_size * h.head_v_dim;
				for ( int v = 0; v < h.v_heads / h.k_heads; ++v ) cols.push_back( { v * h.k_heads * h.head_v_dim + rank * run, run } );
			}
			else cols.push_back( { rank * ( k / tp_size ), k / tp_size } );
			return gguf_linear( shard, meta, arena, { { 0, n } }, cols, 0, stream, code_arena );
		}
		pod::TensorView w = load_slice( name_buf, arena, slice_dim == 1 ? 0 : 1, rank, tp_size, stream, pod::DType::FP16 );
		return pod::LinearWeight::make_fp16( pod::make_tensor_2d( w.data, pod::DType::FP16, w.shape[1], w.shape[0] ) );
	}
	if ( const pod::SafeTensorMeta* tm = find( name_buf ); tm != nullptr && tm->ndim == 3 && tp_size > 1 ) {
		// The input and output Hadamards act on 128-wide blocks: a rank's slice has to be whole blocks (8 tiles of 16)
		ENGINE_ASSERT( tm->shape[slice_dim] % tp_size == 0 && ( tm->shape[slice_dim] / tp_size ) % 8 == 0,
			"load_linear_slice: a rank's slice of an EXL3 tensor must be whole 128-wide Hadamard blocks" );
	}
	pod::TensorView tr = load_slice( name_buf, arena, slice_dim, rank, tp_size, stream );

	std::snprintf( name_buf, sizeof( name_buf ), "%s.mul1", base_name );
	int dummy_shard = -1;
	bool has_mul1 = ( find( name_buf, &dummy_shard ) != nullptr );
	std::snprintf( name_buf, sizeof( name_buf ), "%s.mcg", base_name );
	const bool has_mcg = ( find( name_buf, &dummy_shard ) != nullptr );

	int actual_bits = ( tr.data && tr.ndim >= 3 ) ? ( engine::modules::trellis_code_from_u16( tr.shape[2] ) ) : bits;

	pod::LinearWeight lw;
	if ( slice_dim == 0 ) {
		std::snprintf( name_buf, sizeof( name_buf ), "%s.suh", base_name );
		pod::TensorView suh = load_slice( name_buf, arena, 0, rank, tp_size, stream, pod::DType::FP16 );

		std::snprintf( name_buf, sizeof( name_buf ), "%s.svh", base_name );
		pod::TensorView svh = load( name_buf, arena, stream, pod::DType::FP16 );

		lw = tiles( pod::LinearWeight::make_trellis( tr, suh, svh, actual_bits, has_mul1, has_mcg ), base_name );
	}
	else {
		std::snprintf( name_buf, sizeof( name_buf ), "%s.suh", base_name );
		pod::TensorView suh = load( name_buf, arena, stream, pod::DType::FP16 );

		std::snprintf( name_buf, sizeof( name_buf ), "%s.svh", base_name );
		pod::TensorView svh = load_slice( name_buf, arena, 0, rank, tp_size, stream, pod::DType::FP16 );

		lw = tiles( pod::LinearWeight::make_trellis( tr, suh, svh, actual_bits, has_mul1, has_mcg ), base_name );
	}
	// The extended trellis' tile companions ([K / 16, N / 16], per tile like the codes' first two dimensions): the same rows or columns of tiles
	std::snprintf( name_buf, sizeof( name_buf ), "%s.tstart", base_name );
	if ( find( name_buf, &dummy_shard ) != nullptr ) lw.w3 = load_slice( name_buf, arena, slice_dim, rank, tp_size, stream );
	std::snprintf( name_buf, sizeof( name_buf ), "%s.tgain", base_name );
	if ( runtime::options().tile_gains && find( name_buf, &dummy_shard ) != nullptr ) lw.w4 = load_slice( name_buf, arena, slice_dim, rank, tp_size, stream );
	return lw;
}

pod::LinearWeight SafeTensorsRepo::load_linear_range(
	const char* base_name,
	pod::MemoryArena& arena,
	int slice_dim,
	int begin16,
	int count16,
	backend::stream_t stream,
	pod::MemoryArena* code_arena
) {
	refuse_companions( base_name, "load_linear_range" );
	char name_buf[512];
	std::snprintf( name_buf, sizeof( name_buf ), "%s.weight", base_name );
	int shard = -1;
	if ( const auto* gm = find( name_buf, &shard ); gm != nullptr && gm->dtype == pod::DType::GGUF ) {
		// a GGUF weight [n, k]: output rows or input columns (whole blocks of the type) [begin16 * 16, (begin16 + count16) * 16)
		ENGINE_ASSERT( slice_dim == 0 || slice_dim == 1, "load_linear_range: slice_dim is 0 (input) or 1 (output)" );
		const int n = gm->shape[0], k = gm->shape[1];
		if ( slice_dim == 1 ) return gguf_linear( shard, gm, arena, { { begin16 * 16, count16 * 16 } }, { { 0, k } }, 0, stream, code_arena );
		return gguf_linear( shard, gm, arena, { { 0, n } }, { { begin16 * 16, count16 * 16 } }, 0, stream, code_arena );
	}
	std::snprintf( name_buf, sizeof( name_buf ), "%s.trellis", base_name );
	const auto* meta = find( name_buf, &shard );
	if ( !meta || shard < 0 || meta->ndim != 3 ) return pod::LinearWeight{};
	const int BK = meta->shape[0], BN = meta->shape[1], B = meta->shape[2];
	ENGINE_ASSERT( slice_dim == 0 || slice_dim == 1, "load_linear_range: slice_dim is 0 (input) or 1 (output)" );
	const int total = slice_dim == 0 ? BK : BN;
	ENGINE_ASSERT( begin16 >= 0 && count16 > 0 && begin16 + count16 <= total && begin16 % 8 == 0 && count16 % 8 == 0, "load_linear_range: the range must be whole 128-wide spans inside the tensor" );
	const int shape[3] = { slice_dim == 0 ? count16 : BK, slice_dim == 1 ? count16 : BN, B };
	pod::TensorView tr = engine::core::arena_alloc_tensor( code_arena ? *code_arena : arena, pod::DType::UINT16, shape, 3 );
	if ( !tr.data ) return pod::LinearWeight{};
	const uint16_t* src = static_cast<const uint16_t*>( safetensors_get_data_cpu( shards_[shard], meta ) );
	ENGINE_ASSERT( src != nullptr, "load_linear_range: no CPU data for the trellis tensor" );
	if ( slice_dim == 0 ) {
		// whole rows: one contiguous run
		backend::memcpy_async( tr.data, src + static_cast<size_t>( begin16 ) * BN * B, static_cast<size_t>( count16 ) * BN * B * sizeof( uint16_t ), backend::Copy::Auto, stream );
	}
	else {
		std::vector<uint16_t> h( static_cast<size_t>( BK ) * count16 * B );
		for ( int k = 0; k < BK; ++k ) std::memcpy( h.data() + static_cast<size_t>( k ) * count16 * B, src + ( static_cast<size_t>( k ) * BN + begin16 ) * B, static_cast<size_t>( count16 ) * B * sizeof( uint16_t ) );
		backend::memcpy_async( tr.data, h.data(), h.size() * sizeof( uint16_t ), backend::Copy::Auto, stream );
		backend::stream_sync( stream );
	}
	// scale vectors (FP16 in the checkpoint; converted when not): the sliced one by its elements, the other whole
	auto scales = [&]( const char* suffix, bool sliced ) {
		std::snprintf( name_buf, sizeof( name_buf ), "%s.%s", base_name, suffix );
		int sh = -1;
		const auto* m = find( name_buf, &sh );
		ENGINE_ASSERT( m != nullptr && sh >= 0 && m->ndim == 1, "load_linear_range: missing scale vector" );
		const size_t first = sliced ? static_cast<size_t>( begin16 ) * 16 : 0, n = sliced ? static_cast<size_t>( count16 ) * 16 : static_cast<size_t>( m->shape[0] );
		ENGINE_ASSERT( first + n <= static_cast<size_t>( m->shape[0] ), "load_linear_range: scale vector shorter than the range" );
		const void* d = safetensors_get_data_cpu( shards_[sh], m );
		ENGINE_ASSERT( d != nullptr, "load_linear_range: no CPU data for a scale vector" );
		std::vector<backend::half_t> h( n );
		if ( m->dtype == pod::DType::FP16 ) std::memcpy( h.data(), static_cast<const backend::half_t*>( d ) + first, n * sizeof( backend::half_t ) );
		else if ( m->dtype == pod::DType::BF16 ) for ( size_t i = 0; i < n; ++i ) h[i] = bf16_to_fp16( static_cast<const uint16_t*>( d )[first + i] );
		else if ( m->dtype == pod::DType::FP32 ) for ( size_t i = 0; i < n; ++i ) h[i] = backend::from_float_half( static_cast<const float*>( d )[first + i] );
		else ENGINE_ASSERT( false, "load_linear_range: scale vector dtype" );
		const int sh1[1] = { static_cast<int>( n ) };
		pod::TensorView t = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, sh1, 1 );
		ENGINE_ASSERT( t.data != nullptr, "load_linear_range: arena full" );
		backend::memcpy_async( t.data, h.data(), n * sizeof( backend::half_t ), backend::Copy::Auto, stream );
		backend::stream_sync( stream );
		return t;
	};
	pod::TensorView suh = scales( "suh", slice_dim == 0 );
	pod::TensorView svh = scales( "svh", slice_dim == 1 );
	std::snprintf( name_buf, sizeof( name_buf ), "%s.mul1", base_name );
	int dummy = -1;
	const bool has_mul1 = find( name_buf, &dummy ) != nullptr;
	std::snprintf( name_buf, sizeof( name_buf ), "%s.mcg", base_name );
	const bool has_mcg = find( name_buf, &dummy ) != nullptr;
	return tiles( pod::LinearWeight::make_trellis( tr, suh, svh, engine::modules::trellis_code_from_u16( B ), has_mul1, has_mcg ), base_name );   // the width code (half-integer rates too)
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
	refuse_companions( base_name, "load_linear_slice_padded" );
	char name_buf[512];
	std::snprintf( name_buf, sizeof( name_buf ), "%s.weight", base_name );
	int gguf_shard = -1;
	if ( const auto* gm = find( name_buf, &gguf_shard ); gm != nullptr && gm->dtype == pod::DType::GGUF ) {
		// this rank's vocabulary rows, then the pad rows (zero blocks: exactly 0 logits)
		const int n = gm->shape[0], local = ( n + tp_size - 1 ) / tp_size;
		const int begin = std::min( n, rank * local ), count = std::min( n, begin + local ) - begin;
		return gguf_linear( gguf_shard, gm, arena, { { begin, count } }, { { 0, gm->shape[1] } }, local - count + pad_cols, stream );
	}
	std::snprintf( name_buf, sizeof( name_buf ), "%s.trellis", base_name );
	const auto* meta = find( name_buf );
	if ( !meta ) {
		// a dense [n, k] weight (an NVFP4 checkpoint keeps the head BF16): this rank's rows as FP16, then the zero pad rows
		std::snprintf( name_buf, sizeof( name_buf ), "%s.weight", base_name );
		const auto* dm = find( name_buf );
		if ( !dm || dm->ndim != 2 ) return pod::LinearWeight{};
		ENGINE_ASSERT( dm->shape[0] % tp_size == 0, "load_linear_slice_padded: the dense head's rows must divide across ranks" );
		const int local = dm->shape[0] / tp_size;
		return load_dense_rows_padded( name_buf, arena, rank * local, local, pad_cols, stream );
	}
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
	refuse_companions( base_name, "load_linear_cols_padded" );
	char name_buf[512];
	int shard = -1;
	std::snprintf( name_buf, sizeof( name_buf ), "%s.weight", base_name );
	if ( const auto* gm = find( name_buf, &shard ); gm != nullptr && gm->dtype == pod::DType::GGUF ) {
		// the output rows of the column blocks (16 per block), then the pad rows (zero blocks: exactly 0 logits)
		ENGINE_ASSERT( ( col_block_begin + col_blocks ) * 16 <= gm->shape[0], "column range outside the tensor" );
		return gguf_linear( shard, gm, arena, { { col_block_begin * 16, col_blocks * 16 } }, { { 0, gm->shape[1] } }, pad_cols, stream );
	}

	// A dense [n, k] weight (an unquantized checkpoint's head): the rows of the column blocks as FP16, then the zero pad rows
	std::snprintf( name_buf, sizeof( name_buf ), "%s.trellis", base_name );
	if ( find( name_buf ) == nullptr ) {
		std::snprintf( name_buf, sizeof( name_buf ), "%s.weight", base_name );
		return load_dense_rows_padded( name_buf, arena, col_block_begin * 16, col_blocks * 16, pad_cols, stream );
	}

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

	int actual_bits = engine::modules::trellis_code_from_u16( B );
	return tiles( pod::LinearWeight::make_trellis( tr, suh, svh, actual_bits ? actual_bits : bits, has_mul1, has_mcg ), base_name );
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
	refuse_companions( base_name_a, "load_linear_slice_fused" );
	refuse_companions( base_name_b, "load_linear_slice_fused" );
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

	int actual_bits = ( tr.data && tr.ndim >= 3 ) ? ( engine::modules::trellis_code_from_u16( tr.shape[2] ) ) : bits;
	return tiles( pod::LinearWeight::make_trellis( tr, suh, svh, actual_bits ), base_name_a );
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
	refuse_companions( base_name, "load_grouped_trellis_slice" );
	char name_buf[512];
	std::snprintf( name_buf, sizeof( name_buf ), "%s.weight", base_name );
	int gguf_shard = -1;
	if ( const auto* meta = find( name_buf, &gguf_shard ); meta != nullptr && meta->dtype == pod::DType::GGUF ) {
		// the groups' output rows (16 per block), this rank's share of each
		std::vector<std::pair<int, int>> rows;
		for ( int g = 0, start = 0; g < num_groups; start += group_channel_blocks[g++] * 16 ) rows.push_back( { start + rank * rank_channel_blocks[g] * 16, rank_channel_blocks[g] * 16 } );
		return gguf_linear( gguf_shard, meta, arena, rows, { { 0, meta->shape[1] } }, 0, stream );
	}
	std::snprintf( name_buf, sizeof( name_buf ), "%s.trellis", base_name );
	if ( find( name_buf ) == nullptr ) {
		// a dense [n, k] weight (an NVFP4 checkpoint keeps it BF16): the groups' output rows, this rank's share of each, as one FP16 matrix
		std::snprintf( name_buf, sizeof( name_buf ), "%s.weight", base_name );
		const auto* dm = find( name_buf );
		if ( !dm || dm->ndim != 2 ) return pod::LinearWeight{};
		std::vector<int> group_rows( num_groups ), rank_rows( num_groups );
		for ( int g = 0; g < num_groups; ++g ) {
			group_rows[g] = group_channel_blocks[g] * 16;
			rank_rows[g] = rank_channel_blocks[g] * 16;
		}
		pod::TensorView w = load_grouped_slice( name_buf, arena, group_rows.data(), rank_rows.data(), num_groups, rank, 0, stream, pod::DType::FP16 );
		if ( !w.data ) return pod::LinearWeight{};
		std::swap( w.shape[0], w.shape[1] );   // [n, k] in memory, tagged [k, n] for make_fp16
		return pod::LinearWeight::make_fp16( w );
	}
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

	int actual_bits = ( tr.data && tr.ndim >= 3 ) ? ( engine::modules::trellis_code_from_u16( tr.shape[2] ) ) : 5;
	return tiles( pod::LinearWeight::make_trellis( tr, suh, svh, actual_bits, has_mul1, has_mcg ), base_name );
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
	meta = engine::loader::gguf_float_view( shards_[shard_idx], meta, stream );
	FloatView fp8_hold;
	meta = fp8_float_view( name, meta, fp8_hold );

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
