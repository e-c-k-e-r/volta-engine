// GGUF reader: the header, metadata and tensor table of a GGUF file (version 2 or 3) into a SafeTensorsFile, under the Hugging Face names
// of the file's architecture adapter (gguf_arch.h), with the converter's transforms undone on host copies.
#include "engine/loader/gguf.h"
#include "engine/modules/gguf.h"
#include "engine/core/log.h"
#include "engine/backend/runtime.h"
#include "engine/backend/numeric.h"
#include "gguf_arch.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace engine::loader {

namespace {

using engine::modules::GgmlType;

struct GgufState {
	std::string arch_name;
	GgufArch arch;
	GgufKv kv;
	std::deque<std::vector<uint8_t>> owned;                                          // transformed host copies the metas point at
	std::map<const pod::SafeTensorMeta*, std::unique_ptr<pod::SafeTensorMeta>> views;   // gguf_float_view's FP32 metas
	std::mutex mu;
};

GgufState* state_of( const pod::SafeTensorsFile& st ) { return static_cast<GgufState*>( st.gguf ); }

// A bounds-checked reader over the mapped header
struct Reader {
	const uint8_t* p;
	size_t size, pos = 0;
	bool ok = true;
	template <typename T> T get() {
		T v{};
		if ( pos + sizeof( T ) > size ) { ok = false; return v; }
		std::memcpy( &v, p + pos, sizeof( T ) );
		pos += sizeof( T );
		return v;
	}
	std::string str() {
		const uint64_t n = get<uint64_t>();
		if ( !ok || pos + n > size ) { ok = false; return std::string(); }
		std::string s( reinterpret_cast<const char*>( p + pos ), n );
		pos += n;
		return s;
	}
};

enum KvType : uint32_t { KV_U8 = 0, KV_I8, KV_U16, KV_I16, KV_U32, KV_I32, KV_F32, KV_BOOL, KV_STRING, KV_ARRAY, KV_U64, KV_I64, KV_F64 };

// One scalar of `type` as an integer or float (false: a string or array)
bool read_scalar( Reader& r, uint32_t type, int64_t* i, double* f, bool* is_float ) {
	*is_float = false;
	switch ( type ) {
		case KV_U8: case KV_BOOL: *i = r.get<uint8_t>(); return true;
		case KV_I8: *i = r.get<int8_t>(); return true;
		case KV_U16: *i = r.get<uint16_t>(); return true;
		case KV_I16: *i = r.get<int16_t>(); return true;
		case KV_U32: *i = r.get<uint32_t>(); return true;
		case KV_I32: *i = r.get<int32_t>(); return true;
		case KV_U64: *i = static_cast<int64_t>( r.get<uint64_t>() ); return true;
		case KV_I64: *i = r.get<int64_t>(); return true;
		case KV_F32: *f = r.get<float>(); *is_float = true; return true;
		case KV_F64: *f = r.get<double>(); *is_float = true; return true;
		default: return false;
	}
}

bool read_kv( Reader& r, GgufKv* kv ) {
	const std::string key = r.str();
	const uint32_t type = r.get<uint32_t>();
	int64_t i = 0;
	double f = 0.0;
	bool is_float = false;
	if ( type == KV_STRING ) kv->strings[key] = r.str();
	else if ( type == KV_ARRAY ) {
		const uint32_t et = r.get<uint32_t>();
		const uint64_t n = r.get<uint64_t>();
		std::vector<int64_t> ints;
		for ( uint64_t k = 0; k < n && r.ok; ++k ) {
			if ( et == KV_STRING ) r.str();
			else if ( read_scalar( r, et, &i, &f, &is_float ) ) { if ( !is_float ) ints.push_back( i ); }
			else return false;   // nested arrays: not in the files this reads
		}
		if ( !ints.empty() ) kv->int_arrays[key] = std::move( ints );
	}
	else if ( read_scalar( r, type, &i, &f, &is_float ) ) {
		if ( is_float ) kv->floats[key] = f;
		else kv->ints[key] = i;
	}
	else return false;
	return r.ok;
}

// Rows [first, first + v_heads * head_dim) of a tensor of `rows` rows of row_bytes, from the converter's tiled V-head order back to grouped
void untile_rows( std::vector<uint8_t>& data, size_t row_bytes, int first, const GgufArch& a, int head_dim ) {
	if ( a.k_heads == 0 ) return;
	const int vp = a.v_heads / a.k_heads;
	const std::vector<uint8_t> src( data.begin() + static_cast<ptrdiff_t>( first * row_bytes ),
		data.begin() + static_cast<ptrdiff_t>( ( first + a.v_heads * head_dim ) * row_bytes ) );
	for ( int kh = 0; kh < a.k_heads; ++kh ) {
		for ( int v = 0; v < vp; ++v ) {
			for ( int d = 0; d < head_dim; ++d ) {
				const size_t grouped = ( static_cast<size_t>( kh ) * vp + v ) * head_dim + d, tiled = ( static_cast<size_t>( v ) * a.k_heads + kh ) * head_dim + d;
				std::memcpy( data.data() + ( first + grouped ) * row_bytes, src.data() + tiled * row_bytes, row_bytes );
			}
		}
	}
}

} // namespace

bool gguf_open( pod::SafeTensorsFile& st, const char* path ) {
	st = pod::SafeTensorsFile{};
	st.fd = ::open( path, O_RDONLY );
	if ( st.fd < 0 ) return false;
	struct stat sb{};
	if ( fstat( st.fd, &sb ) != 0 || sb.st_size < 32 ) { ::close( st.fd ); st.fd = -1; return false; }
	st.file_size = static_cast<size_t>( sb.st_size );
	st.mmap_addr = mmap( nullptr, st.file_size, PROT_READ, MAP_SHARED, st.fd, 0 );
	if ( st.mmap_addr == MAP_FAILED ) { st.mmap_addr = nullptr; ::close( st.fd ); st.fd = -1; return false; }
	auto fail = [&]( const char* why ) {
		LOG_ERROR( "[gguf] %s: %s\n", path, why );
		safetensors_close( st );
		return false;
	};
	Reader r{ static_cast<const uint8_t*>( st.mmap_addr ), st.file_size };
	if ( r.get<uint32_t>() != 0x46554747u ) return fail( "not a GGUF file" );
	const uint32_t version = r.get<uint32_t>();
	if ( version < 2 || version > 3 ) return fail( "unsupported GGUF version" );
	const uint64_t n_tensors = r.get<uint64_t>(), n_kv = r.get<uint64_t>();
	auto* s = new GgufState();
	st.gguf = s;
	for ( uint64_t k = 0; k < n_kv; ++k ) {
		if ( !read_kv( r, &s->kv ) ) return fail( "unreadable metadata" );
	}
	s->arch_name = s->kv.strings["general.architecture"];
	bool have_arch = false;
	if ( s->arch_name == "qwen35" ) have_arch = gguf_arch_qwen35( s->kv, &s->arch );
	if ( !have_arch ) {
		LOG_ERROR( "[gguf] %s: architecture '%s' has no adapter\n", path, s->arch_name.c_str() );
		safetensors_close( st );
		return false;
	}

	struct Info { std::string name; std::vector<uint64_t> ne; uint32_t type; uint64_t offset; };
	std::vector<Info> infos( n_tensors );
	for ( auto& t : infos ) {
		t.name = r.str();
		const uint32_t nd = r.get<uint32_t>();
		if ( nd < 1 || nd > 4 ) return fail( "bad tensor rank" );
		t.ne.resize( nd );
		for ( auto& d : t.ne ) d = r.get<uint64_t>();
		t.type = r.get<uint32_t>();
		t.offset = r.get<uint64_t>();
		if ( !r.ok ) return fail( "truncated tensor table" );
	}
	const uint64_t align = static_cast<uint64_t>( s->kv.get( "general.alignment", 32 ) );
	const uint64_t data_start = ( r.pos + align - 1 ) / align * align;

	st.tensor_capacity = static_cast<int>( n_tensors ) + 1;
	st.tensors = static_cast<pod::SafeTensorMeta*>( std::calloc( st.tensor_capacity, sizeof( pod::SafeTensorMeta ) ) );
	int skipped = 0;
	for ( const Info& t : infos ) {
		GgufMapped m;
		if ( !s->arch.map( s->arch, s->kv, t.name, &m ) ) { ++skipped; continue; }
		const auto type = static_cast<GgmlType>( t.type );
		const engine::modules::GgmlBlock blk = engine::modules::ggml_block( type );
		if ( blk.elems == 0 || t.ne[0] % blk.elems != 0 ) {
			LOG_WARN( "[gguf] %s: type %s not readable, skipped\n", t.name.c_str(), engine::modules::ggml_type_name( type ) );
			++skipped;
			continue;
		}
		pod::SafeTensorMeta meta{};
		std::strncpy( meta.name, m.name.c_str(), sizeof( meta.name ) - 1 );
		meta.dtype = type == GgmlType::F32 ? pod::DType::FP32 : type == GgmlType::F16 ? pod::DType::FP16 : type == GgmlType::BF16 ? pod::DType::BF16 : pod::DType::GGUF;
		meta.gguf_type = t.type;
		// Hugging Face dimension order: outermost first (ne[0] is the innermost)
		const int nd = static_cast<int>( t.ne.size() );
		std::vector<int> shape( nd );
		for ( int d = 0; d < nd; ++d ) shape[d] = static_cast<int>( t.ne[nd - 1 - d] );
		if ( !m.shape.empty() ) {
			std::vector<int> hs = m.shape;
			size_t from = 0;
			for ( int& d : hs ) if ( d < 0 ) d = from < shape.size() ? shape[from++] : 1;
			shape = hs;
		}
		meta.ndim = static_cast<int>( shape.size() );
		for ( int d = 0; d < meta.ndim; ++d ) meta.shape[d] = shape[d];
		uint64_t rows = 1;
		for ( size_t d = 1; d < t.ne.size(); ++d ) rows *= t.ne[d];
		const size_t row_bytes = static_cast<size_t>( t.ne[0] / blk.elems * blk.bytes );
		meta.size_bytes = static_cast<size_t>( rows ) * row_bytes;
		meta.data_offset_bytes = static_cast<size_t>( data_start + t.offset );
		if ( meta.data_offset_bytes + meta.size_bytes > st.file_size ) return fail( "tensor data past the end of the file" );
		if ( m.transform == GgufTransform::ColsTiled && s->arch.k_heads > 0 ) meta.gguf_flags |= GGUF_COLS_V_TILED;
		if ( m.transform != GgufTransform::None && m.transform != GgufTransform::ColsTiled ) {
			// A host copy: rows are the outermost dimension (a 1-D tensor is rows of one value)
			const uint8_t* src = static_cast<const uint8_t*>( st.mmap_addr ) + meta.data_offset_bytes;
			std::vector<uint8_t> data( src, src + meta.size_bytes );
			const size_t n_rows = t.ne.size() == 1 ? static_cast<size_t>( t.ne[0] ) : static_cast<size_t>( rows );
			const size_t rb = meta.size_bytes / n_rows;
			if ( m.transform == GgufTransform::NormMinusOne || m.transform == GgufTransform::ALogFromNegExp ) {
				if ( type != GgmlType::F32 ) return fail( "a transformed norm / A_log tensor is not F32" );
				if ( m.transform == GgufTransform::ALogFromNegExp ) untile_rows( data, rb, 0, s->arch, 1 );
				float* v = reinterpret_cast<float*>( data.data() );
				for ( size_t i = 0; i < meta.size_bytes / sizeof( float ); ++i ) v[i] = m.transform == GgufTransform::NormMinusOne ? v[i] - 1.0f : std::log( -v[i] );
			}
			else {
				const int first = m.transform == GgufTransform::RowsUntileVPart ? m.row_offset : 0;
				if ( s->arch.k_heads > 0 && static_cast<size_t>( first + s->arch.v_heads * m.head_dim ) != n_rows ) return fail( "V-head rows do not match the tensor" );
				untile_rows( data, rb, first, s->arch, m.head_dim );
			}
			s->owned.push_back( std::move( data ) );
			meta.host = s->owned.back().data();
		}
		st.tensors[st.tensor_count++] = meta;
	}
	LOG_INFO( "[gguf] %s: architecture %s, %d tensors mapped, %d not used\n", path, s->arch_name.c_str(), st.tensor_count, skipped );
	return true;
}

void gguf_release( pod::SafeTensorsFile& st ) {
	delete state_of( st );
	st.gguf = nullptr;
}

const pod::SafeTensorMeta* gguf_float_view( pod::SafeTensorsFile& st, const pod::SafeTensorMeta* meta, backend::stream_t stream ) {
	GgufState* s = state_of( st );
	if ( s == nullptr || meta == nullptr || meta->dtype != pod::DType::GGUF ) return meta;
	std::lock_guard<std::mutex> lk( s->mu );
	auto it = s->views.find( meta );
	if ( it != s->views.end() ) return it->second.get();
	size_t elems = 1;
	for ( int d = 0; d < meta->ndim; ++d ) elems *= static_cast<size_t>( meta->shape[d] );
	// in runs of whole rows of about 64 MiB of output: the arena may hold most of the device by now
	const size_t row_elems = static_cast<size_t>( meta->shape[meta->ndim - 1] ), rows = elems / row_elems, row_bytes = meta->size_bytes / rows;
	const size_t run = std::max<size_t>( 1, std::min( rows, ( static_cast<size_t>( 64 ) << 20 ) / ( row_elems * sizeof( backend::half_t ) ) ) );
	const uint8_t* src = static_cast<const uint8_t*>( safetensors_get_data_cpu( st, meta ) );
	void* d_blocks = backend::device_malloc( run * row_bytes );
	auto* d_half = static_cast<backend::half_t*>( backend::device_malloc( run * row_elems * sizeof( backend::half_t ) ) );
	std::vector<uint8_t> f16( elems * sizeof( backend::half_t ) );
	for ( size_t r = 0; r < rows; r += run ) {
		const size_t n = std::min( run, rows - r );
		backend::memcpy_sync( d_blocks, src + r * row_bytes, n * row_bytes, backend::Copy::H2D );
		engine::modules::LinearGguf::dequant( d_blocks, static_cast<GgmlType>( meta->gguf_type ), static_cast<int64_t>( n * row_elems ), d_half, stream );
		backend::stream_sync( stream );
		backend::memcpy_sync( f16.data() + r * row_elems * sizeof( backend::half_t ), d_half, n * row_elems * sizeof( backend::half_t ), backend::Copy::D2H );
	}
	backend::device_free( d_blocks );
	backend::device_free( d_half );
	s->owned.push_back( std::move( f16 ) );
	auto view = std::make_unique<pod::SafeTensorMeta>( *meta );
	view->dtype = pod::DType::FP16;
	view->size_bytes = elems * sizeof( backend::half_t );
	view->host = s->owned.back().data();
	return ( s->views[meta] = std::move( view ) ).get();
}

GgufVHeads gguf_v_heads( const pod::SafeTensorsFile& st ) {
	GgufVHeads h;
	if ( const GgufState* s = state_of( st ) ) {
		h.k_heads = s->arch.k_heads;
		h.v_heads = s->arch.v_heads;
		h.head_v_dim = s->arch.head_v_dim;
	}
	return h;
}

std::string gguf_architecture( const pod::SafeTensorsFile& st ) {
	const GgufState* s = state_of( st );
	return s ? s->arch_name : std::string();
}

} // namespace engine::loader
