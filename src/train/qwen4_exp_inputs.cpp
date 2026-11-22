// The end-to-end tuner's per-sequence inputs from token ids, the way the engine makes them (Qwen4ExpPipeline::embed_tokens, ple_prepare,
// ple_forward): the token embeddings from the BF16 table the engine decodes into mapped host memory, gathered to FP16 rows; the PLE layer's
// n-gram rows from the hashed n-gram table (per position the n-grams ending at it inside its eos segment, positions from 0 per sequence),
// FP16, through the key and value projections into FP16 rows.
#include "engine/train/qwen4_exp.h"
#include "engine/config.h"
#include "engine/core/log.h"
#include "engine/loader/ngram_table.h"
#include "engine/loader/safetensors.h"
#include "engine/modules/embedding.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mutex>

namespace engine::train::qwen4_exp {

namespace {

constexpr int kEos = 248044;   // text_config eos_token_id: n-grams never span it

engine::loader::NgramTable& ngram_table() {
	static engine::loader::NgramTable table;
	return table;
}

cudaStream_t cu( backend::stream_t s ) { return static_cast<cudaStream_t>( s ); }

} // namespace

struct Inputs::Impl {
	engine::loader::SafeTensorsRepo repo;
	pod::TensorView table;               // BF16 [vocab, hidden], mapped host memory
	void *key_w = nullptr, *value_w = nullptr;   // FP16 [hc hidden, D], [hidden, D] on the device
	int H = 0, flat = 0, D = 0, device = 0;
	std::unique_ptr<Blas> blas;
	backend::stream_t st = nullptr;
	int* d_ids = nullptr;
	size_t ids_cap = 0;
	void* d16 = nullptr;                 // FP16 staging
	size_t d16_cap = 0;
	int* ids( size_t n ) {
		if ( n > ids_cap ) {
			if ( d_ids ) cudaFree( d_ids );
			CUDA_CHECK( cudaMalloc( &d_ids, n * sizeof( int ) ) );
			ids_cap = n;
		}
		return d_ids;
	}
	void* half_buf( size_t n ) {
		if ( n > d16_cap ) {
			if ( d16 ) cudaFree( d16 );
			CUDA_CHECK( cudaMalloc( &d16, n * 2 ) );
			d16_cap = n;
		}
		return d16;
	}
	~Impl() {
		for ( void* p : { static_cast<void*>( d_ids ), d16, key_w, value_w } ) if ( p ) cudaFree( p );
	}
};

Inputs::Inputs() : p_( std::make_unique<Impl>() ) {}
Inputs::~Inputs() = default;

bool Inputs::open( const std::string& dir, int device, backend::stream_t stream ) {
	Impl& m = *p_;
	CUDA_CHECK( cudaSetDevice( device ) );
	Config cfg;
	if ( !cfg.read( dir ) || !m.repo.open( dir.c_str() ) ) return false;
	m.device = device;
	m.st = stream;
	m.H = cfg.hidden;
	m.flat = cfg.hc * cfg.hidden;
	m.table = m.repo.shared_host_bf16( "model.language_model.embed_tokens.weight", stream );
	if ( m.table.data == nullptr ) {
		LOG_ERROR( "[train] %s: no token embedding table", dir.c_str() );
		return false;
	}
	const std::string pre = "model.language_model.layers." + std::to_string( cfg.ple_layer ) + ".ple.";
	auto load16 = [&]( const std::string& name, void** dst, int rows ) {
		const pod::SafeTensorMeta* t = m.repo.find( name.c_str() );
		if ( t == nullptr || t->dtype != pod::DType::FP16 || t->ndim != 2 || t->shape[0] != rows ) return -1;
		CUDA_CHECK( cudaMalloc( dst, t->size_bytes ) );
		CUDA_CHECK( cudaMemcpy( *dst, m.repo.cpu_data( name.c_str() ), t->size_bytes, cudaMemcpyHostToDevice ) );
		return t->shape[1];
	};
	m.D = load16( pre + "key_proj.weight", &m.key_w, m.flat );
	if ( m.D <= 0 || load16( pre + "value_proj.weight", &m.value_w, m.H ) != m.D ) {
		LOG_ERROR( "[train] %s: no FP16 PLE key / value projections", dir.c_str() );
		return false;
	}
	{
		static std::mutex mu;
		std::lock_guard<std::mutex> lk( mu );
		if ( !ngram_table().is_open() && !ngram_table().open( dir + "/ngram_embedding.safetensors" ) ) {
			LOG_ERROR( "[train] %s: no n-gram table", dir.c_str() );
			return false;
		}
	}
	if ( m.D != ngram_table().num_heads() * engine::loader::NgramTable::ROW_DIM ) {
		LOG_ERROR( "[train] the n-gram table's rows (%d) are not the PLE projections' input (%d)", ngram_table().num_heads() * engine::loader::NgramTable::ROW_DIM, m.D );
		return false;
	}
	m.blas = std::make_unique<Blas>( stream );
	return true;
}

int Inputs::ple_dim() const { return p_->D; }

void Inputs::embed( const int* ids, int n, float* out ) {
	Impl& m = *p_;
	CUDA_CHECK( cudaSetDevice( m.device ) );
	int* d = m.ids( n );
	CUDA_CHECK( cudaMemcpyAsync( d, ids, static_cast<size_t>( n ) * sizeof( int ), cudaMemcpyHostToDevice, cu( m.st ) ) );
	void* h = m.half_buf( static_cast<size_t>( n ) * m.H );
	int shape[2] = { n, m.H };
	pod::TensorView o = pod::make_tensor_2d( h, pod::DType::FP16, shape[0], shape[1] );
	engine::modules::Embedding::forward( m.table, d, n, o, m.st );
	half_to_float( h, out, static_cast<size_t>( n ) * m.H, m.st );
	CUDA_CHECK( cudaStreamSynchronize( cu( m.st ) ) );
}

void Inputs::ple_rows( const int* ids, int n, int part, int parts, uint16_t* emb ) const {
	const engine::loader::NgramTable& t = ngram_table();
	const int NH = t.num_heads(), D = p_->D, ng = t.ngram_size();
	std::vector<int> seg( n );   // the start of each position's eos segment
	for ( int pos = 0; pos < n; ++pos ) seg[pos] = pos > 0 ? ( ids[pos - 1] == kEos ? pos : seg[pos - 1] ) : 0;
	std::vector<uint64_t> uids( static_cast<size_t>( n ) * NH );
	for ( int pos = part; pos < n; pos += parts ) {
		int64_t tok[8];
		for ( int s = 0; s < ng; ++s ) tok[s] = ( s == 0 || ( pos - s >= 0 && pos - seg[pos] >= s ) ) ? ids[pos - s] : kEos;
		t.row_ids( tok, uids.data() + static_cast<size_t>( pos ) * NH );
		for ( int h = 0; h < NH; ++h ) t.prefetch_row( uids[static_cast<size_t>( pos ) * NH + h] );
	}
	float row[engine::loader::NgramTable::ROW_DIM];
	for ( int pos = part; pos < n; pos += parts ) {
		for ( int h = 0; h < NH; ++h ) {
			t.decode_row( uids[static_cast<size_t>( pos ) * NH + h], h, row );
			uint16_t* dst = emb + static_cast<size_t>( pos ) * D + static_cast<size_t>( h ) * engine::loader::NgramTable::ROW_DIM;
			for ( int i = 0; i < engine::loader::NgramTable::ROW_DIM; ++i ) dst[i] = __half_as_ushort( __float2half_rn( row[i] ) );
		}
	}
}

void Inputs::ple_project( const uint16_t* emb, int n, float* key, float* value ) {
	Impl& m = *p_;
	CUDA_CHECK( cudaSetDevice( m.device ) );
	void* e = m.half_buf( static_cast<size_t>( n ) * m.D );
	CUDA_CHECK( cudaMemcpyAsync( e, emb, static_cast<size_t>( n ) * m.D * 2, cudaMemcpyHostToDevice, cu( m.st ) ) );
	m.blas->gemm_h( false, true, n, m.flat, m.D, e, m.key_w, 0.0f, key );
	m.blas->gemm_h( false, true, n, m.H, m.D, e, m.value_w, 0.0f, value );
	round_to_half( key, static_cast<size_t>( n ) * m.flat, m.st );   // the engine's rows are FP16
	round_to_half( value, static_cast<size_t>( n ) * m.H, m.st );
	CUDA_CHECK( cudaStreamSynchronize( cu( m.st ) ) );
}

} // namespace engine::train::qwen4_exp
