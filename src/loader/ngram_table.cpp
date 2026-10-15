#include "engine/loader/ngram_table.h"
#include "engine/quant/fp4_fp8.h"
#include "engine/loader/safetensors.h"
#include "engine/core/log.h"
#include "engine/backend/numeric.h"
#include <nlohmann/json.hpp>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace engine::loader {

namespace {

constexpr uint32_t MUL1 = 0x83DCD12Du;

float bf16_bits_to_float( uint16_t bits ) {
	const uint32_t u = static_cast<uint32_t>( bits ) << 16;
	float f;
	std::memcpy( &f, &u, 4 );
	return f;
}

float half_bits_to_float( uint16_t bits ) {
	backend::half_t h;
	std::memcpy( &h, &bits, sizeof( h ) );
	return backend::to_float( h );
}

// Rounds to the nearest fp16 and back
float round_half( float v ) { return backend::to_float( backend::from_float_half( v ) ); }

} // namespace

NgramTable::~NgramTable() {
	if ( base_ ) munmap( base_, map_bytes_ );
	if ( fd_ >= 0 ) close( fd_ );
}

bool NgramTable::open( const std::string& path ) {
	fd_ = ::open( path.c_str(), O_RDONLY );
	if ( fd_ < 0 ) {
		LOG_ERROR( "NgramTable: cannot open %s\n", path.c_str() );
		return false;
	}
	struct stat st;
	fstat( fd_, &st );
	map_bytes_ = static_cast<size_t>( st.st_size );
	uint64_t header_len = 0;
	if ( pread( fd_, &header_len, 8, 0 ) != 8 || header_len > ( 64u << 20 ) ) return false;
	std::string header( header_len, '\0' );
	if ( pread( fd_, header.data(), header_len, 8 ) != static_cast<ssize_t>( header_len ) ) return false;
	const auto j = nlohmann::json::parse( header, nullptr, false );
	if ( j.is_discarded() ) return false;
	base_ = mmap( nullptr, map_bytes_, PROT_READ, MAP_PRIVATE, fd_, 0 );
	if ( base_ == MAP_FAILED ) { base_ = nullptr; return false; }
	madvise( base_, map_bytes_, MADV_RANDOM );
	const char* data = static_cast<const char*>( base_ ) + 8 + header_len;
	auto tensor = [&]( const std::string& suffix, const char** ptr, std::vector<int64_t>* shape ) {
		for ( auto it = j.begin(); it != j.end(); ++it ) {
			const std::string& k = it.key();
			if ( k.size() >= suffix.size() && k.compare( k.size() - suffix.size(), suffix.size(), suffix ) == 0 ) {
				*ptr = data + it.value()["data_offsets"][0].get<size_t>();
				if ( shape ) *shape = it.value()["shape"].get<std::vector<int64_t>>();
				return true;
			}
		}
		return false;
	};
	const auto& meta = j["__metadata__"];
	K_ = std::atoi( meta.value( "K", "6" ).c_str() );
	words_ = 1 + ROW_DIM * K_ / 16;
	const char* p = nullptr;
	std::vector<int64_t> shape;
	ENGINE_ASSERT( tensor( ".head_offsets", &p, &shape ), "NgramTable: head_offsets" );
	num_heads_ = static_cast<int>( shape[0] );
	head_offsets_.assign( reinterpret_cast<const int64_t*>( p ), reinterpret_cast<const int64_t*>( p ) + num_heads_ );
	ENGINE_ASSERT( tensor( ".head_vocab_sizes", &p, nullptr ), "NgramTable: head_vocab_sizes" );
	head_sizes_.assign( reinterpret_cast<const int64_t*>( p ), reinterpret_cast<const int64_t*>( p ) + num_heads_ );
	ENGINE_ASSERT( tensor( ".layer_multipliers", &p, &shape ), "NgramTable: layer_multipliers" );
	multipliers_.assign( reinterpret_cast<const int64_t*>( p ), reinterpret_cast<const int64_t*>( p ) + shape[0] );
	ngram_size_ = static_cast<int>( shape[0] );
	heads_per_ngram_ = num_heads_ / ( ngram_size_ - 1 );
	ENGINE_ASSERT( tensor( ".head_bias", &p, nullptr ), "NgramTable: head_bias" );
	head_bias_.resize( static_cast<size_t>( num_heads_ ) * ROW_DIM );
	for ( size_t i = 0; i < head_bias_.size(); ++i ) head_bias_[i] = half_bits_to_float( reinterpret_cast<const uint16_t*>( p )[i] );
	// shard_N.trellis, N = 0 ..
	for ( int s = 0;; ++s ) {
		char suffix[64];
		std::snprintf( suffix, sizeof( suffix ), ".shard_%d.trellis", s );
		if ( !tensor( suffix, &p, &shape ) ) break;
		if ( s == 0 ) rows_per_shard_ = shape[0];
		shards_.push_back( reinterpret_cast<const uint16_t*>( p ) );
	}
	ENGINE_ASSERT( !shards_.empty() && rows_per_shard_ > 0, "NgramTable: no shards" );
	// mul1 codebook, bit-exact with decode_3inst<2> (fp16): (1024 + byte sum of s * MUL1) * k_inv + k_bias
	const float k_inv = half_bits_to_float( 0x1eee ), k_bias = half_bits_to_float( 0xc931 );
	codebook_.resize( 65536 );
	for ( uint32_t s = 0; s < 65536; ++s ) {
		const uint32_t prod = s * MUL1;
		const uint32_t bsum = ( prod & 255 ) + ( ( prod >> 8 ) & 255 ) + ( ( prod >> 16 ) & 255 ) + ( prod >> 24 );
		const float h = static_cast<float>( 1024 + bsum );
		volatile float t = h * k_inv;
		codebook_[s] = round_half( t + k_bias );
	}
	LOG_INFO( "NgramTable: %zu shards x %lld rows, %d heads, K=%d, ngram %d\n", shards_.size(), static_cast<long long>( rows_per_shard_ ), num_heads_, K_, ngram_size_ );
	return true;
}

bool NgramTable::open_fp8( const SafeTensorsRepo& repo, const std::string& prefix ) {
	auto aux = [&]( const char* suffix, std::vector<int64_t>& out ) {
		const std::string name = prefix + suffix;
		const auto* m = repo.find( name.c_str() );
		const void* p = repo.cpu_data( name.c_str() );
		if ( !m || !p || m->ndim != 1 ) return false;
		out.assign( static_cast<const int64_t*>( p ), static_cast<const int64_t*>( p ) + m->shape[0] );
		return true;
	};
	if ( !aux( ".ngram_heads_offsets", head_offsets_ ) || !aux( ".ngram_heads_vocab_sizes", head_sizes_ ) || !aux( ".layer_multipliers", multipliers_ ) ) {
		LOG_ERROR( "NgramTable: hash parameters not found under %s\n", prefix.c_str() );
		return false;
	}
	num_heads_ = static_cast<int>( head_offsets_.size() );
	ngram_size_ = static_cast<int>( multipliers_.size() );
	heads_per_ngram_ = num_heads_ / ( ngram_size_ - 1 );
	head_bias_.assign( static_cast<size_t>( num_heads_ ) * ROW_DIM, 0.0f );
	const std::string sc = prefix + ".ngram_embedding.weight_scale";
	if ( const void* s = repo.cpu_data( sc.c_str() ) ) fp8_scale_ = bf16_bits_to_float( *static_cast<const uint16_t*>( s ) );
	for ( int s = 0;; ++s ) {
		char suffix[64];
		std::snprintf( suffix, sizeof( suffix ), ".ngram_embedding.shard_%d.weight", s );
		const std::string name = prefix + suffix;
		const auto* m = repo.find( name.c_str() );
		if ( !m ) break;
		ENGINE_ASSERT( m->ndim == 2 && m->shape[1] == ROW_DIM, "NgramTable: FP8 shard shape" );
		if ( s == 0 ) rows_per_shard_ = m->shape[0];
		const void* p = repo.cpu_data( name.c_str() );
		if ( !p ) return false;
		shards8_.push_back( static_cast<const uint8_t*>( p ) );
	}
	ENGINE_ASSERT( !shards8_.empty() && rows_per_shard_ > 0, "NgramTable: no FP8 shards" );
	fp8_ = true;
	LOG_INFO( "NgramTable: %zu FP8 shards x %lld rows, %d heads, ngram %d, scale %g\n", shards8_.size(), static_cast<long long>( rows_per_shard_ ), num_heads_, ngram_size_, fp8_scale_ );
	return true;
}

void NgramTable::row_ids( const int64_t* tokens, uint64_t* uids ) const {
	for ( int n = 2; n <= ngram_size_; ++n ) {
		const int lo = ( n - 2 ) * heads_per_ngram_;
		int64_t mixed = tokens[0] * multipliers_[0];
		for ( int s = 1; s < n; ++s ) mixed ^= tokens[s] * multipliers_[s];
		for ( int h = lo; h < lo + heads_per_ngram_; ++h ) {
			int64_t r = mixed % head_sizes_[h];
			if ( r < 0 ) r += head_sizes_[h];
			uids[h] = static_cast<uint64_t>( r + head_offsets_[h] );
		}
	}
}

void NgramTable::decode_row( uint64_t uid, int head, float* out ) const {
	const int64_t shard = static_cast<int64_t>( uid ) / rows_per_shard_;
	const int64_t local = static_cast<int64_t>( uid ) - shard * rows_per_shard_;
	if ( fp8_ ) {
		static const std::vector<float> lut = []() {
			std::vector<float> t( 256 );
			for ( int b = 0; b < 256; ++b ) t[b] = quant::e4m3_to_float( static_cast<unsigned>( b ) );
			return t;
		}();
		const uint8_t* row8 = shards8_[shard] + local * ROW_DIM;
		for ( int i = 0; i < ROW_DIM; ++i ) out[i] = lut[row8[i]] * fp8_scale_;
		(void) head;
		return;
	}
	const uint16_t* row = shards_[shard] + local * words_;
	const float scale = half_bits_to_float( row[0] );
	const uint16_t* w = row + 1;
	const float* bias = head_bias_.data() + static_cast<size_t>( head ) * ROW_DIM;
	const int total_bits = ROW_DIM * K_;
	for ( int i = 0; i < ROW_DIM; ++i ) {
		// state_i bit m lives at stream bit ((i - m / K) mod 160) * K + m % K
		uint32_t state = 0;
		for ( int m = 0; m < 16; ++m ) {
			const int idx = ( ( ( i - m / K_ ) % ROW_DIM + ROW_DIM ) % ROW_DIM ) * K_ + m % K_;
			state |= static_cast<uint32_t>( ( w[idx >> 4] >> ( idx & 15 ) ) & 1 ) << m;
		}
		out[i] = codebook_[state] * scale + bias[i];
	}
	(void) total_bits;
}

} // namespace engine::loader
