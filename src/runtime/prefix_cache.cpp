#include "engine/runtime/prefix_cache.h"
#include "engine/core/log.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>

namespace engine::runtime {

namespace {

constexpr char kDiskMagic[4] = { 'V', 'P', 'C', 'S' };
constexpr int32_t kDiskVersion = 1;

uint64_t fnv1a( const void* data, size_t n, uint64_t h = 1469598103934665603ULL ) {
	const unsigned char* p = static_cast<const unsigned char*>( data );
	for ( size_t i = 0; i < n; ++i ) h = ( h ^ p[i] ) * 1099511628211ULL;
	return h;
}

// File: char[4] "VPCS", int32 version, from, tokens, with_taps, ranks, int64 bytes per rank, int32 tokens[tokens], then every rank's bytes
struct DiskHeader {
	int32_t version, from, tokens, with_taps, ranks;
	int64_t bytes;
};

bool read_header( std::FILE* f, DiskHeader* h, std::vector<int>* tokens ) {
	char magic[4];
	if ( std::fread( magic, 1, 4, f ) != 4 || std::memcmp( magic, kDiskMagic, 4 ) != 0 ) return false;
	int32_t v[5];
	if ( std::fread( v, sizeof( int32_t ), 5, f ) != 5 || std::fread( &h->bytes, sizeof( int64_t ), 1, f ) != 1 ) return false;
	h->version = v[0]; h->from = v[1]; h->tokens = v[2]; h->with_taps = v[3]; h->ranks = v[4];
	if ( h->version != kDiskVersion || h->tokens <= 0 || h->from < 0 || h->from >= h->tokens || h->bytes <= 0 ) return false;
	tokens->resize( static_cast<size_t>( h->tokens ) );
	return std::fread( tokens->data(), sizeof( int ), tokens->size(), f ) == tokens->size();
}

}   // namespace

PrefixCache::~PrefixCache() {
	if ( disk_on() ) {
		// every node the files do not hold yet, so the next start finds the whole tree
		for ( auto& e : entries_ ) if ( !e->on_disk ) spill( e.get(), false );
		drain_writes();
		{
			std::lock_guard<std::mutex> lock( disk_mtx_ );
			writer_stop_ = true;
		}
		disk_cv_.notify_all();
		if ( writer_.joinable() ) writer_.join();
	}
	clear();
}

bool PrefixCache::enable_disk( const std::string& dir, const std::string& fingerprint, size_t disk_budget ) {
	char sub[24];
	std::snprintf( sub, sizeof( sub ), "%016llx", static_cast<unsigned long long>( fnv1a( fingerprint.data(), fingerprint.size() ) ) );
	const std::string path = dir + "/" + sub;
	std::error_code ec;
	std::filesystem::create_directories( path, ec );
	if ( ec ) {
		LOG_ERROR( "[prefix] disk tier: cannot create %s: %s\n", path.c_str(), ec.message().c_str() );
		return false;
	}
	if ( std::FILE* f = std::fopen( ( path + "/fingerprint.txt" ).c_str(), "w" ) ) {
		std::fputs( fingerprint.c_str(), f );
		std::fclose( f );
	}
	disk_dir_ = path;
	disk_budget_ = disk_budget;
	size_t bytes = 0;
	for ( const auto& de : std::filesystem::directory_iterator( path, ec ) ) {
		const std::string p = de.path().string();
		if ( de.path().extension() == ".tmp" ) { std::filesystem::remove( de.path(), ec ); continue; }   // a write the last run did not finish
		if ( de.path().extension() != ".snap" ) continue;
		std::FILE* f = std::fopen( p.c_str(), "rb" );
		if ( !f ) continue;
		DiskHeader h{};
		DiskEntry d;
		const bool ok = read_header( f, &h, &d.tokens ) && h.ranks == cluster_.tp();
		std::fclose( f );
		if ( !ok ) continue;
		d.from = h.from;
		d.with_taps = h.with_taps != 0;
		d.bytes = static_cast<size_t>( h.bytes );
		d.path = p;
		d.last_used = static_cast<uint64_t>( std::filesystem::last_write_time( de.path(), ec ).time_since_epoch().count() );
		d.ready = true;
		bytes += d.bytes * cluster_.tp();
		disk_.push_back( std::move( d ) );
	}
	writer_ = std::thread( &PrefixCache::writer_loop, this );
	LOG_INFO( "[prefix] disk tier %s: %zu snapshots, %.1f GiB (budget %.1f GiB)\n", path.c_str(), disk_.size(), bytes / 1073741824.0, disk_budget_ / 1073741824.0 );
	return true;
}

std::string PrefixCache::disk_path( const std::vector<int>& tokens ) const {
	char name[24];
	std::snprintf( name, sizeof( name ), "%016llx", static_cast<unsigned long long>( fnv1a( tokens.data(), tokens.size() * sizeof( int ) ) ) );
	return disk_dir_ + "/" + name + ".snap";
}

void PrefixCache::disk_evict_for( size_t incoming ) {
	size_t held = 0;
	for ( const DiskEntry& d : disk_ ) held += d.bytes * cluster_.tp();
	// The other model directories under the same root (other checkpoints, or this one with other options) share the budget: their
	// snapshots count by file size and age by modification time (a read refreshes it), so the oldest file goes whichever server wrote it
	struct Other { std::filesystem::path path; size_t bytes; uint64_t used; };
	std::vector<Other> others;
	std::error_code ec;
	for ( const auto& sub : std::filesystem::directory_iterator( std::filesystem::path( disk_dir_ ).parent_path(), ec ) ) {
		std::error_code ec2;
		if ( !sub.is_directory( ec2 ) || std::filesystem::equivalent( sub.path(), disk_dir_, ec2 ) ) continue;
		for ( const auto& de : std::filesystem::directory_iterator( sub.path(), ec2 ) ) {
			std::error_code ec3;
			if ( de.path().extension() != ".snap" ) continue;
			const size_t bytes = static_cast<size_t>( de.file_size( ec3 ) );
			const auto t = de.last_write_time( ec3 );
			if ( ec3 ) continue;
			others.push_back( { de.path(), bytes, static_cast<uint64_t>( t.time_since_epoch().count() ) } );
			held += bytes;
		}
	}
	while ( held + incoming > disk_budget_ ) {
		size_t victim = disk_.size();
		for ( size_t k = 0; k < disk_.size(); ++k ) {
			if ( disk_[k].ready && ( victim == disk_.size() || disk_[k].last_used < disk_[victim].last_used ) ) victim = k;
		}
		size_t other = others.size();
		for ( size_t k = 0; k < others.size(); ++k ) {
			if ( other == others.size() || others[k].used < others[other].used ) other = k;
		}
		if ( other < others.size() && ( victim == disk_.size() || others[other].used < disk_[victim].last_used ) ) {
			std::filesystem::remove( others[other].path, ec );
			held -= others[other].bytes;
			others.erase( others.begin() + static_cast<long>( other ) );
			continue;
		}
		if ( victim == disk_.size() ) return;   // only writes in flight: the budget is exceeded until they land
		std::filesystem::remove( disk_[victim].path, ec );
		held -= disk_[victim].bytes * cluster_.tp();
		disk_.erase( disk_.begin() + static_cast<long>( victim ) );
	}
}

void PrefixCache::spill( Entry* e, bool own_buffers ) {
	if ( e->on_disk || !e->snap.host[0] ) return;
	WriteJob job;
	job.tokens = e->tokens;
	job.snap = e->snap;
	job.path = disk_path( e->tokens );
	job.free_after = own_buffers;
	{
		std::lock_guard<std::mutex> lock( disk_mtx_ );
		for ( const DiskEntry& d : disk_ ) {
			if ( d.path == job.path && d.tokens == e->tokens ) { e->on_disk = true; return; }
		}
		disk_evict_for( e->snap.bytes * cluster_.tp() );
		DiskEntry d;
		d.tokens = e->tokens;
		d.from = e->snap.from;
		d.with_taps = e->snap.with_taps;
		d.bytes = e->snap.bytes;
		d.path = job.path;
		d.last_used = static_cast<uint64_t>( std::filesystem::file_time_type::clock::now().time_since_epoch().count() );
		disk_.push_back( std::move( d ) );
		writes_.push_back( std::move( job ) );
	}
	disk_cv_.notify_all();
	e->on_disk = true;
	if ( own_buffers ) for ( void*& p : e->snap.host ) p = nullptr;   // the writer frees them once the file is written
}

bool PrefixCache::write_file( const WriteJob& job ) {
	const std::string tmp = job.path + ".tmp";
	std::FILE* f = std::fopen( tmp.c_str(), "wb" );
	if ( !f ) return false;
	const int32_t v[5] = { kDiskVersion, job.snap.from, job.snap.tokens, job.snap.with_taps ? 1 : 0, cluster_.tp() };
	const int64_t bytes = static_cast<int64_t>( job.snap.bytes );
	bool ok = std::fwrite( kDiskMagic, 1, 4, f ) == 4 && std::fwrite( v, sizeof( int32_t ), 5, f ) == 5 && std::fwrite( &bytes, sizeof( int64_t ), 1, f ) == 1 &&
		std::fwrite( job.tokens.data(), sizeof( int ), job.tokens.size(), f ) == job.tokens.size();
	for ( int d = 0; ok && d < cluster_.tp(); ++d ) ok = std::fwrite( job.snap.host[d], 1, job.snap.bytes, f ) == job.snap.bytes;
	ok = std::fclose( f ) == 0 && ok;
	std::error_code ec;
	if ( ok ) std::filesystem::rename( tmp, job.path, ec );
	if ( !ok || ec ) std::filesystem::remove( tmp, ec );
	return ok && !ec;
}

void PrefixCache::writer_loop() {
	std::unique_lock<std::mutex> lock( disk_mtx_ );
	for ( ;; ) {
		disk_cv_.wait( lock, [&]() { return writer_stop_ || !writes_.empty(); } );
		if ( writes_.empty() ) return;
		WriteJob job = std::move( writes_.front() );
		writes_.pop_front();
		++writes_busy_;
		lock.unlock();
		const bool ok = write_file( job );
		if ( job.free_after ) cluster_.free_snapshot( &job.snap );
		lock.lock();
		for ( size_t k = 0; k < disk_.size(); ++k ) {
			if ( disk_[k].path != job.path || disk_[k].ready ) continue;
			if ( ok ) disk_[k].ready = true;
			else disk_.erase( disk_.begin() + static_cast<long>( k ) );
			break;
		}
		if ( !ok ) LOG_ERROR( "[prefix] disk tier: writing %s failed\n", job.path.c_str() );
		--writes_busy_;
		disk_cv_.notify_all();
	}
}

void PrefixCache::drain_writes() {
	std::unique_lock<std::mutex> lock( disk_mtx_ );
	disk_cv_.wait( lock, [&]() { return writes_.empty() && writes_busy_ == 0; } );
}

PrefixCache::Entry* PrefixCache::load_disk( size_t index, Entry* parent ) {
	DiskEntry d;
	{
		std::lock_guard<std::mutex> lock( disk_mtx_ );
		if ( index >= disk_.size() ) return nullptr;
		d = disk_[index];
	}
	ModelCluster::SeqSnapshot snap;
	if ( !cluster_.alloc_snapshot( d.from, static_cast<int>( d.tokens.size() ), d.with_taps, d.bytes, &snap ) ) return nullptr;
	std::FILE* f = std::fopen( d.path.c_str(), "rb" );
	if ( !f ) {
		// another server sharing the disk budget evicted it
		cluster_.free_snapshot( &snap );
		std::lock_guard<std::mutex> lock( disk_mtx_ );
		for ( size_t k = 0; k < disk_.size(); ++k ) {
			if ( disk_[k].path == d.path ) { disk_.erase( disk_.begin() + static_cast<long>( k ) ); break; }
		}
		LOG_INFO( "[prefix] disk tier: %s is gone (evicted by another directory's server)\n", d.path.c_str() );
		return nullptr;
	}
	DiskHeader h{};
	std::vector<int> tokens;
	bool ok = read_header( f, &h, &tokens ) && tokens == d.tokens && static_cast<size_t>( h.bytes ) == d.bytes;
	for ( int r = 0; ok && r < cluster_.tp(); ++r ) ok = std::fread( snap.host[r], 1, d.bytes, f ) == d.bytes;
	std::fclose( f );
	if ( !ok ) {
		cluster_.free_snapshot( &snap );
		LOG_ERROR( "[prefix] disk tier: reading %s failed\n", d.path.c_str() );
		return nullptr;
	}
	std::error_code ec;
	std::filesystem::last_write_time( d.path, std::filesystem::file_time_type::clock::now(), ec );   // least recently used order across restarts
	{
		std::lock_guard<std::mutex> lock( disk_mtx_ );
		for ( DiskEntry& x : disk_ ) if ( x.path == d.path ) x.last_used = static_cast<uint64_t>( std::filesystem::file_time_type::clock::now().time_since_epoch().count() );
	}
	Entry* e = insert( parent, d.tokens, snap, false );
	if ( e ) e->on_disk = true;
	return e;
}

PrefixCache::Entry* PrefixCache::promote( const std::vector<int>& prompt, Entry* ram_best ) {
	const size_t ram_len = ram_best ? ram_best->tokens.size() : 0;
	std::vector<size_t> chain;   // disk nodes to read, deepest first
	Entry* top = nullptr;        // the RAM node the chain hangs from
	{
		std::lock_guard<std::mutex> lock( disk_mtx_ );
		size_t best = disk_.size();
		for ( size_t k = 0; k < disk_.size(); ++k ) {
			const DiskEntry& d = disk_[k];
			const size_t n = d.tokens.size();
			if ( !d.ready || n >= prompt.size() || n <= ram_len || ( best < disk_.size() && n <= disk_[best].tokens.size() ) ) continue;
			if ( std::equal( d.tokens.begin(), d.tokens.end(), prompt.begin() ) ) best = k;
		}
		if ( best == disk_.size() ) return ram_best;
		for ( size_t k = best;; ) {
			chain.push_back( k );
			const DiskEntry& d = disk_[k];
			if ( d.from == 0 ) break;
			const std::vector<int> parent( d.tokens.begin(), d.tokens.begin() + d.from );
			if ( ( top = exact( parent ) ) ) break;
			size_t p = disk_.size();
			for ( size_t j = 0; j < disk_.size(); ++j ) if ( disk_[j].ready && disk_[j].tokens == parent ) { p = j; break; }
			if ( p == disk_.size() ) return ram_best;   // the chain is broken (an ancestor was dropped): nothing to resume from
			k = p;
		}
	}
	const auto t0 = std::chrono::steady_clock::now();
	Entry* e = top;
	size_t read = 0;
	for ( auto it = chain.rbegin(); it != chain.rend(); ++it ) {
		Entry* next = load_disk( *it, e );
		if ( !next ) return ram_best;
		read += next->snap.bytes * cluster_.tp();
		e = next;
	}
	stats_.disk_hits++;
	LOG_INFO( "[prefix] %zu tokens read back from disk (%zu node(s), %zu MiB, %.0f ms)\n", e->tokens.size(), chain.size(), read >> 20,
		std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - t0 ).count() );
	return e;
}

void PrefixCache::touch_chain( Entry* e ) {
	const uint64_t t = ++clock_;
	for ( ; e; e = e->parent ) e->last_used = t;
}

PrefixCache::Entry* PrefixCache::find( const std::vector<int>& prompt ) {
	stats_.lookups++;
	Entry* best = nullptr;
	for ( auto& e : entries_ ) {
		const size_t n = e->tokens.size();
		if ( n >= prompt.size() || ( best && n <= best->tokens.size() ) ) continue;
		if ( std::equal( e->tokens.begin(), e->tokens.end(), prompt.begin() ) ) best = e.get();
	}
	if ( disk_on() ) best = promote( prompt, best );
	if ( best ) {
		touch_chain( best );
		stats_.hits++;
		stats_.hit_tokens += best->tokens.size();
	}
	return best;
}

PrefixCache::Entry* PrefixCache::exact( const std::vector<int>& tokens ) {
	for ( auto& e : entries_ ) {
		if ( e->tokens == tokens ) return e.get();
	}
	return nullptr;
}

bool PrefixCache::restore( int slot, Entry* entry ) {
	std::vector<Entry*> chain;
	for ( Entry* e = entry; e; e = e->parent ) chain.push_back( e );
	std::reverse( chain.begin(), chain.end() );
	for ( size_t k = 0; k + 1 < chain.size(); ++k ) {
		if ( !cluster_.load_snapshot( slot, chain[k]->snap, true ) ) return false;
	}
	return cluster_.load_snapshot( slot, entry->snap, false );
}

void PrefixCache::remove( size_t index ) {
	Entry* e = entries_[index].get();
	if ( e->parent ) e->parent->children--;
	cluster_.free_snapshot( &e->snap );
	entries_.erase( entries_.begin() + static_cast<long>( index ) );
}

bool PrefixCache::evict_for( size_t incoming, const Entry* protect ) {
	size_t held = stats().bytes;
	while ( held + incoming > budget_ ) {
		size_t victim = entries_.size();
		for ( size_t k = 0; k < entries_.size(); ++k ) {
			const Entry* e = entries_[k].get();
			if ( e->pinned || e->children > 0 ) continue;
			bool guarded = false;
			for ( const Entry* p = protect; p; p = p->parent ) guarded = guarded || p == e;
			if ( guarded ) continue;
			if ( victim == entries_.size() || e->last_used < entries_[victim]->last_used ) victim = k;
		}
		if ( victim == entries_.size() ) return false; // nothing evictable
		held -= cost( entries_[victim]->snap, cluster_.tp() );
		if ( disk_on() ) spill( entries_[victim].get(), true );   // to a file instead of dropping it (its buffers go with the write)
		remove( victim );
		stats_.evictions++;
	}
	return true;
}

PrefixCache::Entry* PrefixCache::insert( Entry* parent, std::vector<int> tokens, ModelCluster::SeqSnapshot snap, bool pinned ) {
	const size_t incoming = cost( snap, cluster_.tp() );
	if ( !evict_for( incoming, parent ) ) {
		cluster_.free_snapshot( &snap );
		return nullptr;
	}
	auto e = std::make_unique<Entry>();
	e->tokens = std::move( tokens );
	e->parent = parent;
	e->snap = snap;
	e->pinned = pinned;
	if ( parent ) parent->children++;
	Entry* raw = e.get();
	entries_.push_back( std::move( e ) );
	touch_chain( raw );
	stats_.inserts++;
	return raw;
}

void PrefixCache::clear() {
	for ( auto& e : entries_ ) cluster_.free_snapshot( &e->snap );
	entries_.clear();
}

PrefixCache::Stats PrefixCache::stats() const {
	Stats s = stats_;
	s.entries = entries_.size();
	s.bytes = 0;
	for ( const auto& e : entries_ ) s.bytes += cost( e->snap, cluster_.tp() );
	if ( disk_on() ) {
		std::lock_guard<std::mutex> lock( disk_mtx_ );
		s.disk_entries = disk_.size();
		for ( const DiskEntry& d : disk_ ) s.disk_bytes += d.bytes * cluster_.tp();
	}
	return s;
}

} // namespace engine::runtime
