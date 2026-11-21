// volta_e2e: end-to-end tuning of a Flash-Next (qwen4_exp) EXL3 checkpoint's scales against a teacher's distributions: every EXL3 tensor's
// input and output scales and the norms' weights (the trellis codes and the other dense weights frozen) by Adam on KL( teacher || model )
// over token sequences, the model in training form (engine/train/qwen4_exp.h), each layer recomputed from its saved input in the backward.
// With --tp 1 (the default) the devices run it tensor parallel, one host thread each; --tp 0 splits the 48 layers over them instead (a
// layer whole on one device, the streams and their gradients crossing at the boundaries, one device busy at a time). The last sequences are
// held out: their KL is measured before training and every --eval-every steps, and the values of the best held-out KL are written as a new
// checkpoint (the input's files, its shards rewritten with the new FP16 scales and norms). The learning rate warms up linearly over
// --warmup steps, then follows a cosine to --lr-min (a fraction of --lr) at the last step. --fp16 0 runs the tuned tensors' products in
// FP32 instead of on the tensor cores; --norms 0 keeps the norms' weights as they are; --dense 1 trains the small dense tensors too (the
// hyper-connection sites' inject mixes, GDN's in_proj_a / b, A_log, dt_bias and convolution, the shared expert's gate, PLE's convolution;
// additive deltas like the norms, written back as FP16) at --dense-lr times the learning rate (a row of them feeds a dot product, so
// Adam's per-entry step adds up over its width); --router 1 trains the routers' weights at that rate as well. --train N trains on the first
// N sequences only
// (the held-out ones unchanged: how the gain grows with the data). --adam-bits 8 keeps Adam's moments as 8-bit block-wise codes; --opt-host 1
// keeps the optimizer's state in pinned host memory (the devices hold the current values, their gradients and one part of the state: each
// step streams the state through in parts). --seqs N runs up to N of a step's --batch sequences (and of the held-out ones) through each
// layer call together (the same gradients; more rows per product, the layer inputs and the workspace N times larger). The memory per
// component is logged after the first step.
//
//   volta_e2e --model <EXL3 dir> --inputs <train_inputs file> --teacher <topk file> --out <dir> [--devices 0,1,2,3] [--tp 1] [--fp16 1]
//             [--norms 1] [--dense 0] [--dense-lr 0.1] [--router 0] [--steps 64] [--batch 4] [--seqs 1] [--lr 1e-3] [--warmup 2] [--lr-min 0.1] [--hold 4] [--train N] [--eval-every 16] [--write 0|1]
//             [--adam-bits 32|8] [--opt-host 0|1] [--timing 0|1] [--check-trace FILE]
//   --check-trace: a trace_layers file of the model's engine forward over the first sequence (test_ppl --calibrate --chunk 512): every layer of
//     the training form fed the engine's input streams of that layer, its output against the engine's, and the first sequence's embedding
//     streams against the engine's; then exits (the layer split)
//
// The inputs come from the model itself (test_ppl --flash --calibrate --file TEXT --seq S --tokens N --chunk 512 with train_inputs set:
// its token embeddings and the PLE layer's n-gram rows), the teacher from the reference model on the same text (test_ppl --flash --file TEXT
// --seq S --tokens N --chunk 16 --topk-out FILE). --inputs and --teacher take comma-separated lists (pairwise, one text each): their sequences
// in that order, so the held-out ones are the last file's last.
#include "engine/config.h"
#include "engine/core/log.h"
#include "engine/train/qwen4_exp.h"
#include "quant/st_io.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#if defined( ENGINE_HAVE_NCCL )
#include <nccl.h>
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace tq = engine::train::qwen4_exp;
namespace q = engine::tools::quant;

struct Args {
	std::string model, inputs, teacher, out, check_trace;
	std::vector<int> devices = { 0, 1, 2, 3 };
	int steps = 64, batch = 4, seqs = 1, hold = 4, eval_every = 16, warmup = 2, train = 0, adam_bits = 32;
	bool write = true, timing = false, tp = true, fp16 = true, norms = true, dense = false, router = false, opt_host = false;
	float lr = 1e-3f, lr_min = 0.1f, dense_lr = 0.1f;
};

// The learning rate of a step (from 1): linear warmup, then a cosine from lr (the first step after the warmup) to lr * lr_min (the last step)
float lr_at( const Args& a, int step ) {
	if ( step <= a.warmup ) return a.lr * static_cast<float>( step ) / static_cast<float>( a.warmup + 1 );
	const int span = std::max( 1, a.steps - a.warmup - 1 );
	const double t = std::min( 1.0, static_cast<double>( step - a.warmup - 1 ) / span );
	return static_cast<float>( a.lr * ( a.lr_min + ( 1.0 - a.lr_min ) * 0.5 * ( 1.0 + std::cos( 3.14159265358979 * t ) ) ) );
}

// The kept values to write: per EXL3 tensor its (suh, svh), per tuned norm weight its values (FP16 each)
struct Best {
	std::map<std::string, std::pair<std::vector<uint16_t>, std::vector<uint16_t>>> scales;
	std::map<std::string, std::vector<uint16_t>> dense;
};

void add_best( tq::Model& m, Best& out ) {
	std::vector<std::pair<std::string, std::pair<std::vector<uint16_t>, std::vector<uint16_t>>>> v;
	m.best_scales( v );
	for ( auto& e : v ) out.scales[e.first] = std::move( e.second );
	std::vector<std::pair<std::string, std::vector<uint16_t>>> d;
	m.best_dense( d );
	for ( auto& e : d ) out.dense[e.first] = std::move( e.second );
}

bool parse( int argc, char** argv, Args& a ) {
	for ( int i = 1; i + 1 < argc; i += 2 ) {
		const std::string k = argv[i], v = argv[i + 1];
		if ( k == "--model" ) a.model = v;
		else if ( k == "--inputs" ) a.inputs = v;
		else if ( k == "--teacher" ) a.teacher = v;
		else if ( k == "--out" ) a.out = v;
		else if ( k == "--steps" ) a.steps = std::stoi( v );
		else if ( k == "--batch" ) a.batch = std::stoi( v );
		else if ( k == "--seqs" ) a.seqs = std::stoi( v );
		else if ( k == "--hold" ) a.hold = std::stoi( v );
		else if ( k == "--train" ) a.train = std::stoi( v );
		else if ( k == "--eval-every" ) a.eval_every = std::stoi( v );
		else if ( k == "--lr" ) a.lr = std::stof( v );
		else if ( k == "--lr-min" ) a.lr_min = std::stof( v );
		else if ( k == "--warmup" ) a.warmup = std::stoi( v );
		else if ( k == "--tp" ) a.tp = v != "0";
		else if ( k == "--fp16" ) a.fp16 = v != "0";
		else if ( k == "--norms" ) a.norms = v != "0";
		else if ( k == "--dense" ) a.dense = v != "0";
		else if ( k == "--dense-lr" ) a.dense_lr = std::stof( v );
		else if ( k == "--router" ) a.router = v != "0";
		else if ( k == "--opt-host" ) a.opt_host = v != "0";
		else if ( k == "--adam-bits" ) a.adam_bits = std::stoi( v );
		else if ( k == "--write" ) a.write = v != "0";
		else if ( k == "--timing" ) a.timing = v != "0";
		else if ( k == "--check-trace" ) a.check_trace = v;
		else if ( k == "--devices" ) {
			a.devices.clear();
			for ( size_t p = 0; p <= v.size(); ) {
				const size_t e = std::min( v.find( ',', p ), v.size() );
				if ( e > p ) a.devices.push_back( std::stoi( v.substr( p, e - p ) ) );
				p = e + 1;
			}
		}
		else {
			LOG_ERROR( "[volta_e2e] unknown argument %s", k.c_str() );
			return false;
		}
	}
	if ( a.model.empty() || a.inputs.empty() || a.teacher.empty() || a.out.empty() || ( a.adam_bits != 32 && a.adam_bits != 8 ) || a.seqs < 1 ) {
		LOG_ERROR( "usage: volta_e2e --model <EXL3 dir> --inputs <train_inputs file> --teacher <topk file> --out <dir> [--devices 0,1,2,3] [--tp 1] "
			"[--fp16 1] [--norms 1] [--dense 0] [--dense-lr 0.1] [--router 0] [--steps 64] [--batch 4] [--seqs 1] [--lr 1e-3] [--warmup 2] [--lr-min 0.1] [--hold 4] [--train N] [--eval-every 16] [--write 0|1] "
			"[--adam-bits 32|8] [--opt-host 0|1] [--timing 0|1] [--check-trace FILE]" );
		return false;
	}
	return true;
}

// The teacher file (test_ppl --topk-out) and the model's own inputs (train_inputs), by sequence
struct Data {
	int n = 0, seq = 0, k = 0, H = 0, flat = 0;
	std::vector<int> tokens;
	std::vector<int> ids;            // [n, k]
	std::vector<float> p, tail;      // [n, k], [n]
	std::vector<uint16_t> emb;       // [n, H] FP16
	std::vector<uint16_t> pk, pv;    // [n, flat], [n, H] FP16
	int sequences() const { return seq > 0 ? n / seq : 0; }
};

bool read_teacher( const std::string& path, Data& d ) {
	FILE* f = std::fopen( path.c_str(), "rb" );
	if ( !f ) return false;
	int hdr[4];
	bool ok = std::fread( hdr, sizeof( int ), 4, f ) == 4 && hdr[0] == 0x4b504f54;
	if ( ok ) {
		d.n = hdr[1];
		d.seq = hdr[2];
		d.k = hdr[3];
		d.tokens.resize( d.n );
		ok = std::fread( d.tokens.data(), sizeof( int ), d.n, f ) == static_cast<size_t>( d.n );
		d.ids.resize( static_cast<size_t>( d.n ) * d.k );
		d.p.resize( d.ids.size() );
		d.tail.resize( d.n );
		for ( int r = 0; ok && r < d.n; ++r ) {
			ok = std::fread( d.ids.data() + static_cast<size_t>( r ) * d.k, sizeof( int ), d.k, f ) == static_cast<size_t>( d.k ) &&
				std::fread( d.p.data() + static_cast<size_t>( r ) * d.k, sizeof( float ), d.k, f ) == static_cast<size_t>( d.k ) &&
				std::fread( &d.tail[r], sizeof( float ), 1, f ) == 1;
		}
	}
	std::fclose( f );
	return ok && d.seq > 0;
}

bool read_inputs( const std::string& path, int H, int flat, Data& d ) {
	FILE* f = std::fopen( path.c_str(), "rb" );
	if ( !f ) return false;
	d.H = H;
	d.flat = flat;
	d.emb.clear();
	d.pk.clear();
	d.pv.clear();
	int hdr[5];
	while ( std::fread( hdr, sizeof( int ), 5, f ) == 5 ) {
		const int kind = hdr[0], m = hdr[1], ca = hdr[3], cb = hdr[4];
		std::vector<uint16_t> a( static_cast<size_t>( m ) * ca ), b( static_cast<size_t>( m ) * cb );
		if ( std::fread( a.data(), 2, a.size(), f ) != a.size() || ( cb > 0 && std::fread( b.data(), 2, b.size(), f ) != b.size() ) ) break;
		if ( kind == 0 && ca == H ) d.emb.insert( d.emb.end(), a.begin(), a.end() );
		if ( kind == 1 && ca == flat && cb == H ) {
			d.pk.insert( d.pk.end(), a.begin(), a.end() );
			d.pv.insert( d.pv.end(), b.begin(), b.end() );
		}
	}
	std::fclose( f );
	const size_t rows = d.emb.size() / H;
	if ( rows < static_cast<size_t>( d.n ) || d.pk.size() / flat < static_cast<size_t>( d.n ) ) {
		LOG_ERROR( "[volta_e2e] %s: %zu embedding and %zu PLE rows for %d teacher positions", path.c_str(), rows, d.pk.size() / flat, d.n );
		return false;
	}
	return true;
}

// Comma-separated teacher and inputs files, pairwise (the same text each), as one data set in their order: the held-out sequences are the
// last file's last ones
bool read_data( const std::string& teachers, const std::string& inputs, int H, int flat, Data& d ) {
	auto split = []( const std::string& s ) {
		std::vector<std::string> out;
		for ( size_t p = 0; p <= s.size(); ) {
			const size_t e = std::min( s.find( ',', p ), s.size() );
			if ( e > p ) out.push_back( s.substr( p, e - p ) );
			p = e + 1;
		}
		return out;
	};
	const std::vector<std::string> ts = split( teachers ), is = split( inputs );
	if ( ts.empty() || ts.size() != is.size() ) {
		LOG_ERROR( "[volta_e2e] %zu teacher files for %zu inputs files", ts.size(), is.size() );
		return false;
	}
	for ( size_t f = 0; f < ts.size(); ++f ) {
		Data e;
		if ( !read_teacher( ts[f], e ) || !read_inputs( is[f], H, flat, e ) ) return false;
		if ( f == 0 ) {
			const size_t n = static_cast<size_t>( e.n );
			e.emb.resize( n * H );
			e.pk.resize( n * flat );
			e.pv.resize( n * H );
			d = std::move( e );
			continue;
		}
		if ( e.seq != d.seq || e.k != d.k || e.n % e.seq != 0 || d.n % d.seq != 0 ) {
			LOG_ERROR( "[volta_e2e] %s: sequences of %d with top %d after %d / %d (or not whole sequences)", ts[f].c_str(), e.seq, e.k, d.seq, d.k );
			return false;
		}
		const size_t n = static_cast<size_t>( e.n );
		d.tokens.insert( d.tokens.end(), e.tokens.begin(), e.tokens.end() );
		d.ids.insert( d.ids.end(), e.ids.begin(), e.ids.end() );
		d.p.insert( d.p.end(), e.p.begin(), e.p.end() );
		d.tail.insert( d.tail.end(), e.tail.begin(), e.tail.end() );
		d.emb.insert( d.emb.end(), e.emb.begin(), e.emb.begin() + n * H );
		d.pk.insert( d.pk.end(), e.pk.begin(), e.pk.begin() + n * flat );
		d.pv.insert( d.pv.end(), e.pv.begin(), e.pv.begin() + n * H );
		d.n += e.n;
	}
	return true;
}

struct Stage {
	std::unique_ptr<tq::Model> m;
	int dev = 0;
	float *x = nullptr, *y = nullptr, *dx = nullptr, *dy = nullptr;   // streams and their gradients [T, flat]
	std::vector<float*> ckpt;                                          // the input of each of the stage's layers
	float *pk = nullptr, *pv = nullptr;                                // PLE rows (the stage with the PLE layer)
	uint16_t* staging = nullptr;                                       // FP16 upload buffer
	int *t_ids = nullptr;
	float *t_p = nullptr, *t_tail = nullptr;                           // teacher (last stage)
};

void up_half( Stage& s, const uint16_t* src, size_t n, float* dst ) {
	CUDA_CHECK( cudaMemcpyAsync( s.staging, src, n * 2, cudaMemcpyHostToDevice, static_cast<cudaStream_t>( s.m->stream() ) ) );
	tq::half_to_float( s.staging, dst, n, s.m->stream() );
}

// Rows [r0, r1) of the sequences sqs (T rows each) laid end to end, from the host's FP16 rows of cols values into dst on the device; returns
// the rows copied
size_t seq_rows( const uint16_t* src, size_t cols, int T, const std::vector<int>& sqs, size_t r0, size_t r1, uint16_t* dst, cudaStream_t st ) {
	size_t o = 0;
	for ( size_t i = 0; i < sqs.size(); ++i ) {
		const size_t s0 = i * T, lo = std::max( r0, s0 ), hi = std::min( r1, s0 + T );
		if ( lo >= hi ) continue;
		CUDA_CHECK( cudaMemcpyAsync( dst + o * cols, src + ( static_cast<size_t>( sqs[i] ) * T + lo - s0 ) * cols, ( hi - lo ) * cols * 2,
			cudaMemcpyHostToDevice, st ) );
		o += hi - lo;
	}
	return o;
}

// The teacher's rows of the sequences sqs, end to end
void seq_teacher( const Data& d, int T, const std::vector<int>& sqs, int* ids, float* p, float* tail, cudaStream_t st ) {
	const size_t k = static_cast<size_t>( d.k ), TR = static_cast<size_t>( T );
	for ( size_t i = 0; i < sqs.size(); ++i ) {
		const size_t base = static_cast<size_t>( sqs[i] ) * TR, o = i * TR;
		CUDA_CHECK( cudaMemcpyAsync( ids + o * k, d.ids.data() + base * k, TR * k * sizeof( int ), cudaMemcpyHostToDevice, st ) );
		CUDA_CHECK( cudaMemcpyAsync( p + o * k, d.p.data() + base * k, TR * k * sizeof( float ), cudaMemcpyHostToDevice, st ) );
		CUDA_CHECK( cudaMemcpyAsync( tail + o, d.tail.data() + base, TR * sizeof( float ), cudaMemcpyHostToDevice, st ) );
	}
}

// The output checkpoint: every file of the input, the shards with their tuned scales and norm weights replaced (the norms as FP16, whatever
// the checkpoint stored them in: the engine computes with them in FP16)
bool write_out( const Args& a, const Best& best, int& replaced ) {
	const auto& best_scales = best.scales;
	std::filesystem::create_directories( a.out );
	replaced = 0;
	for ( const auto& e : std::filesystem::directory_iterator( a.model ) ) {
		const std::string fn = e.path().filename().string(), to = a.out + "/" + fn;
		if ( e.path().extension() != ".safetensors" || fn == "ngram_embedding.safetensors" ) {
			std::error_code ec;
			std::filesystem::remove( to, ec );
			std::filesystem::create_symlink( std::filesystem::canonical( e.path() ), to, ec );
			if ( ec ) std::filesystem::copy_file( e.path(), to, std::filesystem::copy_options::overwrite_existing );
			continue;
		}
		q::StFile f;
		if ( !f.open( e.path().string() ) ) return false;
		std::vector<q::StOut> outs;
		for ( const auto& name : f.names() ) {
			const q::StEntry* en = f.find( name );
			q::StOut o2;
			o2.name = name;
			o2.dtype = en->dtype;
			o2.shape = en->shape;
			const size_t dot = name.rfind( '.' );
			const std::string base = name.substr( 0, dot ), suf = name.substr( dot + 1 );
			const auto it = best_scales.find( base );
			const auto dt = best.dense.find( name );
			if ( it != best_scales.end() && ( suf == "suh" || suf == "svh" ) && en->dtype == "F16" ) {
				const auto& v = suf == "suh" ? it->second.first : it->second.second;
				ENGINE_ASSERT( v.size() * 2 == en->bytes, "volta_e2e: a scale vector's size differs from the checkpoint's" );
				o2.owned.resize( en->bytes );
				std::memcpy( o2.owned.data(), v.data(), en->bytes );
				++replaced;
			}
			else if ( dt != best.dense.end() ) {
				size_t numel = 1;
				for ( auto d : en->shape ) numel *= static_cast<size_t>( d );
				ENGINE_ASSERT( dt->second.size() == numel, "volta_e2e: a norm's size differs from the checkpoint's" );
				o2.dtype = "F16";
				o2.owned.resize( numel * 2 );
				std::memcpy( o2.owned.data(), dt->second.data(), numel * 2 );
				++replaced;
			}
			else {
				o2.view = en->data;
				o2.view_bytes = en->bytes;
			}
			outs.push_back( std::move( o2 ) );
		}
		if ( !q::st_write( to, outs, f.metadata() ) ) return false;
		LOG_INFO( "[volta_e2e] wrote %s", fn.c_str() );
	}
	return true;
}

class Barrier {
public:
	explicit Barrier( int n ) : n_( n ) {}
	void wait() {
		std::unique_lock<std::mutex> lk( m_ );
		const int gen = gen_;
		if ( ++count_ == n_ ) {
			count_ = 0;
			++gen_;
			cv_.notify_all();
		}
		else cv_.wait( lk, [&] { return gen != gen_; } );
	}

private:
	int n_, count_ = 0, gen_ = 0;
	std::mutex m_;
	std::condition_variable cv_;
};

#if defined( ENGINE_HAVE_NCCL )
// Tensor parallel over the devices (engine/train/qwen4_exp.h): one host thread per rank, all of them running the same steps; every rank's
// held-out and training KL is the whole sum, rank 0 reports it
int train_tp( const Args& a, const tq::Config& cfg, const Data& d ) {
	const int n = static_cast<int>( a.devices.size() ), T = d.seq, S = d.sequences(), train_n = S - a.hold, H = cfg.hidden, hc = cfg.hc, flat = hc * H;
	const int train_used = a.train > 0 ? std::min( a.train, train_n ) : train_n;
	if ( T % n != 0 ) {
		LOG_ERROR( "[volta_e2e] %d-token sequences do not split over %d ranks", T, n );
		return 1;
	}
	const int P = a.seqs, R = P * T, Rl = R / n;   // a pass's sequences and rows (all, the rank's)
	std::vector<ncclComm_t> comms( n, nullptr );
	std::vector<int> devs = a.devices;
	if ( ncclCommInitAll( comms.data(), n, devs.data() ) != ncclSuccess ) {
		LOG_ERROR( "[volta_e2e] NCCL init failed" );
		return 1;
	}
	Barrier bar( n );
	std::vector<int> ok( n, 1 );
	std::vector<Best> kept( n );
	double best = 0.0;
	int best_step = 0;
	const auto t0 = std::chrono::steady_clock::now();
	std::vector<std::thread> th;
	for ( int r = 0; r < n; ++r ) th.emplace_back( [&, r] {
		const int dev = a.devices[r];
		CUDA_CHECK( cudaSetDevice( dev ) );
		auto m = std::make_unique<tq::Model>();
		m->set_optimizer( a.opt_host, a.adam_bits );
		m->set_train_dense( a.dense, a.dense_lr );
		m->set_train_router( a.router );
		ok[r] = m->load( a.model, 0, cfg.layers, dev, R, r, n, comms[r] ) ? 1 : 0;
		m->set_fp16( a.fp16 );
		m->set_train_norms( a.norms );
		bar.wait();
		for ( int g = 0; g < n; ++g ) if ( !ok[g] ) return;
		const cudaStream_t st = static_cast<cudaStream_t>( m->stream() );
		const size_t rows_l = static_cast<size_t>( Rl ) * flat;
		float *x, *dx, *dy, *emb, *pk, *pv, *t_p, *t_tail;
		int* t_ids;
		uint16_t* staging;
		for ( float** p : { &x, &dx, &dy } ) CUDA_CHECK( cudaMalloc( p, rows_l * sizeof( float ) ) );
		CUDA_CHECK( cudaMalloc( &emb, static_cast<size_t>( Rl ) * H * sizeof( float ) ) );
		CUDA_CHECK( cudaMalloc( &pk, static_cast<size_t>( R ) * flat * sizeof( float ) ) );
		CUDA_CHECK( cudaMalloc( &pv, static_cast<size_t>( R ) * H * sizeof( float ) ) );
		CUDA_CHECK( cudaMalloc( &staging, static_cast<size_t>( R ) * flat * 2 ) );
		CUDA_CHECK( cudaMalloc( &t_ids, static_cast<size_t>( R ) * d.k * sizeof( int ) ) );
		CUDA_CHECK( cudaMalloc( &t_p, static_cast<size_t>( R ) * d.k * sizeof( float ) ) );
		CUDA_CHECK( cudaMalloc( &t_tail, static_cast<size_t>( R ) * sizeof( float ) ) );
		std::vector<float*> ckpt( cfg.layers );
		for ( float*& c : ckpt ) CUDA_CHECK( cudaMalloc( &c, rows_l * sizeof( float ) ) );
		{
			size_t fr = 0, tot = 0;
			CUDA_CHECK( cudaMemGetInfo( &fr, &tot ) );
			LOG_INFO( "[volta_e2e] rank %d (device %d): all layers, experts %d-%d of %d, %.1f GiB free", r, dev, cfg.experts * r / n, cfg.experts * ( r + 1 ) / n - 1,
				cfg.experts, fr / 1073741824.0 );
		}
		// rows [r0, r1) of the sequences sqs laid end to end (cols FP16 values per row) into dst as FP32
		auto up = [&]( const uint16_t* src, size_t cols, const std::vector<int>& sqs, size_t r0, size_t r1, float* dst ) {
			const size_t got = seq_rows( src, cols, T, sqs, r0, r1, staging, st );
			tq::half_to_float( staging, dst, got * cols, st );
			CUDA_CHECK( cudaStreamSynchronize( st ) );
		};
		// Sequences sqs (up to --seqs) in one pass: the forward over all layers (saving each layer's input), the KL head, optionally the backward
		auto run = [&]( const std::vector<int>& sqs, bool train ) -> double {
			const int c = static_cast<int>( sqs.size() ), Rc = c * T, Rcl = Rc / n;
			up( d.emb.data(), H, sqs, static_cast<size_t>( r ) * Rcl, static_cast<size_t>( r + 1 ) * Rcl, emb );
			for ( int h = 0; h < hc; ++h ) tq::copy_cols( emb, H, 0, x, flat, h * H, Rcl, H, false, st );
			up( d.pk.data(), flat, sqs, 0, Rc, pk );
			up( d.pv.data(), H, sqs, 0, Rc, pv );
			const size_t xl = static_cast<size_t>( Rcl ) * flat;
			for ( int L = 0; L < cfg.layers; ++L ) {
				CUDA_CHECK( cudaMemcpyAsync( ckpt[L], x, xl * sizeof( float ), cudaMemcpyDeviceToDevice, st ) );
				m->forward( L, ckpt[L], x, Rc, pk, pv, train, c );
			}
			seq_teacher( d, T, sqs, t_ids, t_p, t_tail, st );
			tq::Teacher t;
			t.ids = t_ids;
			t.p = t_p;
			t.tail = t_tail;
			t.tk = d.k;
			const double kl = m->head( x, Rc, t, train ? dx : nullptr );
			if ( !train ) return kl;
			for ( int L = cfg.layers - 1; L >= 0; --L ) {
				m->backward( L, ckpt[L], dx, dy, Rc, pk, pv, c );
				std::swap( dx, dy );
			}
			return kl;
		};
		auto held_out = [&]() {
			double kl = 0.0;
			for ( int s0 = train_n; s0 < S; s0 += P ) {
				std::vector<int> sqs;
				for ( int sq = s0; sq < std::min( S, s0 + P ); ++sq ) sqs.push_back( sq );
				kl += run( sqs, false );
			}
			return kl / ( static_cast<double>( a.hold ) * T );
		};
		const auto th0 = std::chrono::steady_clock::now();
		double my_best = held_out();
		if ( r == 0 )
			LOG_INFO( "[volta_e2e] held-out KL before tuning %.6f per token (%.1f s)", my_best, std::chrono::duration<double>( std::chrono::steady_clock::now() - th0 ).count() );
		m->keep_best();
		if ( a.timing ) {
			m->set_timing( true );
			m->timing_report();
		}
		int my_step = 0, sq = 0;
		engine::train::AdamOptions o;
		for ( int step = 1; step <= a.steps; ++step ) {
			const auto ts = std::chrono::steady_clock::now();
			m->zero_grads();
			double kl = 0.0;
			for ( int b = 0; b < a.batch; b += P ) {
				std::vector<int> sqs;
				for ( int i = b; i < std::min( a.batch, b + P ); ++i ) {
					sqs.push_back( sq );
					sq = ( sq + 1 ) % train_used;
				}
				kl += run( sqs, true );
			}
			o.lr = lr_at( a, step );
			m->adam( o, step, 1.0f / ( static_cast<float>( a.batch ) * T ) );
			if ( step == 1 ) {
				// the memory by component once the workspace has seen a whole step, in rank order
				const double G = 1073741824.0;
				const double own = ( ( 3.0 + cfg.layers ) * rows_l + static_cast<double>( Rl ) * H + static_cast<double>( R ) * ( flat + H + 1 ) +
					static_cast<double>( R ) * d.k ) * sizeof( float ) + static_cast<double>( R ) * flat * 2 + static_cast<double>( R ) * d.k * sizeof( int );
				const std::string rep = m->memory_report();
				bar.wait();
				for ( int g = 0; g < n; ++g ) {
					if ( g == r ) LOG_INFO( "[volta_e2e] rank %d memory after step 1:\n%s  layer inputs and streams (volta_e2e)  %6.2f GiB", r, rep.c_str(), own / G );
					bar.wait();
				}
			}
			if ( a.timing ) {
				const std::string rep = m->timing_report();
				bar.wait();   // the reports in rank order
				for ( int g = 0; g < n; ++g ) {
					if ( g == r ) LOG_INFO( "[volta_e2e] step %d, rank %d:\n%s", step, r, rep.c_str() );
					bar.wait();
				}
			}
			if ( r == 0 )
				LOG_INFO( "[volta_e2e] step %d: train KL %.6f per token, lr %.3g, %.1f s", step, kl / ( static_cast<double>( a.batch ) * T ), o.lr,
					std::chrono::duration<double>( std::chrono::steady_clock::now() - ts ).count() );
			if ( step % a.eval_every == 0 || step == a.steps ) {
				const bool was = a.timing;
				if ( was ) m->set_timing( false );
				const double ho = held_out();
				if ( was ) m->set_timing( true );
				if ( r == 0 )
					LOG_INFO( "[volta_e2e] step %d: held-out KL %.6f per token (best %.6f at step %d)", step, ho, std::min( ho, my_best ), ho < my_best ? step : my_step );
				if ( ho < my_best ) {
					my_best = ho;
					my_step = step;
					m->keep_best();
				}
			}
		}
		add_best( *m, kept[r] );
		if ( r == 0 ) {
			best = my_best;
			best_step = my_step;
		}
		for ( float* p : { x, dx, dy, emb, pk, pv, t_p, t_tail } ) cudaFree( p );
		cudaFree( t_ids );
		cudaFree( staging );
		for ( float* c : ckpt ) cudaFree( c );
		bar.wait();   // every rank done with the communicators before any model goes
	} );
	for ( auto& t : th ) t.join();
	for ( ncclComm_t c : comms ) if ( c ) ncclCommDestroy( c );
	for ( int g = 0; g < n; ++g ) if ( !ok[g] ) return 1;
	if ( !a.write ) {
		LOG_INFO( "[volta_e2e] done (--write 0: no checkpoint written): best held-out KL %.6f per token at step %d", best, best_step );
		return 0;
	}
	Best all;
	for ( Best& s : kept ) {
		for ( auto& e : s.scales ) all.scales[e.first] = std::move( e.second );
		for ( auto& e : s.dense ) all.dense[e.first] = std::move( e.second );
	}
	int replaced = 0;
	if ( !write_out( a, all, replaced ) ) return 1;
	LOG_INFO( "[volta_e2e] done: %s, %d scale vectors and norms from step %d (held-out KL %.6f per token), %.1f min", a.out.c_str(), replaced, best_step, best,
		std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count() / 60.0 );
	return 0;
}
#endif

} // namespace

int main( int argc, char** argv ) {
	std::setvbuf( stdout, nullptr, _IOLBF, 0 );
	Args a;
	if ( !parse( argc, argv, a ) ) return 2;
	tq::Config cfg;
	if ( !cfg.read( a.model ) ) {
		LOG_ERROR( "[volta_e2e] %s: no config.json", a.model.c_str() );
		return 1;
	}
	const int H = cfg.hidden, hc = cfg.hc, flat = hc * H;
	Data d;
	if ( !read_data( a.teacher, a.inputs, H, flat, d ) ) {
		LOG_ERROR( "[volta_e2e] cannot read the teacher (%s) or the inputs (%s)", a.teacher.c_str(), a.inputs.c_str() );
		return 1;
	}
	const int T = d.seq, S = d.sequences(), train_n = S - a.hold;
	if ( train_n < 1 ) {
		LOG_ERROR( "[volta_e2e] %d sequences of %d tokens: none left for training after %d held out", S, T, a.hold );
		return 1;
	}
	const int train_used = a.train > 0 ? std::min( a.train, train_n ) : train_n;
	LOG_INFO( "[volta_e2e] %d sequences of %d tokens (%d held out, %d of the others trained on), teacher top %d", S, T, a.hold, train_used, d.k );
	const int G = static_cast<int>( a.devices.size() );
#if defined( ENGINE_HAVE_NCCL )
	if ( a.tp && G > 1 && a.check_trace.empty() ) return train_tp( a, cfg, d );
#endif
	// Stages: the layers split evenly over the devices
	const int P = a.check_trace.empty() ? a.seqs : 1, R = P * T;   // a pass's sequences and rows
	std::vector<Stage> st( G );
	for ( int g = 0; g < G; ++g ) {
		Stage& s = st[g];
		s.dev = a.devices[g];
		const int first = cfg.layers * g / G, last = cfg.layers * ( g + 1 ) / G;
		s.m = std::make_unique<tq::Model>();
		s.m->set_optimizer( a.opt_host, a.adam_bits );
		s.m->set_train_dense( a.dense, a.dense_lr );
		s.m->set_train_router( a.router );
		if ( !s.m->load( a.model, first, last, s.dev, R ) ) return 1;
		s.m->set_fp16( a.fp16 );
		s.m->set_train_norms( a.norms );
		CUDA_CHECK( cudaSetDevice( s.dev ) );
		const size_t bytes = static_cast<size_t>( R ) * flat * sizeof( float );
		for ( float** p : { &s.x, &s.y, &s.dx, &s.dy } ) CUDA_CHECK( cudaMalloc( p, bytes ) );
		for ( int L = first; L < last; ++L ) {
			float* c = nullptr;
			CUDA_CHECK( cudaMalloc( &c, bytes ) );
			s.ckpt.push_back( c );
		}
		CUDA_CHECK( cudaMalloc( &s.staging, static_cast<size_t>( R ) * flat * 2 ) );
		if ( cfg.ple_layer >= first && cfg.ple_layer < last ) {
			CUDA_CHECK( cudaMalloc( &s.pk, bytes ) );
			CUDA_CHECK( cudaMalloc( &s.pv, static_cast<size_t>( R ) * H * sizeof( float ) ) );
		}
		if ( s.m->has_head() ) {
			CUDA_CHECK( cudaMalloc( &s.t_ids, static_cast<size_t>( R ) * d.k * sizeof( int ) ) );
			CUDA_CHECK( cudaMalloc( &s.t_p, static_cast<size_t>( R ) * d.k * sizeof( float ) ) );
			CUDA_CHECK( cudaMalloc( &s.t_tail, static_cast<size_t>( R ) * sizeof( float ) ) );
		}
		size_t fr = 0, tot = 0;
		CUDA_CHECK( cudaMemGetInfo( &fr, &tot ) );
		LOG_INFO( "[volta_e2e] device %d: layers %d-%d%s, %.1f GiB free", s.dev, first, last - 1, s.m->has_head() ? " and the head" : "", fr / 1073741824.0 );
	}
	// Streams between stages: the source's work done, the copy ordered on the receiving stage's stream (the model streams are non-blocking, so a
	// plain cudaMemcpyPeer, ordered against the legacy stream only, races with the receiver's next kernels)
	auto handoff = [&]( Stage& to, float* dst, Stage& from, const float* src, int rows ) {
		CUDA_CHECK( cudaSetDevice( from.dev ) );
		CUDA_CHECK( cudaStreamSynchronize( static_cast<cudaStream_t>( from.m->stream() ) ) );
		CUDA_CHECK( cudaSetDevice( to.dev ) );
		CUDA_CHECK( cudaMemcpyPeerAsync( dst, to.dev, src, from.dev, static_cast<size_t>( rows ) * flat * sizeof( float ), static_cast<cudaStream_t>( to.m->stream() ) ) );
	};
	if ( !a.check_trace.empty() ) {
		// The engine's traces of the first T rows: tag -> rows (FP32), streams tags [T, flat], others [T, H]
		std::map<int, std::vector<float>> tr;
		FILE* f = std::fopen( a.check_trace.c_str(), "rb" );
		if ( !f ) return 1;
		int hdr[3];
		while ( std::fread( hdr, sizeof( int ), 3, f ) == 3 ) {
			const int tag = hdr[0], m = hdr[1], p0 = hdr[2];
			const bool streams = tag == 1000 || tag == 800 || ( tag >= 300 && tag < 400 ) || ( tag >= 600 && tag < 700 );
			const size_t cols = tag == 700 ? 2 * cfg.top_k : streams ? flat : H;
			std::vector<float> v( static_cast<size_t>( m ) * cols );
			if ( std::fread( v.data(), sizeof( float ), v.size(), f ) != v.size() ) break;
			if ( p0 >= T || tag == 700 ) continue;
			auto& dst = tr[tag];
			dst.resize( static_cast<size_t>( T ) * cols );
			std::memcpy( dst.data() + static_cast<size_t>( p0 ) * cols, v.data(), std::min<size_t>( m, T - p0 ) * cols * sizeof( float ) );
		}
		std::fclose( f );
		auto relerr = [&]( const std::vector<float>& a1, const std::vector<float>& b1 ) {
			double d = 0.0, n2 = 0.0;
			for ( size_t i = 0; i < b1.size(); ++i ) { d += ( a1[i] - b1[i] ) * ( a1[i] - b1[i] ); n2 += b1[i] * b1[i]; }
			return std::sqrt( d / ( n2 + 1e-30 ) );
		};
		std::vector<float> host( static_cast<size_t>( T ) * flat );
		{
			// the embedding streams from the inputs against the engine's initial streams
			Stage& s0 = st[0];
			CUDA_CHECK( cudaSetDevice( s0.dev ) );
			up_half( s0, d.emb.data(), static_cast<size_t>( T ) * H, s0.dy );
			for ( int h = 0; h < hc; ++h ) tq::copy_cols( s0.dy, H, 0, s0.x, flat, h * H, T, H, false, s0.m->stream() );
			CUDA_CHECK( cudaStreamSynchronize( static_cast<cudaStream_t>( s0.m->stream() ) ) );
			CUDA_CHECK( cudaMemcpy( host.data(), s0.x, host.size() * sizeof( float ), cudaMemcpyDeviceToHost ) );
			LOG_INFO( "[volta_e2e] check: initial streams from the inputs against the engine's: relative error %.3g", relerr( host, tr[1000] ) );
		}
		for ( Stage& s : st ) {
			CUDA_CHECK( cudaSetDevice( s.dev ) );
			if ( s.pk ) {
				up_half( s, d.pk.data(), static_cast<size_t>( T ) * flat, s.pk );
				CUDA_CHECK( cudaStreamSynchronize( static_cast<cudaStream_t>( s.m->stream() ) ) );
				up_half( s, d.pv.data(), static_cast<size_t>( T ) * H, s.pv );
			}
			for ( int L = s.m->first(); L < s.m->last(); ++L ) {
				const std::vector<float>& in = L == 0 ? tr[1000] : tr[600 + L - 1];
				CUDA_CHECK( cudaMemcpy( s.x, in.data(), in.size() * sizeof( float ), cudaMemcpyHostToDevice ) );
				s.m->forward( L, s.x, s.y, T, s.pk, s.pv );
				CUDA_CHECK( cudaStreamSynchronize( static_cast<cudaStream_t>( s.m->stream() ) ) );
				CUDA_CHECK( cudaMemcpy( host.data(), s.y, host.size() * sizeof( float ), cudaMemcpyDeviceToHost ) );
				// the layer's own change against the engine's (the streams carry the input through the residual)
				std::vector<float> dt( host.size() ), de( host.size() );
				const std::vector<float>& ref = tr[600 + L];
				for ( size_t i = 0; i < host.size(); ++i ) { dt[i] = host[i] - in[i]; de[i] = ref[i] - in[i]; }
				LOG_INFO( "[volta_e2e] check: layer %2d (%s%s): output streams %.3g, the layer's change %.3g", L, cfg.full_attention( L ) ? "attention" : "GDN",
					L == cfg.ple_layer ? " + PLE" : "", relerr( host, ref ), relerr( dt, de ) );
			}
		}
		{
			// the chained forward of the first sequence (each layer on the previous layer's output) against the engine's streams
			Stage& s0 = st[0];
			CUDA_CHECK( cudaSetDevice( s0.dev ) );
			up_half( s0, d.emb.data(), static_cast<size_t>( T ) * H, s0.dy );
			for ( int h = 0; h < hc; ++h ) tq::copy_cols( s0.dy, H, 0, s0.x, flat, h * H, T, H, false, s0.m->stream() );
			for ( int g = 0; g < G; ++g ) {
				Stage& s = st[g];
				CUDA_CHECK( cudaSetDevice( s.dev ) );
				for ( int L = s.m->first(); L < s.m->last(); ++L ) {
					float* c = s.ckpt[L - s.m->first()];
					CUDA_CHECK( cudaMemcpyAsync( c, s.x, static_cast<size_t>( T ) * flat * sizeof( float ), cudaMemcpyDeviceToDevice,
						static_cast<cudaStream_t>( s.m->stream() ) ) );
					s.m->forward( L, c, s.x, T, s.pk, s.pv );
					CUDA_CHECK( cudaStreamSynchronize( static_cast<cudaStream_t>( s.m->stream() ) ) );
					CUDA_CHECK( cudaMemcpy( host.data(), s.x, host.size() * sizeof( float ), cudaMemcpyDeviceToHost ) );
					if ( L % 4 == 0 || L == cfg.layers - 1 ) LOG_INFO( "[volta_e2e] check: chained, after layer %2d: streams %.3g", L, relerr( host, tr[600 + L] ) );
				}
				if ( g + 1 < G ) handoff( st[g + 1], st[g + 1].x, s, s.x, T );
			}
		}
		{
			// the head on the engine's final streams of the first sequence
			Stage& sl = st[G - 1];
			CUDA_CHECK( cudaSetDevice( sl.dev ) );
			const std::vector<float>& fin = tr[600 + cfg.layers - 1];
			CUDA_CHECK( cudaMemcpy( sl.x, fin.data(), fin.size() * sizeof( float ), cudaMemcpyHostToDevice ) );
			CUDA_CHECK( cudaMemcpy( sl.t_ids, d.ids.data(), static_cast<size_t>( T ) * d.k * sizeof( int ), cudaMemcpyHostToDevice ) );
			CUDA_CHECK( cudaMemcpy( sl.t_p, d.p.data(), static_cast<size_t>( T ) * d.k * sizeof( float ), cudaMemcpyHostToDevice ) );
			CUDA_CHECK( cudaMemcpy( sl.t_tail, d.tail.data(), static_cast<size_t>( T ) * sizeof( float ), cudaMemcpyHostToDevice ) );
			tq::Teacher t;
			t.ids = sl.t_ids; t.p = sl.t_p; t.tail = sl.t_tail; t.tk = d.k;
			LOG_INFO( "[volta_e2e] check: head on the engine's final streams, first sequence: KL %.6f per token", sl.m->head( sl.x, T, t, nullptr ) / T );
		}
		return 0;
	}
	// Sequences sqs (up to --seqs) in one pass: forward through the stages (saving each layer's input), the KL head, optionally the backward
	auto up_seqs = [&]( Stage& s, const uint16_t* src, size_t cols, const std::vector<int>& sqs, float* dst ) {
		const cudaStream_t ss = static_cast<cudaStream_t>( s.m->stream() );
		const size_t got = seq_rows( src, cols, T, sqs, 0, sqs.size() * T, s.staging, ss );
		tq::half_to_float( s.staging, dst, got * cols, ss );
		CUDA_CHECK( cudaStreamSynchronize( ss ) );
	};
	auto run = [&]( const std::vector<int>& sqs, bool train ) -> double {
		const int c = static_cast<int>( sqs.size() ), Rc = c * T;
		const size_t xb = static_cast<size_t>( Rc ) * flat * sizeof( float );
		Stage& s0 = st[0];
		CUDA_CHECK( cudaSetDevice( s0.dev ) );
		// the embedding row into every stream
		{
			float* e = s0.dy;   // [Rc, H] scratch
			up_seqs( s0, d.emb.data(), H, sqs, e );
			for ( int h = 0; h < hc; ++h ) tq::copy_cols( e, H, 0, s0.x, flat, h * H, Rc, H, false, s0.m->stream() );
		}
		for ( int g = 0; g < G; ++g ) {
			Stage& s = st[g];
			CUDA_CHECK( cudaSetDevice( s.dev ) );
			if ( s.pk ) {
				up_seqs( s, d.pk.data(), flat, sqs, s.pk );
				up_seqs( s, d.pv.data(), H, sqs, s.pv );
			}
			for ( int L = s.m->first(); L < s.m->last(); ++L ) {
				float* ck = s.ckpt[L - s.m->first()];
				CUDA_CHECK( cudaMemcpyAsync( ck, s.x, xb, cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>( s.m->stream() ) ) );
				s.m->forward( L, ck, s.x, Rc, s.pk, s.pv, false, c );
			}
			CUDA_CHECK( cudaStreamSynchronize( static_cast<cudaStream_t>( s.m->stream() ) ) );
			if ( g + 1 < G ) handoff( st[g + 1], st[g + 1].x, s, s.x, Rc );
		}
		Stage& sl = st[G - 1];
		CUDA_CHECK( cudaSetDevice( sl.dev ) );
		seq_teacher( d, T, sqs, sl.t_ids, sl.t_p, sl.t_tail, static_cast<cudaStream_t>( sl.m->stream() ) );
		tq::Teacher t;
		t.ids = sl.t_ids;
		t.p = sl.t_p;
		t.tail = sl.t_tail;
		t.tk = d.k;
		const double kl = sl.m->head( sl.x, Rc, t, train ? sl.dx : nullptr );
		if ( !train ) return kl;
		for ( int g = G - 1; g >= 0; --g ) {
			Stage& s = st[g];
			CUDA_CHECK( cudaSetDevice( s.dev ) );
			for ( int L = s.m->last() - 1; L >= s.m->first(); --L ) {
				s.m->backward( L, s.ckpt[L - s.m->first()], s.dx, s.dy, Rc, s.pk, s.pv, c );
				std::swap( s.dx, s.dy );
			}
			CUDA_CHECK( cudaStreamSynchronize( static_cast<cudaStream_t>( s.m->stream() ) ) );
			if ( g > 0 ) handoff( st[g - 1], st[g - 1].dx, s, s.dx, Rc );
		}
		return kl;
	};
	auto held_out = [&]() {
		double kl = 0.0;
		for ( int s0 = train_n; s0 < S; s0 += P ) {
			std::vector<int> sqs;
			for ( int sq = s0; sq < std::min( S, s0 + P ); ++sq ) sqs.push_back( sq );
			kl += run( sqs, false );
		}
		return kl / ( static_cast<double>( a.hold ) * T );
	};
	const auto t0 = std::chrono::steady_clock::now();
	double best = held_out();
	LOG_INFO( "[volta_e2e] held-out KL before tuning %.6f per token (%.1f s)", best,
		std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count() );
	for ( Stage& s : st ) s.m->keep_best();
	if ( a.timing ) for ( Stage& s : st ) { s.m->set_timing( true ); s.m->timing_report(); }
	int best_step = 0;
	engine::train::AdamOptions o;
	int sq = 0;
	for ( int step = 1; step <= a.steps; ++step ) {
		const auto ts = std::chrono::steady_clock::now();
		for ( Stage& s : st ) s.m->zero_grads();
		double kl = 0.0;
		for ( int b = 0; b < a.batch; b += P ) {
			std::vector<int> sqs;
			for ( int i = b; i < std::min( a.batch, b + P ); ++i ) {
				sqs.push_back( sq );
				sq = ( sq + 1 ) % train_used;
			}
			kl += run( sqs, true );
		}
		const float gs = 1.0f / ( static_cast<float>( a.batch ) * T );
		o.lr = lr_at( a, step );
		for ( Stage& s : st ) s.m->adam( o, step, gs );
		if ( a.timing ) {
			for ( Stage& s : st ) LOG_INFO( "[volta_e2e] step %d, device %d (layers %d-%d):\n%s", step, s.dev, s.m->first(), s.m->last() - 1, s.m->timing_report().c_str() );
		}
		LOG_INFO( "[volta_e2e] step %d: train KL %.6f per token, lr %.3g, %.1f s", step, kl / ( static_cast<double>( a.batch ) * T ), o.lr,
			std::chrono::duration<double>( std::chrono::steady_clock::now() - ts ).count() );
		if ( step % a.eval_every == 0 || step == a.steps ) {
			const double ho = held_out();
			LOG_INFO( "[volta_e2e] step %d: held-out KL %.6f per token (best %.6f at step %d)", step, ho, std::min( ho, best ), ho < best ? step : best_step );
			if ( ho < best ) {
				best = ho;
				best_step = step;
				for ( Stage& s : st ) s.m->keep_best();
			}
		}
	}
	if ( !a.write ) {
		LOG_INFO( "[volta_e2e] done (--write 0: no checkpoint written): best held-out KL %.6f per token at step %d", best, best_step );
		return 0;
	}
	Best best_scales;
	for ( Stage& s : st ) add_best( *s.m, best_scales );
	int replaced = 0;
	if ( !write_out( a, best_scales, replaced ) ) return 1;
	LOG_INFO( "[volta_e2e] done: %s, %d scale vectors and norms from step %d (held-out KL %.6f per token), %.1f min", a.out.c_str(), replaced, best_step, best,
		std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count() / 60.0 );
	return 0;
}
