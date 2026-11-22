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
// Adam's per-entry step adds up over its width); --router 1 trains the routers' weights at that rate as well. --gains 1 trains the tensors'
// 8-bit tile gains (straight through: FP32 values, the codes set from them after each step, written back as codes) at --gains-lr times the
// learning rate. --reencode L moves layer L's routed experts' codes instead of training (with --tp 1): their tile gradients and routed-input
// Hessians summed over the training sequences, then each --reencode-steps size re-encoded (each tile on its own toward R - eta G, or with
// --reencode-feedback 1 the LDL feedback toward R - eta M^-1 G), measured on the held-out sequences and undone (Model::reencode). --train N
// trains on the first N sequences only
// (the held-out ones unchanged: how the gain grows with the data). --adam-bits 8 keeps Adam's moments as 8-bit block-wise codes; --opt-host 1
// keeps the optimizer's state in pinned host memory (the devices hold the current values, their gradients and one part of the state: each
// step streams the state through in parts). --seqs N runs up to N of a step's --batch sequences (and of the held-out ones) through each
// layer call together (the same gradients; more rows per product, the layer inputs and the workspace N times larger). The memory per
// component is logged after the first step.
//
//   volta_e2e --model <EXL3 dir> --inputs <train_inputs file> --teacher <topk file> --out <dir> [--devices 0,1,2,3] [--tp 1] [--fp16 1]
//             [--norms 1] [--dense 0] [--dense-lr 0.1] [--router 0] [--hc 0] [--hc-lr 0.1] [--gains 0] [--gains-lr 1] [--reencode L] [--reencode-steps 0,0.003,0.01,0.03,0.1] [--reencode-feedback 0] [--steps 64] [--batch 4] [--seqs 1] [--lr 1e-3] [--warmup 2] [--lr-min 0.1] [--hold 4] [--train N] [--eval-every 16] [--write 0|1]
//             [--adam-bits 32|8] [--opt-host 0|1] [--timing 0|1] [--check-trace FILE]
//   --check-trace: a trace_layers file of the model's engine forward over the first sequence (test_ppl --calibrate --chunk 512): every layer of
//     the training form fed the engine's input streams of that layer, its output against the engine's, and the first sequence's embedding
//     streams against the engine's; then exits (the layer split). The trace may come from another checkpoint of the model (the same text
//     and inputs): each layer's own deviation from it. --check-routes FILE (a route_dump of that run): per layer, fed that run's input and
//     chained, the share of its routed experts this model picks too
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
#include <random>
#include <thread>
#include <vector>

namespace {

namespace tq = engine::train::qwen4_exp;
namespace q = engine::tools::quant;

struct Args {
	std::string model, inputs, teacher, out, check_trace, check_routes, router_teacher;
	float router_kl = 0.0f;                    // --router-kl: the routing distillation term's weight (0: off)
	bool router_calib = false;                 // --router-calib 1: router calibration (forward passes, the routers alone; with --router 1)
	int router_l0 = 0, router_l1 = 1 << 20;    // --router-kl-layers A-B: its layers
	std::vector<int> devices = { 0, 1, 2, 3 };
	int steps = 64, batch = 4, seqs = 1, hold = 4, eval_every = 16, warmup = 2, train = 0, adam_bits = 32, ckpt_host = -1, keep = -1, check_inputs = 0, route_cand = 0;
	bool write = true, timing = false, tp = true, fp16 = true, norms = true, dense = false, router = false, opt_host = false, qsa = true, shuffle = false, hc = false,
		gains = false;
	float lr = 1e-3f, lr_min = 0.1f, dense_lr = 0.1f, hc_lr = 0.1f, gains_lr = 1.0f;
	int reencode = -1;                                         // code updates of this layer's routed experts instead of training
	bool reencode_feedback = false;                            // with the LDL feedback (else each tile on its own)
	std::vector<float> reencode_steps = { 0.0f, 0.003f, 0.01f, 0.03f, 0.1f };
};

// The learning rate of a step (from 1): linear warmup, then a cosine from lr (the first step after the warmup) to lr * lr_min (the last step)
float lr_at( const Args& a, int step ) {
	if ( step <= a.warmup ) return a.lr * static_cast<float>( step ) / static_cast<float>( a.warmup + 1 );
	const int span = std::max( 1, a.steps - a.warmup - 1 );
	const double t = std::min( 1.0, static_cast<double>( step - a.warmup - 1 ) / span );
	return static_cast<float>( a.lr * ( a.lr_min + ( 1.0 - a.lr_min ) * 0.5 * ( 1.0 + std::cos( 3.14159265358979 * t ) ) ) );
}

// The training sequences in the order they are used: i-th of the run (cycling over n), in file order, or with shuffle in a new permutation
// each pass over them (fixed seeds: every rank and every run the same order)
struct TrainOrder {
	int n = 1;
	bool shuffle = false;
	int epoch = -1;
	std::vector<int> perm;
	int at( long i ) {
		if ( !shuffle ) return static_cast<int>( i % n );
		const int e = static_cast<int>( i / n );
		if ( e != epoch ) {
			perm.resize( n );
			for ( int j = 0; j < n; ++j ) perm[j] = j;
			std::mt19937 rng( 1234u + static_cast<unsigned>( e ) );
			std::shuffle( perm.begin(), perm.end(), rng );
			epoch = e;
		}
		return perm[i % n];
	}
};

// The kept values to write: per EXL3 tensor its (suh, svh), per tuned norm weight its values (FP16 each)
struct Best {
	std::map<std::string, std::pair<std::vector<uint16_t>, std::vector<uint16_t>>> scales;
	std::map<std::string, std::vector<uint16_t>> dense;
	std::map<std::string, std::vector<uint8_t>> gains;   // per EXL3 tensor its tile gains' codes (--gains)
};

void add_best( tq::Model& m, Best& out ) {
	std::vector<std::pair<std::string, std::pair<std::vector<uint16_t>, std::vector<uint16_t>>>> v;
	m.best_scales( v );
	for ( auto& e : v ) out.scales[e.first] = std::move( e.second );
	std::vector<std::pair<std::string, std::vector<uint16_t>>> d;
	m.best_dense( d );
	for ( auto& e : d ) out.dense[e.first] = std::move( e.second );
	std::vector<std::pair<std::string, std::vector<uint8_t>>> gv;
	m.best_gains( gv );
	for ( auto& e : gv ) out.gains[e.first] = std::move( e.second );
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
		else if ( k == "--ckpt-host" ) a.ckpt_host = std::stoi( v );
		else if ( k == "--keep" ) a.keep = std::stoi( v );
		else if ( k == "--qsa" ) a.qsa = v != "0";
		else if ( k == "--shuffle" ) a.shuffle = v != "0";
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
		else if ( k == "--hc" ) a.hc = v != "0";
		else if ( k == "--hc-lr" ) a.hc_lr = std::stof( v );
		else if ( k == "--route-cand" ) a.route_cand = std::stoi( v );
		else if ( k == "--reencode" ) a.reencode = std::stoi( v );
		else if ( k == "--reencode-feedback" ) a.reencode_feedback = v != "0";
		else if ( k == "--reencode-steps" ) {
			a.reencode_steps.clear();
			for ( size_t p = 0; p <= v.size(); ) {
				const size_t e = std::min( v.find( ',', p ), v.size() );
				if ( e > p ) a.reencode_steps.push_back( std::stof( v.substr( p, e - p ) ) );
				p = e + 1;
			}
		}
		else if ( k == "--gains" ) a.gains = v != "0";
		else if ( k == "--gains-lr" ) a.gains_lr = std::stof( v );
		else if ( k == "--opt-host" ) a.opt_host = v != "0";
		else if ( k == "--adam-bits" ) a.adam_bits = std::stoi( v );
		else if ( k == "--write" ) a.write = v != "0";
		else if ( k == "--timing" ) a.timing = v != "0";
		else if ( k == "--check-trace" ) a.check_trace = v;
		else if ( k == "--check-routes" ) a.check_routes = v;
		else if ( k == "--router-teacher" ) a.router_teacher = v;
		else if ( k == "--router-kl" ) a.router_kl = std::stof( v );
		else if ( k == "--router-calib" ) a.router_calib = v != "0";
		else if ( k == "--router-kl-layers" ) {
			const size_t dash = v.find( '-' );
			a.router_l0 = std::stoi( v.substr( 0, dash ) );
			a.router_l1 = dash == std::string::npos ? a.router_l0 + 1 : std::stoi( v.substr( dash + 1 ) ) + 1;
		}
		else if ( k == "--check-inputs" ) a.check_inputs = std::stoi( v );
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
	if ( a.model.empty() || a.teacher.empty() || a.out.empty() || ( a.adam_bits != 32 && a.adam_bits != 8 ) || a.seqs < 1 ) {
		LOG_ERROR( "usage: volta_e2e --model <EXL3 dir> --teacher <topk file> --out <dir> [--inputs <train_inputs file>] [--devices 0,1,2,3] [--tp 1] "
			"[--fp16 1] [--norms 1] [--dense 0] [--dense-lr 0.1] [--router 0] [--hc 0] [--hc-lr 0.1] [--gains 0] [--gains-lr 1] [--reencode L] [--reencode-steps 0,0.003,0.01,0.03,0.1] [--reencode-feedback 0] [--steps 64] [--batch 4] [--seqs 1] [--lr 1e-3] [--warmup 2] [--lr-min 0.1] [--hold 4] [--train N] [--eval-every 16] [--write 0|1] "
			"[--adam-bits 32|8] [--opt-host 0|1] [--ckpt-host -1|0|1] [--keep -1|0|1] [--qsa 1] [--timing 0|1] [--check-trace FILE]" );
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
	// the routing teacher (--router-teacher): per row and layer its rk most likely experts and their probabilities
	int rk = 0, rl = 0;
	std::vector<int16_t> r_ids;      // [n, rl, rk]
	std::vector<float> r_p;          // [n, rl, rk]
	int sequences() const { return seq > 0 ? n / seq : 0; }
	bool dumped() const { return !emb.empty(); }   // the inputs read from files, else made from the tokens (tq::Inputs)
	// the tokens of the sequences sqs end to end
	std::vector<int> tokens_of( const std::vector<int>& sqs ) const {
		std::vector<int> t;
		for ( int s : sqs ) t.insert( t.end(), tokens.begin() + static_cast<size_t>( s ) * seq, tokens.begin() + static_cast<size_t>( s + 1 ) * seq );
		return t;
	}
};

// router_dump files (comma-separated, the teacher files' texts in their order, --seq sequences, chunks of any size): per row of every
// sequence and layer the 16 largest router logits as probabilities (their exp less the log-sum-exp). A layer-0 record at position 0 opens
// the next sequence; records past the model's layers (the MTP block's) are skipped
bool read_router_teacher( const std::string& list, int layers, Data& d ) {
	d.rl = layers;
	d.rk = 16;
	d.r_ids.assign( static_cast<size_t>( d.n ) * layers * d.rk, -1 );
	d.r_p.assign( d.r_ids.size(), 0.0f );
	int s = -1;
	for ( size_t p0 = 0; p0 <= list.size(); ) {
		const size_t e = std::min( list.find( ',', p0 ), list.size() );
		const std::string path = list.substr( p0, e - p0 );
		p0 = e + 1;
		if ( path.empty() ) continue;
		FILE* f = std::fopen( path.c_str(), "rb" );
		if ( !f ) return false;
		int hdr[4];
		while ( std::fread( hdr, sizeof( int ), 4, f ) == 4 ) {
			const int L = hdr[0], m = hdr[1], pos = hdr[2], K = hdr[3];
			if ( K != d.rk ) {
				std::fclose( f );
				return false;
			}
			std::vector<int16_t> ids( static_cast<size_t>( m ) * K );
			std::vector<uint16_t> lg( ids.size() );
			std::vector<float> lse( m );
			if ( std::fread( ids.data(), 2, ids.size(), f ) != ids.size() || std::fread( lg.data(), 2, lg.size(), f ) != lg.size() ||
				std::fread( lse.data(), 4, lse.size(), f ) != lse.size() )
				break;
			if ( L == 0 && pos == 0 ) ++s;
			if ( L >= layers || s < 0 || ( s + 1 ) * d.seq > d.n ) continue;
			for ( int r = 0; r < m && pos + r < d.seq; ++r ) {
				const size_t at = ( ( static_cast<size_t>( s ) * d.seq + pos + r ) * layers + L ) * K;
				for ( int j = 0; j < K; ++j ) {
					__half_raw hr;
					hr.x = lg[static_cast<size_t>( r ) * K + j];
					d.r_ids[at + j] = ids[static_cast<size_t>( r ) * K + j];
					d.r_p[at + j] = std::exp( __half2float( __half( hr ) ) - lse[r] );
				}
			}
		}
		std::fclose( f );
	}
	if ( ( s + 1 ) * d.seq != d.n ) {
		LOG_ERROR( "[volta_e2e] the routing teacher holds %d sequences, the teacher %d", s + 1, d.sequences() );
		return false;
	}
	return true;
}

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
	const bool dumped = !is.empty();   // else the inputs are made from the tokens
	if ( ts.empty() || ( dumped && ts.size() != is.size() ) ) {
		LOG_ERROR( "[volta_e2e] %zu teacher files for %zu inputs files", ts.size(), is.size() );
		return false;
	}
	for ( size_t f = 0; f < ts.size(); ++f ) {
		Data e;
		if ( !read_teacher( ts[f], e ) || ( dumped && !read_inputs( is[f], H, flat, e ) ) ) return false;
		if ( f == 0 ) {
			const size_t n = dumped ? static_cast<size_t>( e.n ) : 0;
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
		const size_t n = dumped ? static_cast<size_t>( e.n ) : 0;
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
			const auto gt = suf == "tgain" ? best.gains.find( base ) : best.gains.end();
			if ( gt != best.gains.end() ) {
				ENGINE_ASSERT( gt->second.size() == en->bytes, "volta_e2e: a tensor's tile gains differ in count from the checkpoint's" );
				o2.owned.assign( gt->second.begin(), gt->second.end() );
				++replaced;
			}
			else if ( it != best_scales.end() && ( suf == "suh" || suf == "svh" ) && en->dtype == "F16" ) {
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
	// the layer inputs in host memory past 4 GiB of them a device (--ckpt-host -1); the kept sublayer outputs only within 2 GiB (--keep -1)
	const double in_bytes = static_cast<double>( cfg.layers ) * Rl * flat * sizeof( float ), kept_bytes = 2.0 * cfg.layers * Rl * H * sizeof( float );
	const bool ckpt_host = a.ckpt_host < 0 ? in_bytes > 4.0 * ( 1 << 30 ) : a.ckpt_host > 0;
	const bool keep = a.keep < 0 ? kept_bytes <= 2.0 * ( 1 << 30 ) : a.keep > 0;
	LOG_INFO( "[volta_e2e] %d rows a pass, %d a rank: layer inputs %.1f GiB (%s), kept sublayer outputs %s", R, Rl, in_bytes / ( 1 << 30 ),
		ckpt_host ? "host" : "device", keep ? "on" : "off (recomputed)" );
	std::vector<uint16_t> ple16;   // the pass's n-gram rows (inputs made from the tokens), shared by the ranks
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
		m->set_train_hc( a.hc, a.hc_lr );
		m->set_train_gains( a.gains, a.gains_lr );
		m->set_route_candidates( a.route_cand );
		m->set_keep_outputs( keep );
		ok[r] = m->load( a.model, 0, cfg.layers, dev, R, r, n, comms[r] ) ? 1 : 0;
		m->set_fp16( a.fp16 );
		m->set_train_norms( a.norms );
		m->set_qsa( a.qsa );
		m->set_router_calibration( a.router_calib );
		std::unique_ptr<tq::Inputs> inp;
		if ( ok[r] && !d.dumped() ) {
			inp = std::make_unique<tq::Inputs>();
			if ( !inp->open( a.model, dev, m->stream() ) ) ok[r] = 0;
			else if ( r == 0 ) ple16.resize( static_cast<size_t>( R ) * inp->ple_dim() );
		}
		bar.wait();
		for ( int g = 0; g < n; ++g ) if ( !ok[g] ) return;
		const cudaStream_t st = static_cast<cudaStream_t>( m->stream() );
		const size_t rows_l = static_cast<size_t>( Rl ) * flat;
		float *x, *dx, *dy, *emb, *pk, *pv, *t_p, *t_tail;
		int* t_ids;
		uint16_t* staging;
		for ( float** p : { &x, &dx, &dy } ) CUDA_CHECK( cudaMalloc( p, rows_l * sizeof( float ) ) );
		// the routing teacher of a pass's rows (rank 0: it alone adds the term), per layer its ids, probabilities and the mass outside them
		const bool routing = a.router_kl > 0.0f && !d.r_ids.empty() && r == 0;
		int *r_ids = nullptr;
		float *r_p = nullptr, *r_tail = nullptr;
		std::vector<tq::Teacher> r_layers;
		if ( routing ) {
			const size_t per = static_cast<size_t>( cfg.layers ) * R * d.rk;
			CUDA_CHECK( cudaMalloc( &r_ids, per * sizeof( int ) ) );
			CUDA_CHECK( cudaMalloc( &r_p, per * sizeof( float ) ) );
			CUDA_CHECK( cudaMalloc( &r_tail, static_cast<size_t>( cfg.layers ) * R * sizeof( float ) ) );
		}
		// the routing teacher's rows of the sequences sqs (end to end) into its buffers, then handed to the model
		auto routing_targets = [&]( const std::vector<int>& sqs ) {
			if ( !routing ) return;
			const int Rc = static_cast<int>( sqs.size() ) * T, K = d.rk;
			std::vector<int> hi( static_cast<size_t>( cfg.layers ) * Rc * K );
			std::vector<float> hp( hi.size() ), ht( static_cast<size_t>( cfg.layers ) * Rc );
			for ( int L = 0; L < cfg.layers; ++L )
				for ( size_t si = 0; si < sqs.size(); ++si )
					for ( int t = 0; t < T; ++t ) {
						const size_t row = si * T + t, src = ( ( static_cast<size_t>( sqs[si] ) * T + t ) * d.rl + L ) * K, dst = ( L * static_cast<size_t>( Rc ) + row ) * K;
						float sum = 0.0f;
						for ( int j = 0; j < K; ++j ) {
							hi[dst + j] = d.r_ids[src + j];
							sum += hp[dst + j] = d.r_p[src + j];
						}
						ht[L * static_cast<size_t>( Rc ) + row] = std::max( 0.0f, 1.0f - sum );
					}
			CUDA_CHECK( cudaMemcpyAsync( r_ids, hi.data(), hi.size() * sizeof( int ), cudaMemcpyHostToDevice, st ) );
			CUDA_CHECK( cudaMemcpyAsync( r_p, hp.data(), hp.size() * sizeof( float ), cudaMemcpyHostToDevice, st ) );
			CUDA_CHECK( cudaMemcpyAsync( r_tail, ht.data(), ht.size() * sizeof( float ), cudaMemcpyHostToDevice, st ) );
			CUDA_CHECK( cudaStreamSynchronize( st ) );
			r_layers.assign( cfg.layers, tq::Teacher{} );
			for ( int L = std::max( 0, a.router_l0 ); L < std::min( cfg.layers, a.router_l1 ); ++L ) {
				tq::Teacher& tt = r_layers[L];
				tt.ids = r_ids + static_cast<size_t>( L ) * Rc * K;
				tt.p = r_p + static_cast<size_t>( L ) * Rc * K;
				tt.tail = r_tail + static_cast<size_t>( L ) * Rc;
				tt.tk = K;
			}
			m->set_router_teacher( r_layers, a.router_kl );
		};
		const int r_layers_n = std::max( 0, std::min( cfg.layers, a.router_l1 ) - std::max( 0, a.router_l0 ) );
		CUDA_CHECK( cudaMalloc( &emb, static_cast<size_t>( Rl ) * H * sizeof( float ) ) );
		CUDA_CHECK( cudaMalloc( &pk, static_cast<size_t>( R ) * flat * sizeof( float ) ) );
		CUDA_CHECK( cudaMalloc( &pv, static_cast<size_t>( R ) * H * sizeof( float ) ) );
		CUDA_CHECK( cudaMalloc( &staging, static_cast<size_t>( R ) * flat * 2 ) );
		CUDA_CHECK( cudaMalloc( &t_ids, static_cast<size_t>( R ) * d.k * sizeof( int ) ) );
		CUDA_CHECK( cudaMalloc( &t_p, static_cast<size_t>( R ) * d.k * sizeof( float ) ) );
		CUDA_CHECK( cudaMalloc( &t_tail, static_cast<size_t>( R ) * sizeof( float ) ) );
		// Every layer's input for the backward: on the device, or with ckpt_host in pinned host memory, a layer's input copied out on a side
		// stream while the layer runs (two device buffers taking turns) and copied back ahead of its backward
		std::vector<float*> ckpt( cfg.layers );
		float* cbuf[2] = { nullptr, nullptr };
		cudaStream_t cst = nullptr;
		cudaEvent_t ev_in[2], ev_cp[2], ev_free[2];
		if ( ckpt_host ) {
			for ( float*& c : ckpt ) CUDA_CHECK( cudaMallocHost( &c, rows_l * sizeof( float ) ) );
			for ( float*& b : cbuf ) CUDA_CHECK( cudaMalloc( &b, rows_l * sizeof( float ) ) );
			CUDA_CHECK( cudaStreamCreateWithFlags( &cst, cudaStreamNonBlocking ) );
			for ( int i = 0; i < 2; ++i ) for ( cudaEvent_t* e : { &ev_in[i], &ev_cp[i], &ev_free[i] } ) CUDA_CHECK( cudaEventCreateWithFlags( e, cudaEventDisableTiming ) );
		}
		else for ( float*& c : ckpt ) CUDA_CHECK( cudaMalloc( &c, rows_l * sizeof( float ) ) );
		{
			size_t fr = 0, tot = 0;
			CUDA_CHECK( cudaMemGetInfo( &fr, &tot ) );
			LOG_INFO( "[volta_e2e] rank %d (device %d): all layers, experts %d-%d of %d, %.1f GiB free%s", r, dev, cfg.experts * r / n, cfg.experts * ( r + 1 ) / n - 1,
				cfg.experts, fr / 1073741824.0, ckpt_host ? " (layer inputs in host memory)" : "" );
		}
		// rows [r0, r1) of the sequences sqs laid end to end (cols FP16 values per row) into dst as FP32
		auto up = [&]( const uint16_t* src, size_t cols, const std::vector<int>& sqs, size_t r0, size_t r1, float* dst ) {
			const size_t got = seq_rows( src, cols, T, sqs, r0, r1, staging, st );
			tq::half_to_float( staging, dst, got * cols, st );
			CUDA_CHECK( cudaStreamSynchronize( st ) );
		};
		// A pass's inputs: this rank's rows' embeddings, every row's PLE rows (made from the tokens: the n-gram lookups split over the ranks)
		auto inputs = [&]( const std::vector<int>& sqs, int Rc, int Rcl ) {
			if ( d.dumped() ) {
				up( d.emb.data(), H, sqs, static_cast<size_t>( r ) * Rcl, static_cast<size_t>( r + 1 ) * Rcl, emb );
				up( d.pk.data(), flat, sqs, 0, Rc, pk );
				up( d.pv.data(), H, sqs, 0, Rc, pv );
				return;
			}
			const std::vector<int> ids = d.tokens_of( sqs );
			inp->embed( ids.data() + static_cast<size_t>( r ) * Rcl, Rcl, emb );
			const size_t D = inp->ple_dim();
			bar.wait();   // every rank's projection of the previous pass is done with the buffer
			for ( size_t s = 0; s < sqs.size(); ++s ) inp->ple_rows( ids.data() + s * T, T, r, n, ple16.data() + s * T * D );
			bar.wait();
			inp->ple_project( ple16.data(), Rc, pk, pv );
		};
		// Sequences sqs (up to --seqs) in one pass: the forward over all layers (saving each layer's input), the KL head, optionally the backward
		auto run = [&]( const std::vector<int>& sqs, bool train ) -> double {
			const int c = static_cast<int>( sqs.size() ), Rc = c * T, Rcl = Rc / n;
			inputs( sqs, Rc, Rcl );
			routing_targets( sqs );
			const size_t xb = static_cast<size_t>( Rcl ) * flat * sizeof( float );
			float* x0 = ckpt_host ? cbuf[0] : x;
			for ( int h = 0; h < hc; ++h ) tq::copy_cols( emb, H, 0, x0, flat, h * H, Rcl, H, false, st );
			for ( int L = 0; L < cfg.layers; ++L ) {
				if ( !ckpt_host ) {
					CUDA_CHECK( cudaMemcpyAsync( ckpt[L], x, xb, cudaMemcpyDeviceToDevice, st ) );
					m->forward( L, ckpt[L], x, Rc, pk, pv, train, c );
					continue;
				}
				float *in = cbuf[L & 1], *out = cbuf[( L + 1 ) & 1];
				if ( train ) {
					if ( L >= 1 ) CUDA_CHECK( cudaStreamWaitEvent( st, ev_cp[( L + 1 ) & 1], 0 ) );   // out was the previous layer's input: copied out
					CUDA_CHECK( cudaEventRecord( ev_in[L & 1], st ) );
					CUDA_CHECK( cudaStreamWaitEvent( cst, ev_in[L & 1], 0 ) );
					CUDA_CHECK( cudaMemcpyAsync( ckpt[L], in, xb, cudaMemcpyDeviceToHost, cst ) );
					CUDA_CHECK( cudaEventRecord( ev_cp[L & 1], cst ) );
				}
				m->forward( L, in, out, Rc, pk, pv, train, c );
			}
			float* xf = ckpt_host ? cbuf[cfg.layers & 1] : x;
			seq_teacher( d, T, sqs, t_ids, t_p, t_tail, st );
			tq::Teacher t;
			t.ids = t_ids;
			t.p = t_p;
			t.tail = t_tail;
			t.tk = d.k;
			const double kl = m->head( xf, Rc, t, train ? dx : nullptr );
			if ( !train ) return kl;
			if ( !ckpt_host ) {
				for ( int L = cfg.layers - 1; L >= 0; --L ) {
					m->backward( L, ckpt[L], dx, dy, Rc, pk, pv, c );
					std::swap( dx, dy );
				}
				return kl;
			}
			// the inputs back, a layer ahead: into the buffer its layer's parity owns, once that buffer's last reader is done
			for ( int i = 0; i < 2; ++i ) CUDA_CHECK( cudaEventRecord( ev_free[i], st ) );
			auto fetch = [&]( int L ) {
				CUDA_CHECK( cudaStreamWaitEvent( cst, ev_free[L & 1], 0 ) );
				CUDA_CHECK( cudaMemcpyAsync( cbuf[L & 1], ckpt[L], xb, cudaMemcpyHostToDevice, cst ) );
				CUDA_CHECK( cudaEventRecord( ev_in[L & 1], cst ) );
			};
			fetch( cfg.layers - 1 );
			for ( int L = cfg.layers - 1; L >= 0; --L ) {
				if ( L >= 1 ) fetch( L - 1 );
				CUDA_CHECK( cudaStreamWaitEvent( st, ev_in[L & 1], 0 ) );
				m->backward( L, cbuf[L & 1], dx, dy, Rc, pk, pv, c );
				CUDA_CHECK( cudaEventRecord( ev_free[L & 1], st ) );
				std::swap( dx, dy );
			}
			return kl;
		};
		// with the routing teacher, the routing KL per token and layer of the last held-out pass (rank 0)
		double ho_router = 0.0, ho_shared = 0.0, tr_shared = 0.0;
		auto held_out = [&]() {
			double kl = 0.0;
			m->router_kl();
			for ( int s0 = train_n; s0 < S; s0 += P ) {
				std::vector<int> sqs;
				for ( int sq = s0; sq < std::min( S, s0 + P ); ++sq ) sqs.push_back( sq );
				kl += run( sqs, false );
			}
			ho_router = m->router_kl( &ho_shared ) / ( static_cast<double>( a.hold ) * T * std::max( r_layers_n, 1 ) );
			return kl / ( static_cast<double>( a.hold ) * T );
		};
		auto routing_note = [&]( double v, double sh ) {
			std::string s;
			if ( routing ) {
				char b[96];
				std::snprintf( b, sizeof( b ), ", routing KL %.4f per layer (%.1f%% of the teacher's experts picked)", v, 100.0 * sh );
				s = b;
			}
			return s;
		};
		const auto th0 = std::chrono::steady_clock::now();
		double my_best = held_out();
		if ( r == 0 )
			LOG_INFO( "[volta_e2e] held-out KL before tuning %.6f per token%s (%.1f s)", my_best, routing_note( ho_router, ho_shared ).c_str(),
				std::chrono::duration<double>( std::chrono::steady_clock::now() - th0 ).count() );
		m->keep_best();
		if ( a.reencode >= 0 ) {
			// Code updates of one layer's routed experts: their gradients with respect to the tiles and their inputs' Hessians summed over the
			// training sequences, then each step size (a fraction of the tiles' norm in the quantization metric) re-encoded, measured, undone
			const auto ta = std::chrono::steady_clock::now();
			m->set_accumulate( a.reencode );
			m->zero_grads();
			for ( int s0 = 0; s0 < train_used; s0 += P ) {
				std::vector<int> sqs;
				for ( int sq = s0; sq < std::min( train_used, s0 + P ); ++sq ) sqs.push_back( sq );
				run( sqs, true );
			}
			const double rho = m->reencode_prepare( a.reencode_feedback );
			// the KL on the first (up to) 20 training sequences too: whether a step lowers the loss it was taken on
			const int tn = std::min( train_used, 20 );
			auto train_kl = [&]() {
				double kl = 0.0;
				for ( int s0 = 0; s0 < tn; s0 += P ) {
					std::vector<int> sqs;
					for ( int sq = s0; sq < std::min( tn, s0 + P ); ++sq ) sqs.push_back( sq );
					kl += run( sqs, false );
				}
				return kl / ( static_cast<double>( tn ) * T );
			};
			const double tr0 = train_kl();
			if ( r == 0 )
				LOG_INFO( "[volta_e2e] layer %d: gradient and Hessian sums over %d sequences, %.0f s; |D| / |R| %.4g (%s); KL on %d of them %.6f",
					a.reencode, train_used, std::chrono::duration<double>( std::chrono::steady_clock::now() - ta ).count(), rho,
					a.reencode_feedback ? "D = M^-1 G, in the metric" : "D = G, plain", tn, tr0 );
			for ( float s : a.reencode_steps ) {
				const auto tr = std::chrono::steady_clock::now();
				const float eta = rho > 0.0 ? static_cast<float>( s / rho ) : 0.0f;
				double lin = 0.0, quad = 0.0;
				const double pe = m->reencode( eta, a.reencode_feedback, &lin, &quad );
				const double ho = held_out(), tk = train_kl();
				if ( r == 0 )
					LOG_INFO( "[volta_e2e] layer %d re-encoded at a step of %.3g of |R| (eta %.4g%s): proxy error %.4g (rank 0), first-order change %.3g per "
						"training token (dQ^T M dQ %.3g), held-out KL %.6f (before %.6f), training KL %.6f (before %.6f), %.0f s", a.reencode, s, eta,
						a.reencode_feedback ? ", with the feedback" : "", pe, lin / ( static_cast<double>( train_used ) * T ), quad, ho, my_best, tk, tr0,
						std::chrono::duration<double>( std::chrono::steady_clock::now() - tr ).count() );
				m->restore_codes();
			}
			m->set_accumulate( -1 );
		}
		if ( a.timing ) {
			m->set_timing( true );
			m->timing_report();
		}
		int my_step = 0;
		long used = 0;
		TrainOrder order;
		order.n = train_used;
		order.shuffle = a.shuffle;
		engine::train::AdamOptions o;
		for ( int step = 1; step <= ( a.reencode >= 0 ? 0 : a.steps ); ++step ) {
			const auto ts = std::chrono::steady_clock::now();
			m->zero_grads();
			double kl = 0.0;
			for ( int b = 0; b < a.batch; b += P ) {
				std::vector<int> sqs;
				for ( int i = b; i < std::min( a.batch, b + P ); ++i ) sqs.push_back( order.at( used++ ) );
				kl += run( sqs, !a.router_calib );   // router calibration: forward passes (the routers' gradient comes with them)
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
			const double tr_router = m->router_kl( &tr_shared ) / ( static_cast<double>( a.batch ) * T * std::max( r_layers_n, 1 ) );
			if ( r == 0 )
				LOG_INFO( "[volta_e2e] step %d: train KL %.6f per token%s, lr %.3g, %.1f s", step, kl / ( static_cast<double>( a.batch ) * T ),
					routing_note( tr_router, tr_shared ).c_str(), o.lr, std::chrono::duration<double>( std::chrono::steady_clock::now() - ts ).count() );
			if ( step % a.eval_every == 0 || step == a.steps ) {
				const bool was = a.timing;
				if ( was ) m->set_timing( false );
				const double ho = held_out();
				if ( was ) m->set_timing( true );
				if ( r == 0 )
					LOG_INFO( "[volta_e2e] step %d: held-out KL %.6f per token%s (best %.6f at step %d)", step, ho, routing_note( ho_router, ho_shared ).c_str(),
						std::min( ho, my_best ), ho < my_best ? step : my_step );
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
		for ( float* p : { x, dx, dy, emb, pk, pv, t_p, t_tail, r_p, r_tail } ) if ( p ) cudaFree( p );
		if ( r_ids ) cudaFree( r_ids );
		cudaFree( t_ids );
		cudaFree( staging );
		for ( float* c : ckpt ) ckpt_host ? cudaFreeHost( c ) : cudaFree( c );
		if ( ckpt_host ) {
			for ( float* b : cbuf ) cudaFree( b );
			for ( int i = 0; i < 2; ++i ) for ( cudaEvent_t e : { ev_in[i], ev_cp[i], ev_free[i] } ) cudaEventDestroy( e );
			cudaStreamDestroy( cst );
		}
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
	if ( !a.router_teacher.empty() && !read_router_teacher( a.router_teacher, cfg.layers, d ) ) {
		LOG_ERROR( "[volta_e2e] cannot read the routing teacher (%s)", a.router_teacher.c_str() );
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
	if ( a.check_inputs > 0 ) {
		// The inputs made from the tokens against the dumped ones (--inputs) for the first N sequences: relative error per kind
		if ( !d.dumped() ) return 1;
		CUDA_CHECK( cudaSetDevice( a.devices[0] ) );
		cudaStream_t s0;
		CUDA_CHECK( cudaStreamCreate( &s0 ) );
		tq::Inputs inp;
		if ( !inp.open( a.model, a.devices[0], s0 ) ) return 1;
		float *e, *k, *v;
		CUDA_CHECK( cudaMalloc( &e, static_cast<size_t>( T ) * H * sizeof( float ) ) );
		CUDA_CHECK( cudaMalloc( &k, static_cast<size_t>( T ) * flat * sizeof( float ) ) );
		CUDA_CHECK( cudaMalloc( &v, static_cast<size_t>( T ) * H * sizeof( float ) ) );
		std::vector<uint16_t> p16( static_cast<size_t>( T ) * inp.ple_dim() );
		auto rel = [&]( const float* dev, const uint16_t* ref, size_t cnt ) {
			std::vector<float> h( cnt );
			CUDA_CHECK( cudaMemcpy( h.data(), dev, cnt * sizeof( float ), cudaMemcpyDeviceToHost ) );
			double dd = 0.0, nn = 0.0;
			for ( size_t i = 0; i < cnt; ++i ) {
				__half_raw hr;
				hr.x = ref[i];
				const double r = __half2float( __half( hr ) );
				dd += ( h[i] - r ) * ( h[i] - r );
				nn += r * r;
			}
			return std::sqrt( dd / ( nn + 1e-30 ) );
		};
		for ( int sq = 0; sq < std::min( a.check_inputs, S ); ++sq ) {
			const std::vector<int> ids = d.tokens_of( { sq } );
			const size_t b = static_cast<size_t>( sq ) * T;
			inp.embed( ids.data(), T, e );
			inp.ple_rows( ids.data(), T, 0, 1, p16.data() );
			inp.ple_project( p16.data(), T, k, v );
			LOG_INFO( "[volta_e2e] check: sequence %d inputs from its tokens against the dumped ones: embeddings %.3g, PLE keys %.3g, values %.3g", sq,
				rel( e, d.emb.data() + b * H, static_cast<size_t>( T ) * H ), rel( k, d.pk.data() + b * flat, static_cast<size_t>( T ) * flat ),
				rel( v, d.pv.data() + b * H, static_cast<size_t>( T ) * H ) );
		}
		return 0;
	}
#if defined( ENGINE_HAVE_NCCL )
	if ( a.tp && G > 1 && a.check_trace.empty() ) return train_tp( a, cfg, d );
#endif
	if ( !d.dumped() ) {
		LOG_ERROR( "[volta_e2e] inputs made from the tokens need --tp 1 over several devices (else give --inputs)" );
		return 1;
	}
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
		s.m->set_train_hc( a.hc, a.hc_lr );
		s.m->set_train_gains( a.gains, a.gains_lr );
		s.m->set_route_candidates( a.route_cand );
		s.m->set_keep_outputs( a.keep != 0 );
		if ( !s.m->load( a.model, first, last, s.dev, R ) ) return 1;
		s.m->set_fp16( a.fp16 );
		s.m->set_train_norms( a.norms );
		s.m->set_qsa( a.qsa );
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
		// --check-routes: another run's route_dump of the same text and chunking (its routed experts per layer and row), compared with this
		// model's selections: per row the share of that run's top-k experts this one picks too
		std::map<int, std::vector<int>> ref_ids;
		if ( !a.check_routes.empty() ) {
			FILE* fr = std::fopen( a.check_routes.c_str(), "rb" );
			if ( !fr ) return 1;
			int rh[4];
			while ( std::fread( rh, sizeof( int ), 4, fr ) == 4 ) {
				const int layer = rh[0], m = rh[1], p0 = rh[2], K = rh[3];
				std::vector<int> v( static_cast<size_t>( m ) * K );
				if ( std::fread( v.data(), sizeof( int ), v.size(), fr ) != v.size() ) break;
				std::fseek( fr, static_cast<long>( 2 ) * m * H, SEEK_CUR );   // the router input rows (FP16)
				if ( K != cfg.top_k || p0 >= T ) continue;
				auto& dst = ref_ids[layer];
				dst.resize( static_cast<size_t>( T ) * K );
				std::memcpy( dst.data() + static_cast<size_t>( p0 ) * K, v.data(), std::min( m, T - p0 ) * K * sizeof( int ) );
			}
			std::fclose( fr );
		}
		auto overlap = [&]( const tq::Model& mm, int L ) -> double {
			const auto it = ref_ids.find( L );
			if ( it == ref_ids.end() ) return -1.0;
			const std::vector<int>& mine = mm.selection( L );
			const int K = cfg.top_k;
			if ( mine.size() < static_cast<size_t>( T ) * K ) return -1.0;
			long hit = 0;
			for ( int rr = 0; rr < T; ++rr )
				for ( int j = 0; j < K; ++j )
					for ( int q = 0; q < K; ++q ) hit += mine[static_cast<size_t>( rr ) * K + q] == it->second[static_cast<size_t>( rr ) * K + j];
			return static_cast<double>( hit ) / ( static_cast<double>( T ) * K );
		};
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
				CUDA_CHECK( cudaDeviceSynchronize() );   // a pageable copy's DMA may outlive the call; the model's stream does not wait for it
				s.m->forward( L, s.x, s.y, T, s.pk, s.pv );
				CUDA_CHECK( cudaStreamSynchronize( static_cast<cudaStream_t>( s.m->stream() ) ) );
				CUDA_CHECK( cudaMemcpy( host.data(), s.y, host.size() * sizeof( float ), cudaMemcpyDeviceToHost ) );
				// the layer's own change against the engine's (the streams carry the input through the residual)
				std::vector<float> dt( host.size() ), de( host.size() );
				const std::vector<float>& ref = tr[600 + L];
				for ( size_t i = 0; i < host.size(); ++i ) { dt[i] = host[i] - in[i]; de[i] = ref[i] - in[i]; }
				char rt[64] = "";
				if ( overlap( *s.m, L ) >= 0.0 ) std::snprintf( rt, sizeof( rt ), ", routes shared %.1f%%", 100.0 * overlap( *s.m, L ) );
				LOG_INFO( "[volta_e2e] check: layer %2d (%s%s): output streams %.3g, the layer's change %.3g%s", L, cfg.full_attention( L ) ? "attention" : "GDN",
					L == cfg.ple_layer ? " + PLE" : "", relerr( host, ref ), relerr( dt, de ), rt );
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
					char rt[64] = "";
					if ( overlap( *s.m, L ) >= 0.0 ) std::snprintf( rt, sizeof( rt ), ", routes shared %.1f%%", 100.0 * overlap( *s.m, L ) );
					if ( L % 4 == 0 || L == cfg.layers - 1 || rt[0] )
						LOG_INFO( "[volta_e2e] check: chained, after layer %2d: streams %.3g%s", L, relerr( host, tr[600 + L] ), rt );
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
			CUDA_CHECK( cudaDeviceSynchronize() );
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
	long used = 0;
	TrainOrder order;
	order.n = train_used;
	order.shuffle = a.shuffle;
	for ( int step = 1; step <= a.steps; ++step ) {
		const auto ts = std::chrono::steady_clock::now();
		for ( Stage& s : st ) s.m->zero_grads();
		double kl = 0.0;
		for ( int b = 0; b < a.batch; b += P ) {
			std::vector<int> sqs;
			for ( int i = b; i < std::min( a.batch, b + P ); ++i ) sqs.push_back( order.at( used++ ) );
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
