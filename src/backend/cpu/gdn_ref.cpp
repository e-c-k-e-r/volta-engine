// CPU backend: the remaining GDN / KDA entry points. The single-row Gated DeltaNet forms (causal_conv1d_silu, recurrent_step) are
// composed from the existing Slang row kernels (same state layouts as the CUDA kernels). The KDA gate projection and the KDA
// recurrences are plain C++ mirroring src/modules/gdn.cu (FP32 state and accumulation, FP16 storage).
// cpu_kernels.h first: the Slang prelude defines int4, which the backend shim only declares when the prelude is absent
#include "cpu_kernels.h"
#include "engine/modules/gdn.h"
#include "engine/backend/tiers.h"
#include <cmath>
#include <vector>

namespace engine::modules {

namespace {

using h16 = backend::half_t;

// One-row batch for a SlotRef: the rows kernels resolve the slot through BatchRows::slot[0]
pod::BatchRows one_row( const pod::SlotRef& slot, int m ) {
	pod::BatchRows r;
	r.count = 1;
	r.slot[0] = slot.rows != nullptr ? slot.rows->slot[slot.row + m] : 0;
	r.pos[0] = slot.rows != nullptr ? slot.rows->pos[slot.row + m] : 0;
	return r;
}

float dot_h( const h16* a, const h16* b, int n ) {
	float s = 0.0f;
	for ( int i = 0; i < n; ++i ) s = std::fma( static_cast<float>( a[i] ), static_cast<float>( b[i] ), s );
	return s;
}

// One KDA step for one head and one row: S = diag(exp(g)) S; delta = beta (v - S^T k); S += k delta^T; y = S^T q. State layout [row][col].
// y is the unnormalized core output (128 columns).
void kda_head_step( float* st, const h16* q, const h16* k, const h16* v, const float* g, float beta, float* y ) {
	float gexp[128], kf[128], qf[128];
	for ( int r = 0; r < 128; ++r ) {
		gexp[r] = std::exp( g[r] );
		kf[r] = static_cast<float>( k[r] );
		qf[r] = static_cast<float>( q[r] );
	}
	for ( int c = 0; c < 128; ++c ) {
		float kv_mem = 0.0f;
		for ( int r = 0; r < 128; ++r ) {
			const float s = st[r * 128 + c] * gexp[r];
			st[r * 128 + c] = s;
			kv_mem = std::fma( s, kf[r], kv_mem );
		}
		const float delta = ( static_cast<float>( v[c] ) - kv_mem ) * beta;
		float yc = 0.0f;
		for ( int r = 0; r < 128; ++r ) {
			const float s = std::fma( kf[r], delta, st[r * 128 + c] );
			st[r * 128 + c] = s;
			yc = std::fma( s, qf[r], yc );
		}
		y[c] = yc;
	}
}

void gate_norm_out( const float* y, const h16* o_norm, const h16* z, float eps, h16* out ) {
	float sq = 0.0f;
	for ( int c = 0; c < 128; ++c ) sq += y[c] * y[c];
	const float inv_rms = 1.0f / std::sqrt( sq / 128.0f + eps );
	for ( int c = 0; c < 128; ++c ) {
		const float gamma = o_norm != nullptr ? static_cast<float>( o_norm[c] ) : 1.0f;
		const float zv = z != nullptr ? static_cast<float>( z[c] ) : 0.0f;
		out[c] = static_cast<h16>( y[c] * inv_rms * gamma * ( 1.0f / ( 1.0f + std::exp( -zv ) ) ) );
	}
}

}

void GDN::causal_conv1d_silu( pod::TensorView& qkv, pod::TensorView& conv_history, const pod::TensorView& conv_weights, int channels, backend::stream_t stream,
	const pod::SlotRef& slot, float* snap, const int* snap_on ) {
	pod::BatchRows br = one_row( slot, 0 );
	causal_conv1d_silu_rows( qkv, conv_history.ptr<float>(), conv_weights, channels, 1, &br, slot.stride, snap, 0, 1, snap_on, stream );
}

void GDN::recurrent_step( float* gdn_state, const pod::TensorView& qkv, const pod::TensorView& gate, const pod::TensorView& beta, pod::TensorView& attn_out,
	int num_k_heads, int num_v_heads, int head_dim, backend::stream_t stream, const pod::SlotRef& slot, float* snap, const int* snap_on ) {
	const int tokens = ( qkv.ndim >= 1 && qkv.shape[0] > 0 ) ? qkv.shape[0] : 1;
	const size_t qkv_stride = static_cast<size_t>( 2 * num_k_heads + num_v_heads ) * head_dim;
	// Token by token, each resolving its own slot row like the CUDA path
	for ( int m = 0; m < tokens; ++m ) {
		pod::BatchRows br = one_row( slot, m );
		pod::TensorView qkv_m = qkv, gate_m = gate, beta_m = beta, out_m = attn_out;
		qkv_m.data = const_cast<h16*>( qkv.ptr<const h16>() ) + m * qkv_stride;
		gate_m.data = const_cast<float*>( gate.ptr<const float>() ) + static_cast<size_t>( m ) * num_v_heads;
		beta_m.data = const_cast<float*>( beta.ptr<const float>() ) + static_cast<size_t>( m ) * num_v_heads;
		out_m.data = attn_out.ptr<h16>() + static_cast<size_t>( m ) * num_v_heads * head_dim;
		recurrent_step_rows( gdn_state, qkv_m, gate_m, beta_m, out_m, num_k_heads, num_v_heads, head_dim, 1, &br, slot.stride, snap, 0, 1, snap_on, stream );
	}
}

void GDN::project_kda_gates( const pod::TensorView& norm_hidden, const pod::TensorView& b_proj, const pod::TensorView& f_a_proj, const pod::TensorView& f_b_proj,
	const pod::TensorView& g_a_proj, const pod::TensorView& g_b_proj, const pod::TensorView& A_log, const pod::TensorView& dt_bias, pod::TensorView& beta_out,
	pod::TensorView& g_decay_out, pod::TensorView& z_out, pod::TensorView& f_a_buf, pod::TensorView& g_a_buf, int num_heads, int head_dim, int hidden_size,
	backend::stream_t, int rows ) {
	ENGINE_ASSERT( num_heads == 16 && head_dim == 128, "GDN::project_kda_gates (CPU): 16 heads of 128" );
	const int channels = num_heads * head_dim;
	const h16* x = norm_hidden.ptr<const h16>();
	for ( int r = 0; r < rows; ++r ) {
		const h16* xr = x + static_cast<size_t>( r ) * hidden_size;
		float* beta = beta_out.ptr<float>() + r * 16;
		h16* fa = f_a_buf.ptr<h16>() + r * 128;
		h16* ga = g_a_buf.ptr<h16>() + r * 128;
		if ( b_proj.data != nullptr ) {
			for ( int i = 0; i < 16; ++i ) beta[i] = 1.0f / ( 1.0f + std::exp( -dot_h( xr, b_proj.ptr<const h16>() + static_cast<size_t>( i ) * hidden_size, hidden_size ) ) );
		}
		if ( f_a_proj.data != nullptr ) {
			for ( int i = 0; i < 128; ++i ) fa[i] = static_cast<h16>( dot_h( xr, f_a_proj.ptr<const h16>() + static_cast<size_t>( i ) * hidden_size, hidden_size ) );
		}
		if ( g_a_proj.data != nullptr ) {
			for ( int i = 0; i < 128; ++i ) ga[i] = static_cast<h16>( dot_h( xr, g_a_proj.ptr<const h16>() + static_cast<size_t>( i ) * hidden_size, hidden_size ) );
		}
		for ( int j = 0; j < channels; ++j ) {
			if ( f_b_proj.data != nullptr ) {
				const float f_val = dot_h( fa, f_b_proj.ptr<const h16>() + static_cast<size_t>( j ) * 128, 128 );
				const float dt = dt_bias.data != nullptr ? static_cast<float>( dt_bias.ptr<const h16>()[j] ) : 0.0f;
				const float a_val = A_log.data != nullptr ? static_cast<float>( A_log.ptr<const h16>()[j / 128] ) : 0.0f;
				g_decay_out.ptr<float>()[static_cast<size_t>( r ) * channels + j] = -5.0f / ( 1.0f + std::exp( -std::exp( a_val ) * ( f_val + dt ) ) );
			}
			if ( g_b_proj.data != nullptr ) {
				z_out.ptr<h16>()[static_cast<size_t>( r ) * channels + j] = static_cast<h16>( dot_h( ga, g_b_proj.ptr<const h16>() + static_cast<size_t>( j ) * 128, 128 ) );
			}
		}
	}
}

// Unfused single step: the state is decayed while kv_mem is read from the undecayed column (as src/modules/gdn.cu does); no norm
void GDN::recurrent_step_kda( float* gdn_state, const pod::TensorView& qkv, const pod::TensorView& g_decay, const pod::TensorView& beta, pod::TensorView& attn_out,
	int num_heads, int head_dim, backend::stream_t ) {
	const h16* base = qkv.ptr<const h16>();
	for ( int h = 0; h < num_heads; ++h ) {
		float* st = gdn_state + static_cast<size_t>( h ) * head_dim * head_dim;
		const h16* q = base + h * head_dim;
		const h16* k = base + num_heads * head_dim + h * head_dim;
		const h16* v = base + 2 * num_heads * head_dim + h * head_dim;
		const float* g = g_decay.ptr<const float>() + h * head_dim;
		const float b = beta.ptr<const float>()[h];
		for ( int c = 0; c < head_dim; ++c ) {
			float kv_mem = 0.0f;
			for ( int r = 0; r < head_dim; ++r ) {
				const float sp = st[r * head_dim + c];
				kv_mem = std::fma( sp, static_cast<float>( k[r] ), kv_mem );
				st[r * head_dim + c] = sp * std::exp( g[r] );
			}
			const float delta = ( static_cast<float>( v[c] ) - kv_mem ) * b;
			float acc = 0.0f;
			for ( int r = 0; r < head_dim; ++r ) {
				const float s = std::fma( static_cast<float>( k[r] ), delta, st[r * head_dim + c] );
				st[r * head_dim + c] = s;
				acc = std::fma( s, static_cast<float>( q[r] ), acc );
			}
			attn_out.ptr<h16>()[h * head_dim + c] = static_cast<h16>( acc );
		}
	}
}

void GDN::recurrent_step_kda_fused( float* gdn_state, const pod::TensorView& qkv, const pod::TensorView& g_decay, const pod::TensorView& beta, const pod::TensorView& z,
	const pod::TensorView& o_norm, pod::TensorView& attn_out, int num_heads, int head_dim, float eps, backend::stream_t, const pod::SlotRef& slot ) {
	ENGINE_ASSERT( head_dim == 128, "GDN::recurrent_step_kda_fused: head_dim must be 128" );
	float* state = slot.apply( gdn_state );
	const int channels = num_heads * 128;
	const h16* base = qkv.ptr<const h16>();
	float y[128];
	for ( int h = 0; h < num_heads; ++h ) {
		kda_head_step( state + static_cast<size_t>( h ) * 128 * 128, base + h * 128, base + channels + h * 128, base + 2 * channels + h * 128,
			g_decay.ptr<const float>() + h * 128, beta.ptr<const float>()[h], y );
		gate_norm_out( y, o_norm.data != nullptr ? o_norm.ptr<const h16>() : nullptr, z.data != nullptr ? z.ptr<const h16>() + h * 128 : nullptr, eps, attn_out.ptr<h16>() + h * 128 );
	}
}

bool GDN::kda_journal_supported() { return false; }
size_t GDN::kda_journal_row_bytes( int ) { return 0; }
void GDN::recurrent_step_kda_rows_journal( float*, const pod::TensorView&, const pod::TensorView&, const pod::TensorView&, const pod::TensorView&, const pod::TensorView&,
	pod::TensorView&, int, int, float, int, const pod::BatchRows*, size_t, unsigned char*, size_t, int, const int*, backend::stream_t ) {
	ENGINE_UNSUPPORTED( "GDN::recurrent_step_kda_rows_journal" );
}
void GDN::kda_replay( float*, size_t, size_t, int, int, const unsigned char*, size_t, size_t, int, int, int, backend::stream_t ) {
	ENGINE_UNSUPPORTED( "GDN::kda_replay" );
}

void GDN::recurrent_step_kda_rows( float* gdn_state, const pod::TensorView& qkv, const pod::TensorView& g_decay, const pod::TensorView& beta, const pod::TensorView& z,
	const pod::TensorView& o_norm, pod::TensorView& attn_out, int num_heads, int head_dim, float eps, int rows, const pod::BatchRows* d_rows, size_t slot_stride,
	float* snap, size_t snap_stride, int snap_rows, const int* snap_on, backend::stream_t ) {
	ENGINE_ASSERT( head_dim == 128, "GDN::recurrent_step_kda_rows: head_dim must be 128" );
	const int channels = num_heads * 128;
	const bool do_snap = snap != nullptr && *snap_on;
	float y[128];
	for ( int m = 0; m < rows; ++m ) {
		float* state = gdn_state + static_cast<size_t>( d_rows->slot[m] ) * slot_stride;
		const h16* q = qkv.ptr<const h16>() + static_cast<size_t>( m ) * 3 * channels;
		for ( int h = 0; h < num_heads; ++h ) {
			float* st = state + static_cast<size_t>( h ) * 128 * 128;
			kda_head_step( st, q + h * 128, q + channels + h * 128, q + 2 * channels + h * 128, g_decay.ptr<const float>() + static_cast<size_t>( m ) * channels + h * 128,
				beta.ptr<const float>()[m * num_heads + h], y );
			if ( do_snap && m < snap_rows ) {
				float* sp = snap + m * snap_stride + static_cast<size_t>( h ) * 128 * 128;
				for ( int i = 0; i < 128 * 128; ++i ) sp[i] = st[i];
			}
			gate_norm_out( y, o_norm.data != nullptr ? o_norm.ptr<const h16>() : nullptr, z.data != nullptr ? z.ptr<const h16>() + static_cast<size_t>( m ) * channels + h * 128 : nullptr,
				eps, attn_out.ptr<h16>() + static_cast<size_t>( m ) * channels + h * 128 );
		}
	}
}

} // namespace engine::modules
