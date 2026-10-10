#include "engine/server/monitor.h"

// The monitor page: one self-contained document (no external requests). It polls /api/stats every second and /api/experts every few.
namespace engine::server {

const char* monitor_html() {
	return R"MONITOR(<!DOCTYPE html>
<html lang="en" data-theme="monokai">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>volta engine</title>
<style>
:root[data-theme=monokai] {
	--bg:#272822; --card:#1e1f1c; --line:#49483e; --fg:#f8f8f2; --dim:#75715e;
	--good:#a6e22e; --info:#66d9ef; --bad:#f92672; --warn:#fd971f; --hl:#e6db74; --alt:#ae81ff;
}
:root[data-theme=tomorrow] {
	--bg:#1d1f21; --card:#282a2e; --line:#373b41; --fg:#c5c8c6; --dim:#969896;
	--good:#b5bd68; --info:#81a2be; --bad:#cc6666; --warn:#de935f; --hl:#f0c674; --alt:#b294bb;
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
header { display:flex; flex-wrap:wrap; align-items:center; gap:10px; padding:10px 16px; background:var(--card); border-bottom:1px solid var(--line); position:sticky; top:0; z-index:5; }
header h1 { font-size:15px; margin:0 8px 0 0; color:var(--good); font-weight:600; }
.chip { border:1px solid var(--line); border-radius:3px; padding:1px 7px; color:var(--dim); white-space:nowrap; }
.chip b { color:var(--fg); font-weight:500; }
.chip.ok b { color:var(--good); } .chip.warn b { color:var(--warn); } .chip.bad b { color:var(--bad); }
.sp { flex:1; }
button, select { background:var(--bg); color:var(--fg); border:1px solid var(--line); border-radius:3px; padding:2px 9px; font:inherit; cursor:pointer; }
button:hover, select:hover { border-color:var(--info); }
main { display:grid; grid-template-columns:repeat(12, 1fr); gap:12px; padding:12px 16px 40px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:4px; padding:10px 12px; min-width:0; }
.card h2 { font-size:12px; font-weight:600; margin:0 0 8px; color:var(--info); text-transform:uppercase; letter-spacing:.06em; display:flex; justify-content:space-between; gap:8px; }
.card h2 small { color:var(--dim); text-transform:none; letter-spacing:0; font-weight:400; }
.c12 { grid-column:span 12; } .c8 { grid-column:span 8; } .c6 { grid-column:span 6; } .c4 { grid-column:span 4; } .c3 { grid-column:span 3; }
@media (max-width:1100px) { .c8,.c6,.c4,.c3 { grid-column:span 12; } }
canvas { width:100%; display:block; }
.big { font-size:22px; line-height:1.1; }
.tiles { display:grid; grid-template-columns:repeat(auto-fill, minmax(118px, 1fr)); gap:8px; }
.tile { border:1px solid var(--line); border-radius:3px; padding:5px 8px; }
.tile span { display:block; color:var(--dim); font-size:11px; }
.tile b { font-size:16px; font-weight:500; }
table { width:100%; border-collapse:collapse; }
th { color:var(--dim); font-weight:400; text-align:right; padding:2px 6px; border-bottom:1px solid var(--line); white-space:nowrap; }
td { text-align:right; padding:2px 6px; white-space:nowrap; }
th:first-child, td:first-child { text-align:left; }
tr:hover td { background:rgba(128,128,128,.08); }
.bar { height:8px; background:var(--bg); border:1px solid var(--line); border-radius:2px; overflow:hidden; position:relative; min-width:60px; }
.bar i { display:block; height:100%; background:var(--info); }
.bar i.g { background:var(--good); } .bar i.w { background:var(--warn); } .bar i.b { background:var(--bad); } .bar i.p { background:var(--alt); }
.gpus { display:grid; grid-template-columns:repeat(auto-fit, minmax(210px, 1fr)); gap:10px; }
.gpu { border:1px solid var(--line); border-radius:3px; padding:6px 8px; }
.gpu .row { display:flex; justify-content:space-between; gap:6px; }
.gpu .row span:first-child { color:var(--dim); }
.tag { display:inline-block; border-radius:2px; padding:0 5px; margin:1px 2px 0 0; font-size:11px; background:var(--bg); border:1px solid var(--line); }
.tag.warn { color:var(--warn); border-color:var(--warn); } .tag.bad { color:var(--bad); border-color:var(--bad); }
.phase-decode { color:var(--good); } .phase-prefill { color:var(--warn); } .phase-thinking { color:var(--alt); } .phase-idle { color:var(--dim); }
.dim { color:var(--dim); }
#ehelp { color:var(--dim); font-size:11px; margin-top:6px; }
#tip { position:fixed; pointer-events:none; background:var(--bg); border:1px solid var(--info); border-radius:3px; padding:3px 7px; display:none; z-index:20; white-space:pre; }
.legend span { margin-right:12px; } .legend i { display:inline-block; width:10px; height:3px; margin-right:4px; vertical-align:middle; }
</style>
</head>
<body>
<header>
	<h1>volta engine</h1>
	<span class="chip" id="h-model"><b>-</b></span>
	<span class="chip" id="h-tp">tp <b>-</b></span>
	<span class="chip" id="h-ctx">context <b>-</b></span>
	<span class="chip" id="h-spec">speculation <b>-</b></span>
	<span class="chip" id="h-kv">kv <b>-</b></span>
	<span class="chip" id="h-up">up <b>-</b></span>
	<span class="chip" id="h-conn">link <b>-</b></span>
	<span class="sp"></span>
	<select id="interval" title="refresh"><option value="1000">1 s</option><option value="2000">2 s</option><option value="5000">5 s</option></select>
	<button id="pause">pause</button>
	<button id="theme">theme</button>
</header>
<main>
	<section class="card c8">
		<h2>Throughput <small id="thr-now"></small></h2>
		<canvas id="ch-thr" height="170"></canvas>
		<div class="legend"><span><i style="background:var(--good)"></i>decode tok/s</span><span><i style="background:var(--info)"></i>prefill tok/s</span><span><i style="background:var(--dim)"></i>active slots</span></div>
	</section>
	<section class="card c4">
		<h2>Totals</h2>
		<div class="tiles" id="totals"></div>
	</section>

	<section class="card c12">
		<h2>Slots <small id="slots-note"></small></h2>
		<table id="slots"><thead><tr><th>slot</th><th>phase</th><th>request</th><th>context</th><th style="width:28%">fill</th><th>prompt</th><th>cached</th><th>generated</th><th>tok/s</th><th>age</th></tr></thead><tbody></tbody></table>
	</section>

	<section class="card c12">
		<h2>GPUs <small id="gpu-note"></small></h2>
		<div class="gpus" id="gpus"></div>
	</section>

	<section class="card c6">
		<h2>Token gap <small>time between bursts of tokens a client sees, last <span id="gap-n">0</span> samples</small></h2>
		<div class="tiles" id="gap-tiles"></div>
		<canvas id="ch-gap" height="110" style="margin-top:8px"></canvas>
	</section>
	<section class="card c3">
		<h2>Speculation</h2>
		<div class="tiles" id="spec"></div>
	</section>
	<section class="card c3">
		<h2>Prefix cache</h2>
		<div class="tiles" id="prefix"></div>
	</section>

	<section class="card c12">
		<h2>Experts <small id="exp-note"></small>
			<span><select id="exp-mode"><option value="recent">recent (about a minute)</option><option value="total">since boot</option></select></span></h2>
		<div id="exp-empty" class="dim">no routing counters from this model (dense model, or expert_stats off)</div>
		<div id="exp-body" style="display:none">
			<div style="display:grid; grid-template-columns:minmax(0, 2.4fr) minmax(0, 1fr); gap:14px">
				<div>
					<canvas id="ch-heat" height="260"></canvas>
					<div id="ehelp">rows are the MoE layers (top = first), columns the experts grouped by rank; brightness is log routes (counts are exact: every prompt and generated token routes to its top-k experts in each layer; rejected drafts are not counted). Hover for counts.</div>
				</div>
				<div>
					<div class="tiles" id="exp-tiles"></div>
					<div style="margin-top:10px" class="dim">load per rank (share of routes)</div>
					<div id="exp-ranks"></div>
					<div style="margin-top:10px" class="dim">hottest experts</div>
					<div id="exp-top"></div>
				</div>
			</div>
		</div>
	</section>

	<section class="card c8">
		<h2>Recent requests</h2>
		<table id="recent"><thead><tr><th>request</th><th>ago</th><th>prompt</th><th>cached</th><th>generated</th><th>ttft</th><th>total</th><th>tok/s</th><th>end</th></tr></thead><tbody></tbody></table>
	</section>
	<section class="card c4">
		<h2>Host</h2>
		<div class="tiles" id="host"></div>
		<canvas id="ch-cpu" height="90" style="margin-top:8px"></canvas>
	</section>
</main>
<div id="tip"></div>
<script>
"use strict";
const $ = (id) => document.getElementById(id);
const fmt = (v, d = 0) => (v === undefined || v === null || isNaN(v)) ? '-' : Number(v).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d });
const gib = (b) => (b / 1073741824).toFixed(1) + ' GiB';
const mib = (b) => (b / 1048576).toFixed(0) + ' MiB';
const dur = (s) => { s = Math.max(0, Math.round(s)); const d = Math.floor(s / 86400), h = Math.floor(s % 86400 / 3600), m = Math.floor(s % 3600 / 60); return (d ? d + 'd ' : '') + (h || d ? h + 'h ' : '') + m + 'm ' + (s % 60) + 's'; };
const ms = (v) => v >= 1000 ? (v / 1000).toFixed(2) + ' s' : v.toFixed(v < 10 ? 1 : 0) + ' ms';
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
const tile = (label, value, cls) => '<div class="tile"><span>' + label + '</span><b' + (cls ? ' style="color:var(--' + cls + ')"' : '') + '>' + value + '</b></div>';
const bar = (frac, cls) => '<div class="bar"><i class="' + (cls || '') + '" style="width:' + Math.max(0, Math.min(100, frac * 100)).toFixed(1) + '%"></i></div>';

let paused = false, timer = null, expTimer = null, stats = null, gpuHist = [], cpuHist = [], failures = 0;

// ---- charts ----
function setup(canvas) {
	// the logical height is kept in data-h: the height attribute becomes the backing-store size, which must not feed back into the layout
	if (!canvas.dataset.h) { canvas.dataset.h = canvas.getAttribute('height'); canvas.style.height = canvas.dataset.h + 'px'; }
	const dpr = window.devicePixelRatio || 1, w = canvas.clientWidth, h = parseInt(canvas.dataset.h);
	if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
	const g = canvas.getContext('2d');
	g.setTransform(dpr, 0, 0, dpr, 0, 0);
	g.clearRect(0, 0, w, h);
	return { g, w, h };
}
function lineChart(canvas, series, opt) {
	const { g, w, h } = setup(canvas);
	const L = 44, R = 8, T = 6, B = 16, pw = w - L - R, ph = h - T - B;
	let max = opt.max || 0;
	if (!opt.max) for (const s of series) for (const v of s.data) if (v > max) max = v;
	max = max <= 0 ? 1 : max * 1.1;
	g.font = '10px ' + getComputedStyle(document.body).fontFamily;
	g.strokeStyle = css('--line'); g.fillStyle = css('--dim'); g.lineWidth = 1;
	const steps = ph < 60 ? 1 : 4;   // sparklines get a top and a bottom label only
	for (let i = 0; i <= steps; i++) {
		const y = T + ph * i / steps;
		g.beginPath(); g.moveTo(L, y + .5); g.lineTo(w - R, y + .5); g.stroke();
		g.textAlign = 'right'; g.fillText(fmt(max * (1 - i / steps), max < 10 ? 1 : 0), L - 4, y + 3);
	}
	const n = opt.n || Math.max(...series.map((s) => s.data.length));
	g.textAlign = 'left'; g.fillText(opt.left || '', L, h - 3);
	g.textAlign = 'right'; g.fillText(opt.right || 'now', w - R, h - 3);
	for (const s of series) {
		const d = s.data; if (!d.length) continue;
		g.strokeStyle = css(s.color); g.lineWidth = s.width || 1.5; g.beginPath();
		for (let i = 0; i < d.length; i++) {
			const x = L + pw * (n - d.length + i) / Math.max(1, n - 1), y = T + ph * (1 - d[i] / max);
			i ? g.lineTo(x, y) : g.moveTo(x, y);
		}
		g.stroke();
		if (s.fill) { g.lineTo(L + pw, T + ph); g.lineTo(L + pw * (n - d.length) / Math.max(1, n - 1), T + ph); g.closePath(); g.globalAlpha = .12; g.fillStyle = css(s.color); g.fill(); g.globalAlpha = 1; }
	}
}
function barChart(canvas, values, labels, color) {
	const { g, w, h } = setup(canvas);
	const B = 24, T = 6, n = values.length, mx = Math.max(1, ...values), bw = w / n;
	g.font = '10px ' + getComputedStyle(document.body).fontFamily; g.textAlign = 'center';
	for (let i = 0; i < n; i++) {
		const bh = (h - B - T) * values[i] / mx;
		g.fillStyle = css(color); g.fillRect(i * bw + 3, h - B - bh, bw - 6, bh);
		g.fillStyle = css('--dim'); g.fillText(labels[i], i * bw + bw / 2, h - 11);
		if (values[i]) { g.fillStyle = css('--fg'); g.fillText(values[i], i * bw + bw / 2, Math.max(T + 8, h - B - bh - 3)); }
	}
}

// ---- stats ----
function render(s) {
	stats = s;
	const sv = s.server;
	$('h-model').innerHTML = '<b>' + esc(sv.model) + '</b>';
	$('h-tp').innerHTML = 'tp <b>' + sv.tp + '</b>';
	$('h-ctx').innerHTML = 'context <b>' + fmt(sv.max_context) + '</b> x ' + sv.slots + ' slots';
	$('h-spec').innerHTML = 'speculation <b>' + (sv.speculation ? esc(sv.speculator || 'on') : 'off') + '</b>';
	$('h-kv').innerHTML = 'kv <b>' + esc(sv.kv_format) + '</b>';
	$('h-up').innerHTML = 'up <b>' + dur(sv.uptime_s) + '</b>';

	const h = s.history, last = (a) => a.length ? a[a.length - 1] : 0;
	const nowDec = last(h.decode), nowPf = last(h.prefill);
	$('thr-now').innerHTML = '<span class="big" style="color:var(--good)">' + fmt(nowDec, 1) + '</span> decode &nbsp; <span class="big" style="color:var(--info)">' + fmt(nowPf, 0) + '</span> prefill tok/s';
	const span = h.t.length ? h.t[h.t.length - 1] - h.t[0] : 0;
	lineChart($('ch-thr'), [{ data: h.decode, color: '--good', fill: true }, { data: h.prefill, color: '--info' }, { data: h.active.map((v) => v * Math.max(1, Math.max(...h.decode, ...h.prefill) / Math.max(1, sv.slots)) * 0.4), color: '--dim', width: 1 }],
		{ left: '-' + dur(span), n: 900 });

	const t = s.totals;
	const hit = t.prompt_tokens ? t.cached_tokens / t.prompt_tokens : 0;
	$('totals').innerHTML = tile('requests', fmt(t.requests)) + tile('completed', fmt(t.completed)) + tile('queued', fmt(sv.queued), sv.queued ? 'warn' : '') +
		tile('tokens out', fmt(t.tokens_out)) + tile('prefilled', fmt(t.tokens_prefilled)) + tile('prompt tokens', fmt(t.prompt_tokens)) +
		tile('cached share', fmt(hit * 100, 1) + '%', 'good') + tile('avg ttft', ms(s.averages.ttft_ms || 0)) +
		tile('stop / length', fmt(t.finished.stop) + ' / ' + fmt(t.finished.length)) + tile('cancelled', fmt(t.finished.cancelled), t.finished.cancelled ? 'warn' : '');

	// slots
	let body = '', busy = 0;
	for (const x of s.slots) {
		if (x.active) busy++;
		const frac = sv.max_context ? x.pos / sv.max_context : 0;
		body += '<tr><td>' + x.slot + '</td><td class="phase-' + x.phase + '">' + x.phase + '</td><td>' + (x.active ? '#' + x.req_id : '-') + '</td><td>' + (x.active ? fmt(x.pos) : '-') + '</td><td>' +
			bar(x.active ? frac : 0, frac > .9 ? 'b' : frac > .6 ? 'w' : x.phase === 'prefill' ? 'w' : 'g') + '</td><td>' + (x.active ? fmt(x.prompt_tokens) : '-') + '</td><td>' + (x.active ? fmt(x.cached_tokens) : '-') +
			'</td><td>' + (x.active ? fmt(x.generated) : '-') + '</td><td>' + (x.active && x.decode_tok_s ? fmt(x.decode_tok_s, 1) : '-') + '</td><td>' + (x.active ? dur(x.age_s) : '-') + '</td></tr>';
	}
	document.querySelector('#slots tbody').innerHTML = body;
	$('slots-note').textContent = busy + ' of ' + sv.slots + ' busy';

	// gpus (history is kept here)
	gpuHist.push(s.gpus.map((g) => g.power_w)); if (gpuHist.length > 120) gpuHist.shift();
	cpuHist.push(s.host.cpu_pct); if (cpuHist.length > 120) cpuHist.shift();
	if (!s.gpus.length) $('gpus').innerHTML = '<div class="dim">NVML not available</div>';
	else {
		let totalW = 0, html = '';
		s.gpus.forEach((g, i) => {
			totalW += g.power_w;
			const pf = g.power_limit_w ? g.power_w / g.power_limit_w : 0, tf = g.temp_c / 90, mf = g.mem_total ? g.mem_used / g.mem_total : 0;
			const tags = g.throttle.filter((x) => x !== 'gpu idle').map((x) => '<span class="tag ' + (x.indexOf('hw') === 0 ? 'bad' : 'warn') + '">' + esc(x) + '</span>').join('');
			html += '<div class="gpu"><div class="row"><b>GPU ' + i + '</b><span>' + fmt(g.power_w, 0) + ' / ' + fmt(g.power_limit_w, 0) + ' W</span></div>' + bar(pf, pf > .95 ? 'w' : 'g') +
				'<div class="row"><span>temp</span><span style="color:var(--' + (g.temp_c > 80 ? 'bad' : g.temp_c > 70 ? 'warn' : 'fg') + ')">' + g.temp_c + ' C</span></div>' + bar(tf, g.temp_c > 80 ? 'b' : g.temp_c > 70 ? 'w' : 'p') +
				'<div class="row"><span>util</span><span>' + g.util_gpu + '% gpu / ' + g.util_mem + '% mem</span></div>' +
				'<div class="row"><span>clocks</span><span>' + g.sm_mhz + ' sm / ' + g.mem_mhz + ' mem MHz</span></div>' +
				'<div class="row"><span>vram</span><span>' + gib(g.mem_used) + ' / ' + gib(g.mem_total) + '</span></div>' + bar(mf, mf > .95 ? 'w' : 'i') +
				'<canvas id="ch-gpu' + i + '" height="44" style="margin-top:4px"></canvas><div>' + tags + '</div></div>';
		});
		$('gpus').innerHTML = html;
		s.gpus.forEach((g, i) => lineChart($('ch-gpu' + i), [{ data: gpuHist.map((p) => p[i]), color: '--warn', fill: true }], { max: Math.max(g.power_limit_w, 1), left: '', right: '' }));
		$('gpu-note').textContent = fmt(totalW, 0) + ' W total';
	}

	// token gap
	const L = s.latency;
	$('gap-n').textContent = fmt(L.gap_samples);
	$('gap-tiles').innerHTML = tile('p50', ms(L.gap_ms[0])) + tile('p90', ms(L.gap_ms[1])) + tile('p99', ms(L.gap_ms[2]), L.gap_ms[2] > 500 ? 'warn' : '') + tile('p99.9', ms(L.gap_ms[3]), L.gap_ms[3] > 1000 ? 'warn' : '') + tile('max', ms(L.gap_ms[4])) +
		tile('ttft p50 / p99', ms(L.ttft_ms[0]) + ' / ' + ms(L.ttft_ms[2]));
	const labels = ['<2', '4', '8', '16', '32', '64', '128', '256', '512', '1k', '2k', '>2k'];
	barChart($('ch-gap'), L.gap_hist, labels, '--info');

	// speculation, prefix cache
	const sp = s.speculation;
	$('spec').innerHTML = sv.speculation ? tile('accept rate', fmt(sp.accept_rate * 100, 1) + '%', sp.accept_rate > .6 ? 'good' : 'warn') + tile('tokens / round', fmt(sp.tokens_per_round, 2)) + tile('rounds', fmt(sp.rounds)) + tile('drafted', fmt(sp.proposed)) : '<span class="dim">speculation is off</span>';
	const pc = s.prefix_cache;
	$('prefix').innerHTML = pc ? tile('hit rate', fmt(pc.lookups ? pc.hits / pc.lookups * 100 : 0, 0) + '%', 'good') + tile('tokens saved', fmt(pc.hit_tokens)) + tile('entries', fmt(pc.entries)) + tile('evictions', fmt(pc.evictions), pc.evictions ? 'warn' : '') +
		'<div class="tile" style="grid-column:1/-1"><span>pinned host RAM ' + gib(pc.bytes) + ' of ' + gib(pc.budget_bytes) + '</span>' + bar(pc.budget_bytes ? pc.bytes / pc.budget_bytes : 0, 'p') + '</div>' : '<span class="dim">prefix cache is off</span>';

	// recent
	document.querySelector('#recent tbody').innerHTML = s.recent.slice(0, 14).map((r) => '<tr><td>#' + r.req_id + '</td><td>' + dur(r.age_s) + '</td><td>' + fmt(r.prompt_tokens) + '</td><td>' + fmt(r.cached_tokens) + '</td><td>' + fmt(r.generated) +
		'</td><td>' + ms(r.ttft_ms) + '</td><td>' + ms(r.total_ms) + '</td><td>' + fmt(r.decode_tok_s, 1) + '</td><td class="' + (r.reason === 'stop' ? '' : 'dim') + '">' + r.reason + '</td></tr>').join('') || '<tr><td class="dim" colspan="9">no finished requests yet</td></tr>';

	// host
	const ho = s.host;
	$('host').innerHTML = tile('cpu', fmt(ho.cpu_pct, 0) + '%') + tile('load 1m', fmt(ho.load1, 2)) + tile('ram', gib(ho.mem_used) + ' / ' + gib(ho.mem_total)) + tile('server rss', gib(ho.rss));
	lineChart($('ch-cpu'), [{ data: cpuHist, color: '--alt', fill: true }], { max: 100, left: '', right: '' });
}

async function poll() {
	if (paused) return;
	try {
		const r = await fetch('/api/stats', { cache: 'no-store' });
		render(await r.json());
		failures = 0;
		$('h-conn').className = 'chip ok'; $('h-conn').innerHTML = 'link <b>live</b>';
	} catch (e) {
		failures++;
		$('h-conn').className = 'chip bad'; $('h-conn').innerHTML = 'link <b>lost (' + failures + ')</b>';
	}
}

// ---- experts ----
let expData = null, heatGeom = null;
function ramp(f) {
	const stops = [css('--card'), css('--info'), css('--hl'), css('--bad')];
	const p = Math.min(.999, Math.max(0, f)) * (stops.length - 1), i = Math.floor(p), k = p - i;
	return mix(stops[i], stops[i + 1], k);
}
function rgb(c) {
	if (c[0] === '#') return [parseInt(c.substr(1, 2), 16), parseInt(c.substr(3, 2), 16), parseInt(c.substr(5, 2), 16)];
	const m = c.match(/\d+/g); return [+m[0], +m[1], +m[2]];
}
function mix(a, b, k) { const x = rgb(a), y = rgb(b); return 'rgb(' + [0, 1, 2].map((i) => Math.round(x[i] + (y[i] - x[i]) * k)).join(',') + ')'; }

function renderExperts() {
	const e = expData;
	if (!e || !e.layers || !e.experts) return;
	$('exp-empty').style.display = 'none'; $('exp-body').style.display = 'block';
	const v = $('exp-mode').value === 'total' ? e.total : e.recent;
	const R = e.ranks, Lc = e.layers, E = e.experts, cols = R * E;
	const at = (r, l, x) => v[(r * Lc + l) * E + x];
	let max = 0, sum = 0, live = 0;
	const perRank = new Array(R).fill(0), perExpert = new Array(cols).fill(0), perLayerImb = [];
	for (let l = 0; l < Lc; l++) {
		let lsum = 0, lmax = 0;
		for (let r = 0; r < R; r++) for (let x = 0; x < E; x++) {
			const c = at(r, l, x); sum += c; lsum += c; perRank[r] += c; perExpert[r * E + x] += c;
			if (c > max) max = c; if (c > lmax) lmax = c;
		}
		perLayerImb.push(lsum ? lmax / (lsum / cols) : 0);
	}
	for (let i = 0; i < cols; i++) if (perExpert[i] > 0) live++;
	const imb = perLayerImb.reduce((a, b) => a + b, 0) / Math.max(1, perLayerImb.filter((x) => x > 0).length);
	$('exp-note').textContent = fmt(sum) + ' routes, ' + Lc + ' layers x ' + cols + ' experts' + (R > 1 ? ' on ' + R + ' ranks' : '');
	// heatmap
	const canvas = $('ch-heat'), { g, w, h } = setup(canvas);
	const cw = w / cols, ch = (h - 2) / Lc, lg = Math.log1p(max);
	for (let l = 0; l < Lc; l++) for (let r = 0; r < R; r++) for (let x = 0; x < E; x++) {
		const c = at(r, l, x);
		g.fillStyle = ramp(max ? Math.log1p(c) / lg : 0);
		g.fillRect((r * E + x) * cw, l * ch, Math.ceil(cw), Math.ceil(ch));
	}
	g.strokeStyle = css('--line');
	for (let r = 1; r < R; r++) { g.beginPath(); g.moveTo(r * E * cw + .5, 0); g.lineTo(r * E * cw + .5, h); g.stroke(); }
	heatGeom = { cw, ch, E, R, Lc, at };
	// tiles
	const top = perExpert.map((c, i) => [c, i]).sort((a, b) => b[0] - a[0]);
	const top10 = top.slice(0, 10).reduce((a, b) => a + b[0], 0);
	$('exp-tiles').innerHTML = tile('experts used', fmt(live) + ' / ' + cols, live < cols ? 'warn' : 'good') + tile('layer imbalance', fmt(imb, 2) + 'x') + tile('top 10 share', fmt(sum ? top10 / sum * 100 : 0, 1) + '%') + tile('even share', fmt(100 / cols * 10, 1) + '%');
	$('exp-ranks').innerHTML = perRank.map((c, r) => '<div style="display:flex; gap:8px; align-items:center"><span style="width:52px">rank ' + r + '</span><div style="flex:1">' + bar(sum ? c / sum * R / 2 : 0, Math.abs(c / sum * R - 1) > .15 ? 'w' : 'g') + '</div><span style="width:48px; text-align:right">' + fmt(sum ? c / sum * 100 : 0, 1) + '%</span></div>').join('');
	$('exp-top').innerHTML = '<table>' + top.slice(0, 10).map((t) => '<tr><td>expert ' + t[1] + (R > 1 ? ' <span class="dim">(rank ' + Math.floor(t[1] / E) + ')</span>' : '') + '</td><td>' + fmt(t[0]) + '</td><td>' + fmt(sum ? t[0] / sum * 100 : 0, 2) + '%</td></tr>').join('') + '</table>';
}
async function pollExperts() {
	if (paused) return;
	try {
		expData = await (await fetch('/api/experts', { cache: 'no-store' })).json();
		renderExperts();
		$('exp-empty').textContent = 'no routing counters yet (they start with the first request, or this model has none)';
	} catch (e) { /* the stats poll reports the link state */ }
}
$('ch-heat').addEventListener('mousemove', (ev) => {
	if (!heatGeom) return;
	const rc = ev.target.getBoundingClientRect(), x = Math.floor((ev.clientX - rc.left) / heatGeom.cw), l = Math.floor((ev.clientY - rc.top) / heatGeom.ch);
	const tip = $('tip');
	if (x < 0 || l < 0 || l >= heatGeom.Lc || x >= heatGeom.R * heatGeom.E) { tip.style.display = 'none'; return; }
	const r = Math.floor(x / heatGeom.E), ex = x % heatGeom.E;
	tip.textContent = 'layer ' + l + '  expert ' + x + (heatGeom.R > 1 ? '  (rank ' + r + ' #' + ex + ')' : '') + '\n' + fmt(heatGeom.at(r, l, ex)) + ' routes';
	tip.style.display = 'block'; tip.style.left = (ev.clientX + 14) + 'px'; tip.style.top = (ev.clientY + 14) + 'px';
});
$('ch-heat').addEventListener('mouseleave', () => { $('tip').style.display = 'none'; });
$('exp-mode').addEventListener('change', renderExperts);

// ---- controls ----
function schedule() {
	clearInterval(timer); clearInterval(expTimer);
	const iv = parseInt($('interval').value);
	timer = setInterval(poll, iv); expTimer = setInterval(pollExperts, Math.max(3000, iv * 3));
}
$('interval').addEventListener('change', () => { localStorage.setItem('volta-interval', $('interval').value); schedule(); });
$('pause').addEventListener('click', () => { paused = !paused; $('pause').textContent = paused ? 'resume' : 'pause'; if (!paused) { poll(); pollExperts(); } });
function theme(t) {
	document.documentElement.setAttribute('data-theme', t); localStorage.setItem('volta-theme', t);
	if (stats) render(stats); renderExperts();
}
$('theme').addEventListener('click', () => theme(document.documentElement.getAttribute('data-theme') === 'monokai' ? 'tomorrow' : 'monokai'));
window.addEventListener('resize', () => { if (stats) render(stats); renderExperts(); });
(function init() {
	const t = localStorage.getItem('volta-theme'); if (t) document.documentElement.setAttribute('data-theme', t);
	const iv = localStorage.getItem('volta-interval'); if (iv) $('interval').value = iv;
	schedule(); poll(); pollExperts();
})();
</script>
</body>
</html>
)MONITOR";
}

} // namespace engine::server
