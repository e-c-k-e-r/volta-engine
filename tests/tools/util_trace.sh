#!/usr/bin/env bash
# Samples GPU 0 (utilization, SM clock, power, power-cap throttle flag) every 250 ms while a command runs, then prints a compact timeline
# and summary. A utilization that swings between ~100% and a low value with power following (64 <-> 150 W) means the GPU waits on host work
# between bursts (look for host-side stages in the per-step path); a steady ~100% with the clock wandering and sw_power_cap active is the
# power cap. nvprof's own gap analysis is not reliable for host stalls (its per-launch overhead adds ~2 ms holes per decode step).
#   tests/tools/util_trace.sh <command...>      e.g.  BENCH_FLASH=1 tests/tools/util_trace.sh taskset -c 0-9 build/bench_prefill 8192 512
set -u
[ $# -ge 1 ] || { echo "usage: $0 <command...>"; exit 2; }
LOG="$(mktemp -u "${TMPDIR:-/tmp}/util_trace_XXXXXX.csv")"
nvidia-smi --query-gpu=index,utilization.gpu,clocks.sm,power.draw,clocks_throttle_reasons.sw_power_cap --format=csv,noheader -lms 250 > "$LOG" &
SMI=$!
"$@"
RC=$?
kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
python3 -I - "$LOG" <<'PY'
import sys, collections
rows = [l.strip().split(', ') for l in open(sys.argv[1]) if l.strip()]
g = [r for r in rows if r[0] == '0']
act = [r for r in g if float(r[1].strip(' %')) > 0]
if not act:
    print('no GPU activity sampled'); sys.exit(0)
u = [float(r[1].strip(' %')) for r in act]
w = [float(r[3].split()[0]) for r in act]
c = [float(r[2].split()[0]) for r in act]
cap = sum(1 for r in act if r[4] == 'Active')
print('GPU 0, %d active samples (250 ms): util mean %.0f%%, <50%% in %d, power mean %.0f W (min %.0f, max %.0f), sm clock %.0f-%.0f MHz, power cap active %d%%' %
      (len(act), sum(u) / len(u), sum(1 for x in u if x < 50), sum(w) / len(w), min(w), max(w), min(c), max(c), 100 * cap // len(act)))
print('util%:', ' '.join('%d' % x for x in u[-60:]))
PY
exit $RC
