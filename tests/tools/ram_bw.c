// Host RAM bandwidth per NUMA node and for the whole machine (standalone; no engine headers):
//   cc -O3 -march=native -fopenmp tests/tools/ram_bw.c -o ram_bw && ./ram_bw [GiB per array, default 2]
// For each node: one thread, one thread per physical core, every hardware thread; then every hardware thread of every node. Threads are
// pinned to their CPUs and fill their own slices first, so the pages sit on the node of the thread that reads them. Per setup: read (a
// sum), write (a fill) and triad (a = b + s * c), best of 5. Reading is what the CPU expert kernels do with their codes; plain stores read
// each line before writing it, so write is lower than read. The node layout comes from /sys/devices/system/node.
#define _GNU_SOURCE
#include <omp.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_CPUS 1024

// a cpulist ("0-13,28-41") into cpus[], returns the count
static int parse_list(const char* s, int* cpus) {
	int n = 0;
	while (*s && *s != '\n') {
		char* end;
		int a = (int)strtol(s, &end, 10), b = a;
		if (*end == '-') b = (int)strtol(end + 1, &end, 10);
		for (int c = a; c <= b && n < MAX_CPUS; ++c) cpus[n++] = c;
		s = *end == ',' ? end + 1 : end;
	}
	return n;
}

static int read_list(const char* path, int* cpus) {
	char buf[4096] = {0};
	FILE* f = fopen(path, "r");
	if (!f) return 0;
	if (!fgets(buf, sizeof buf, f)) buf[0] = 0;
	fclose(f);
	return parse_list(buf, cpus);
}

// the first hardware thread of every core in `cpus`
static int physical(const int* cpus, int n, int* out) {
	int k = 0;
	for (int i = 0; i < n; ++i) {
		char path[256];
		int sib[MAX_CPUS];
		snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpus[i]);
		const int ns = read_list(path, sib);
		if (ns == 0 || sib[0] == cpus[i]) out[k++] = cpus[i];
	}
	return k;
}

static void pin(int cpu) {
	cpu_set_t set;
	CPU_ZERO(&set);
	CPU_SET(cpu, &set);
	sched_setaffinity(0, sizeof set, &set);
}

static void run(const char* label, const int* cpus, int nt, size_t n) {
	double* a = aligned_alloc(64, n * sizeof(double));
	double* b = aligned_alloc(64, n * sizeof(double));
	double* c = aligned_alloc(64, n * sizeof(double));
	if (!a || !b || !c) { printf("%-34s allocation failed\n", label); free(a); free(b); free(c); return; }
	omp_set_num_threads(nt);
	#pragma omp parallel
	{
		pin(cpus[omp_get_thread_num()]);
		#pragma omp for schedule(static)
		for (size_t i = 0; i < n; ++i) { a[i] = 1.0; b[i] = 2.0; c[i] = 0.5; }
	}
	const double gb = (double)n * sizeof(double) / 1e9;
	double best_r = 0, best_w = 0, best_t = 0, sink = 0;
	for (int rep = 0; rep < 5; ++rep) {
		double s = 0, t0 = omp_get_wtime();
		#pragma omp parallel
		{
			pin(cpus[omp_get_thread_num()]);
			#pragma omp for simd schedule(static) reduction(+ : s)   // simd: the sum may be split into vector lanes
			for (size_t i = 0; i < n; ++i) s += a[i];
		}
		double t1 = omp_get_wtime();
		sink += s;
		#pragma omp parallel
		{
			pin(cpus[omp_get_thread_num()]);
			#pragma omp for schedule(static)
			for (size_t i = 0; i < n; ++i) c[i] = (double)rep;
		}
		double t2 = omp_get_wtime();
		#pragma omp parallel
		{
			pin(cpus[omp_get_thread_num()]);
			#pragma omp for schedule(static)
			for (size_t i = 0; i < n; ++i) a[i] = b[i] + 0.5 * c[i];
		}
		double t3 = omp_get_wtime();
		if (gb / (t1 - t0) > best_r) best_r = gb / (t1 - t0);
		if (gb / (t2 - t1) > best_w) best_w = gb / (t2 - t1);
		if (3 * gb / (t3 - t2) > best_t) best_t = 3 * gb / (t3 - t2);
	}
	printf("%-34s %7.1f %7.1f %7.1f\n", label, best_r, best_w, best_t);
	if (sink == 42.0) printf(" ");   // keeps the reads
	free(a); free(b); free(c);
}

int main(int argc, char** argv) {
	const double gib = argc > 1 ? atof(argv[1]) : 2.0;
	const size_t n = (size_t)(gib * (1ull << 30)) / sizeof(double);
	static int node_cpus[64][MAX_CPUS], node_n[64], all[MAX_CPUS];
	int nodes = 0, nall = 0;
	for (; nodes < 64; ++nodes) {
		char path[256];
		snprintf(path, sizeof path, "/sys/devices/system/node/node%d/cpulist", nodes);
		node_n[nodes] = read_list(path, node_cpus[nodes]);
		if (node_n[nodes] == 0) break;
	}
	if (nodes == 0) {   // no NUMA information: one node of every online CPU
		node_n[0] = read_list("/sys/devices/system/cpu/online", node_cpus[0]);
		nodes = 1;
	}
	printf("%d node(s), %.1f GiB per array; GB/s, best of 5\n%-34s %7s %7s %7s\n", nodes, gib, "setup", "read", "write", "triad");
	for (int d = 0; d < nodes; ++d) {
		int phys[MAX_CPUS];
		const int np = physical(node_cpus[d], node_n[d], phys);
		char label[64];
		snprintf(label, sizeof label, "node %d, 1 thread", d);
		run(label, node_cpus[d], 1, n);
		snprintf(label, sizeof label, "node %d, %d threads (cores)", d, np);
		run(label, phys, np, n);
		if (node_n[d] > np) {
			snprintf(label, sizeof label, "node %d, %d threads (hw threads)", d, node_n[d]);
			run(label, node_cpus[d], node_n[d], n);
		}
		for (int i = 0; i < node_n[d] && nall < MAX_CPUS; ++i) all[nall++] = node_cpus[d][i];
	}
	if (nodes > 1) {
		char label[64];
		snprintf(label, sizeof label, "all nodes, %d threads", nall);
		run(label, all, nall, n);
	}
	return 0;
}
