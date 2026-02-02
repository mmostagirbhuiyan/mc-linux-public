/*
 * Sampling Interval Sweep Test (M2 Response)
 *
 * Measures detection latency vs overhead tradeoff for 1-20ms polling intervals.
 * This addresses TPDS Review M2: "What is the detection latency distribution?"
 *
 * Build (Pi):
 *   gcc -O3 -march=native test_sampling_interval_sweep.c -lpthread -lm -o test_sampling_sweep
 *
 * Requires: Linux with perf_event_open (sudo or perf_event_paranoid=0)
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <fcntl.h>
#include <sys/ioctl.h>

#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/syscall.h>
#endif

/* Test parameters */
#define NUM_TRIALS 50
#define WARMUP_TRIALS 5
#define GEMM_SIZE 256
#define NUM_INTERFERERS 4

/* Sampling intervals to test (milliseconds) */
static int intervals_ms[] = {1, 2, 5, 10, 20};
#define NUM_INTERVALS (sizeof(intervals_ms) / sizeof(intervals_ms[0]))

/* Interferer state */
static char *memory_buffers[NUM_INTERFERERS] = {NULL};
static _Atomic int interferer_running = 0;
static _Atomic int interferer_should_stop = 0;
static _Atomic int interferer_throttled = 0;

/* Detection state */
static _Atomic int interference_detected = 0;
static _Atomic uint64_t detection_timestamp = 0;
static _Atomic int profiling_active = 0;

/* PMU state */
#ifdef __linux__
static int g_fd_cycles = -1;
static int g_fd_instructions = -1;
static double g_ipc_threshold = 0.0;
static int g_poll_interval_ms = 10;
#endif

typedef struct {
    int core;
    int id;
} interferer_arg_t;

static int cmp_double(const void *a, const void *b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

static inline uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

/* GEMM workload */
static void gemm(float *C, float *A, float *B, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

static float* alloc_matrix(int N) {
    float *ptr = (float*)aligned_alloc(64, N * N * sizeof(float));
    if (!ptr) return NULL;
    for (int i = 0; i < N * N; i++) {
        ptr[i] = (float)(rand() % 1000) / 1000.0f;
    }
    return ptr;
}

/* Interferer thread - cache pollution */
static void* interferer_fn(void *arg) {
    interferer_arg_t *iarg = (interferer_arg_t*)arg;
    int core = iarg->core;
    int id = iarg->id;

#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#endif

    atomic_fetch_add(&interferer_running, 1);
    unsigned int seed = (unsigned int)get_time_ns() + id;
    char *buffer = memory_buffers[id];

    while (!atomic_load(&interferer_should_stop)) {
        if (atomic_load(&interferer_throttled)) {
            usleep(1000);
            continue;
        }
        /* Random memory access to pollute cache */
        for (int i = 0; i < 10000; i++) {
            int idx = rand_r(&seed) % (8 * 1024 * 1024);
            buffer[idx] = buffer[(idx + 64) % (8 * 1024 * 1024)];
        }
    }

    return NULL;
}

#ifdef __linux__
static int perf_event_open(struct perf_event_attr *attr, pid_t pid,
                           int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}

/* Polling-based profiler thread */
static void* profiler_polling_thread(void *arg) {
    (void)arg;

    /* Initial enable */
    ioctl(g_fd_cycles, PERF_EVENT_IOC_RESET, 0);
    ioctl(g_fd_instructions, PERF_EVENT_IOC_RESET, 0);
    ioctl(g_fd_cycles, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);

    while (atomic_load(&profiling_active)) {
        /* Sleep for configured interval */
        usleep(g_poll_interval_ms * 1000);

        /* Read PMU counters */
        ioctl(g_fd_cycles, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);

        uint64_t cycles = 0, instructions = 0;
        read(g_fd_cycles, &cycles, sizeof(cycles));
        read(g_fd_instructions, &instructions, sizeof(instructions));

        double ipc = (cycles > 0) ? (double)instructions / cycles : 0;

        /* Check for interference */
        if (ipc < g_ipc_threshold && ipc > 0 &&
            !atomic_load(&interference_detected)) {
            atomic_store(&detection_timestamp, get_time_ns());
            atomic_store(&interference_detected, 1);
            atomic_store(&interferer_throttled, 1);
        }

        /* Reset and re-enable */
        ioctl(g_fd_cycles, PERF_EVENT_IOC_RESET, 0);
        ioctl(g_fd_instructions, PERF_EVENT_IOC_RESET, 0);
        ioctl(g_fd_cycles, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
    }

    ioctl(g_fd_cycles, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
    return NULL;
}

static int setup_pmu(void) {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.size = sizeof(pe);
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    pe.type = PERF_TYPE_HARDWARE;
    pe.config = PERF_COUNT_HW_CPU_CYCLES;

    g_fd_cycles = perf_event_open(&pe, 0, -1, -1, 0);
    if (g_fd_cycles < 0) return -1;

    pe.config = PERF_COUNT_HW_INSTRUCTIONS;
    g_fd_instructions = perf_event_open(&pe, 0, -1, g_fd_cycles, 0);
    if (g_fd_instructions < 0) {
        close(g_fd_cycles);
        return -1;
    }

    return 0;
}

static double calibrate_ipc_threshold(float *A, float *B, float *C) {
    /* Run isolated GEMM to get baseline IPC */
    ioctl(g_fd_cycles, PERF_EVENT_IOC_RESET, 0);
    ioctl(g_fd_instructions, PERF_EVENT_IOC_RESET, 0);
    /* Enable both counters (group leader enables group) */
    ioctl(g_fd_cycles, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);

    gemm(C, A, B, GEMM_SIZE);

    ioctl(g_fd_cycles, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);

    uint64_t cycles = 0, instructions = 0;
    if (read(g_fd_cycles, &cycles, sizeof(cycles)) != sizeof(cycles)) {
        printf("Warning: cycles read failed\n");
    }
    if (read(g_fd_instructions, &instructions, sizeof(instructions)) != sizeof(instructions)) {
        printf("Warning: instructions read failed\n");
    }

    double baseline_ipc = (cycles > 0) ? (double)instructions / cycles : 0;
    printf("Calibration: %lu cycles, %lu instructions, IPC=%.3f\n",
           cycles, instructions, baseline_ipc);

    /* Use 60% of baseline as threshold */
    return baseline_ipc * 0.60;
}
#endif

/* Results structure */
typedef struct {
    int interval_ms;
    double det_latency_p50;
    double det_latency_p99;
    double overhead_pct;
    double detection_rate;
    double recovery_pct;
} interval_result_t;

int main(void) {
    printf("================================================================\n");
    printf("SAMPLING INTERVAL SWEEP TEST (M2 Response)\n");
    printf("================================================================\n\n");

#ifndef __linux__
    printf("ERROR: This test requires Linux PMU support.\n");
    return 1;
#else

    /* Allocate matrices */
    float *A = alloc_matrix(GEMM_SIZE);
    float *B = alloc_matrix(GEMM_SIZE);
    float *C = alloc_matrix(GEMM_SIZE);

    if (!A || !B || !C) {
        printf("ERROR: Matrix allocation failed\n");
        return 1;
    }

    /* Setup PMU */
    if (setup_pmu() < 0) {
        printf("ERROR: PMU setup failed (need root or perf_event_paranoid=0)\n");
        return 1;
    }

    /* Allocate interferer buffers */
    for (int i = 0; i < NUM_INTERFERERS; i++) {
        memory_buffers[i] = malloc(8 * 1024 * 1024);
        if (!memory_buffers[i]) {
            printf("ERROR: Buffer allocation failed\n");
            return 1;
        }
        memset(memory_buffers[i], 0, 8 * 1024 * 1024);
    }

    /* Warmup */
    printf("Warmup (%d iterations)...\n", WARMUP_TRIALS);
    for (int i = 0; i < WARMUP_TRIALS; i++) {
        gemm(C, A, B, GEMM_SIZE);
    }

    /* Calibrate IPC threshold */
    g_ipc_threshold = calibrate_ipc_threshold(A, B, C);
    printf("Baseline IPC threshold: %.3f\n\n", g_ipc_threshold);

    /* Measure isolated baseline */
    printf("Measuring isolated baseline...\n");
    double isolated_times[NUM_TRIALS];
    for (int t = 0; t < NUM_TRIALS; t++) {
        uint64_t start = get_time_ns();
        gemm(C, A, B, GEMM_SIZE);
        isolated_times[t] = (get_time_ns() - start) / 1e6;
    }

    double isolated_sum = 0;
    for (int t = 0; t < NUM_TRIALS; t++) isolated_sum += isolated_times[t];
    double isolated_mean = isolated_sum / NUM_TRIALS;
    printf("Isolated baseline: %.2f ms\n\n", isolated_mean);

    /* Measure interfered baseline (no detection) */
    printf("Measuring interfered baseline (no detection)...\n");

    atomic_store(&interferer_should_stop, 0);
    atomic_store(&interferer_running, 0);
    pthread_t interferer_threads[NUM_INTERFERERS];
    interferer_arg_t iargs[NUM_INTERFERERS];

    for (int i = 0; i < NUM_INTERFERERS; i++) {
        iargs[i].core = i % 4;
        iargs[i].id = i;
        pthread_create(&interferer_threads[i], NULL, interferer_fn, &iargs[i]);
    }
    while (atomic_load(&interferer_running) < NUM_INTERFERERS) sched_yield();
    usleep(10000);

    double interfered_times[NUM_TRIALS];
    for (int t = 0; t < NUM_TRIALS; t++) {
        uint64_t start = get_time_ns();
        gemm(C, A, B, GEMM_SIZE);
        interfered_times[t] = (get_time_ns() - start) / 1e6;
    }

    atomic_store(&interferer_should_stop, 1);
    for (int i = 0; i < NUM_INTERFERERS; i++) {
        pthread_join(interferer_threads[i], NULL);
    }

    double interfered_sum = 0;
    for (int t = 0; t < NUM_TRIALS; t++) interfered_sum += interfered_times[t];
    double interfered_mean = interfered_sum / NUM_TRIALS;
    printf("Interfered baseline: %.2f ms (+%.0f%% slowdown)\n\n",
           interfered_mean, (interfered_mean - isolated_mean) / isolated_mean * 100);

    /* Results storage */
    interval_result_t results[NUM_INTERVALS];

    /* Sweep through intervals */
    printf("================================================================\n");
    printf("INTERVAL SWEEP (%d trials per interval)\n", NUM_TRIALS);
    printf("================================================================\n\n");

    for (int i = 0; i < (int)NUM_INTERVALS; i++) {
        int interval = intervals_ms[i];
        g_poll_interval_ms = interval;

        printf("Testing %dms interval...\n", interval);

        double detection_latencies[NUM_TRIALS];
        double task_times[NUM_TRIALS];
        int detections = 0;
        double overhead_sum = 0;

        /* Start interferers */
        atomic_store(&interferer_should_stop, 0);
        atomic_store(&interferer_running, 0);
        for (int j = 0; j < NUM_INTERFERERS; j++) {
            pthread_create(&interferer_threads[j], NULL, interferer_fn, &iargs[j]);
        }
        while (atomic_load(&interferer_running) < NUM_INTERFERERS) sched_yield();
        usleep(10000);

        for (int t = 0; t < NUM_TRIALS; t++) {
            atomic_store(&interference_detected, 0);
            atomic_store(&detection_timestamp, 0);
            atomic_store(&interferer_throttled, 0);

            /* Start profiler thread */
            atomic_store(&profiling_active, 1);
            pthread_t profiler;
            pthread_create(&profiler, NULL, profiler_polling_thread, NULL);

            /* Small delay to let profiler start */
            usleep(100);

            /* Run GEMM */
            uint64_t start = get_time_ns();
            gemm(C, A, B, GEMM_SIZE);
            uint64_t end = get_time_ns();

            /* Stop profiler */
            atomic_store(&profiling_active, 0);
            pthread_join(profiler, NULL);

            task_times[t] = (end - start) / 1e6;

            if (atomic_load(&interference_detected)) {
                uint64_t det_ts = atomic_load(&detection_timestamp);
                detection_latencies[detections] = (det_ts - start) / 1e6;
                detections++;
            }
        }

        /* Stop interferers */
        atomic_store(&interferer_should_stop, 1);
        for (int j = 0; j < NUM_INTERFERERS; j++) {
            pthread_join(interferer_threads[j], NULL);
        }

        /* Measure overhead (isolated with profiler) */
        for (int t = 0; t < NUM_TRIALS; t++) {
            atomic_store(&profiling_active, 1);
            pthread_t profiler;
            pthread_create(&profiler, NULL, profiler_polling_thread, NULL);
            usleep(100);

            uint64_t start = get_time_ns();
            gemm(C, A, B, GEMM_SIZE);
            uint64_t end = get_time_ns();

            atomic_store(&profiling_active, 0);
            pthread_join(profiler, NULL);

            overhead_sum += (end - start) / 1e6;
        }
        double overhead_mean = overhead_sum / NUM_TRIALS;

        /* Compute statistics */
        results[i].interval_ms = interval;
        results[i].detection_rate = (double)detections / NUM_TRIALS * 100;
        results[i].overhead_pct = (overhead_mean - isolated_mean) / isolated_mean * 100;

        /* Detection latency percentiles */
        if (detections > 0) {
            qsort(detection_latencies, detections, sizeof(double), cmp_double);
            results[i].det_latency_p50 = detection_latencies[detections / 2];
            int p99_idx = (int)(detections * 0.99);
            if (p99_idx >= detections) p99_idx = detections - 1;
            results[i].det_latency_p99 = detection_latencies[p99_idx];
        } else {
            results[i].det_latency_p50 = 0;
            results[i].det_latency_p99 = 0;
        }

        /* Recovery calculation */
        double task_sum = 0;
        for (int t = 0; t < NUM_TRIALS; t++) task_sum += task_times[t];
        double task_mean = task_sum / NUM_TRIALS;

        double recovery = 0;
        if (interfered_mean > isolated_mean) {
            recovery = (interfered_mean - task_mean) / (interfered_mean - isolated_mean) * 100;
            if (recovery < 0) recovery = 0;
            if (recovery > 100) recovery = 100;
        }
        results[i].recovery_pct = recovery;

        printf("  Detection rate: %.0f%%, p50: %.1fms, p99: %.1fms, overhead: %.1f%%, recovery: %.0f%%\n",
               results[i].detection_rate, results[i].det_latency_p50,
               results[i].det_latency_p99, results[i].overhead_pct, results[i].recovery_pct);
    }

    /* Summary table */
    printf("\n================================================================\n");
    printf("SUMMARY TABLE (for paper)\n");
    printf("================================================================\n\n");
    printf("Interval | Det-p50 | Det-p99 | Overhead | Det Rate | Recovery\n");
    printf("---------|---------|---------|----------|----------|----------\n");
    for (int i = 0; i < (int)NUM_INTERVALS; i++) {
        printf("  %3dms  | %6.1fms| %6.1fms|   %5.1f%% |   %5.0f%% |   %5.0f%%\n",
               results[i].interval_ms,
               results[i].det_latency_p50,
               results[i].det_latency_p99,
               results[i].overhead_pct,
               results[i].detection_rate,
               results[i].recovery_pct);
    }

    /* LaTeX table output */
    printf("\n\\begin{table}[!htbp]\n");
    printf("\\centering\n");
    printf("\\caption{Sampling interval tradeoffs (M2)}\n");
    printf("\\label{tab:sampling}\n");
    printf("\\footnotesize\n");
    printf("\\begin{tabular}{@{}rrrrrr@{}}\n");
    printf("\\toprule\n");
    printf("Interval & Det-p50 & Det-p99 & Overhead & Det Rate & Recovery \\\\\n");
    printf("\\midrule\n");
    for (int i = 0; i < (int)NUM_INTERVALS; i++) {
        printf("%dms & %.1fms & %.1fms & %.1f\\%% & %.0f\\%% & %.0f\\%% \\\\\n",
               results[i].interval_ms,
               results[i].det_latency_p50,
               results[i].det_latency_p99,
               results[i].overhead_pct,
               results[i].detection_rate,
               results[i].recovery_pct);
    }
    printf("\\bottomrule\n");
    printf("\\end{tabular}\n");
    printf("\\end{table}\n");

    /* Recommendations */
    printf("\n================================================================\n");
    printf("ANALYSIS\n");
    printf("================================================================\n\n");

    /* Find sweet spot: highest recovery with acceptable overhead */
    int best_idx = 0;
    double best_score = 0;
    for (int i = 0; i < (int)NUM_INTERVALS; i++) {
        /* Score = recovery - overhead (favor low overhead) */
        double score = results[i].recovery_pct - results[i].overhead_pct;
        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    printf("Sweet spot: %dms interval\n", results[best_idx].interval_ms);
    printf("  - Detection latency: %.1fms (p50), %.1fms (p99)\n",
           results[best_idx].det_latency_p50, results[best_idx].det_latency_p99);
    printf("  - Overhead: %.1f%%\n", results[best_idx].overhead_pct);
    printf("  - Recovery: %.0f%%\n", results[best_idx].recovery_pct);

    /* Cleanup */
    close(g_fd_cycles);
    close(g_fd_instructions);
    free(A);
    free(B);
    free(C);
    for (int i = 0; i < NUM_INTERFERERS; i++) {
        free(memory_buffers[i]);
    }

    return 0;
#endif
}
