/*
 * Memcached Real Workload Test
 * Compare with Redis to show technique works across different key-value stores
 *
 * Build:
 *   gcc -O3 -march=native -DUSE_MEMCACHED test_memcached_workload.c -lpthread -lmemcached -lm -o test_memcached
 *   gcc -O3 -march=native test_memcached_workload.c -lpthread -lm -o test_memcached_sim  # simulated
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
#include <signal.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>

#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/syscall.h>
#endif

#ifdef USE_MEMCACHED
#include <libmemcached/memcached.h>
#endif

#define BUFFER_SIZE (8 * 1024 * 1024)
#define NUM_OPERATIONS 5000
#define NUM_TRIALS 30
#define BACKGROUND_ITERS 100000
#define VALUE_SIZE 256
#define OPS_PER_ACCESS 500

static inline uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static char *memory_buffer = NULL;
static _Atomic int background_should_pause = 0;
static _Atomic int background_running = 0;
static _Atomic int test_complete = 0;
static _Atomic int interference_detected = 0;
static _Atomic uint64_t detection_time_ns = 0;

static double g_ipc_threshold = 0.30;
static int g_fd_cycles = -1;
static int g_fd_instructions = -1;
static _Atomic int interrupt_count = 0;

static int compare_double(const void *a, const void *b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}

static double percentile(double *sorted, int n, double p) {
    if (n == 0) return 0;
    double idx = (p / 100.0) * (n - 1);
    int lower = (int)idx;
    int upper = lower + 1;
    if (upper >= n) return sorted[n - 1];
    double frac = idx - lower;
    return sorted[lower] * (1 - frac) + sorted[upper] * frac;
}

#ifdef __linux__
static int perf_event_open(struct perf_event_attr *attr, pid_t pid,
                           int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}

static void perf_overflow_handler(int signum, siginfo_t *info, void *context) {
    (void)signum; (void)context;
    if (info->si_fd != g_fd_cycles) return;

    atomic_fetch_add(&interrupt_count, 1);

    uint64_t cycles = 0, instructions = 0;
    read(g_fd_cycles, &cycles, sizeof(cycles));
    read(g_fd_instructions, &instructions, sizeof(instructions));

    double ipc = (cycles > 0) ? (double)instructions / cycles : 0;

    if (ipc < g_ipc_threshold && !atomic_load(&interference_detected)) {
        atomic_store(&interference_detected, 1);
        atomic_store(&detection_time_ns, get_time_ns());
        atomic_store(&background_should_pause, 1);
    }

    ioctl(g_fd_cycles, PERF_EVENT_IOC_REFRESH, 1);
}

static int setup_detection(int cycles_per_check) {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.size = sizeof(pe);
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    pe.type = PERF_TYPE_HARDWARE;
    pe.config = PERF_COUNT_HW_CPU_CYCLES;
    pe.sample_period = cycles_per_check;
    pe.wakeup_events = 1;

    g_fd_cycles = perf_event_open(&pe, 0, -1, -1, 0);
    if (g_fd_cycles < 0) return -1;

    pe.sample_period = 0;
    pe.config = PERF_COUNT_HW_INSTRUCTIONS;
    g_fd_instructions = perf_event_open(&pe, 0, -1, g_fd_cycles, 0);
    if (g_fd_instructions < 0) {
        close(g_fd_cycles);
        return -1;
    }

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = perf_overflow_handler;
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigaction(SIGIO, &sa, NULL);

    fcntl(g_fd_cycles, F_SETFL, O_ASYNC);
    fcntl(g_fd_cycles, F_SETSIG, SIGIO);
    fcntl(g_fd_cycles, F_SETOWN, getpid());

    return 0;
}

static void cleanup_detection(void) {
    if (g_fd_cycles >= 0) close(g_fd_cycles);
    if (g_fd_instructions >= 0) close(g_fd_instructions);
    g_fd_cycles = -1;
    g_fd_instructions = -1;
}

static void start_monitoring(void) {
    atomic_store(&interrupt_count, 0);
    atomic_store(&interference_detected, 0);
    atomic_store(&detection_time_ns, 0);

    ioctl(g_fd_cycles, PERF_EVENT_IOC_RESET, 0);
    ioctl(g_fd_instructions, PERF_EVENT_IOC_RESET, 0);
    ioctl(g_fd_cycles, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(g_fd_instructions, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(g_fd_cycles, PERF_EVENT_IOC_REFRESH, 1);
}

static void stop_monitoring(void) {
    ioctl(g_fd_cycles, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(g_fd_instructions, PERF_EVENT_IOC_DISABLE, 0);
}
#endif

static void memory_workload(int iterations) {
    unsigned int seed = (unsigned int)get_time_ns();
    volatile char sum = 0;
    for (int i = 0; i < iterations; i++) {
        int idx = rand_r(&seed) % BUFFER_SIZE;
        sum += memory_buffer[idx];
        memory_buffer[(idx + 1) % BUFFER_SIZE] = sum;
    }
}

typedef struct {
    pthread_t thread;
    int core_id;
    int with_throttling;
} background_worker_t;

static void* background_worker(void *arg) {
    background_worker_t *w = (background_worker_t*)arg;

#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(w->core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#endif

    atomic_fetch_add(&background_running, 1);

    while (!atomic_load(&test_complete)) {
        if (w->with_throttling && atomic_load(&background_should_pause)) {
            usleep(100);  /* More effective than sched_yield: 84% vs 68% recovery */
            continue;
        }
        memory_workload(BACKGROUND_ITERS / 10);
    }

    atomic_fetch_sub(&background_running, 1);
    return NULL;
}

typedef struct {
    double total_time_ms;
    double *op_latencies_us;
    int num_ops;
    int detected;
    double detection_latency_us;
} workload_result_t;

#ifdef USE_MEMCACHED
static workload_result_t run_memcached_workload(memcached_st *memc, int with_detection) {
    workload_result_t result = {0};
    result.op_latencies_us = malloc(NUM_OPERATIONS * sizeof(double));
    result.num_ops = NUM_OPERATIONS;

    char key[64];
    char value[VALUE_SIZE];
    memset(value, 'x', VALUE_SIZE - 1);
    value[VALUE_SIZE - 1] = '\0';

#ifdef __linux__
    if (with_detection) {
        start_monitoring();
    }
#endif

    uint64_t total_start = get_time_ns();

    for (int i = 0; i < NUM_OPERATIONS; i++) {
        snprintf(key, sizeof(key), "test:key:%d", i);
        uint64_t op_start = get_time_ns();

        if (i % 2 == 0) {
            memcached_set(memc, key, strlen(key), value, VALUE_SIZE - 1, 0, 0);
        } else {
            size_t val_len;
            uint32_t flags;
            memcached_return_t rc;
            char *ret = memcached_get(memc, key, strlen(key), &val_len, &flags, &rc);
            if (ret) free(ret);
        }

        result.op_latencies_us[i] = (get_time_ns() - op_start) / 1e3;
    }

    result.total_time_ms = (get_time_ns() - total_start) / 1e6;

#ifdef __linux__
    if (with_detection) {
        stop_monitoring();
        result.detected = atomic_load(&interference_detected);
        uint64_t det_time = atomic_load(&detection_time_ns);
        if (det_time > 0 && det_time > total_start) {
            result.detection_latency_us = (det_time - total_start) / 1e3;
        }
    }
#endif

    return result;
}
#endif

/* Simulated memcached-like workload */
static workload_result_t run_simulated_memcached(int with_detection) {
    workload_result_t result = {0};
    result.op_latencies_us = malloc(NUM_OPERATIONS * sizeof(double));
    result.num_ops = NUM_OPERATIONS;

    char **keys = malloc(NUM_OPERATIONS * sizeof(char*));
    char **values = malloc(NUM_OPERATIONS * sizeof(char*));
    for (int i = 0; i < NUM_OPERATIONS; i++) {
        keys[i] = malloc(64);
        values[i] = malloc(VALUE_SIZE);
        snprintf(keys[i], 64, "test:key:%d", i);
        memset(values[i], 'x', VALUE_SIZE - 1);
        values[i][VALUE_SIZE - 1] = '\0';
    }

#ifdef __linux__
    if (with_detection) {
        start_monitoring();
    }
#endif

    uint64_t total_start = get_time_ns();

    for (int i = 0; i < NUM_OPERATIONS; i++) {
        uint64_t op_start = get_time_ns();

        if (i % 2 == 0) {
            unsigned int seed = (unsigned int)i;
            for (int j = 0; j < OPS_PER_ACCESS; j++) {
                int idx = rand_r(&seed) % BUFFER_SIZE;
                memory_buffer[idx] = values[i][j % VALUE_SIZE];
            }
        } else {
            unsigned int seed = (unsigned int)i;
            volatile char sum = 0;
            for (int j = 0; j < OPS_PER_ACCESS; j++) {
                int idx = rand_r(&seed) % BUFFER_SIZE;
                sum += memory_buffer[idx];
            }
        }

        result.op_latencies_us[i] = (get_time_ns() - op_start) / 1e3;
    }

    result.total_time_ms = (get_time_ns() - total_start) / 1e6;

#ifdef __linux__
    if (with_detection) {
        stop_monitoring();
        result.detected = atomic_load(&interference_detected);
        uint64_t det_time = atomic_load(&detection_time_ns);
        if (det_time > 0 && det_time > total_start) {
            result.detection_latency_us = (det_time - total_start) / 1e3;
        }
    }
#endif

    for (int i = 0; i < NUM_OPERATIONS; i++) {
        free(keys[i]);
        free(values[i]);
    }
    free(keys);
    free(values);

    return result;
}

int main(void) {
    printf("================================================================\n");
    printf("MEMCACHED WORKLOAD TEST\n");
#ifdef USE_MEMCACHED
    printf("Mode: Memcached (libmemcached)\n");
#else
    printf("Mode: Simulated (compile with -DUSE_MEMCACHED for real)\n");
#endif
    printf("================================================================\n");

#ifndef __linux__
    printf("This test requires Linux\n");
    return 1;
#endif

    memory_buffer = aligned_alloc(64, BUFFER_SIZE);
    if (!memory_buffer) {
        printf("Failed to allocate memory\n");
        return 1;
    }
    memset(memory_buffer, 0, BUFFER_SIZE);

#ifdef USE_MEMCACHED
    memcached_st *memc = memcached_create(NULL);
    memcached_server_add(memc, "127.0.0.1", 11211);
    memcached_return_t rc = memcached_set(memc, "test", 4, "test", 4, 0, 0);
    if (rc != MEMCACHED_SUCCESS) {
        printf("Failed to connect to memcached: %s\n", memcached_strerror(memc, rc));
        printf("Make sure memcached is running: memcached -d\n");
        memcached_free(memc);
        free(memory_buffer);
        return 1;
    }
    printf("\nConnected to Memcached\n");
#endif

    background_worker_t workers[3];
    for (int i = 0; i < 3; i++) {
        workers[i].core_id = i + 1;
    }

#ifdef __linux__
    if (setup_detection(240000) < 0) {
        printf("Failed to setup detection\n");
#ifdef USE_MEMCACHED
        memcached_free(memc);
#endif
        free(memory_buffer);
        return 1;
    }

    /* Auto-calibrate */
    printf("\nCalibrating...\n");
    ioctl(g_fd_cycles, PERF_EVENT_IOC_RESET, 0);
    ioctl(g_fd_instructions, PERF_EVENT_IOC_RESET, 0);
    ioctl(g_fd_cycles, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(g_fd_instructions, PERF_EVENT_IOC_ENABLE, 0);

#ifdef USE_MEMCACHED
    char cal_key[64], cal_value[256];
    memset(cal_value, 'x', 255);
    cal_value[255] = '\0';
    for (int i = 0; i < 500; i++) {
        snprintf(cal_key, sizeof(cal_key), "cal:key:%d", i);
        memcached_set(memc, cal_key, strlen(cal_key), cal_value, 255, 0, 0);
        size_t vl; uint32_t fl; memcached_return_t r;
        char *v = memcached_get(memc, cal_key, strlen(cal_key), &vl, &fl, &r);
        if (v) free(v);
    }
#else
    memory_workload(PRIORITY_ITERS / 4);
#endif

    uint64_t cal_cyc = 0, cal_ins = 0;
    read(g_fd_cycles, &cal_cyc, sizeof(cal_cyc));
    read(g_fd_instructions, &cal_ins, sizeof(cal_ins));
    ioctl(g_fd_cycles, PERF_EVENT_IOC_DISABLE, 0);

    double baseline_ipc = (cal_cyc > 0) ? (double)cal_ins / cal_cyc : 0.5;
    g_ipc_threshold = baseline_ipc * 0.60;
    printf("Baseline IPC: %.3f, Threshold: %.3f\n", baseline_ipc, g_ipc_threshold);
#endif

    printf("\n================================================================\n");
    printf("BASELINE: Isolated\n");
    printf("================================================================\n");

    double isolated_totals[NUM_TRIALS];
    double *isolated_ops[NUM_TRIALS];

    for (int trial = 0; trial < NUM_TRIALS; trial++) {
#ifdef USE_MEMCACHED
        workload_result_t r = run_memcached_workload(memc, 0);
#else
        workload_result_t r = run_simulated_memcached(0);
#endif
        isolated_totals[trial] = r.total_time_ms;
        isolated_ops[trial] = r.op_latencies_us;
    }

    double *all_isolated = malloc(NUM_TRIALS * NUM_OPERATIONS * sizeof(double));
    for (int t = 0; t < NUM_TRIALS; t++) {
        for (int i = 0; i < NUM_OPERATIONS; i++) {
            all_isolated[t * NUM_OPERATIONS + i] = isolated_ops[t][i];
        }
    }
    qsort(all_isolated, NUM_TRIALS * NUM_OPERATIONS, sizeof(double), compare_double);
    qsort(isolated_totals, NUM_TRIALS, sizeof(double), compare_double);

    printf("Total (p50): %.1f ms\n", percentile(isolated_totals, NUM_TRIALS, 50));
    printf("Op (p50):    %.1f μs\n", percentile(all_isolated, NUM_TRIALS * NUM_OPERATIONS, 50));
    printf("Op (p99):    %.1f μs\n", percentile(all_isolated, NUM_TRIALS * NUM_OPERATIONS, 99));

    printf("\n================================================================\n");
    printf("WITH INTERFERENCE\n");
    printf("================================================================\n");

    double interference_totals[NUM_TRIALS];
    double *interference_ops[NUM_TRIALS];

    for (int trial = 0; trial < NUM_TRIALS; trial++) {
        atomic_store(&test_complete, 0);
        atomic_store(&background_should_pause, 0);

        for (int i = 0; i < 3; i++) {
            workers[i].with_throttling = 0;
            pthread_create(&workers[i].thread, NULL, background_worker, &workers[i]);
        }
        while (atomic_load(&background_running) < 3) sched_yield();

#ifdef USE_MEMCACHED
        workload_result_t r = run_memcached_workload(memc, 0);
#else
        workload_result_t r = run_simulated_memcached(0);
#endif

        atomic_store(&test_complete, 1);
        for (int i = 0; i < 3; i++) pthread_join(workers[i].thread, NULL);

        interference_totals[trial] = r.total_time_ms;
        interference_ops[trial] = r.op_latencies_us;
    }

    double *all_interference = malloc(NUM_TRIALS * NUM_OPERATIONS * sizeof(double));
    for (int t = 0; t < NUM_TRIALS; t++) {
        for (int i = 0; i < NUM_OPERATIONS; i++) {
            all_interference[t * NUM_OPERATIONS + i] = interference_ops[t][i];
        }
    }
    qsort(all_interference, NUM_TRIALS * NUM_OPERATIONS, sizeof(double), compare_double);
    qsort(interference_totals, NUM_TRIALS, sizeof(double), compare_double);

    double iso_p50 = percentile(isolated_totals, NUM_TRIALS, 50);
    double int_p50 = percentile(interference_totals, NUM_TRIALS, 50);

    printf("Total (p50): %.1f ms (+%.0f%%)\n", int_p50, ((int_p50 - iso_p50) / iso_p50) * 100);
    printf("Op (p50):    %.1f μs\n", percentile(all_interference, NUM_TRIALS * NUM_OPERATIONS, 50));
    printf("Op (p99):    %.1f μs\n", percentile(all_interference, NUM_TRIALS * NUM_OPERATIONS, 99));

    printf("\n================================================================\n");
    printf("WITH PROTECTION\n");
    printf("================================================================\n");

    double protected_totals[NUM_TRIALS];
    double *protected_ops[NUM_TRIALS];
    int detected = 0;
    double total_det = 0;

    for (int trial = 0; trial < NUM_TRIALS; trial++) {
        atomic_store(&test_complete, 0);
        atomic_store(&background_should_pause, 0);

        for (int i = 0; i < 3; i++) {
            workers[i].with_throttling = 1;
            pthread_create(&workers[i].thread, NULL, background_worker, &workers[i]);
        }
        while (atomic_load(&background_running) < 3) sched_yield();

#ifdef USE_MEMCACHED
        workload_result_t r = run_memcached_workload(memc, 1);
#else
        workload_result_t r = run_simulated_memcached(1);
#endif

        atomic_store(&test_complete, 1);
        for (int i = 0; i < 3; i++) pthread_join(workers[i].thread, NULL);

        protected_totals[trial] = r.total_time_ms;
        protected_ops[trial] = r.op_latencies_us;
        if (r.detected) {
            detected++;
            total_det += r.detection_latency_us;
        }
        atomic_store(&background_should_pause, 0);
    }

    double *all_protected = malloc(NUM_TRIALS * NUM_OPERATIONS * sizeof(double));
    for (int t = 0; t < NUM_TRIALS; t++) {
        for (int i = 0; i < NUM_OPERATIONS; i++) {
            all_protected[t * NUM_OPERATIONS + i] = protected_ops[t][i];
        }
    }
    qsort(all_protected, NUM_TRIALS * NUM_OPERATIONS, sizeof(double), compare_double);
    qsort(protected_totals, NUM_TRIALS, sizeof(double), compare_double);

    double prot_p50 = percentile(protected_totals, NUM_TRIALS, 50);
    double lat_inc = int_p50 - iso_p50;
    double recovered = int_p50 - prot_p50;
    double recovery_pct = (recovered / lat_inc) * 100;

    printf("Total (p50): %.1f ms\n", prot_p50);
    printf("Op (p50):    %.1f μs\n", percentile(all_protected, NUM_TRIALS * NUM_OPERATIONS, 50));
    printf("Op (p99):    %.1f μs\n", percentile(all_protected, NUM_TRIALS * NUM_OPERATIONS, 99));
    printf("\nDetection:   %d/%d (%.0f%%)\n", detected, NUM_TRIALS, (double)detected/NUM_TRIALS*100);
    printf("Avg det:     %.0f μs\n", detected > 0 ? total_det / detected : 0);
    printf("Recovery:    %.1f%%\n", recovery_pct);

    double iso_p99 = percentile(all_isolated, NUM_TRIALS * NUM_OPERATIONS, 99);
    double int_p99 = percentile(all_interference, NUM_TRIALS * NUM_OPERATIONS, 99);
    double prot_p99 = percentile(all_protected, NUM_TRIALS * NUM_OPERATIONS, 99);

    printf("\n================================================================\n");
    printf("SUMMARY\n");
    printf("================================================================\n");
    printf("%-20s %10s %10s %10s\n", "Scenario", "Time-p50", "Op-p50", "Op-p99");
    printf("%-20s %10.1fms %10.1fμs %10.1fμs\n", "Isolated", iso_p50,
           percentile(all_isolated, NUM_TRIALS * NUM_OPERATIONS, 50), iso_p99);
    printf("%-20s %10.1fms %10.1fμs %10.1fμs\n", "Interference", int_p50,
           percentile(all_interference, NUM_TRIALS * NUM_OPERATIONS, 50), int_p99);
    printf("%-20s %10.1fms %10.1fμs %10.1fμs\n", "Protected", prot_p50,
           percentile(all_protected, NUM_TRIALS * NUM_OPERATIONS, 50), prot_p99);

    printf("\np99 reduction: %.0f%% (%.1f → %.1f μs)\n",
           ((int_p99 - prot_p99) / (int_p99 - iso_p99)) * 100, int_p99, prot_p99);

#ifdef __linux__
    cleanup_detection();
#endif

#ifdef USE_MEMCACHED
    memcached_free(memc);
#endif

    for (int t = 0; t < NUM_TRIALS; t++) {
        free(isolated_ops[t]);
        free(interference_ops[t]);
        free(protected_ops[t]);
    }
    free(all_isolated);
    free(all_interference);
    free(all_protected);
    free(memory_buffer);

    return 0;
}
