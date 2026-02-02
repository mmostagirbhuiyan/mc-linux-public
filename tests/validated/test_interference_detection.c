/*
 * Interference Detection Test
 *
 * HYPOTHESIS: When two memory-bound tasks run simultaneously on cores
 * sharing L3 cache, their IPC drops due to cache contention.
 * MC's per-task telemetry can detect this interference in real-time.
 *
 * EXPERIMENT:
 * 1. Run memory-bound task ALONE on core 0 → measure baseline IPC
 * 2. Run memory-bound task on core 0 WHILE core 1 also runs memory-bound → measure contended IPC
 * 3. Compare: if contended IPC < baseline IPC, we've detected interference
 *
 * SUCCESS CRITERIA:
 * - Measurable IPC drop (>10%) when memory tasks run simultaneously
 * - Consistent detection across multiple trials
 *
 * Pi 5 Architecture:
 * - 4x Cortex-A76 cores
 * - Private L2 (512KB per core)
 * - Shared L3 (2MB)
 * - Memory bandwidth shared across all cores
 *
 * Build:
 *   gcc -O3 -march=native test_interference_detection.c -lpthread -lm -o test_interference
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

#ifdef __linux__
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>
#endif

/* Memory buffer - must be larger than L3 to cause cache misses */
#define MEMORY_SIZE (8 * 1024 * 1024)  /* 8MB > 2MB L3 */
#define MEMORY_ITERS 500000            /* Enough iterations for stable measurement */

static char *memory_buffer_0 = NULL;   /* For core 0 */
static char *memory_buffer_1 = NULL;   /* For core 1 - separate to avoid false sharing */

/* Timing */
static inline uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

/* ========== Perf counters ========== */
#ifdef __linux__
static int perf_event_open(struct perf_event_attr *attr, pid_t pid,
                           int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}

typedef struct {
    int fd_cycles;
    int fd_instructions;
    int fd_cache_misses;
} perf_t;

static int perf_init(perf_t *p) {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.size = sizeof(pe);
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;

    pe.type = PERF_TYPE_HARDWARE;
    pe.config = PERF_COUNT_HW_CPU_CYCLES;
    p->fd_cycles = perf_event_open(&pe, 0, -1, -1, 0);

    pe.config = PERF_COUNT_HW_INSTRUCTIONS;
    p->fd_instructions = perf_event_open(&pe, 0, -1, p->fd_cycles, 0);

    pe.config = PERF_COUNT_HW_CACHE_MISSES;
    p->fd_cache_misses = perf_event_open(&pe, 0, -1, p->fd_cycles, 0);

    return (p->fd_cycles >= 0);
}

static void perf_start(perf_t *p) {
    ioctl(p->fd_cycles, PERF_EVENT_IOC_RESET, 0);
    ioctl(p->fd_instructions, PERF_EVENT_IOC_RESET, 0);
    ioctl(p->fd_cache_misses, PERF_EVENT_IOC_RESET, 0);
    ioctl(p->fd_cycles, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(p->fd_instructions, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(p->fd_cache_misses, PERF_EVENT_IOC_ENABLE, 0);
}

static void perf_stop(perf_t *p, uint64_t *cycles, uint64_t *instructions, uint64_t *cache_misses) {
    ioctl(p->fd_cycles, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(p->fd_instructions, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(p->fd_cache_misses, PERF_EVENT_IOC_DISABLE, 0);
    read(p->fd_cycles, cycles, sizeof(*cycles));
    read(p->fd_instructions, instructions, sizeof(*instructions));
    read(p->fd_cache_misses, cache_misses, sizeof(*cache_misses));
}

static void perf_close(perf_t *p) {
    if (p->fd_cycles >= 0) close(p->fd_cycles);
    if (p->fd_instructions >= 0) close(p->fd_instructions);
    if (p->fd_cache_misses >= 0) close(p->fd_cache_misses);
}
#endif

/* ========== Memory-bound workload ========== */
static volatile char sink = 0;

static void memory_workload(char *buffer, int iterations) {
    unsigned int seed = (unsigned int)get_time_ns();
    volatile char sum = 0;
    for (int i = 0; i < iterations; i++) {
        int idx = rand_r(&seed) % MEMORY_SIZE;
        sum += buffer[idx];
        buffer[(idx + 64) % MEMORY_SIZE] = sum;  /* Write to different cache line */
    }
    sink = sum;
}

/* ========== Test structures ========== */
typedef struct {
    double ipc;
    uint64_t cycles;
    uint64_t instructions;
    uint64_t cache_misses;
    double time_ms;
} measurement_t;

typedef struct {
    int core_id;
    char *buffer;
    int iterations;
    _Atomic int *start_flag;
    _Atomic int *done_flag;
    measurement_t result;
} worker_arg_t;

/* Worker that runs memory workload and measures its own performance */
static void* measured_worker(void *arg) {
    worker_arg_t *w = (worker_arg_t *)arg;

#ifdef __linux__
    /* Pin to core */
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(w->core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

    perf_t perf;
    perf_init(&perf);

    /* Wait for start signal */
    while (!atomic_load(w->start_flag)) {
        __asm__ __volatile__("yield" ::: "memory");
    }

    uint64_t start_time = get_time_ns();
    perf_start(&perf);

    memory_workload(w->buffer, w->iterations);

    uint64_t cycles, instructions, cache_misses;
    perf_stop(&perf, &cycles, &instructions, &cache_misses);
    uint64_t end_time = get_time_ns();

    w->result.cycles = cycles;
    w->result.instructions = instructions;
    w->result.cache_misses = cache_misses;
    w->result.ipc = (cycles > 0) ? (double)instructions / cycles : 0;
    w->result.time_ms = (end_time - start_time) / 1e6;

    perf_close(&perf);
#endif

    atomic_store(w->done_flag, 1);
    return NULL;
}

/* Interfering worker - just runs memory workload, no measurement */
static void* interfering_worker(void *arg) {
    worker_arg_t *w = (worker_arg_t *)arg;

#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(w->core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#endif

    /* Wait for start signal */
    while (!atomic_load(w->start_flag)) {
        __asm__ __volatile__("yield" ::: "memory");
    }

    memory_workload(w->buffer, w->iterations);

    atomic_store(w->done_flag, 1);
    return NULL;
}

/* ========== Test runners ========== */

/* Run memory workload ALONE on core 0 */
static measurement_t run_isolated(void) {
    _Atomic int start_flag = 0;
    _Atomic int done_flag = 0;

    worker_arg_t arg = {
        .core_id = 0,
        .buffer = memory_buffer_0,
        .iterations = MEMORY_ITERS,
        .start_flag = &start_flag,
        .done_flag = &done_flag
    };

    pthread_t thread;
    pthread_create(&thread, NULL, measured_worker, &arg);

    usleep(1000);  /* Let thread start */
    atomic_store(&start_flag, 1);

    pthread_join(thread, NULL);
    return arg.result;
}

/* Run memory workload on core 0 WHILE core 1 also runs memory workload */
static measurement_t run_with_interference(void) {
    _Atomic int start_flag = 0;
    _Atomic int done_flag_0 = 0;
    _Atomic int done_flag_1 = 0;

    worker_arg_t arg0 = {
        .core_id = 0,
        .buffer = memory_buffer_0,
        .iterations = MEMORY_ITERS,
        .start_flag = &start_flag,
        .done_flag = &done_flag_0
    };

    worker_arg_t arg1 = {
        .core_id = 1,
        .buffer = memory_buffer_1,
        .iterations = MEMORY_ITERS,
        .start_flag = &start_flag,
        .done_flag = &done_flag_1
    };

    pthread_t thread0, thread1;
    pthread_create(&thread0, NULL, measured_worker, &arg0);
    pthread_create(&thread1, NULL, interfering_worker, &arg1);

    usleep(1000);
    atomic_store(&start_flag, 1);  /* Start both simultaneously */

    pthread_join(thread0, NULL);
    pthread_join(thread1, NULL);

    return arg0.result;  /* Return core 0's measurement */
}

/* Run with interference from 2 cores (cores 1 and 2) */
static measurement_t run_with_heavy_interference(void) {
    _Atomic int start_flag = 0;
    _Atomic int done_flags[3] = {0, 0, 0};

    worker_arg_t args[3];

    /* Core 0: measured */
    args[0] = (worker_arg_t){
        .core_id = 0,
        .buffer = memory_buffer_0,
        .iterations = MEMORY_ITERS,
        .start_flag = &start_flag,
        .done_flag = &done_flags[0]
    };

    /* Core 1: interferer */
    args[1] = (worker_arg_t){
        .core_id = 1,
        .buffer = memory_buffer_1,
        .iterations = MEMORY_ITERS,
        .start_flag = &start_flag,
        .done_flag = &done_flags[1]
    };

    /* Core 2: interferer (reuse buffer 1, different random seed) */
    args[2] = (worker_arg_t){
        .core_id = 2,
        .buffer = memory_buffer_1,
        .iterations = MEMORY_ITERS,
        .start_flag = &start_flag,
        .done_flag = &done_flags[2]
    };

    pthread_t threads[3];
    pthread_create(&threads[0], NULL, measured_worker, &args[0]);
    pthread_create(&threads[1], NULL, interfering_worker, &args[1]);
    pthread_create(&threads[2], NULL, interfering_worker, &args[2]);

    usleep(1000);
    atomic_store(&start_flag, 1);

    for (int i = 0; i < 3; i++) {
        pthread_join(threads[i], NULL);
    }

    return args[0].result;
}

/* ========== Main ========== */
int main(void) {
    printf("================================================================\n");
    printf("INTERFERENCE DETECTION TEST\n");
    printf("================================================================\n");
#ifndef __linux__
    printf("This test requires Linux for perf_event_open\n");
    return 1;
#endif

    printf("Pi 5: 4x Cortex-A76, 512KB L2/core, 2MB shared L3\n");
    printf("Buffer: %d MB (larger than L3)\n", MEMORY_SIZE / (1024 * 1024));
    printf("Iterations: %d random accesses per test\n\n", MEMORY_ITERS);

    printf("HYPOTHESIS: Memory-bound tasks running simultaneously will\n");
    printf("show lower IPC due to shared L3/memory bandwidth contention.\n\n");

    /* Allocate separate buffers to avoid false sharing */
    memory_buffer_0 = aligned_alloc(64, MEMORY_SIZE);
    memory_buffer_1 = aligned_alloc(64, MEMORY_SIZE);
    if (!memory_buffer_0 || !memory_buffer_1) {
        printf("Failed to allocate memory\n");
        return 1;
    }

    /* Initialize with different patterns */
    for (int i = 0; i < MEMORY_SIZE; i++) {
        memory_buffer_0[i] = (char)(i ^ 0xAA);
        memory_buffer_1[i] = (char)(i ^ 0x55);
    }

    /* Warmup */
    printf("Warming up...\n");
    run_isolated();
    run_with_interference();

    /* Run tests */
    int NUM_TRIALS = 5;
    measurement_t isolated[5], interfered[5], heavy[5];

    printf("\nRunning %d trials of each scenario...\n\n", NUM_TRIALS);

    for (int i = 0; i < NUM_TRIALS; i++) {
        printf("Trial %d: ", i + 1);

        isolated[i] = run_isolated();
        printf("isolated=%.3f ", isolated[i].ipc);

        interfered[i] = run_with_interference();
        printf("1-interferer=%.3f ", interfered[i].ipc);

        heavy[i] = run_with_heavy_interference();
        printf("2-interferers=%.3f\n", heavy[i].ipc);

        usleep(100000);  /* Cool down between trials */
    }

    /* Compute averages */
    double iso_ipc = 0, int_ipc = 0, heavy_ipc = 0;
    double iso_misses = 0, int_misses = 0, heavy_misses = 0;
    double iso_time = 0, int_time = 0, heavy_time = 0;

    for (int i = 0; i < NUM_TRIALS; i++) {
        iso_ipc += isolated[i].ipc;
        int_ipc += interfered[i].ipc;
        heavy_ipc += heavy[i].ipc;
        iso_misses += isolated[i].cache_misses;
        int_misses += interfered[i].cache_misses;
        heavy_misses += heavy[i].cache_misses;
        iso_time += isolated[i].time_ms;
        int_time += interfered[i].time_ms;
        heavy_time += heavy[i].time_ms;
    }
    iso_ipc /= NUM_TRIALS;
    int_ipc /= NUM_TRIALS;
    heavy_ipc /= NUM_TRIALS;
    iso_misses /= NUM_TRIALS;
    int_misses /= NUM_TRIALS;
    heavy_misses /= NUM_TRIALS;
    iso_time /= NUM_TRIALS;
    int_time /= NUM_TRIALS;
    heavy_time /= NUM_TRIALS;

    printf("\n================================================================\n");
    printf("RESULTS\n");
    printf("================================================================\n\n");

    printf("%-20s %10s %12s %12s\n", "Scenario", "Avg IPC", "Cache Misses", "Time (ms)");
    printf("------------------------------------------------------------------------\n");
    printf("%-20s %10.3f %12.0f %12.1f\n", "Isolated (baseline)", iso_ipc, iso_misses, iso_time);
    printf("%-20s %10.3f %12.0f %12.1f\n", "1 Interferer", int_ipc, int_misses, int_time);
    printf("%-20s %10.3f %12.0f %12.1f\n", "2 Interferers", heavy_ipc, heavy_misses, heavy_time);

    printf("\n================================================================\n");
    printf("INTERFERENCE ANALYSIS\n");
    printf("================================================================\n\n");

    double ipc_drop_1 = (iso_ipc - int_ipc) / iso_ipc * 100;
    double ipc_drop_2 = (iso_ipc - heavy_ipc) / iso_ipc * 100;
    double miss_increase_1 = (int_misses - iso_misses) / iso_misses * 100;
    double miss_increase_2 = (heavy_misses - iso_misses) / iso_misses * 100;
    double time_increase_1 = (int_time - iso_time) / iso_time * 100;
    double time_increase_2 = (heavy_time - iso_time) / iso_time * 100;

    printf("With 1 interferer:\n");
    printf("  IPC drop:           %+.1f%%\n", -ipc_drop_1);
    printf("  Cache miss increase: %+.1f%%\n", miss_increase_1);
    printf("  Time increase:       %+.1f%%\n", time_increase_1);

    printf("\nWith 2 interferers:\n");
    printf("  IPC drop:           %+.1f%%\n", -ipc_drop_2);
    printf("  Cache miss increase: %+.1f%%\n", miss_increase_2);
    printf("  Time increase:       %+.1f%%\n", time_increase_2);

    printf("\n================================================================\n");
    printf("CONCLUSION\n");
    printf("================================================================\n\n");

    if (ipc_drop_1 > 10) {
        printf("SUCCESS: Interference detected!\n\n");
        printf("- IPC drops %.0f%% with 1 interferer, %.0f%% with 2 interferers\n",
               ipc_drop_1, ipc_drop_2);
        printf("- MC can detect this in real-time via per-task IPC monitoring\n");
        printf("- This enables: throttling interferers, task migration, SLA protection\n");

        printf("\nDETECTION THRESHOLD:\n");
        printf("- Baseline IPC: %.3f\n", iso_ipc);
        printf("- If task IPC < %.3f (80%% of baseline), interference likely\n", iso_ipc * 0.8);
    } else if (ipc_drop_1 > 5) {
        printf("PARTIAL: Interference detectable but marginal (%.0f%% IPC drop)\n", ipc_drop_1);
        printf("- Detection may have false positives\n");
        printf("- Consider using cache miss increase (%.0f%%) as secondary signal\n", miss_increase_1);
    } else {
        printf("NO INTERFERENCE DETECTED: IPC drop only %.1f%%\n", ipc_drop_1);
        printf("- Possible causes:\n");
        printf("  - L3 cache large enough for both workloads\n");
        printf("  - Memory bandwidth not saturated\n");
        printf("  - Need more aggressive workload\n");
    }

    free(memory_buffer_0);
    free(memory_buffer_1);
    return 0;
}
