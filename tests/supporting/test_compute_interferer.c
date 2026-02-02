/*
 * Compute-Bound Interferer Test
 *
 * Test detection when interferers are CPU-bound (high IPC) instead of memory-bound (low IPC)
 * Key question: Can we still detect interference when aggressor has HIGH IPC?
 *
 * Build:
 *   gcc -O3 -march=native test_compute_interferer.c -lpthread -lm -o test_compute
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
#include <math.h>

#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/syscall.h>
#endif

#define BUFFER_SIZE (8 * 1024 * 1024)
#define PRIORITY_ITERS 500000
#define NUM_TRIALS 30

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

#ifdef __linux__
static int perf_event_open(struct perf_event_attr *attr, pid_t pid,
                           int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}

static void perf_overflow_handler(int signum, siginfo_t *info, void *context) {
    (void)signum; (void)context;
    if (info->si_fd != g_fd_cycles) return;

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
}

static void start_monitoring(void) {
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

/* Memory-bound workload (victim) - random memory access */
static void memory_workload(int iterations) {
    unsigned int seed = (unsigned int)get_time_ns();
    volatile char sum = 0;
    for (int i = 0; i < iterations; i++) {
        int idx = rand_r(&seed) % BUFFER_SIZE;
        sum += memory_buffer[idx];
        memory_buffer[(idx + 1) % BUFFER_SIZE] = sum;
    }
}

/* Compute-bound workload (aggressor) - high IPC, cache-friendly */
static void compute_workload(int iterations) {
    volatile double result = 0;
    for (int i = 0; i < iterations; i++) {
        result += sin((double)i) * cos((double)i);
        result *= 1.0001;
    }
}

/* Mixed workload - some compute, some memory */
static void mixed_workload(int iterations) {
    unsigned int seed = (unsigned int)get_time_ns();
    volatile double result = 0;
    volatile char sum = 0;
    for (int i = 0; i < iterations; i++) {
        if (i % 4 == 0) {
            int idx = rand_r(&seed) % BUFFER_SIZE;
            sum += memory_buffer[idx];
        } else {
            result += sin((double)i);
        }
    }
}

typedef struct {
    pthread_t thread;
    int core_id;
    int with_throttling;
    int workload_type;  /* 0=memory, 1=compute, 2=mixed */
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
            sched_yield();
            continue;
        }
        switch (w->workload_type) {
            case 0: memory_workload(10000); break;
            case 1: compute_workload(50000); break;
            case 2: mixed_workload(20000); break;
        }
    }

    atomic_fetch_sub(&background_running, 1);
    return NULL;
}

static double run_isolated(void) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#endif

    uint64_t start = get_time_ns();
    memory_workload(PRIORITY_ITERS);
    return (get_time_ns() - start) / 1e6;
}

typedef struct {
    double time_ms;
    int detected;
    double detection_us;
} result_t;

static double run_interference(background_worker_t *workers, int n, int workload_type) {
    atomic_store(&test_complete, 0);
    atomic_store(&background_should_pause, 0);

    for (int i = 0; i < n; i++) {
        workers[i].with_throttling = 0;
        workers[i].workload_type = workload_type;
        pthread_create(&workers[i].thread, NULL, background_worker, &workers[i]);
    }
    while (atomic_load(&background_running) < n) sched_yield();

#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#endif

    uint64_t start = get_time_ns();
    memory_workload(PRIORITY_ITERS);
    double time_ms = (get_time_ns() - start) / 1e6;

    atomic_store(&test_complete, 1);
    for (int i = 0; i < n; i++) pthread_join(workers[i].thread, NULL);

    return time_ms;
}

static result_t run_protected(background_worker_t *workers, int n, int workload_type) {
    result_t result = {0};

#ifdef __linux__
    atomic_store(&test_complete, 0);
    atomic_store(&background_should_pause, 0);

    for (int i = 0; i < n; i++) {
        workers[i].with_throttling = 1;
        workers[i].workload_type = workload_type;
        pthread_create(&workers[i].thread, NULL, background_worker, &workers[i]);
    }
    while (atomic_load(&background_running) < n) sched_yield();

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

    start_monitoring();

    uint64_t start = get_time_ns();
    memory_workload(PRIORITY_ITERS);
    uint64_t end = get_time_ns();

    stop_monitoring();

    atomic_store(&test_complete, 1);
    for (int i = 0; i < n; i++) pthread_join(workers[i].thread, NULL);

    result.time_ms = (end - start) / 1e6;
    result.detected = atomic_load(&interference_detected);
    uint64_t det = atomic_load(&detection_time_ns);
    if (det > 0 && det > start) {
        result.detection_us = (det - start) / 1e3;
    }
    atomic_store(&background_should_pause, 0);
#endif

    return result;
}

int main(void) {
    printf("================================================================\n");
    printf("COMPUTE-BOUND INTERFERER TEST\n");
    printf("Question: Can we detect interference from CPU-bound aggressors?\n");
    printf("================================================================\n");

#ifndef __linux__
    printf("This test requires Linux\n");
    return 1;
#endif

    memory_buffer = aligned_alloc(64, BUFFER_SIZE);
    memset(memory_buffer, 0, BUFFER_SIZE);

    background_worker_t workers[3];
    for (int i = 0; i < 3; i++) workers[i].core_id = i + 1;

    if (setup_detection(240000) < 0) {
        printf("Failed to setup detection\n");
        free(memory_buffer);
        return 1;
    }

    /* Calibrate on memory workload */
    memory_workload(PRIORITY_ITERS / 4);

    ioctl(g_fd_cycles, PERF_EVENT_IOC_RESET, 0);
    ioctl(g_fd_instructions, PERF_EVENT_IOC_RESET, 0);
    ioctl(g_fd_cycles, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(g_fd_instructions, PERF_EVENT_IOC_ENABLE, 0);
    memory_workload(PRIORITY_ITERS / 4);
    uint64_t cyc = 0, ins = 0;
    read(g_fd_cycles, &cyc, sizeof(cyc));
    read(g_fd_instructions, &ins, sizeof(ins));
    ioctl(g_fd_cycles, PERF_EVENT_IOC_DISABLE, 0);

    double baseline_ipc = (cyc > 0) ? (double)ins / cyc : 0.5;
    g_ipc_threshold = baseline_ipc * 0.75;
    printf("\nBaseline IPC: %.3f, Threshold: %.3f\n", baseline_ipc, g_ipc_threshold);

    /* Measure isolated */
    double isolated_avg = 0;
    for (int i = 0; i < NUM_TRIALS; i++) isolated_avg += run_isolated();
    isolated_avg /= NUM_TRIALS;
    printf("\nIsolated: %.1f ms\n", isolated_avg);

    const char *workload_names[] = {"Memory-bound", "Compute-bound", "Mixed"};

    printf("\n================================================================\n");
    printf("RESULTS BY INTERFERER TYPE\n");
    printf("================================================================\n");
    printf("%-15s %10s %10s %10s %10s %10s\n",
           "Interferer", "Int-Time", "Prot-Time", "Det%", "Det-μs", "Recovery");
    printf("------------------------------------------------------------------------\n");

    for (int wtype = 0; wtype < 3; wtype++) {
        double int_avg = 0, prot_avg = 0, det_avg = 0;
        int detected = 0;

        for (int trial = 0; trial < NUM_TRIALS; trial++) {
            int_avg += run_interference(workers, 3, wtype);
        }
        int_avg /= NUM_TRIALS;

        for (int trial = 0; trial < NUM_TRIALS; trial++) {
            result_t r = run_protected(workers, 3, wtype);
            prot_avg += r.time_ms;
            if (r.detected) {
                detected++;
                det_avg += r.detection_us;
            }
        }
        prot_avg /= NUM_TRIALS;
        if (detected > 0) det_avg /= detected;

        double lat_inc = int_avg - isolated_avg;
        double recovered = int_avg - prot_avg;
        double recovery = (recovered / lat_inc) * 100;

        printf("%-15s %10.1fms %10.1fms %10.0f%% %10.0fμs %10.1f%%\n",
               workload_names[wtype], int_avg, prot_avg,
               (double)detected / NUM_TRIALS * 100, det_avg, recovery);
    }

    printf("\n================================================================\n");
    printf("ANALYSIS\n");
    printf("================================================================\n");
    printf("Memory-bound: Aggressors compete for memory bandwidth → detection works\n");
    printf("Compute-bound: Aggressors use CPU, less memory pressure → detection may fail\n");
    printf("Mixed: Partial memory pressure → intermediate detection\n");
    printf("\nKey insight: IPC-based detection works when aggressors cause memory contention\n");

    cleanup_detection();
    free(memory_buffer);
    return 0;
}
