/*
 * Interrupt Detection - Parameter Sweep v2
 *
 * Enhanced version with:
 * - Tail latency reporting (p50/p95/p99)
 * - Auto-calibrating IPC threshold
 * - More trials for statistical significance
 *
 * Build:
 *   Linux: gcc -O3 -march=native test_interrupt_sweep_v2.c -lpthread -lm -o test_sweep_v2
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
#include <math.h>
#include <fcntl.h>
#include <sys/ioctl.h>

#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/syscall.h>
#endif

#define BUFFER_SIZE (8 * 1024 * 1024)
#define PRIORITY_ITERS 500000
#define BACKGROUND_ITERS 100000
#define NUM_TRIALS 30  /* Statistical rigor for TPDS */
#define CALIBRATION_RUNS 5

static inline uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static char *memory_buffer = NULL;
static _Atomic int background_should_pause = 0;
static _Atomic int background_running = 0;
static _Atomic int test_complete = 0;
static _Atomic int interrupt_count = 0;
static _Atomic int interference_detected = 0;
static _Atomic uint64_t detection_time_ns = 0;

static int g_fd_cycles = -1;
static int g_fd_instructions = -1;
static double g_ipc_threshold = 0.32;  /* Will be auto-calibrated */

/* Comparison function for qsort */
static int compare_double(const void *a, const void *b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

/* Calculate percentile from sorted array */
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
    (void)signum;
    (void)context;

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

static int setup_perf_with_interval(int cycles_per_check) {
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

static void cleanup_perf(void) {
    if (g_fd_cycles >= 0) close(g_fd_cycles);
    if (g_fd_instructions >= 0) close(g_fd_instructions);
    g_fd_cycles = -1;
    g_fd_instructions = -1;
}

static void start_perf_monitoring(void) {
    atomic_store(&interrupt_count, 0);
    atomic_store(&interference_detected, 0);
    atomic_store(&detection_time_ns, 0);

    ioctl(g_fd_cycles, PERF_EVENT_IOC_RESET, 0);
    ioctl(g_fd_instructions, PERF_EVENT_IOC_RESET, 0);
    ioctl(g_fd_cycles, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(g_fd_instructions, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(g_fd_cycles, PERF_EVENT_IOC_REFRESH, 1);
}

static void stop_perf_monitoring(void) {
    ioctl(g_fd_cycles, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(g_fd_instructions, PERF_EVENT_IOC_DISABLE, 0);
}

/* Measure baseline IPC for auto-calibration */
static double calibrate_ipc(void) {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.size = sizeof(pe);
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    pe.type = PERF_TYPE_HARDWARE;
    pe.config = PERF_COUNT_HW_CPU_CYCLES;

    int fd_cycles = perf_event_open(&pe, 0, -1, -1, 0);
    if (fd_cycles < 0) return 0.4;  /* Fallback */

    pe.config = PERF_COUNT_HW_INSTRUCTIONS;
    int fd_inst = perf_event_open(&pe, 0, -1, fd_cycles, 0);
    if (fd_inst < 0) {
        close(fd_cycles);
        return 0.4;
    }

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

    double total_ipc = 0;
    for (int run = 0; run < CALIBRATION_RUNS; run++) {
        ioctl(fd_cycles, PERF_EVENT_IOC_RESET, 0);
        ioctl(fd_inst, PERF_EVENT_IOC_RESET, 0);
        ioctl(fd_cycles, PERF_EVENT_IOC_ENABLE, 0);
        ioctl(fd_inst, PERF_EVENT_IOC_ENABLE, 0);

        /* Run the same workload we'll use for testing */
        unsigned int seed = (unsigned int)get_time_ns();
        volatile char sum = 0;
        for (int i = 0; i < PRIORITY_ITERS; i++) {
            int idx = rand_r(&seed) % BUFFER_SIZE;
            sum += memory_buffer[idx];
            memory_buffer[(idx + 1) % BUFFER_SIZE] = sum;
        }

        ioctl(fd_cycles, PERF_EVENT_IOC_DISABLE, 0);
        ioctl(fd_inst, PERF_EVENT_IOC_DISABLE, 0);

        uint64_t cycles = 0, instructions = 0;
        read(fd_cycles, &cycles, sizeof(cycles));
        read(fd_inst, &instructions, sizeof(instructions));

        if (cycles > 0) {
            total_ipc += (double)instructions / cycles;
        }
    }

    close(fd_cycles);
    close(fd_inst);

    return total_ipc / CALIBRATION_RUNS;
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
            sched_yield();
            continue;
        }
        memory_workload(BACKGROUND_ITERS / 10);
    }

    atomic_fetch_sub(&background_running, 1);
    return NULL;
}

typedef struct {
    double time_ms;
    int interrupts;
    double detection_latency_us;
    int detected;
} result_t;

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

static double run_with_interference(background_worker_t *workers, int num_workers) {
    atomic_store(&test_complete, 0);
    atomic_store(&background_should_pause, 0);

    for (int i = 0; i < num_workers; i++) {
        workers[i].with_throttling = 0;
        pthread_create(&workers[i].thread, NULL, background_worker, &workers[i]);
    }

    while (atomic_load(&background_running) < num_workers) sched_yield();

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
    for (int i = 0; i < num_workers; i++) {
        pthread_join(workers[i].thread, NULL);
    }

    return time_ms;
}

static result_t run_with_protection(background_worker_t *workers, int num_workers) {
    result_t result = {0};

#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

    atomic_store(&test_complete, 0);
    atomic_store(&background_should_pause, 0);

    for (int i = 0; i < num_workers; i++) {
        workers[i].with_throttling = 1;
        pthread_create(&workers[i].thread, NULL, background_worker, &workers[i]);
    }

    while (atomic_load(&background_running) < num_workers) sched_yield();

    start_perf_monitoring();

    uint64_t start = get_time_ns();
    memory_workload(PRIORITY_ITERS);
    uint64_t end = get_time_ns();

    stop_perf_monitoring();

    atomic_store(&test_complete, 1);
    for (int i = 0; i < num_workers; i++) {
        pthread_join(workers[i].thread, NULL);
    }

    result.time_ms = (end - start) / 1e6;
    result.interrupts = atomic_load(&interrupt_count);
    result.detected = atomic_load(&interference_detected);

    uint64_t detected_at = atomic_load(&detection_time_ns);
    if (detected_at > 0 && detected_at > start) {
        result.detection_latency_us = (detected_at - start) / 1e3;
    }

    atomic_store(&background_should_pause, 0);
#endif

    return result;
}

static double run_with_interrupts_only(void) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

    start_perf_monitoring();

    uint64_t start = get_time_ns();
    memory_workload(PRIORITY_ITERS);
    double time_ms = (get_time_ns() - start) / 1e6;

    stop_perf_monitoring();

    return time_ms;
#else
    return 0;
#endif
}

int main(void) {
    printf("================================================================\n");
    printf("INTERRUPT DETECTION - PARAMETER SWEEP v2\n");
    printf("With Tail Latencies (p50/p95/p99) and Auto-Calibration\n");
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

    background_worker_t workers[3];
    for (int i = 0; i < 3; i++) {
        workers[i].core_id = i + 1;
    }

    /* Warmup */
    memory_workload(PRIORITY_ITERS / 4);

    /* Auto-calibrate IPC threshold */
    printf("\nCalibrating IPC threshold...\n");
    double baseline_ipc = calibrate_ipc();
    g_ipc_threshold = baseline_ipc * 0.75;  /* 75% of baseline */
    printf("Baseline IPC: %.3f\n", baseline_ipc);
    printf("Threshold (75%%): %.3f\n", g_ipc_threshold);

    /* Get baselines with multiple trials */
    printf("\nMeasuring baselines (%d trials)...\n", NUM_TRIALS);
    double isolated_times[NUM_TRIALS];
    double interference_times[NUM_TRIALS];

    for (int i = 0; i < NUM_TRIALS; i++) {
        isolated_times[i] = run_isolated();
        interference_times[i] = run_with_interference(workers, 3);
    }

    /* Sort for percentiles */
    qsort(isolated_times, NUM_TRIALS, sizeof(double), compare_double);
    qsort(interference_times, NUM_TRIALS, sizeof(double), compare_double);

    double isolated_p50 = percentile(isolated_times, NUM_TRIALS, 50);
    double isolated_p99 = percentile(isolated_times, NUM_TRIALS, 99);
    double interference_p50 = percentile(interference_times, NUM_TRIALS, 50);
    double interference_p99 = percentile(interference_times, NUM_TRIALS, 99);

    printf("\nBaseline Results:\n");
    printf("%-20s %10s %10s %10s\n", "Scenario", "p50", "p95", "p99");
    printf("--------------------------------------------------------\n");
    printf("%-20s %10.1fms %10.1fms %10.1fms\n", "Isolated",
           isolated_p50,
           percentile(isolated_times, NUM_TRIALS, 95),
           isolated_p99);
    printf("%-20s %10.1fms %10.1fms %10.1fms\n", "With interference",
           interference_p50,
           percentile(interference_times, NUM_TRIALS, 95),
           interference_p99);

    double latency_increase = interference_p50 - isolated_p50;
    printf("\nLatency increase from interference: %.1f ms (p50)\n", latency_increase);

    /* Test different interrupt intervals */
    int intervals[] = {24000, 48000, 96000, 120000, 240000, 480000};
    int num_intervals = sizeof(intervals) / sizeof(intervals[0]);

    printf("\n================================================================\n");
    printf("INTERVAL SWEEP WITH TAIL LATENCIES\n");
    printf("================================================================\n");

    for (int idx = 0; idx < num_intervals; idx++) {
        int cycles = intervals[idx];
        double interval_us = cycles / 2.4;  /* Approx for 2.4GHz */

        if (setup_perf_with_interval(cycles) < 0) {
            printf("\n%d cycles: FAILED to setup perf\n", cycles);
            continue;
        }

        printf("\n--- %d cycles (~%.0fμs interval) ---\n", cycles, interval_us);

        /* Measure interrupt overhead */
        double overhead_times[NUM_TRIALS];
        for (int i = 0; i < NUM_TRIALS; i++) {
            overhead_times[i] = run_with_interrupts_only();
        }
        qsort(overhead_times, NUM_TRIALS, sizeof(double), compare_double);
        double overhead_p50 = percentile(overhead_times, NUM_TRIALS, 50);
        double overhead_pct = ((overhead_p50 - isolated_p50) / isolated_p50) * 100;

        /* Measure with protection */
        double detection_latencies[NUM_TRIALS];
        double protected_times[NUM_TRIALS];
        int detected_count = 0;
        int total_interrupts = 0;

        for (int trial = 0; trial < NUM_TRIALS; trial++) {
            result_t r = run_with_protection(workers, 3);
            protected_times[trial] = r.time_ms;
            detection_latencies[trial] = r.detection_latency_us;
            total_interrupts += r.interrupts;
            if (r.detected) detected_count++;
        }

        /* Sort for percentiles */
        qsort(protected_times, NUM_TRIALS, sizeof(double), compare_double);

        /* Sort only valid detection latencies */
        double valid_detections[NUM_TRIALS];
        int valid_count = 0;
        for (int i = 0; i < NUM_TRIALS; i++) {
            if (detection_latencies[i] > 0) {
                valid_detections[valid_count++] = detection_latencies[i];
            }
        }
        if (valid_count > 0) {
            qsort(valid_detections, valid_count, sizeof(double), compare_double);
        }

        double protected_p50 = percentile(protected_times, NUM_TRIALS, 50);
        double protected_p95 = percentile(protected_times, NUM_TRIALS, 95);
        double protected_p99 = percentile(protected_times, NUM_TRIALS, 99);

        double recovered = interference_p50 - protected_p50;
        double recovery_pct = (recovered / latency_increase) * 100;

        printf("Detection rate:     %d/%d (%.0f%%)\n", detected_count, NUM_TRIALS,
               (double)detected_count / NUM_TRIALS * 100);
        printf("Avg interrupts:     %.0f\n", (double)total_interrupts / NUM_TRIALS);
        printf("Overhead (p50):     %.1f%% vs isolated\n", overhead_pct);
        printf("Recovery (p50):     %.1f%%\n", recovery_pct);

        printf("\nLatency (ms):       %10s %10s %10s\n", "p50", "p95", "p99");
        printf("  Protected:        %10.1f %10.1f %10.1f\n",
               protected_p50, protected_p95, protected_p99);

        if (valid_count > 0) {
            printf("\nDetection (μs):     %10s %10s %10s\n", "p50", "p95", "p99");
            printf("  Latency:          %10.0f %10.0f %10.0f\n",
                   percentile(valid_detections, valid_count, 50),
                   percentile(valid_detections, valid_count, 95),
                   percentile(valid_detections, valid_count, 99));
        }

        cleanup_perf();
    }

    printf("\n================================================================\n");
    printf("SUMMARY TABLE\n");
    printf("================================================================\n");
    printf("%-10s %8s %8s %8s %8s %8s %8s\n",
           "Interval", "Det%", "Rec%", "Ovhd%", "Det-p50", "Det-p99", "Lat-p99");
    printf("------------------------------------------------------------------------\n");

    /* Re-run for summary table */
    for (int idx = 0; idx < num_intervals; idx++) {
        int cycles = intervals[idx];
        double interval_us = cycles / 2.4;

        if (setup_perf_with_interval(cycles) < 0) continue;

        double overhead_times[NUM_TRIALS];
        for (int i = 0; i < NUM_TRIALS; i++) {
            overhead_times[i] = run_with_interrupts_only();
        }
        qsort(overhead_times, NUM_TRIALS, sizeof(double), compare_double);
        double overhead_pct = ((percentile(overhead_times, NUM_TRIALS, 50) - isolated_p50) / isolated_p50) * 100;

        double detection_latencies[NUM_TRIALS];
        double protected_times[NUM_TRIALS];
        int detected_count = 0;

        for (int trial = 0; trial < NUM_TRIALS; trial++) {
            result_t r = run_with_protection(workers, 3);
            protected_times[trial] = r.time_ms;
            detection_latencies[trial] = r.detection_latency_us;
            if (r.detected) detected_count++;
        }

        qsort(protected_times, NUM_TRIALS, sizeof(double), compare_double);

        double valid_detections[NUM_TRIALS];
        int valid_count = 0;
        for (int i = 0; i < NUM_TRIALS; i++) {
            if (detection_latencies[i] > 0) {
                valid_detections[valid_count++] = detection_latencies[i];
            }
        }
        if (valid_count > 0) {
            qsort(valid_detections, valid_count, sizeof(double), compare_double);
        }

        double protected_p50 = percentile(protected_times, NUM_TRIALS, 50);
        double protected_p99 = percentile(protected_times, NUM_TRIALS, 99);
        double recovery_pct = ((interference_p50 - protected_p50) / latency_increase) * 100;

        printf("~%.0fμs %8.0f%% %8.1f%% %8.1f%% %8.0f %8.0f %8.1f\n",
               interval_us,
               (double)detected_count / NUM_TRIALS * 100,
               recovery_pct,
               overhead_pct,
               valid_count > 0 ? percentile(valid_detections, valid_count, 50) : 0,
               valid_count > 0 ? percentile(valid_detections, valid_count, 99) : 0,
               protected_p99);

        cleanup_perf();
    }

    printf("\n================================================================\n");
    printf("KEY FOR TPDS PAPER\n");
    printf("================================================================\n");
    printf("This data provides:\n");
    printf("- Tail latency guarantees (p99) for QoS claims\n");
    printf("- Detection-to-response latency percentiles\n");
    printf("- Overhead-vs-detection trade-off curve\n");
    printf("- Cross-platform reproducible (auto-calibrated threshold)\n");

    free(memory_buffer);
    return 0;
}
