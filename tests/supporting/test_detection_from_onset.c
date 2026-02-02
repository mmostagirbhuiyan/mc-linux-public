/*
 * Detection Latency from Interference Onset Test
 *
 * This test measures the time from when interference ACTUALLY BEGINS
 * to when it is detected, rather than from when the priority task starts.
 *
 * The key insight: current tests measure detection latency from task start,
 * which conflates task execution time with actual detection speed.
 *
 * Protocol:
 * 1. Priority task runs in isolation (establishing baseline IPC)
 * 2. At a controlled point, interferers are signaled to start
 * 3. We measure time from interference onset to detection
 *
 * This provides the true "reaction time" of the detection system.
 *
 * Build:
 *   gcc -O3 -march=native test_detection_from_onset.c -lpthread -lm -o test_onset
 *
 * Run (requires perf_event_paranoid <= 1):
 *   sudo sysctl kernel.perf_event_paranoid=0
 *   ./test_onset
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
#include <signal.h>
#include <fcntl.h>
#include <sys/ioctl.h>

#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/syscall.h>
#endif

#define NUM_TRIALS 50
#define INTERFERER_CORES 3  /* Cores 1,2,3 for interferers */
#define PRIORITY_CORE 0     /* Core 0 for priority task */
#define BUFFER_SIZE (4 * 1024 * 1024)  /* 4MB to stress L3 */

/* Synchronization */
static _Atomic int interferers_ready = 0;
static _Atomic int start_interferers = 0;
static _Atomic int interferers_should_stop = 0;
static _Atomic int interference_detected = 0;
static _Atomic uint64_t interference_onset_time = 0;

/* PMU */
#ifdef __linux__
static int g_fd_cycles = -1;
static int g_fd_instructions = -1;
static double g_baseline_ipc = 0.0;
static double g_ipc_threshold = 0.0;  /* 60% of baseline */
#endif

/* Interferer buffers */
static char *interferer_buffers[INTERFERER_CORES] = {NULL};

static inline uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

#ifdef __linux__
static int perf_event_open(struct perf_event_attr *attr, pid_t pid,
                           int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
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
    pe.read_format = PERF_FORMAT_GROUP;

    g_fd_cycles = perf_event_open(&pe, 0, -1, -1, 0);
    if (g_fd_cycles < 0) {
        perror("perf_event_open cycles");
        return -1;
    }

    pe.config = PERF_COUNT_HW_INSTRUCTIONS;
    pe.disabled = 0;
    g_fd_instructions = perf_event_open(&pe, 0, -1, g_fd_cycles, 0);
    if (g_fd_instructions < 0) {
        perror("perf_event_open instructions");
        close(g_fd_cycles);
        return -1;
    }

    return 0;
}

static void reset_pmu(void) {
    ioctl(g_fd_cycles, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
}

static void enable_pmu(void) {
    ioctl(g_fd_cycles, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
}

static void disable_pmu(void) {
    ioctl(g_fd_cycles, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
}

static double read_ipc(void) {
    struct {
        uint64_t nr;
        uint64_t cycles;
        uint64_t instructions;
    } data;

    if (read(g_fd_cycles, &data, sizeof(data)) != sizeof(data)) {
        return 0.0;
    }

    if (data.cycles == 0) return 0.0;
    return (double)data.instructions / (double)data.cycles;
}
#endif

/* Priority task: compute-bound work */
static void priority_work(int iterations) {
    volatile double result = 0.0;
    for (int i = 0; i < iterations; i++) {
        for (int j = 0; j < 1000; j++) {
            result += (double)i * (double)j / (double)(i + j + 1);
        }
    }
    (void)result;
}

/* Interferer thread: pollutes cache */
static void* interferer_fn(void *arg) {
    int core = *(int*)arg;

#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#endif

    char *buffer = interferer_buffers[core - 1];  /* cores 1,2,3 -> buffers 0,1,2 */
    unsigned int seed = (unsigned int)get_time_ns() + core;

    /* Signal ready */
    atomic_fetch_add(&interferers_ready, 1);

    /* Wait for start signal */
    while (!atomic_load(&start_interferers)) {
        sched_yield();
    }

    /* Pollute cache aggressively */
    while (!atomic_load(&interferers_should_stop)) {
        for (int i = 0; i < 10000; i++) {
            int idx = rand_r(&seed) % BUFFER_SIZE;
            buffer[idx] = buffer[(idx + 64) % BUFFER_SIZE];
        }
    }

    return NULL;
}

/* Detection thread: monitors IPC and detects interference */
static void* detector_fn(void *arg) {
    (void)arg;

#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(PRIORITY_CORE, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

    /* Wait for interference to start */
    while (!atomic_load(&start_interferers)) {
        sched_yield();
    }

    /* Now monitor IPC */
    while (!atomic_load(&interferers_should_stop)) {
        reset_pmu();
        enable_pmu();
        usleep(2000);  /* 2ms sampling interval */
        disable_pmu();

        double ipc = read_ipc();

        if (ipc > 0 && ipc < g_ipc_threshold) {
            /* Interference detected! */
            atomic_store(&interference_detected, 1);
            return NULL;
        }
    }
#endif

    return NULL;
}

/* Calibrate baseline IPC */
static double calibrate_baseline(void) {
#ifdef __linux__
    double ipcs[10];

    for (int i = 0; i < 10; i++) {
        reset_pmu();
        enable_pmu();
        priority_work(100);
        disable_pmu();
        ipcs[i] = read_ipc();
    }

    /* Use median */
    qsort(ipcs, 10, sizeof(double), cmp_double);
    double baseline = ipcs[5];

    printf("Calibrated baseline IPC: %.3f\n", baseline);
    return baseline;
#else
    return 1.0;
#endif
}

/* Run one trial: measure onset-to-detection latency */
static double run_trial(void) {
#ifdef __linux__
    /* Reset state */
    atomic_store(&interferers_ready, 0);
    atomic_store(&start_interferers, 0);
    atomic_store(&interferers_should_stop, 0);
    atomic_store(&interference_detected, 0);
    atomic_store(&interference_onset_time, 0);

    /* Start interferer threads (but they wait for signal) */
    pthread_t interferer_threads[INTERFERER_CORES];
    int cores[INTERFERER_CORES];
    for (int i = 0; i < INTERFERER_CORES; i++) {
        cores[i] = i + 1;  /* Cores 1, 2, 3 */
        pthread_create(&interferer_threads[i], NULL, interferer_fn, &cores[i]);
    }

    /* Wait for interferers to be ready */
    while (atomic_load(&interferers_ready) < INTERFERER_CORES) {
        sched_yield();
    }

    /* Start detector thread */
    pthread_t detector_thread;
    pthread_create(&detector_thread, NULL, detector_fn, NULL);

    /* Brief delay to let detector start monitoring */
    usleep(5000);

    /* NOW start interference and record onset time */
    uint64_t onset = get_time_ns();
    atomic_store(&interference_onset_time, onset);
    atomic_store(&start_interferers, 1);

    /* Wait for detection (with timeout) */
    uint64_t timeout = onset + 500000000ULL;  /* 500ms timeout */
    while (!atomic_load(&interference_detected)) {
        if (get_time_ns() > timeout) {
            /* Detection failed - timeout */
            atomic_store(&interferers_should_stop, 1);
            for (int i = 0; i < INTERFERER_CORES; i++) {
                pthread_join(interferer_threads[i], NULL);
            }
            pthread_join(detector_thread, NULL);
            return -1.0;  /* Indicate failure */
        }
        usleep(100);
    }

    uint64_t detection_time = get_time_ns();

    /* Cleanup */
    atomic_store(&interferers_should_stop, 1);
    for (int i = 0; i < INTERFERER_CORES; i++) {
        pthread_join(interferer_threads[i], NULL);
    }
    pthread_join(detector_thread, NULL);

    /* Calculate latency from onset */
    double latency_ms = (detection_time - onset) / 1e6;
    return latency_ms;
#else
    return -1.0;
#endif
}

int main(void) {
    printf("================================================================\n");
    printf("DETECTION LATENCY FROM INTERFERENCE ONSET TEST\n");
    printf("================================================================\n\n");

#ifndef __linux__
    printf("ERROR: This test requires Linux perf_event API\n");
    return 1;
#else

    /* Pin to priority core */
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(PRIORITY_CORE, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

    /* Allocate interferer buffers */
    printf("Allocating interferer buffers...\n");
    for (int i = 0; i < INTERFERER_CORES; i++) {
        interferer_buffers[i] = malloc(BUFFER_SIZE);
        if (!interferer_buffers[i]) {
            perror("malloc");
            return 1;
        }
        memset(interferer_buffers[i], 0, BUFFER_SIZE);
    }

    /* Setup PMU */
    printf("Setting up PMU...\n");
    if (setup_pmu() < 0) {
        printf("ERROR: PMU setup failed. Try:\n");
        printf("  sudo sysctl kernel.perf_event_paranoid=0\n");
        return 1;
    }

    /* Calibrate baseline */
    printf("Calibrating baseline IPC...\n");
    g_baseline_ipc = calibrate_baseline();
    g_ipc_threshold = g_baseline_ipc * 0.60;  /* 60% of baseline */
    printf("Detection threshold: %.3f (60%% of baseline)\n\n", g_ipc_threshold);

    if (g_baseline_ipc < 0.1) {
        printf("ERROR: Baseline IPC too low (%.3f). PMU may not be working.\n", g_baseline_ipc);
        return 1;
    }

    /* Run trials */
    printf("Running %d trials...\n", NUM_TRIALS);
    printf("Trial | Onset-to-Detection (ms)\n");
    printf("------|------------------------\n");

    double latencies[NUM_TRIALS];
    int successful = 0;

    for (int t = 0; t < NUM_TRIALS; t++) {
        double lat = run_trial();
        if (lat >= 0) {
            latencies[successful++] = lat;
            printf("  %3d |        %7.2f\n", t + 1, lat);
        } else {
            printf("  %3d |        TIMEOUT\n", t + 1);
        }

        /* Brief delay between trials */
        usleep(50000);
    }

    if (successful == 0) {
        printf("\nERROR: All trials failed to detect interference.\n");
        printf("Check PMU configuration and thresholds.\n");
        return 1;
    }

    /* Compute statistics */
    qsort(latencies, successful, sizeof(double), cmp_double);

    double sum = 0;
    for (int i = 0; i < successful; i++) sum += latencies[i];
    double mean = sum / successful;

    double p50 = latencies[successful / 2];
    double p95 = latencies[(int)(successful * 0.95)];
    double p99 = latencies[(int)(successful * 0.99)];

    printf("\n================================================================\n");
    printf("RESULTS: Detection Latency from Interference Onset\n");
    printf("================================================================\n");
    printf("Successful trials: %d/%d (%.0f%%)\n", successful, NUM_TRIALS,
           (double)successful / NUM_TRIALS * 100);
    printf("\n");
    printf("Latency Statistics (ms):\n");
    printf("  Mean:  %7.2f\n", mean);
    printf("  p50:   %7.2f\n", p50);
    printf("  p95:   %7.2f\n", p95);
    printf("  p99:   %7.2f\n", p99);
    printf("  Min:   %7.2f\n", latencies[0]);
    printf("  Max:   %7.2f\n", latencies[successful - 1]);
    printf("\n");
    printf("Sampling interval: 2ms\n");
    printf("Detection threshold: %.3f (60%% of baseline %.3f)\n",
           g_ipc_threshold, g_baseline_ipc);

    /* Cleanup */
    close(g_fd_cycles);
    close(g_fd_instructions);
    for (int i = 0; i < INTERFERER_CORES; i++) {
        free(interferer_buffers[i]);
    }

    return 0;
#endif
}
