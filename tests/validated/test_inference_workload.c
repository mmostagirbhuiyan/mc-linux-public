/*
 * ML Inference Workload Test
 *
 * Simulates MobileNetV2-like inference using GEMM layers with realistic dimensions.
 * MobileNetV2 consists of:
 * - Initial 3x3 conv (32 filters)
 * - 17 inverted residual bottleneck blocks
 * - Final 1x1 conv + avgpool + FC
 *
 * Core operation is depthwise-separable convolution = depthwise + pointwise
 * Pointwise = 1x1 conv = GEMM when im2col'd
 *
 * This test measures interference detection for inference workloads.
 *
 * Build:
 *   gcc -O3 -march=native test_inference_workload.c -lpthread -lm -o test_inference
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

#define NUM_TRIALS 30
#define BATCH_SIZE 1

/* MobileNetV2-like layer dimensions (simplified) */
typedef struct {
    int input_channels;
    int output_channels;
    int spatial_size;  /* HxW flattened for GEMM */
} layer_config_t;

/* Simplified MobileNet-like layers (consistent dimensions for GEMM chain) */
static layer_config_t mobilenet_layers[] = {
    {256, 256, 256},      /* Layer 1: 256x256 GEMM */
    {256, 256, 256},      /* Layer 2 */
    {256, 256, 256},      /* Layer 3 */
    {256, 256, 256},      /* Layer 4 */
    {256, 128, 256},      /* Layer 5: reduce channels */
    {128, 128, 128},      /* Layer 6 */
    {128, 128, 128},      /* Layer 7 */
    {128, 64, 128},       /* Layer 8: reduce again */
    {64, 64, 64},         /* Layer 9 */
    {64, 32, 64},         /* Layer 10 */
    {32, 10, 32},         /* Layer 11: classifier */
};
#define NUM_LAYERS (sizeof(mobilenet_layers) / sizeof(mobilenet_layers[0]))

static int cmp_double(const void *a, const void *b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

static inline uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

/* GEMM layer (C = A * B) */
static void gemm_layer(float *C, float *A, float *B, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum > 0 ? sum : 0;  /* ReLU activation */
        }
    }
}

/* Allocate and initialize layer weights/activations */
static float* alloc_rand(size_t count) {
    float *ptr = (float*)aligned_alloc(64, count * sizeof(float));
    if (!ptr) return NULL;
    for (size_t i = 0; i < count; i++) {
        ptr[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    }
    return ptr;
}

/* Run single inference pass */
static double run_inference(float **weights, float **activations, int num_layers) {
    uint64_t start = get_time_ns();

    for (int l = 0; l < num_layers; l++) {
        layer_config_t *cfg = &mobilenet_layers[l];
        int M = cfg->spatial_size;
        int K = cfg->input_channels;
        int N = cfg->output_channels;

        gemm_layer(activations[l+1], activations[l], weights[l], M, K, N);
    }

    return (get_time_ns() - start) / 1e6;  /* ms */
}

/* Background interferer */
#define NUM_INTERFERERS 4
static char *memory_buffers[NUM_INTERFERERS] = {NULL};
static _Atomic int interferer_running = 0;
static _Atomic int interferer_should_stop = 0;
static _Atomic int interferer_throttled = 0;  /* When set, interferers pause */

typedef struct {
    int core;
    int id;
} interferer_arg_t;

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
        /* Check if throttled */
        if (atomic_load(&interferer_throttled)) {
            usleep(1000);  /* Back off when throttled */
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

/* PMU-based detection */
#ifdef __linux__
static int g_fd_cycles = -1;
static int g_fd_instructions = -1;
static _Atomic int interference_detected = 0;
static double g_ipc_threshold = 0.5;  /* Lower threshold for inference */

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
        atomic_store(&interferer_throttled, 1);  /* Immediately throttle interferers */
    }

    ioctl(g_fd_cycles, PERF_EVENT_IOC_REFRESH, 1);
}

static int setup_detection(void) {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.size = sizeof(pe);
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    pe.type = PERF_TYPE_HARDWARE;
    pe.config = PERF_COUNT_HW_CPU_CYCLES;
    pe.sample_period = 50000000;  /* 50M cycles ~ 20ms at 2.4GHz */
    pe.wakeup_events = 1;

    g_fd_cycles = perf_event_open(&pe, 0, -1, -1, 0);
    if (g_fd_cycles < 0) return -1;

    pe.sample_period = 0;
    pe.config = PERF_COUNT_HW_INSTRUCTIONS;
    g_fd_instructions = perf_event_open(&pe, 0, -1, g_fd_cycles, 0);

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = perf_overflow_handler;
    sa.sa_flags = SA_SIGINFO;
    sigaction(SIGIO, &sa, NULL);

    fcntl(g_fd_cycles, F_SETFL, O_ASYNC | O_NONBLOCK);
    fcntl(g_fd_cycles, F_SETSIG, SIGIO);
    fcntl(g_fd_cycles, F_SETOWN, getpid());

    ioctl(g_fd_cycles, PERF_EVENT_IOC_RESET, 0);
    ioctl(g_fd_instructions, PERF_EVENT_IOC_RESET, 0);
    ioctl(g_fd_cycles, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(g_fd_cycles, PERF_EVENT_IOC_REFRESH, 1);

    return 0;
}
#endif

int main(void) {
    printf("================================================================\n");
    printf("ML INFERENCE WORKLOAD - INTERFERENCE DETECTION TEST\n");
    printf("================================================================\n\n");

    printf("Simulating MobileNetV2-like inference (%d layers)\n", (int)NUM_LAYERS);
    printf("Batch size: %d, Trials: %d\n\n", BATCH_SIZE, NUM_TRIALS);

    /* Allocate weights and activations */
    float *weights[NUM_LAYERS];
    float *activations[NUM_LAYERS + 1];

    /* Input activation */
    activations[0] = alloc_rand(mobilenet_layers[0].spatial_size *
                                 mobilenet_layers[0].input_channels);

    for (int l = 0; l < (int)NUM_LAYERS; l++) {
        layer_config_t *cfg = &mobilenet_layers[l];
        weights[l] = alloc_rand(cfg->input_channels * cfg->output_channels);
        activations[l+1] = alloc_rand(cfg->spatial_size * cfg->output_channels);
    }

    /* Warmup */
    printf("Warmup...\n");
    for (int i = 0; i < 3; i++) {
        run_inference(weights, activations, NUM_LAYERS);
    }

    /* Isolated baseline */
    printf("\n1. ISOLATED INFERENCE (no interference)\n");
    double isolated_times[NUM_TRIALS];
    for (int t = 0; t < NUM_TRIALS; t++) {
        isolated_times[t] = run_inference(weights, activations, NUM_LAYERS);
    }

    double isolated_sum = 0;
    for (int t = 0; t < NUM_TRIALS; t++) isolated_sum += isolated_times[t];
    double isolated_mean = isolated_sum / NUM_TRIALS;
    printf("   Mean latency: %.2f ms\n", isolated_mean);

    /* With interference (no detection) */
    printf("\n2. WITH INTERFERENCE (no detection)\n");

    /* Allocate interferer buffers */
    for (int i = 0; i < NUM_INTERFERERS; i++) {
        memory_buffers[i] = malloc(8 * 1024 * 1024);
        memset(memory_buffers[i], 0, 8 * 1024 * 1024);
    }

    atomic_store(&interferer_should_stop, 0);
    atomic_store(&interferer_running, 0);
    pthread_t interferer_threads[NUM_INTERFERERS];
    interferer_arg_t iargs[NUM_INTERFERERS];

    /* Start interferers on all cores (including core 0 where main runs) */
    for (int i = 0; i < NUM_INTERFERERS; i++) {
        iargs[i].core = i % 4;  /* Distribute across 4 cores */
        iargs[i].id = i;
        pthread_create(&interferer_threads[i], NULL, interferer_fn, &iargs[i]);
    }
    while (atomic_load(&interferer_running) < NUM_INTERFERERS) sched_yield();
    usleep(10000);  /* Let interferers warm up */

    double interfered_times[NUM_TRIALS];
    for (int t = 0; t < NUM_TRIALS; t++) {
        interfered_times[t] = run_inference(weights, activations, NUM_LAYERS);
    }

    atomic_store(&interferer_should_stop, 1);
    for (int i = 0; i < NUM_INTERFERERS; i++) {
        pthread_join(interferer_threads[i], NULL);
    }

    double interfered_sum = 0;
    for (int t = 0; t < NUM_TRIALS; t++) interfered_sum += interfered_times[t];
    double interfered_mean = interfered_sum / NUM_TRIALS;
    printf("   Mean latency: %.2f ms (%.1f%% slowdown)\n",
           interfered_mean, (interfered_mean - isolated_mean) / isolated_mean * 100);

#ifdef __linux__
    /* With detection */
    printf("\n3. WITH DETECTION + THROTTLING\n");

    if (setup_detection() < 0) {
        printf("   PMU setup failed (need root or perf_event_paranoid=0)\n");
    } else {
        atomic_store(&interferer_should_stop, 0);
        atomic_store(&interferer_running, 0);
        for (int i = 0; i < NUM_INTERFERERS; i++) {
            pthread_create(&interferer_threads[i], NULL, interferer_fn, &iargs[i]);
        }
        while (atomic_load(&interferer_running) < NUM_INTERFERERS) sched_yield();
        usleep(10000);

        int detections = 0;
        double detected_times[NUM_TRIALS];

        for (int t = 0; t < NUM_TRIALS; t++) {
            atomic_store(&interference_detected, 0);
            atomic_store(&interferer_throttled, 0);  /* Allow interferers at start */

            detected_times[t] = run_inference(weights, activations, NUM_LAYERS);

            if (atomic_load(&interference_detected)) {
                detections++;
                /* Throttling already done by PMU handler */
            }
        }

        atomic_store(&interferer_should_stop, 1);
        for (int i = 0; i < NUM_INTERFERERS; i++) {
            pthread_join(interferer_threads[i], NULL);
        }

        double detected_sum = 0;
        for (int t = 0; t < NUM_TRIALS; t++) detected_sum += detected_times[t];
        double detected_mean = detected_sum / NUM_TRIALS;

        double recovery = (interfered_mean - detected_mean) /
                          (interfered_mean - isolated_mean) * 100;

        printf("   Mean latency: %.2f ms\n", detected_mean);
        printf("   Detection rate: %d/%d (%.0f%%)\n", detections, NUM_TRIALS,
               (double)detections / NUM_TRIALS * 100);
        printf("   Recovery: %.1f%%\n", recovery > 0 ? recovery : 0);

        close(g_fd_cycles);
        close(g_fd_instructions);
    }
#else
    printf("\n3. Detection requires Linux PMU (skipped on macOS)\n");
#endif

    /* Compute percentiles */
    double isolated_sorted[NUM_TRIALS], interfered_sorted[NUM_TRIALS];
    memcpy(isolated_sorted, isolated_times, sizeof(isolated_sorted));
    memcpy(interfered_sorted, interfered_times, sizeof(interfered_sorted));
    qsort(isolated_sorted, NUM_TRIALS, sizeof(double), cmp_double);
    qsort(interfered_sorted, NUM_TRIALS, sizeof(double), cmp_double);

    int p50_idx = NUM_TRIALS / 2;
    int p95_idx = (int)(NUM_TRIALS * 0.95);
    int p99_idx = (int)(NUM_TRIALS * 0.99);

    /* Summary */
    printf("\n================================================================\n");
    printf("SUMMARY\n");
    printf("================================================================\n");
    printf("                   p50      p95      p99      mean\n");
    printf("Isolated:      %7.2f  %7.2f  %7.2f  %7.2f ms\n",
           isolated_sorted[p50_idx], isolated_sorted[p95_idx],
           isolated_sorted[p99_idx], isolated_mean);
    printf("Interfered:    %7.2f  %7.2f  %7.2f  %7.2f ms (+%.0f%%)\n",
           interfered_sorted[p50_idx], interfered_sorted[p95_idx],
           interfered_sorted[p99_idx], interfered_mean,
           (interfered_mean - isolated_mean) / isolated_mean * 100);

    /* Cleanup */
    for (int l = 0; l < (int)NUM_LAYERS; l++) {
        free(weights[l]);
        free(activations[l+1]);
    }
    free(activations[0]);
    for (int i = 0; i < NUM_INTERFERERS; i++) {
        free(memory_buffers[i]);
    }

    return 0;
}
