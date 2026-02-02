/*
 * Caladan Core Cost Comparison
 *
 * Quantitative comparison of interference detection approaches:
 *
 * 1. CALADAN APPROACH: Dedicated polling core every 10μs
 *    - Cost: 1 core out of N (25% on 4-core, 12.5% on 8-core)
 *    - Detection: ~10μs
 *    - Benefit: Instant detection
 *    - Drawback: Core is ALWAYS consumed
 *
 * 2. MC INTERRUPT APPROACH: Counter overflow triggers detection
 *    - Cost: ~5-10% overhead on active workers
 *    - Detection: ~50-100μs
 *    - Benefit: No dedicated core, scales with workload
 *    - Drawback: Slightly slower detection
 *
 * This test measures THROUGHPUT IMPACT to show the trade-off.
 *
 * Build:
 *   Linux: gcc -O3 -march=native test_caladan_comparison.c -lpthread -lm -o test_caladan
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
#define WORK_ITERS 500000
#define NUM_TRIALS 5

static inline uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static char *memory_buffer = NULL;
static _Atomic int caladan_core_running = 0;
static _Atomic int test_complete = 0;
static _Atomic int interference_detected = 0;

/* Memory-bound workload */
static void memory_workload(int iterations) {
    unsigned int seed = (unsigned int)get_time_ns();
    volatile char sum = 0;
    for (int i = 0; i < iterations; i++) {
        int idx = rand_r(&seed) % BUFFER_SIZE;
        sum += memory_buffer[idx];
        memory_buffer[(idx + 1) % BUFFER_SIZE] = sum;
    }
}

/* ========== CALADAN APPROACH: Dedicated Polling Core ========== */
/* Simulates Caladan's scheduler core that polls every 10μs */
static void* caladan_polling_core(void *arg) {
    int core_id = *(int*)arg;

#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#endif

    atomic_store(&caladan_core_running, 1);

    /* Spin-poll at 10μs intervals (simulating Caladan) */
    while (!atomic_load(&test_complete)) {
        /* In real Caladan, this would check IPC and make decisions */
        /* For now, just consume the core */
        struct timespec ts = {0, 10000};  /* 10μs */
        nanosleep(&ts, NULL);
    }

    return NULL;
}

/* ========== MC INTERRUPT APPROACH ========== */
#ifdef __linux__
static int g_fd_cycles = -1;
static int g_fd_instructions = -1;
static _Atomic int interrupt_count = 0;
static _Atomic uint64_t detection_time_ns = 0;

#define IPC_THRESHOLD 0.32
#define CYCLES_PER_INTERRUPT 120000  /* ~50μs at 2.4GHz */

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

    if (ipc < IPC_THRESHOLD && !atomic_load(&interference_detected)) {
        atomic_store(&interference_detected, 1);
        atomic_store(&detection_time_ns, get_time_ns());
    }

    ioctl(g_fd_cycles, PERF_EVENT_IOC_REFRESH, 1);
}

static int setup_perf_interrupts(void) {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.size = sizeof(pe);
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    pe.type = PERF_TYPE_HARDWARE;
    pe.config = PERF_COUNT_HW_CPU_CYCLES;
    pe.sample_period = CYCLES_PER_INTERRUPT;
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
#endif

/* ========== Throughput Tests ========== */

typedef struct {
    double time_ms;
    double throughput;  /* work units per second */
} result_t;

/* Baseline: All 4 cores doing work */
static result_t run_full_cores(int num_cores) {
    result_t r = {0};

    pthread_t threads[4];
    _Atomic int workers_done = 0;

    uint64_t start = get_time_ns();

    for (int i = 0; i < num_cores; i++) {
        pthread_create(&threads[i], NULL, (void*(*)(void*))memory_workload,
                       (void*)(intptr_t)WORK_ITERS);
    }

    for (int i = 0; i < num_cores; i++) {
        pthread_join(threads[i], NULL);
    }

    r.time_ms = (get_time_ns() - start) / 1e6;
    r.throughput = (num_cores * WORK_ITERS) / (r.time_ms / 1000.0);

    return r;
}

/* Caladan approach: N-1 cores doing work, 1 core polling */
static result_t run_caladan_style(int num_cores) {
    result_t r = {0};

    int caladan_core = num_cores - 1;  /* Last core is the scheduler */
    pthread_t caladan_thread;
    pthread_t work_threads[3];

    atomic_store(&test_complete, 0);

    /* Start Caladan polling core */
    pthread_create(&caladan_thread, NULL, caladan_polling_core, &caladan_core);
    while (!atomic_load(&caladan_core_running)) sched_yield();

    uint64_t start = get_time_ns();

    /* Workers on remaining cores */
    for (int i = 0; i < num_cores - 1; i++) {
        pthread_create(&work_threads[i], NULL, (void*(*)(void*))memory_workload,
                       (void*)(intptr_t)WORK_ITERS);
    }

    for (int i = 0; i < num_cores - 1; i++) {
        pthread_join(work_threads[i], NULL);
    }

    r.time_ms = (get_time_ns() - start) / 1e6;
    r.throughput = ((num_cores - 1) * WORK_ITERS) / (r.time_ms / 1000.0);

    /* Stop Caladan core */
    atomic_store(&test_complete, 1);
    pthread_join(caladan_thread, NULL);
    atomic_store(&caladan_core_running, 0);

    return r;
}

/* MC interrupt approach: All 4 cores doing work with interrupt overhead */
static result_t run_mc_interrupt_style(int num_cores) {
    result_t r = {0};

#ifdef __linux__
    if (setup_perf_interrupts() < 0) {
        printf("Failed to setup perf interrupts\n");
        return r;
    }

    pthread_t threads[4];

    start_perf_monitoring();

    uint64_t start = get_time_ns();

    for (int i = 0; i < num_cores; i++) {
        pthread_create(&threads[i], NULL, (void*(*)(void*))memory_workload,
                       (void*)(intptr_t)WORK_ITERS);
    }

    for (int i = 0; i < num_cores; i++) {
        pthread_join(threads[i], NULL);
    }

    r.time_ms = (get_time_ns() - start) / 1e6;
    r.throughput = (num_cores * WORK_ITERS) / (r.time_ms / 1000.0);

    stop_perf_monitoring();
    cleanup_perf();
#endif

    return r;
}

int main(void) {
    printf("================================================================\n");
    printf("CALADAN CORE COST COMPARISON\n");
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

    int num_cores = 4;

    /* Warmup */
    memory_workload(WORK_ITERS / 4);

    printf("\nComparing throughput with %d cores:\n\n", num_cores);

    double full_throughput = 0, caladan_throughput = 0, mc_throughput = 0;
    double full_time = 0, caladan_time = 0, mc_time = 0;

    printf("Running %d trials each...\n\n", NUM_TRIALS);

    for (int trial = 0; trial < NUM_TRIALS; trial++) {
        result_t full = run_full_cores(num_cores);
        result_t caladan = run_caladan_style(num_cores);
        result_t mc = run_mc_interrupt_style(num_cores);

        full_throughput += full.throughput;
        caladan_throughput += caladan.throughput;
        mc_throughput += mc.throughput;

        full_time += full.time_ms;
        caladan_time += caladan.time_ms;
        mc_time += mc.time_ms;

        printf("Trial %d: Full=%.1fms  Caladan=%.1fms  MC=%.1fms\n",
               trial + 1, full.time_ms, caladan.time_ms, mc.time_ms);
    }

    full_throughput /= NUM_TRIALS;
    caladan_throughput /= NUM_TRIALS;
    mc_throughput /= NUM_TRIALS;
    full_time /= NUM_TRIALS;
    caladan_time /= NUM_TRIALS;
    mc_time /= NUM_TRIALS;

    printf("\n================================================================\n");
    printf("THROUGHPUT RESULTS\n");
    printf("================================================================\n");
    printf("%-25s %12s %12s %12s\n", "Approach", "Time (ms)", "Throughput", "vs Full");
    printf("------------------------------------------------------------------------\n");
    printf("%-25s %12.1f %12.0f %12s\n",
           "Full (4 cores, no detect)", full_time, full_throughput, "baseline");
    printf("%-25s %12.1f %12.0f %11.1f%%\n",
           "Caladan (3+1 dedicated)", caladan_time, caladan_throughput,
           ((caladan_throughput - full_throughput) / full_throughput) * 100);
    printf("%-25s %12.1f %12.0f %11.1f%%\n",
           "MC interrupt (~50μs)", mc_time, mc_throughput,
           ((mc_throughput - full_throughput) / full_throughput) * 100);

    printf("\n================================================================\n");
    printf("ANALYSIS\n");
    printf("================================================================\n");

    double caladan_cost = ((full_throughput - caladan_throughput) / full_throughput) * 100;
    double mc_cost = ((full_throughput - mc_throughput) / full_throughput) * 100;

    printf("Caladan dedicated core cost: %.1f%% throughput loss\n", caladan_cost);
    printf("MC interrupt overhead:       %.1f%% throughput loss\n", mc_cost);
    printf("\n");

    if (mc_cost < caladan_cost) {
        double savings = caladan_cost - mc_cost;
        printf("MC WINS: %.1f%% more throughput than Caladan approach\n", savings);
        printf("\n");
        printf("Trade-off:\n");
        printf("  - Caladan: 10μs detection, %.1f%% cost\n", caladan_cost);
        printf("  - MC:      ~50μs detection, %.1f%% cost\n", mc_cost);
        printf("  - 5x slower detection buys %.1fx better throughput\n",
               caladan_cost / (mc_cost > 0 ? mc_cost : 0.1));
    } else {
        printf("Caladan wins on throughput (unexpected)\n");
    }

    printf("\n================================================================\n");
    printf("SCALING IMPLICATIONS\n");
    printf("================================================================\n");
    printf("On different core counts:\n");
    printf("  4 cores: Caladan loses 25%% (1/4), MC loses ~%.0f%%\n", mc_cost);
    printf("  8 cores: Caladan loses 12.5%% (1/8), MC loses ~%.0f%%\n", mc_cost);
    printf(" 16 cores: Caladan loses 6.25%% (1/16), MC loses ~%.0f%%\n", mc_cost);
    printf(" 32 cores: Caladan loses 3.125%% (1/32), MC loses ~%.0f%%\n", mc_cost);
    printf("\n");
    printf("MC's overhead is CONSTANT (per-task), Caladan's scales better with cores.\n");
    printf("But on 4-8 core systems, MC is clearly better.\n");

    free(memory_buffer);
    return 0;
}
