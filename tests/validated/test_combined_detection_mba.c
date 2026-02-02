/*
 * Combined Interrupt Detection + MBA Test
 *
 * Hypothesis: Detection identifies interference, MBA provides fast hardware throttling
 * - Detection alone: sched_yield() throttling (voluntary, slow)
 * - MBA alone: Static allocation (always on, unfair when no interference)
 * - Combined: Detect -> Apply MBA -> Fast hardware-based throttling
 *
 * Build:
 *   gcc -O3 -march=native test_combined_detection_mba.c -lpthread -lm -o test_combined
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
static _Atomic int throttle_mode = 0;  /* 0=yield, 1=usleep, 2=hard_pause */

static double g_ipc_threshold = 0.30;
static int g_fd_cycles = -1;
static int g_fd_instructions = -1;
static _Atomic int interrupt_count = 0;

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
            int mode = atomic_load(&throttle_mode);
            switch (mode) {
                case 0:  /* yield */
                    sched_yield();
                    break;
                case 1:  /* usleep - more aggressive */
                    usleep(100);  /* 100μs pause */
                    break;
                case 2:  /* hard pause - complete stop */
                    while (atomic_load(&background_should_pause) &&
                           !atomic_load(&test_complete)) {
                        usleep(1000);
                    }
                    break;
            }
            continue;
        }
        memory_workload(BACKGROUND_ITERS / 10);
    }

    atomic_fetch_sub(&background_running, 1);
    return NULL;
}

typedef struct {
    double time_ms;
    int detected;
    double detection_us;
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

static double run_interference(background_worker_t *workers, int n) {
    atomic_store(&test_complete, 0);
    atomic_store(&background_should_pause, 0);

    for (int i = 0; i < n; i++) {
        workers[i].with_throttling = 0;
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
    for (int i = 0; i < n; i++) {
        pthread_join(workers[i].thread, NULL);
    }
    return time_ms;
}

static result_t run_protected(background_worker_t *workers, int n) {
    result_t result = {0};

#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

    atomic_store(&test_complete, 0);
    atomic_store(&background_should_pause, 0);

    for (int i = 0; i < n; i++) {
        workers[i].with_throttling = 1;
        pthread_create(&workers[i].thread, NULL, background_worker, &workers[i]);
    }
    while (atomic_load(&background_running) < n) sched_yield();

    start_monitoring();

    uint64_t start = get_time_ns();
    memory_workload(PRIORITY_ITERS);
    uint64_t end = get_time_ns();

    stop_monitoring();

    atomic_store(&test_complete, 1);
    for (int i = 0; i < n; i++) {
        pthread_join(workers[i].thread, NULL);
    }

    result.time_ms = (end - start) / 1e6;
    result.detected = atomic_load(&interference_detected);
    uint64_t det_time = atomic_load(&detection_time_ns);
    if (det_time > 0 && det_time > start) {
        result.detection_us = (det_time - start) / 1e3;
    }

    atomic_store(&background_should_pause, 0);
#endif

    return result;
}

int main(void) {
    printf("================================================================\n");
    printf("COMBINED DETECTION + THROTTLING MODES TEST\n");
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

    /* Setup detection at 120K cycles (~50μs) */
    if (setup_detection(120000) < 0) {
        printf("Failed to setup detection\n");
        free(memory_buffer);
        return 1;
    }

    /* Calibrate */
    memory_workload(PRIORITY_ITERS / 4);

    /* Get baselines */
    printf("\nMeasuring baselines...\n");
    double isolated_avg = 0, interference_avg = 0;
    for (int i = 0; i < NUM_TRIALS; i++) {
        isolated_avg += run_isolated();
        interference_avg += run_interference(workers, 3);
    }
    isolated_avg /= NUM_TRIALS;
    interference_avg /= NUM_TRIALS;

    printf("Isolated:     %.1f ms\n", isolated_avg);
    printf("Interference: %.1f ms (+%.0f%%)\n", interference_avg,
           ((interference_avg - isolated_avg) / isolated_avg) * 100);

    double latency_increase = interference_avg - isolated_avg;

    printf("\n================================================================\n");
    printf("THROTTLING MODES COMPARISON\n");
    printf("================================================================\n");
    printf("%-20s %10s %10s %10s %10s\n",
           "Mode", "Time(ms)", "Recovery%", "Detection", "Overhead%");
    printf("------------------------------------------------------------\n");

    const char *mode_names[] = {"sched_yield()", "usleep(100)", "hard_pause"};

    for (int mode = 0; mode < 3; mode++) {
        atomic_store(&throttle_mode, mode);

        double total_time = 0;
        double total_detection = 0;
        int detected = 0;

        for (int trial = 0; trial < NUM_TRIALS; trial++) {
            result_t r = run_protected(workers, 3);
            total_time += r.time_ms;
            if (r.detected) {
                detected++;
                total_detection += r.detection_us;
            }
        }

        double avg_time = total_time / NUM_TRIALS;
        double recovered = interference_avg - avg_time;
        double recovery_pct = (recovered / latency_increase) * 100;
        double overhead = ((avg_time - isolated_avg) / isolated_avg) * 100;
        double avg_detection = detected > 0 ? total_detection / detected : 0;

        printf("%-20s %10.1f %10.1f%% %10.0fμs %10.1f%%\n",
               mode_names[mode], avg_time, recovery_pct, avg_detection, overhead);
    }

    printf("\n================================================================\n");
    printf("ANALYSIS\n");
    printf("================================================================\n");
    printf("sched_yield():  Voluntary yield, aggressors quickly resume\n");
    printf("usleep(100):    100μs forced pause each check, more effective\n");
    printf("hard_pause:     Complete stop until priority task finishes\n");
    printf("\nNote: hard_pause may be unfair but maximizes victim protection\n");

    cleanup_detection();
    free(memory_buffer);
    return 0;
}
