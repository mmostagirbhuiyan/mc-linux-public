/*
 * Sustained Load Validation
 *
 * Tests interrupt-driven detection under continuous task streams:
 * - Multiple priority tasks back-to-back
 * - Sustained background interference
 * - Validates detection doesn't degrade over time
 *
 * Build:
 *   Linux: gcc -O3 -march=native test_sustained_load.c -lpthread -lm -o test_sustained
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
#define WORK_ITERS 200000  /* Shorter tasks for more iterations */
#define NUM_PRIORITY_TASKS 20
#define IPC_THRESHOLD 0.32
#define CYCLES_PER_INTERRUPT 120000

static inline uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static char *memory_buffer = NULL;
static _Atomic int background_should_pause = 0;
static _Atomic int background_running = 0;
static _Atomic int test_complete = 0;

/* Per-task metrics */
typedef struct {
    double time_ms;
    double detection_latency_us;
    int interrupts;
    int detected;
} task_metrics_t;

/* Interrupt-driven detection globals */
#ifdef __linux__
static int g_fd_cycles = -1;
static int g_fd_instructions = -1;
static _Atomic int interrupt_count = 0;
static _Atomic int interference_detected = 0;
static _Atomic uint64_t task_start_time = 0;
static _Atomic uint64_t detection_time_ns = 0;

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
        atomic_store(&background_should_pause, 1);
    }

    ioctl(g_fd_cycles, PERF_EVENT_IOC_REFRESH, 1);
}

static int setup_perf(void) {
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

static void reset_task_state(void) {
    atomic_store(&interrupt_count, 0);
    atomic_store(&interference_detected, 0);
    atomic_store(&detection_time_ns, 0);
    atomic_store(&background_should_pause, 0);

    ioctl(g_fd_cycles, PERF_EVENT_IOC_RESET, 0);
    ioctl(g_fd_instructions, PERF_EVENT_IOC_RESET, 0);
}

static void start_monitoring(void) {
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
        if (atomic_load(&background_should_pause)) {
            sched_yield();
            continue;
        }
        memory_workload(10000);
    }

    atomic_fetch_sub(&background_running, 1);
    return NULL;
}

static task_metrics_t run_priority_task(void) {
    task_metrics_t m = {0};

#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

    reset_task_state();
    start_monitoring();

    uint64_t start = get_time_ns();
    atomic_store(&task_start_time, start);

    memory_workload(WORK_ITERS);

    uint64_t end = get_time_ns();
    stop_monitoring();

    m.time_ms = (end - start) / 1e6;
    m.interrupts = atomic_load(&interrupt_count);
    m.detected = atomic_load(&interference_detected);

    uint64_t det_time = atomic_load(&detection_time_ns);
    if (det_time > 0 && det_time > start) {
        m.detection_latency_us = (det_time - start) / 1e3;
    }
#endif

    return m;
}

int main(void) {
    printf("================================================================\n");
    printf("SUSTAINED LOAD VALIDATION\n");
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

    if (setup_perf() < 0) {
        printf("Failed to setup perf\n");
        free(memory_buffer);
        return 1;
    }

    /* Start background workers (sustained interference) */
    background_worker_t workers[3];
    for (int i = 0; i < 3; i++) {
        workers[i].core_id = i + 1;
    }

    atomic_store(&test_complete, 0);
    for (int i = 0; i < 3; i++) {
        pthread_create(&workers[i].thread, NULL, background_worker, &workers[i]);
    }
    while (atomic_load(&background_running) < 3) sched_yield();

    printf("\nRunning %d priority tasks with sustained interference...\n\n", NUM_PRIORITY_TASKS);
    printf("%-6s %10s %12s %10s %10s\n",
           "Task", "Time(ms)", "Detection", "Interrupts", "Detected");
    printf("------------------------------------------------------------\n");

    task_metrics_t all_metrics[NUM_PRIORITY_TASKS];
    double total_time = 0;
    double total_detection = 0;
    int total_interrupts = 0;
    int total_detected = 0;

    for (int task = 0; task < NUM_PRIORITY_TASKS; task++) {
        task_metrics_t m = run_priority_task();
        all_metrics[task] = m;

        printf("%-6d %10.1f %10.0fμs %10d %10s\n",
               task + 1, m.time_ms, m.detection_latency_us,
               m.interrupts, m.detected ? "YES" : "NO");

        total_time += m.time_ms;
        if (m.detected) {
            total_detection += m.detection_latency_us;
            total_detected++;
        }
        total_interrupts += m.interrupts;
    }

    /* Stop background workers */
    atomic_store(&test_complete, 1);
    for (int i = 0; i < 3; i++) {
        pthread_join(workers[i].thread, NULL);
    }

    cleanup_perf();

    printf("\n================================================================\n");
    printf("AGGREGATE STATISTICS\n");
    printf("================================================================\n");
    printf("Total priority tasks:     %d\n", NUM_PRIORITY_TASKS);
    printf("Detection success rate:   %d/%d (%.0f%%)\n",
           total_detected, NUM_PRIORITY_TASKS,
           (double)total_detected / NUM_PRIORITY_TASKS * 100);
    printf("Average task time:        %.1f ms\n", total_time / NUM_PRIORITY_TASKS);
    printf("Average detection:        %.0f μs\n",
           total_detected > 0 ? total_detection / total_detected : 0);
    printf("Average interrupts/task:  %d\n", total_interrupts / NUM_PRIORITY_TASKS);

    /* Check for degradation over time */
    double first_half_detection = 0, second_half_detection = 0;
    int first_half_count = 0, second_half_count = 0;

    for (int i = 0; i < NUM_PRIORITY_TASKS / 2; i++) {
        if (all_metrics[i].detected) {
            first_half_detection += all_metrics[i].detection_latency_us;
            first_half_count++;
        }
    }
    for (int i = NUM_PRIORITY_TASKS / 2; i < NUM_PRIORITY_TASKS; i++) {
        if (all_metrics[i].detected) {
            second_half_detection += all_metrics[i].detection_latency_us;
            second_half_count++;
        }
    }

    printf("\n================================================================\n");
    printf("DEGRADATION CHECK\n");
    printf("================================================================\n");
    double first_avg = first_half_count > 0 ? first_half_detection / first_half_count : 0;
    double second_avg = second_half_count > 0 ? second_half_detection / second_half_count : 0;
    printf("First half avg detection:  %.0f μs\n", first_avg);
    printf("Second half avg detection: %.0f μs\n", second_avg);

    if (second_avg > first_avg * 1.5) {
        printf("WARNING: Detection degraded by %.0f%% over time\n",
               ((second_avg - first_avg) / first_avg) * 100);
    } else if (second_avg < first_avg * 0.5) {
        printf("IMPROVED: Detection got faster by %.0f%%\n",
               ((first_avg - second_avg) / first_avg) * 100);
    } else {
        printf("STABLE: Detection latency consistent across tasks\n");
    }

    printf("\n================================================================\n");
    printf("CONCLUSION\n");
    printf("================================================================\n");
    if (total_detected == NUM_PRIORITY_TASKS) {
        printf("SUCCESS: 100%% detection rate under sustained load!\n");
        printf("- Interrupt-driven approach handles continuous task streams\n");
        printf("- No degradation over %d back-to-back tasks\n", NUM_PRIORITY_TASKS);
    } else if (total_detected >= NUM_PRIORITY_TASKS * 0.9) {
        printf("GOOD: %.0f%% detection rate - mostly working\n",
               (double)total_detected / NUM_PRIORITY_TASKS * 100);
    } else {
        printf("NEEDS WORK: Only %.0f%% detection rate\n",
               (double)total_detected / NUM_PRIORITY_TASKS * 100);
    }

    free(memory_buffer);
    return 0;
}
