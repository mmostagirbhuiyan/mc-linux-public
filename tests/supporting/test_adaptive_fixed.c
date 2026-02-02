/*
 * FIXED Adaptive Scheduling via Real-Time Telemetry
 *
 * BUG IN ORIGINAL: atomic_fetch_add inside retry loop caused 47% of tasks to be skipped.
 *
 * FIX: Use a two-phase approach:
 *   1. Peek at tasks without committing
 *   2. Only advance the queue when we take a task
 *   3. Use CAS to avoid races
 *
 * Build:
 *   Linux: gcc -O3 -march=native test_adaptive_fixed.c -lpthread -lm -o test_adaptive_fixed
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

#define NUM_CORES 4
#define TOTAL_TASKS 200
#define COMPUTE_RATIO 0.5  /* 50% compute, 50% memory */

/* ========== Timing ========== */
static inline uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static inline void cpu_yield(void) {
#if defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield" ::: "memory");
#else
    __asm__ __volatile__("pause" ::: "memory");
#endif
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

    return (p->fd_cycles >= 0);
}

static void perf_start(perf_t *p) {
    ioctl(p->fd_cycles, PERF_EVENT_IOC_RESET, 0);
    ioctl(p->fd_instructions, PERF_EVENT_IOC_RESET, 0);
    ioctl(p->fd_cycles, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(p->fd_instructions, PERF_EVENT_IOC_ENABLE, 0);
}

static double perf_stop_get_ipc(perf_t *p) {
    uint64_t cycles, instructions;
    ioctl(p->fd_cycles, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(p->fd_instructions, PERF_EVENT_IOC_DISABLE, 0);
    read(p->fd_cycles, &cycles, sizeof(cycles));
    read(p->fd_instructions, &instructions, sizeof(instructions));
    return (cycles > 0) ? (double)instructions / cycles : 0;
}

static void perf_close(perf_t *p) {
    if (p->fd_cycles >= 0) close(p->fd_cycles);
    if (p->fd_instructions >= 0) close(p->fd_instructions);
}
#endif

/* ========== Task definitions ========== */
typedef enum { TASK_COMPUTE = 0, TASK_MEMORY = 1 } task_type_t;

/* Compute task: Pure arithmetic */
static volatile double sink = 0;
#define COMPUTE_ITERS 500000

static void do_compute_task(void) {
    double x = 1.0;
    for (int i = 0; i < COMPUTE_ITERS; i++) {
        x = sin(x) * cos(x) + sqrt(fabs(x));
    }
    sink += x;
}

/* Memory task: Random access (defeats prefetcher, causes cache misses) */
static char *memory_buffer = NULL;
#define MEMORY_SIZE (16 * 1024 * 1024)  /* 16MB - larger than L2 */
#define MEMORY_ITERS 100000

static void do_memory_task(void) {
    unsigned int seed = (unsigned int)get_time_ns();
    for (int i = 0; i < MEMORY_ITERS; i++) {
        int idx = rand_r(&seed) % MEMORY_SIZE;
        memory_buffer[idx] = (char)(memory_buffer[idx] + 1);
    }
}

/* ========== Task queue ========== */
typedef struct {
    task_type_t type;
    int id;
    _Atomic int taken;  /* 0 = available, 1 = taken */
} task_t;

static task_t task_queue[TOTAL_TASKS];
static _Atomic int queue_head = 0;
static _Atomic int tasks_completed = 0;

/* BLIND: Simple fetch-and-increment */
static int get_next_task_blind(task_t *out) {
    int idx = atomic_fetch_add(&queue_head, 1);
    if (idx >= TOTAL_TASKS) return 0;
    *out = task_queue[idx];
    return 1;
}

/* ========== FIXED Adaptive scheduler ========== */
static _Atomic double worker_ipc[NUM_CORES];
static _Atomic int adaptive_scan_start = 0;

/*
 * FIXED VERSION: Instead of incrementing and skipping, we:
 * 1. Scan forward looking for a suitable task
 * 2. Use CAS to claim the task we want
 * 3. Never skip tasks - if we can't find a match, take whatever's next
 */
static int get_next_task_adaptive_fixed(task_t *out, int worker_id) {
    double my_last_ipc = atomic_load(&worker_ipc[worker_id]);
    int prefer_compute = (my_last_ipc < 0.4);  /* Low IPC = was memory bound, prefer compute next */

    int scan_start = atomic_load(&adaptive_scan_start);

    /* Phase 1: Try to find a task matching our preference */
    if (prefer_compute) {
        for (int i = scan_start; i < TOTAL_TASKS && i < scan_start + 20; i++) {
            if (task_queue[i].type == TASK_COMPUTE) {
                int expected = 0;
                if (atomic_compare_exchange_strong(&task_queue[i].taken, &expected, 1)) {
                    *out = task_queue[i];
                    /* Advance scan start if we took from near the front */
                    if (i == scan_start) {
                        while (scan_start < TOTAL_TASKS &&
                               atomic_load(&task_queue[scan_start].taken)) {
                            scan_start++;
                        }
                        atomic_store(&adaptive_scan_start, scan_start);
                    }
                    return 1;
                }
            }
        }
    }

    /* Phase 2: Take any available task */
    for (int i = scan_start; i < TOTAL_TASKS; i++) {
        int expected = 0;
        if (atomic_compare_exchange_strong(&task_queue[i].taken, &expected, 1)) {
            *out = task_queue[i];
            /* Advance scan start */
            if (i == scan_start) {
                while (scan_start < TOTAL_TASKS &&
                       atomic_load(&task_queue[scan_start].taken)) {
                    scan_start++;
                }
                atomic_store(&adaptive_scan_start, scan_start);
            }
            return 1;
        }
    }

    return 0;  /* No more tasks */
}

/* ========== Workers ========== */
typedef struct {
    pthread_t thread;
    int worker_id;
    int tasks_completed;
#ifdef __linux__
    perf_t perf;
#endif
} worker_t;

static worker_t workers[NUM_CORES];

static void* worker_blind(void *arg) {
    worker_t *w = (worker_t*)arg;
    task_t task;

#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(w->worker_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#endif

    while (get_next_task_blind(&task)) {
        if (task.type == TASK_COMPUTE) {
            do_compute_task();
        } else {
            do_memory_task();
        }
        w->tasks_completed++;
    }

    return NULL;
}

static void* worker_adaptive(void *arg) {
    worker_t *w = (worker_t*)arg;
    task_t task;

#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(w->worker_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

    perf_init(&w->perf);
#endif

    while (get_next_task_adaptive_fixed(&task, w->worker_id)) {
#ifdef __linux__
        perf_start(&w->perf);
#endif

        if (task.type == TASK_COMPUTE) {
            do_compute_task();
        } else {
            do_memory_task();
        }

#ifdef __linux__
        double ipc = perf_stop_get_ipc(&w->perf);
        atomic_store(&worker_ipc[w->worker_id], ipc);
#endif

        w->tasks_completed++;
        atomic_fetch_add(&tasks_completed, 1);
    }

#ifdef __linux__
    perf_close(&w->perf);
#endif

    return NULL;
}

/* ========== Benchmark runners ========== */
static void reset_queue(void) {
    atomic_store(&queue_head, 0);
    atomic_store(&adaptive_scan_start, 0);
    atomic_store(&tasks_completed, 0);
    for (int i = 0; i < TOTAL_TASKS; i++) {
        atomic_store(&task_queue[i].taken, 0);
    }
    for (int i = 0; i < NUM_CORES; i++) {
        atomic_store(&worker_ipc[i], 1.0);
        workers[i].tasks_completed = 0;
    }
}

static double run_blind(int *total_tasks) {
    reset_queue();

    uint64_t start = get_time_ns();

    for (int i = 0; i < NUM_CORES; i++) {
        workers[i].worker_id = i;
        pthread_create(&workers[i].thread, NULL, worker_blind, &workers[i]);
    }

    for (int i = 0; i < NUM_CORES; i++) {
        pthread_join(workers[i].thread, NULL);
    }

    *total_tasks = 0;
    for (int i = 0; i < NUM_CORES; i++) {
        *total_tasks += workers[i].tasks_completed;
    }

    return (get_time_ns() - start) / 1e6;
}

static double run_adaptive(int *total_tasks) {
    reset_queue();

    uint64_t start = get_time_ns();

    for (int i = 0; i < NUM_CORES; i++) {
        workers[i].worker_id = i;
        pthread_create(&workers[i].thread, NULL, worker_adaptive, &workers[i]);
    }

    for (int i = 0; i < NUM_CORES; i++) {
        pthread_join(workers[i].thread, NULL);
    }

    *total_tasks = atomic_load(&tasks_completed);

    return (get_time_ns() - start) / 1e6;
}

int main(void) {
    printf("================================================================\n");
    printf("FIXED ADAPTIVE SCHEDULING VIA REAL-TIME TELEMETRY\n");
    printf("================================================================\n");
#ifndef __linux__
    printf("This test requires Linux for perf_event_open\n");
    return 1;
#endif
    printf("Cores: %d, Tasks: %d (%.0f%% compute, %.0f%% memory)\n",
           NUM_CORES, TOTAL_TASKS, COMPUTE_RATIO * 100, (1 - COMPUTE_RATIO) * 100);
    printf("\nFIX APPLIED: Using CAS-based task claiming instead of\n");
    printf("atomic_fetch_add inside retry loop (which skipped tasks).\n\n");

    /* Allocate memory buffer */
    memory_buffer = malloc(MEMORY_SIZE);
    if (!memory_buffer) {
        printf("Failed to allocate memory buffer\n");
        return 1;
    }
    memset(memory_buffer, 0, MEMORY_SIZE);

    /* Create shuffled task queue */
    int compute_count = (int)(TOTAL_TASKS * COMPUTE_RATIO);
    for (int i = 0; i < TOTAL_TASKS; i++) {
        task_queue[i].id = i;
        task_queue[i].type = (i < compute_count) ? TASK_COMPUTE : TASK_MEMORY;
        atomic_init(&task_queue[i].taken, 0);
    }

    /* Shuffle */
    srand(42);
    for (int i = TOTAL_TASKS - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        task_t tmp = task_queue[i];
        task_queue[i] = task_queue[j];
        task_queue[j] = tmp;
    }

    /* Run benchmarks */
    int NUM_RUNS = 5;
    double blind_times[5], adaptive_times[5];
    int blind_tasks[5], adaptive_tasks[5];

    printf("Running blind scheduling...\n");
    for (int i = 0; i < NUM_RUNS; i++) {
        blind_times[i] = run_blind(&blind_tasks[i]);
        printf("  Run %d: %.1f ms (%d tasks)\n", i + 1, blind_times[i], blind_tasks[i]);
    }

    printf("\nRunning FIXED adaptive scheduling (with telemetry)...\n");
    for (int i = 0; i < NUM_RUNS; i++) {
        adaptive_times[i] = run_adaptive(&adaptive_tasks[i]);
        printf("  Run %d: %.1f ms (%d tasks)\n", i + 1, adaptive_times[i], adaptive_tasks[i]);
    }

    /* Compute averages */
    double blind_avg = 0, adaptive_avg = 0;
    int blind_avg_tasks = 0, adaptive_avg_tasks = 0;
    for (int i = 0; i < NUM_RUNS; i++) {
        blind_avg += blind_times[i];
        adaptive_avg += adaptive_times[i];
        blind_avg_tasks += blind_tasks[i];
        adaptive_avg_tasks += adaptive_tasks[i];
    }
    blind_avg /= NUM_RUNS;
    adaptive_avg /= NUM_RUNS;
    blind_avg_tasks /= NUM_RUNS;
    adaptive_avg_tasks /= NUM_RUNS;

    printf("\n================================================================\n");
    printf("RESULTS\n");
    printf("================================================================\n");
    printf("Blind scheduling:    %.1f ms avg (%d tasks)\n", blind_avg, blind_avg_tasks);
    printf("Adaptive scheduling: %.1f ms avg (%d tasks)\n", adaptive_avg, adaptive_avg_tasks);

    /* Verify all tasks completed */
    if (blind_avg_tasks != TOTAL_TASKS || adaptive_avg_tasks != TOTAL_TASKS) {
        printf("\nWARNING: Task count mismatch!\n");
        printf("Expected: %d, Blind: %d, Adaptive: %d\n",
               TOTAL_TASKS, blind_avg_tasks, adaptive_avg_tasks);
    } else {
        printf("\n✓ All %d tasks completed in both modes\n", TOTAL_TASKS);
    }

    double improvement = ((blind_avg - adaptive_avg) / blind_avg) * 100;
    printf("\nImprovement: %.1f%%\n", improvement);

    /* Calculate telemetry overhead (adaptive has perf calls, blind doesn't) */
    double overhead = ((adaptive_avg - blind_avg) / blind_avg) * 100;
    if (overhead > 0) {
        printf("Telemetry overhead: %.1f%%\n", overhead);
    }

    printf("\n================================================================\n");
    printf("CONCLUSION\n");
    printf("================================================================\n");
    if (improvement > 5) {
        printf("SUCCESS: Adaptive scheduling shows %.1f%% improvement!\n", improvement);
        printf("- Real-time IPC feedback enables smarter task placement\n");
        printf("- This is a THROUGHPUT advantage, not just observability\n");
    } else if (improvement > 0) {
        printf("MARGINAL: %.1f%% improvement - needs better heuristics\n", improvement);
        printf("- The mechanism works but scheduling policy needs tuning\n");
    } else if (overhead < 10) {
        printf("NO IMPROVEMENT but LOW OVERHEAD: %.1f%% telemetry cost\n", -improvement);
        printf("- Telemetry is nearly free but scheduling doesn't help this workload\n");
    } else {
        printf("NO BENEFIT: Adaptive scheduling doesn't help for this workload\n");
        printf("- Telemetry overhead: %.1f%%\n", overhead);
    }

    free(memory_buffer);
    return 0;
}
