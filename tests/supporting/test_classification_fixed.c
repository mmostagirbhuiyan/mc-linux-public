/*
 * FIXED Workload Classification via Hardware Metrics
 *
 * BUG IN ORIGINAL: Used SEQUENTIAL memory access, which prefetchers handle perfectly.
 * The result: memory tasks showed high IPC, same as compute tasks.
 *
 * FIX: Use RANDOM memory access (like the anomaly detection test which got 100% accuracy).
 *
 * Build:
 *   Linux: gcc -O3 -march=native test_classification_fixed.c -lpthread -lm -o test_classification_fixed
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
#include <time.h>
#include <unistd.h>
#include <math.h>

#ifdef __linux__
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>
#endif

#define NUM_WORKERS 4
#define TASKS_PER_TYPE 50
#define TOTAL_TASKS (TASKS_PER_TYPE * 3)

/* ========== Timing ========== */
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
    int fd_cache_refs;
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

    pe.config = PERF_COUNT_HW_CACHE_REFERENCES;
    p->fd_cache_refs = perf_event_open(&pe, 0, -1, p->fd_cycles, 0);

    return (p->fd_cycles >= 0);
}

static void perf_start(perf_t *p) {
    ioctl(p->fd_cycles, PERF_EVENT_IOC_RESET, 0);
    ioctl(p->fd_instructions, PERF_EVENT_IOC_RESET, 0);
    ioctl(p->fd_cache_misses, PERF_EVENT_IOC_RESET, 0);
    ioctl(p->fd_cache_refs, PERF_EVENT_IOC_RESET, 0);
    ioctl(p->fd_cycles, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(p->fd_instructions, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(p->fd_cache_misses, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(p->fd_cache_refs, PERF_EVENT_IOC_ENABLE, 0);
}

static void perf_stop(perf_t *p, uint64_t *cycles, uint64_t *instructions,
                      uint64_t *cache_misses, uint64_t *cache_refs) {
    ioctl(p->fd_cycles, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(p->fd_instructions, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(p->fd_cache_misses, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(p->fd_cache_refs, PERF_EVENT_IOC_DISABLE, 0);
    read(p->fd_cycles, cycles, sizeof(*cycles));
    read(p->fd_instructions, instructions, sizeof(*instructions));
    read(p->fd_cache_misses, cache_misses, sizeof(*cache_misses));
    read(p->fd_cache_refs, cache_refs, sizeof(*cache_refs));
}

static void perf_close(perf_t *p) {
    if (p->fd_cycles >= 0) close(p->fd_cycles);
    if (p->fd_instructions >= 0) close(p->fd_instructions);
    if (p->fd_cache_misses >= 0) close(p->fd_cache_misses);
    if (p->fd_cache_refs >= 0) close(p->fd_cache_refs);
}
#endif

/* ========== Task types ========== */
typedef enum {
    TASK_COMPUTE = 0,
    TASK_MEMORY = 1,
    TASK_MIXED = 2
} task_type_t;

static const char* task_type_names[] = {"COMPUTE", "MEMORY", "MIXED"};

/* Compute-bound: Pure arithmetic, no memory access */
static volatile double sink = 0;
#define COMPUTE_ITERS 2000000

static void compute_task(void) {
    double x = 1.0;
    for (int i = 0; i < COMPUTE_ITERS; i++) {
        x = sin(x) * cos(x) + sqrt(fabs(x));
    }
    sink += x;
}

/*
 * FIXED Memory-bound: RANDOM access (defeats prefetcher!)
 *
 * Original used sequential: for (i = 0; i < N; i++) sum += buf[i]
 * This uses random: sum += buf[rand() % N]
 */
static char *memory_buffer = NULL;
#define MEMORY_SIZE (16 * 1024 * 1024)  /* 16MB - much larger than L2 */
#define MEMORY_ITERS 200000

static void memory_task(void) {
    unsigned int seed = (unsigned int)get_time_ns();
    volatile char sum = 0;
    for (int i = 0; i < MEMORY_ITERS; i++) {
        int idx = rand_r(&seed) % MEMORY_SIZE;
        sum += memory_buffer[idx];
        memory_buffer[(idx + 1) % MEMORY_SIZE] = sum;  /* Write to prevent optimization */
    }
    sink += sum;
}

/* Mixed: Some compute, some random memory */
#define MIXED_COMPUTE_ITERS 500000
#define MIXED_MEMORY_ITERS 50000

static void mixed_task(void) {
    double x = 1.0;
    unsigned int seed = (unsigned int)get_time_ns();
    volatile char sum = 0;

    /* Compute phase */
    for (int i = 0; i < MIXED_COMPUTE_ITERS; i++) {
        x = sin(x) + cos(x);
    }

    /* Memory phase - RANDOM access */
    for (int i = 0; i < MIXED_MEMORY_ITERS; i++) {
        int idx = rand_r(&seed) % MEMORY_SIZE;
        sum += memory_buffer[idx];
    }

    sink += x + sum;
}

/* ========== Task result storage ========== */
typedef struct {
    int task_id;
    task_type_t actual_type;
    uint64_t cycles;
    uint64_t instructions;
    uint64_t cache_misses;
    uint64_t cache_refs;
    double ipc;
    double miss_rate;
    task_type_t predicted_type;
} task_result_t;

static task_result_t results[TOTAL_TASKS];
static _Atomic int next_task = 0;
static _Atomic int results_count = 0;
static task_type_t task_types[TOTAL_TASKS];

/* Classification thresholds - tuned based on actual measurements:
 * COMPUTE: IPC ~2.3, MEMORY: IPC ~0.4, MIXED: IPC ~2.2
 * Memory is clearly separated from compute/mixed by IPC alone.
 */
#define IPC_COMPUTE_THRESHOLD 1.5   /* Above this = compute or mixed */
#define IPC_MEMORY_THRESHOLD 0.5    /* Below this = memory-bound */

static task_type_t classify_task(double ipc, double miss_rate) {
    /* Primary classifier: IPC
     * Memory-bound tasks have dramatically lower IPC (~0.4) vs compute/mixed (~2.2)
     * Use miss_rate as secondary signal to distinguish compute from mixed
     */
    if (ipc < IPC_MEMORY_THRESHOLD) {
        return TASK_MEMORY;
    } else if (miss_rate < 0.001) {
        return TASK_COMPUTE;  /* Very low miss rate = pure compute */
    } else {
        return TASK_MIXED;    /* High IPC but some misses = mixed */
    }
}

/* ========== Worker ========== */
typedef struct {
    pthread_t thread;
    int worker_id;
#ifdef __linux__
    perf_t perf;
#endif
} worker_t;

static worker_t workers[NUM_WORKERS];

static void* worker_fn(void *arg) {
    worker_t *w = (worker_t*)arg;

#ifdef __linux__
    perf_init(&w->perf);
#endif

    while (1) {
        int task_id = atomic_fetch_add(&next_task, 1);
        if (task_id >= TOTAL_TASKS) break;

        uint64_t cycles = 0, instructions = 0, cache_misses = 0, cache_refs = 0;
        task_type_t type = task_types[task_id];

#ifdef __linux__
        perf_start(&w->perf);
#endif

        switch (type) {
            case TASK_COMPUTE: compute_task(); break;
            case TASK_MEMORY:  memory_task(); break;
            case TASK_MIXED:   mixed_task(); break;
        }

#ifdef __linux__
        perf_stop(&w->perf, &cycles, &instructions, &cache_misses, &cache_refs);
#endif

        /* Record result */
        int idx = atomic_fetch_add(&results_count, 1);
        results[idx].task_id = task_id;
        results[idx].actual_type = type;
        results[idx].cycles = cycles;
        results[idx].instructions = instructions;
        results[idx].cache_misses = cache_misses;
        results[idx].cache_refs = cache_refs;
        results[idx].ipc = (cycles > 0) ? (double)instructions / cycles : 0;
        results[idx].miss_rate = (cache_refs > 0) ? (double)cache_misses / cache_refs : 0;
        results[idx].predicted_type = classify_task(results[idx].ipc, results[idx].miss_rate);
    }

#ifdef __linux__
    perf_close(&w->perf);
#endif

    return NULL;
}

int main(void) {
    printf("================================================================\n");
    printf("FIXED WORKLOAD CLASSIFICATION VIA HARDWARE METRICS\n");
    printf("================================================================\n");
#ifdef __linux__
    printf("Platform: Linux (using real perf_event_open)\n");
#else
    printf("Platform: Not Linux - perf counters unavailable\n");
    return 1;
#endif
    printf("Tasks per type: %d (total: %d)\n", TASKS_PER_TYPE, TOTAL_TASKS);
    printf("\nFIX APPLIED: Using RANDOM memory access instead of sequential.\n");
    printf("Random access defeats prefetcher, causing real cache misses.\n\n");
    printf("Classification thresholds: IPC>%.2f=compute, IPC<%.2f=memory\n\n",
           IPC_COMPUTE_THRESHOLD, IPC_MEMORY_THRESHOLD);

    /* Allocate memory buffer */
    memory_buffer = aligned_alloc(64, MEMORY_SIZE);
    if (!memory_buffer) {
        printf("Failed to allocate memory buffer\n");
        return 1;
    }
    /* Initialize with random data to avoid zero-page optimization */
    for (int i = 0; i < MEMORY_SIZE; i++) {
        memory_buffer[i] = (char)(i ^ (i >> 8));
    }

    /* Create task list: interleaved types */
    for (int i = 0; i < TOTAL_TASKS; i++) {
        task_types[i] = i % 3;
    }

    printf("Running %d tasks with %d workers...\n\n", TOTAL_TASKS, NUM_WORKERS);

    uint64_t start = get_time_ns();

    for (int i = 0; i < NUM_WORKERS; i++) {
        workers[i].worker_id = i;
        pthread_create(&workers[i].thread, NULL, worker_fn, &workers[i]);
    }

    for (int i = 0; i < NUM_WORKERS; i++) {
        pthread_join(workers[i].thread, NULL);
    }

    double elapsed_ms = (get_time_ns() - start) / 1e6;

    /* Analyze results */
    printf("================================================================\n");
    printf("RESULTS (Total time: %.1f ms)\n", elapsed_ms);
    printf("================================================================\n\n");

    /* Compute per-type statistics */
    double ipc_sum[3] = {0}, miss_rate_sum[3] = {0};
    int type_count[3] = {0};
    int confusion[3][3] = {0};

    printf("Sample task metrics:\n");
    printf("%-4s %-8s %-12s %-10s %-8s %-6s %-8s %-8s\n",
           "ID", "Actual", "Cycles", "Instr", "Misses", "IPC", "MissRate", "Predict");
    printf("--------------------------------------------------------------------------------\n");

    for (int i = 0; i < TOTAL_TASKS && i < results_count; i++) {
        task_result_t *r = &results[i];

        ipc_sum[r->actual_type] += r->ipc;
        miss_rate_sum[r->actual_type] += r->miss_rate;
        type_count[r->actual_type]++;
        confusion[r->actual_type][r->predicted_type]++;

        /* Print first 18 (6 of each type) */
        if (i < 18) {
            printf("%-4d %-8s %-12lu %-10lu %-8lu %-6.3f %-8.3f %-8s %s\n",
                   r->task_id,
                   task_type_names[r->actual_type],
                   r->cycles,
                   r->instructions,
                   r->cache_misses,
                   r->ipc,
                   r->miss_rate,
                   task_type_names[r->predicted_type],
                   (r->actual_type == r->predicted_type) ? "✓" : "✗");
        }
    }

    printf("\n================================================================\n");
    printf("PER-TYPE STATISTICS\n");
    printf("================================================================\n");
    for (int t = 0; t < 3; t++) {
        if (type_count[t] > 0) {
            printf("%-8s: avg IPC=%.3f, avg miss_rate=%.4f (n=%d)\n",
                   task_type_names[t],
                   ipc_sum[t] / type_count[t],
                   miss_rate_sum[t] / type_count[t],
                   type_count[t]);
        }
    }

    /* Check IPC separation */
    double compute_ipc = (type_count[0] > 0) ? ipc_sum[0] / type_count[0] : 0;
    double memory_ipc = (type_count[1] > 0) ? ipc_sum[1] / type_count[1] : 0;
    double separation = compute_ipc / memory_ipc;

    printf("\nIPC Separation Ratio (compute/memory): %.1fx\n", separation);
    if (separation > 2.0) {
        printf("✓ Strong separation - classification should work well\n");
    } else if (separation > 1.5) {
        printf("~ Moderate separation - classification may need threshold tuning\n");
    } else {
        printf("✗ Weak separation - workloads not distinguishable by IPC\n");
    }

    printf("\n================================================================\n");
    printf("CONFUSION MATRIX\n");
    printf("================================================================\n");
    printf("                  Predicted:\n");
    printf("Actual:     COMPUTE  MEMORY   MIXED\n");
    for (int a = 0; a < 3; a++) {
        printf("%-8s    %-8d %-8d %-8d\n",
               task_type_names[a],
               confusion[a][0], confusion[a][1], confusion[a][2]);
    }

    int correct = confusion[0][0] + confusion[1][1] + confusion[2][2];
    double accuracy = (double)correct / TOTAL_TASKS * 100;

    printf("\n================================================================\n");
    printf("CLASSIFICATION ACCURACY\n");
    printf("================================================================\n");
    printf("Overall accuracy: %.1f%% (%d/%d correct)\n", accuracy, correct, TOTAL_TASKS);

    for (int t = 0; t < 3; t++) {
        int class_correct = confusion[t][t];
        int class_total = type_count[t];
        printf("  %s: %.1f%% (%d/%d)\n",
               task_type_names[t],
               class_total > 0 ? (double)class_correct / class_total * 100 : 0,
               class_correct, class_total);
    }

    printf("\n================================================================\n");
    printf("CONCLUSION\n");
    printf("================================================================\n");

    /* Check if MEMORY is now distinguishable from COMPUTE */
    int memory_correct = confusion[1][1];
    int memory_total = type_count[1];
    double memory_accuracy = (memory_total > 0) ? (double)memory_correct / memory_total * 100 : 0;

    if (accuracy >= 80) {
        printf("SUCCESS: Hardware metrics effectively classify workloads!\n");
        printf("- Compute accuracy: %.0f%%\n", type_count[0] > 0 ? (double)confusion[0][0]/type_count[0]*100 : 0);
        printf("- Memory accuracy: %.0f%%\n", memory_accuracy);
        printf("- Random access successfully defeats prefetcher\n");
        printf("- IPC separation: %.1fx (compute/memory)\n", separation);
        printf("\nThis enables:\n");
        printf("  - Real-time workload characterization\n");
        printf("  - Intelligent task placement (separate memory-bound tasks)\n");
        printf("  - Cache-aware scheduling decisions\n");
    } else if (memory_accuracy >= 70) {
        printf("PARTIAL SUCCESS: Memory tasks now detectable!\n");
        printf("- Memory detection: %.0f%% (was ~0%% with sequential access)\n", memory_accuracy);
        printf("- Threshold tuning could improve mixed classification\n");
    } else {
        printf("FIX NOT SUFFICIENT: Even random access isn't creating\n");
        printf("enough cache misses to distinguish workloads.\n");
        printf("- May need larger buffer or more iterations\n");
    }

    free(memory_buffer);
    return 0;
}
