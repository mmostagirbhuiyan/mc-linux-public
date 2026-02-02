/*
 * Fair Comparison Test: sched_yield vs yield instruction
 *
 * Tests whether sched_yield's benefit is:
 *   A) Specific to MC architecture, OR
 *   B) Universal (helps any thread pool)
 *
 * 2x2 Matrix:
 *   - Spin-pool + yield instruction
 *   - Spin-pool + sched_yield
 *   - MC + yield instruction
 *   - MC + sched_yield
 *
 * If sched_yield helps BOTH equally → it's just a syscall effect
 * If sched_yield helps MC MORE → there's architectural synergy
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <sched.h>
#include <pthread.h>
#include <float.h>

#define TILE_SIZE 32
#define MATRIX_SIZE 1024
#define WARMUP_RUNS 3
#define BENCHMARK_RUNS 10
#define RING_CAPACITY 256

static inline uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

typedef struct {
    double times[100];
    int count;
    double mean, std, min, max;
} stats_t;

static void stats_init(stats_t *s) {
    memset(s, 0, sizeof(stats_t));
    s->min = DBL_MAX;
    s->max = -DBL_MAX;
}

static void stats_add(stats_t *s, double value) {
    if (s->count < 100) s->times[s->count] = value;
    s->count++;
    if (value < s->min) s->min = value;
    if (value > s->max) s->max = value;
}

static void stats_compute(stats_t *s) {
    if (s->count == 0) return;
    int n = s->count < 100 ? s->count : 100;
    double sum = 0;
    for (int i = 0; i < n; i++) sum += s->times[i];
    s->mean = sum / n;
    if (n > 1) {
        double sq_sum = 0;
        for (int i = 0; i < n; i++) {
            double diff = s->times[i] - s->mean;
            sq_sum += diff * diff;
        }
        s->std = sqrt(sq_sum / (n - 1));
    }
}

/* Two idle mechanisms */
static inline void idle_yield_instruction(void) {
#if defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield" ::: "memory");
#else
    __asm__ __volatile__("pause" ::: "memory");
#endif
}

static inline void idle_sched_yield(void) {
    sched_yield();
}

/* GEMM work */
typedef struct {
    const float *A;
    const float *B;
    float *C;
    int N;
    int row_start, row_end;
} work_t;

static void gemm_tiled(const work_t *w) {
    int rows = w->row_end - w->row_start;
    const float *A_start = w->A + w->row_start * w->N;
    float *C_start = w->C + w->row_start * w->N;
    memset(C_start, 0, rows * w->N * sizeof(float));

    for (int i0 = 0; i0 < rows; i0 += TILE_SIZE) {
        for (int j0 = 0; j0 < w->N; j0 += TILE_SIZE) {
            for (int k0 = 0; k0 < w->N; k0 += TILE_SIZE) {
                int imax = (i0 + TILE_SIZE < rows) ? i0 + TILE_SIZE : rows;
                int jmax = (j0 + TILE_SIZE < w->N) ? j0 + TILE_SIZE : w->N;
                int kmax = (k0 + TILE_SIZE < w->N) ? k0 + TILE_SIZE : w->N;
                for (int i = i0; i < imax; i++) {
                    for (int k = k0; k < kmax; k++) {
                        float a_ik = A_start[i * w->N + k];
                        for (int j = j0; j < jmax; j++) {
                            C_start[i * w->N + j] += a_ik * w->B[k * w->N + j];
                        }
                    }
                }
            }
        }
    }
}

/* ============================================================================
 * SPIN-POOL with configurable idle mechanism
 * ============================================================================ */

typedef struct {
    pthread_t thread;
    _Atomic int has_work;
    _Atomic int done;
    _Atomic int shutdown;
    work_t work;
    int cpu_pin;
    int use_sched_yield;  /* 0 = yield instruction, 1 = sched_yield */
} spin_worker_t;

static spin_worker_t *g_spin = NULL;
static int g_spin_size = 0;

static void *spin_worker_fn(void *arg) {
    spin_worker_t *w = (spin_worker_t *)arg;

    if (w->cpu_pin >= 0) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(w->cpu_pin, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    }

    while (1) {
        while (!atomic_load(&w->has_work) && !atomic_load(&w->shutdown)) {
            if (w->use_sched_yield)
                idle_sched_yield();
            else
                idle_yield_instruction();
        }
        if (atomic_load(&w->shutdown)) break;

        gemm_tiled(&w->work);
        atomic_store(&w->has_work, 0);
        atomic_store(&w->done, 1);
    }
    return NULL;
}

static void spin_init(int n, int *cpus, int use_sched_yield) {
    g_spin = calloc(n, sizeof(spin_worker_t));
    g_spin_size = n;
    for (int i = 0; i < n; i++) {
        g_spin[i].cpu_pin = cpus ? cpus[i] : -1;
        g_spin[i].use_sched_yield = use_sched_yield;
        pthread_create(&g_spin[i].thread, NULL, spin_worker_fn, &g_spin[i]);
    }
    usleep(10000);
}

static void spin_shutdown(void) {
    for (int i = 0; i < g_spin_size; i++) atomic_store(&g_spin[i].shutdown, 1);
    for (int i = 0; i < g_spin_size; i++) pthread_join(g_spin[i].thread, NULL);
    free(g_spin);
    g_spin = NULL;
}

static double run_spin(const float *A, const float *B, float *C, int size, int n) {
    int rows_per = size / n;
    uint64_t start = get_time_ns();

    for (int i = 0; i < n; i++) {
        g_spin[i].work = (work_t){A, B, C, size, i * rows_per,
                                   (i == n - 1) ? size : (i + 1) * rows_per};
        atomic_store(&g_spin[i].done, 0);
        atomic_store(&g_spin[i].has_work, 1);
    }

    for (int i = 0; i < n; i++)
        while (!atomic_load(&g_spin[i].done)) idle_yield_instruction();

    return (get_time_ns() - start) / 1e6;
}

/* ============================================================================
 * MC-STYLE with configurable idle mechanism
 * ============================================================================ */

typedef void *(*task_fn_t)(void *);

typedef struct {
    task_fn_t fn;
    void *arg;
    void (*on_complete)(void *, void *);
    void *result;
} mc_task_t;

typedef struct {
    mc_task_t *buffer;
    size_t capacity;
    size_t mask;
    char _pad1[40];
    atomic_uint_fast64_t head;
    char _pad2[56];
    atomic_uint_fast64_t tail;
    char _pad3[56];
} mc_ring_t;

typedef struct {
    int id;
    int cpu_id;
    mc_ring_t ring;
    pthread_t thread;
    atomic_bool running;
    int use_sched_yield;
} mc_worker_t;

static mc_worker_t *g_mc = NULL;
static int g_mc_size = 0;

static bool ring_push(mc_ring_t *rb, const mc_task_t *t) {
    uint64_t tail = atomic_load_explicit(&rb->tail, memory_order_relaxed);
    uint64_t head = atomic_load_explicit(&rb->head, memory_order_acquire);
    if (tail - head >= rb->capacity) return false;
    rb->buffer[tail & rb->mask] = *t;
    atomic_store_explicit(&rb->tail, tail + 1, memory_order_release);
    return true;
}

static bool ring_pop(mc_ring_t *rb, mc_task_t *t) {
    uint64_t head = atomic_load_explicit(&rb->head, memory_order_relaxed);
    uint64_t tail = atomic_load_explicit(&rb->tail, memory_order_acquire);
    if (head >= tail) return false;
    *t = rb->buffer[head & rb->mask];
    atomic_store_explicit(&rb->head, head + 1, memory_order_release);
    return true;
}

static void *mc_worker_fn(void *arg) {
    mc_worker_t *w = (mc_worker_t *)arg;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(w->cpu_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    mc_task_t task;
    while (atomic_load(&w->running)) {
        if (ring_pop(&w->ring, &task)) {
            if (task.fn) task.result = task.fn(task.arg);
            if (task.on_complete) task.on_complete(task.result, NULL);
        } else {
            if (w->use_sched_yield)
                idle_sched_yield();
            else
                idle_yield_instruction();
        }
    }
    return NULL;
}

static void mc_init(int n, int *cpus, int use_sched_yield) {
    g_mc = calloc(n, sizeof(mc_worker_t));
    g_mc_size = n;

    for (int i = 0; i < n; i++) {
        g_mc[i].id = i;
        g_mc[i].cpu_id = cpus ? cpus[i] : i;
        g_mc[i].use_sched_yield = use_sched_yield;
        g_mc[i].ring.capacity = RING_CAPACITY;
        g_mc[i].ring.mask = RING_CAPACITY - 1;
        g_mc[i].ring.buffer = aligned_alloc(64, RING_CAPACITY * sizeof(mc_task_t));
        memset(g_mc[i].ring.buffer, 0, RING_CAPACITY * sizeof(mc_task_t));
        atomic_init(&g_mc[i].ring.head, 0);
        atomic_init(&g_mc[i].ring.tail, 0);
        atomic_init(&g_mc[i].running, true);
        pthread_create(&g_mc[i].thread, NULL, mc_worker_fn, &g_mc[i]);
    }
    usleep(10000);
}

static void mc_shutdown(void) {
    for (int i = 0; i < g_mc_size; i++) atomic_store(&g_mc[i].running, false);
    for (int i = 0; i < g_mc_size; i++) {
        pthread_join(g_mc[i].thread, NULL);
        free(g_mc[i].ring.buffer);
    }
    free(g_mc);
    g_mc = NULL;
}

static atomic_int g_done_count;
static work_t g_work[64];

static void *gemm_wrapper(void *arg) {
    gemm_tiled((work_t *)arg);
    return NULL;
}

static void on_complete(void *r, void *u) {
    (void)r; (void)u;
    atomic_fetch_add(&g_done_count, 1);
}

static double run_mc(const float *A, const float *B, float *C, int size, int n) {
    int rows_per = size / n;
    atomic_store(&g_done_count, 0);

    uint64_t start = get_time_ns();

    for (int i = 0; i < n; i++) {
        g_work[i] = (work_t){A, B, C, size, i * rows_per,
                             (i == n - 1) ? size : (i + 1) * rows_per};
        mc_task_t task = {
            .fn = gemm_wrapper,
            .arg = &g_work[i],
            .on_complete = on_complete
        };
        int mc_idx = i % g_mc_size;
        while (!ring_push(&g_mc[mc_idx].ring, &task)) idle_yield_instruction();
    }

    while (atomic_load(&g_done_count) < n) idle_yield_instruction();

    return (get_time_ns() - start) / 1e6;
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

static float *alloc_matrix(int n) {
    float *m = aligned_alloc(64, n * n * sizeof(float));
    if (m) for (int i = 0; i < n * n; i++) m[i] = (float)rand() / RAND_MAX;
    return m;
}

int main(void) {
    int num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    int n = num_cpus > 8 ? 8 : num_cpus;

    printf("================================================================\n");
    printf("FAIR COMPARISON: sched_yield vs yield instruction\n");
    printf("================================================================\n\n");

    printf("Question: Is sched_yield's benefit MC-specific or universal?\n\n");
    printf("Configuration: %d CPUs, %d workers, %dx%d matrix\n\n",
           num_cpus, n, MATRIX_SIZE, MATRIX_SIZE);

    float *A = alloc_matrix(MATRIX_SIZE);
    float *B = alloc_matrix(MATRIX_SIZE);
    float *C = alloc_matrix(MATRIX_SIZE);

    if (!A || !B || !C) {
        fprintf(stderr, "Alloc failed\n");
        return 1;
    }

    int cpus[64];
    for (int i = 0; i < n; i++) cpus[i] = i % num_cpus;

    stats_t s_spin_yield, s_spin_sched, s_mc_yield, s_mc_sched;
    stats_init(&s_spin_yield);
    stats_init(&s_spin_sched);
    stats_init(&s_mc_yield);
    stats_init(&s_mc_sched);

    printf("Running benchmarks...\n\n");

    /* 1. Spin-pool + yield instruction */
    printf("  [1/4] Spin-pool + yield instruction...\n");
    spin_init(n, cpus, 0);
    for (int i = 0; i < WARMUP_RUNS; i++) run_spin(A, B, C, MATRIX_SIZE, n);
    for (int i = 0; i < BENCHMARK_RUNS; i++) stats_add(&s_spin_yield, run_spin(A, B, C, MATRIX_SIZE, n));
    spin_shutdown();
    stats_compute(&s_spin_yield);

    /* 2. Spin-pool + sched_yield */
    printf("  [2/4] Spin-pool + sched_yield...\n");
    spin_init(n, cpus, 1);
    for (int i = 0; i < WARMUP_RUNS; i++) run_spin(A, B, C, MATRIX_SIZE, n);
    for (int i = 0; i < BENCHMARK_RUNS; i++) stats_add(&s_spin_sched, run_spin(A, B, C, MATRIX_SIZE, n));
    spin_shutdown();
    stats_compute(&s_spin_sched);

    /* 3. MC + yield instruction */
    printf("  [3/4] MC + yield instruction...\n");
    mc_init(n, cpus, 0);
    for (int i = 0; i < WARMUP_RUNS; i++) run_mc(A, B, C, MATRIX_SIZE, n);
    for (int i = 0; i < BENCHMARK_RUNS; i++) stats_add(&s_mc_yield, run_mc(A, B, C, MATRIX_SIZE, n));
    mc_shutdown();
    stats_compute(&s_mc_yield);

    /* 4. MC + sched_yield */
    printf("  [4/4] MC + sched_yield...\n");
    mc_init(n, cpus, 1);
    for (int i = 0; i < WARMUP_RUNS; i++) run_mc(A, B, C, MATRIX_SIZE, n);
    for (int i = 0; i < BENCHMARK_RUNS; i++) stats_add(&s_mc_sched, run_mc(A, B, C, MATRIX_SIZE, n));
    mc_shutdown();
    stats_compute(&s_mc_sched);

    /* Results */
    printf("\n================================================================\n");
    printf("RESULTS (2x2 Matrix)\n");
    printf("================================================================\n\n");

    printf("%-30s %10s %10s %10s\n", "Configuration", "Mean(ms)", "Std", "Min");
    printf("------------------------------------------------------------------------\n");
    printf("%-30s %10.2f %10.2f %10.2f\n", "Spin-pool + yield instr",
           s_spin_yield.mean, s_spin_yield.std, s_spin_yield.min);
    printf("%-30s %10.2f %10.2f %10.2f\n", "Spin-pool + sched_yield",
           s_spin_sched.mean, s_spin_sched.std, s_spin_sched.min);
    printf("%-30s %10.2f %10.2f %10.2f\n", "MC + yield instr",
           s_mc_yield.mean, s_mc_yield.std, s_mc_yield.min);
    printf("%-30s %10.2f %10.2f %10.2f\n", "MC + sched_yield",
           s_mc_sched.mean, s_mc_sched.std, s_mc_sched.min);

    printf("\n================================================================\n");
    printf("ANALYSIS\n");
    printf("================================================================\n\n");

    double spin_sched_benefit = (s_spin_yield.mean - s_spin_sched.mean) / s_spin_yield.mean * 100;
    double mc_sched_benefit = (s_mc_yield.mean - s_mc_sched.mean) / s_mc_yield.mean * 100;

    printf("sched_yield benefit:\n");
    printf("  Spin-pool: %+.1f%%\n", spin_sched_benefit);
    printf("  MC:        %+.1f%%\n", mc_sched_benefit);

    printf("\nFair comparisons (same idle mechanism):\n");
    double fair_yield = (s_spin_yield.mean - s_mc_yield.mean) / s_spin_yield.mean * 100;
    double fair_sched = (s_spin_sched.mean - s_mc_sched.mean) / s_spin_sched.mean * 100;
    printf("  With yield instr: MC is %+.1f%% vs spin-pool\n", fair_yield);
    printf("  With sched_yield: MC is %+.1f%% vs spin-pool\n", fair_sched);

    printf("\n================================================================\n");
    printf("CONCLUSION\n");
    printf("================================================================\n\n");

    if (fabs(spin_sched_benefit - mc_sched_benefit) < 10) {
        printf("sched_yield benefit is UNIVERSAL (helps both equally)\n");
        printf("The ~40%% speedup is NOT MC-specific architecture advantage.\n");
        printf("It's a syscall-induced desynchronization effect.\n");
    } else if (mc_sched_benefit > spin_sched_benefit + 10) {
        printf("sched_yield benefit is MC-SPECIFIC!\n");
        printf("MC architecture has synergy with kernel scheduling.\n");
        printf("This could be related to ring buffer dispatch pattern.\n");
    } else {
        printf("sched_yield benefit is SPIN-POOL-SPECIFIC!\n");
        printf("This is unexpected and needs investigation.\n");
    }

    printf("\nTRUE MC ADVANTAGE (fair comparison):\n");
    if (fabs(fair_yield) < 5 && fabs(fair_sched) < 5) {
        printf("  MC has NO inherent advantage over spin-pool.\n");
        printf("  The architectures are equivalent.\n");
    } else if (fair_yield > 5 || fair_sched > 5) {
        printf("  MC is genuinely faster: %.1f%% (yield) / %.1f%% (sched)\n",
               fair_yield, fair_sched);
    } else {
        printf("  Spin-pool is actually faster in fair comparison!\n");
    }

    printf("\n");

    free(A);
    free(B);
    free(C);
    return 0;
}
