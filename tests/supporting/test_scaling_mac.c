/*
 * Scaling Efficiency Test (macOS)
 *
 * Tests how MC vs spin-pool scale from 1 to N workers.
 * Looking for: any difference in scaling efficiency that could indicate
 * architectural advantages at higher core counts.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <unistd.h>
#include <math.h>
#include <pthread.h>
#include <float.h>
#include <mach/mach_time.h>

#define TILE_SIZE 32
#define MATRIX_SIZE 1024
#define WARMUP_RUNS 3
#define BENCHMARK_RUNS 30
#define RING_CAPACITY 256

static mach_timebase_info_data_t timebase_info;

static inline uint64_t get_time_ns(void) {
    if (timebase_info.denom == 0) mach_timebase_info(&timebase_info);
    return mach_absolute_time() * timebase_info.numer / timebase_info.denom;
}

/* Idle - use sched_yield for fair comparison */
static inline void idle_wait(void) {
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
 * SPIN-POOL
 * ============================================================================ */

typedef struct {
    pthread_t thread;
    _Atomic int has_work;
    _Atomic int done;
    _Atomic int shutdown;
    work_t work;
} spin_worker_t;

static spin_worker_t *g_spin = NULL;
static int g_spin_size = 0;

static void *spin_worker_fn(void *arg) {
    spin_worker_t *w = (spin_worker_t *)arg;
    while (1) {
        while (!atomic_load(&w->has_work) && !atomic_load(&w->shutdown)) idle_wait();
        if (atomic_load(&w->shutdown)) break;
        gemm_tiled(&w->work);
        atomic_store(&w->has_work, 0);
        atomic_store(&w->done, 1);
    }
    return NULL;
}

static void spin_init(int n) {
    g_spin = calloc(n, sizeof(spin_worker_t));
    g_spin_size = n;
    for (int i = 0; i < n; i++)
        pthread_create(&g_spin[i].thread, NULL, spin_worker_fn, &g_spin[i]);
    usleep(5000);
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
        while (!atomic_load(&g_spin[i].done)) idle_wait();
    return (get_time_ns() - start) / 1e6;
}

/* ============================================================================
 * MC-STYLE (SPSC ring buffer per worker)
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
    mc_ring_t ring;
    pthread_t thread;
    atomic_bool running;
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
    mc_task_t task;
    while (atomic_load(&w->running)) {
        if (ring_pop(&w->ring, &task)) {
            if (task.fn) task.result = task.fn(task.arg);
            if (task.on_complete) task.on_complete(task.result, NULL);
        } else {
            idle_wait();
        }
    }
    return NULL;
}

static void mc_init(int n) {
    g_mc = calloc(n, sizeof(mc_worker_t));
    g_mc_size = n;
    for (int i = 0; i < n; i++) {
        g_mc[i].id = i;
        g_mc[i].ring.capacity = RING_CAPACITY;
        g_mc[i].ring.mask = RING_CAPACITY - 1;
        posix_memalign((void**)&g_mc[i].ring.buffer, 64, RING_CAPACITY * sizeof(mc_task_t));
        memset(g_mc[i].ring.buffer, 0, RING_CAPACITY * sizeof(mc_task_t));
        atomic_init(&g_mc[i].ring.head, 0);
        atomic_init(&g_mc[i].ring.tail, 0);
        atomic_init(&g_mc[i].running, true);
        pthread_create(&g_mc[i].thread, NULL, mc_worker_fn, &g_mc[i]);
    }
    usleep(5000);
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
        mc_task_t task = { .fn = gemm_wrapper, .arg = &g_work[i], .on_complete = on_complete };
        while (!ring_push(&g_mc[i % g_mc_size].ring, &task)) idle_wait();
    }
    while (atomic_load(&g_done_count) < n) idle_wait();
    return (get_time_ns() - start) / 1e6;
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

static float *alloc_matrix(int n) {
    float *m;
    posix_memalign((void**)&m, 64, n * n * sizeof(float));
    if (m) for (int i = 0; i < n * n; i++) m[i] = (float)rand() / RAND_MAX;
    return m;
}

int main(void) {
    int max_cpus = (int)sysconf(_SC_NPROCESSORS_ONLN);

    printf("================================================================\n");
    printf("SCALING EFFICIENCY TEST (macOS M4 Pro)\n");
    printf("================================================================\n\n");
    printf("Hardware: %d CPUs, 1024x1024 GEMM\n", max_cpus);
    printf("Question: Does MC scale differently than spin-pool?\n\n");

    float *A = alloc_matrix(MATRIX_SIZE);
    float *B = alloc_matrix(MATRIX_SIZE);
    float *C = alloc_matrix(MATRIX_SIZE);

    if (!A || !B || !C) { fprintf(stderr, "Alloc failed\n"); return 1; }

    /* Test worker counts */
    int worker_counts[] = {1, 2, 4, 6, 8, 10, 12, 14};
    int num_tests = sizeof(worker_counts) / sizeof(worker_counts[0]);

    double spin_times[16], mc_times[16];
    double baseline = 0;

    printf("%-8s %12s %12s %12s %12s\n", "Workers", "Spin(ms)", "MC(ms)", "Spin Eff", "MC Eff");
    printf("------------------------------------------------------------------------\n");

    for (int t = 0; t < num_tests; t++) {
        int n = worker_counts[t];
        if (n > max_cpus) break;

        /* Spin-pool */
        spin_init(n);
        for (int i = 0; i < WARMUP_RUNS; i++) run_spin(A, B, C, MATRIX_SIZE, n);
        double spin_sum = 0;
        for (int i = 0; i < BENCHMARK_RUNS; i++) spin_sum += run_spin(A, B, C, MATRIX_SIZE, n);
        spin_times[t] = spin_sum / BENCHMARK_RUNS;
        spin_shutdown();

        /* MC */
        mc_init(n);
        for (int i = 0; i < WARMUP_RUNS; i++) run_mc(A, B, C, MATRIX_SIZE, n);
        double mc_sum = 0;
        for (int i = 0; i < BENCHMARK_RUNS; i++) mc_sum += run_mc(A, B, C, MATRIX_SIZE, n);
        mc_times[t] = mc_sum / BENCHMARK_RUNS;
        mc_shutdown();

        if (t == 0) baseline = spin_times[0];  /* Single-threaded baseline */

        double spin_speedup = baseline / spin_times[t];
        double mc_speedup = baseline / mc_times[t];
        double spin_eff = spin_speedup / n * 100;
        double mc_eff = mc_speedup / n * 100;

        printf("%-8d %12.2f %12.2f %11.1f%% %11.1f%%\n",
               n, spin_times[t], mc_times[t], spin_eff, mc_eff);
    }

    printf("\n================================================================\n");
    printf("ANALYSIS\n");
    printf("================================================================\n\n");

    printf("Scaling comparison (speedup relative to 1 worker):\n\n");
    printf("%-8s %12s %12s %12s\n", "Workers", "Spin Speedup", "MC Speedup", "Difference");
    printf("------------------------------------------------------------------------\n");

    for (int t = 0; t < num_tests; t++) {
        int n = worker_counts[t];
        if (n > max_cpus) break;

        double spin_speedup = baseline / spin_times[t];
        double mc_speedup = baseline / mc_times[t];
        double diff = (mc_speedup - spin_speedup) / spin_speedup * 100;

        printf("%-8d %12.2fx %12.2fx %+11.1f%%\n", n, spin_speedup, mc_speedup, diff);
    }

    printf("\n================================================================\n");
    printf("CONCLUSION\n");
    printf("================================================================\n\n");

    /* Check if there's a consistent difference at high core counts */
    int high_count = 0;
    double high_diff_sum = 0;
    for (int t = 0; t < num_tests; t++) {
        int n = worker_counts[t];
        if (n > max_cpus) break;
        if (n >= 8) {
            double spin_speedup = baseline / spin_times[t];
            double mc_speedup = baseline / mc_times[t];
            high_diff_sum += (mc_speedup - spin_speedup) / spin_speedup * 100;
            high_count++;
        }
    }

    if (high_count > 0) {
        double avg_high_diff = high_diff_sum / high_count;
        if (avg_high_diff > 5) {
            printf("MC scales BETTER at high core counts: avg %+.1f%% at 8+ cores\n", avg_high_diff);
        } else if (avg_high_diff < -5) {
            printf("Spin-pool scales BETTER at high core counts: avg %+.1f%% at 8+ cores\n", avg_high_diff);
        } else {
            printf("MC and spin-pool scale EQUIVALENTLY (within 5%% at all core counts)\n");
        }
    }

    free(A); free(B); free(C);
    return 0;
}
