/*
 * Minimal GEMM timing test to understand baseline performance.
 * Tests identical GEMM kernel with both architectures.
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

#define MATRIX_SIZE 1024
#define TILE_SIZE 32
#define NUM_WORKERS 4
#define NUM_RUNS 10

static float *A, *B, *C;

typedef struct {
    const float *A, *B;
    float *C;
    int N;
    int row_start, row_end;
} work_t;

/* Exact same GEMM kernel as V4 */
static void gemm_tiled_range(const work_t *w) {
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

/* Worker threads */
typedef struct {
    pthread_t thread;
    _Atomic int has_work;
    _Atomic int done;
    _Atomic int shutdown;
    work_t work;
    int cpu_pin;
} worker_t;

static worker_t workers[NUM_WORKERS];

static inline void arm_yield(void) {
#if defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield" ::: "memory");
#else
    __asm__ __volatile__("pause" ::: "memory");
#endif
}

static void* worker_fn(void *arg) {
    worker_t *w = (worker_t*)arg;

    /* Pin to CPU */
    if (w->cpu_pin >= 0) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(w->cpu_pin, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    }

    while (1) {
        while (!atomic_load(&w->has_work) && !atomic_load(&w->shutdown)) {
            arm_yield();
        }

        if (atomic_load(&w->shutdown)) break;

        atomic_store(&w->has_work, 0);
        gemm_tiled_range(&w->work);
        atomic_store(&w->done, 1);
    }
    return NULL;
}

static inline uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static double run_gemm(void) {
    int rows_per = MATRIX_SIZE / NUM_WORKERS;

    uint64_t start = get_time_ns();

    /* Submit work */
    for (int i = 0; i < NUM_WORKERS; i++) {
        workers[i].work = (work_t){A, B, C, MATRIX_SIZE,
                                   i * rows_per,
                                   (i == NUM_WORKERS - 1) ? MATRIX_SIZE : (i + 1) * rows_per};
        atomic_store(&workers[i].done, 0);
        atomic_store(&workers[i].has_work, 1);
    }

    /* Wait for completion */
    for (int i = 0; i < NUM_WORKERS; i++) {
        while (!atomic_load(&workers[i].done)) {
            arm_yield();
        }
    }

    return (get_time_ns() - start) / 1e6;
}

int main(void) {
    printf("================================================================\n");
    printf("BASELINE GEMM TIMING TEST\n");
    printf("================================================================\n");
    printf("Matrix: %dx%d, Tile: %d, Workers: %d, Runs: %d\n\n",
           MATRIX_SIZE, MATRIX_SIZE, TILE_SIZE, NUM_WORKERS, NUM_RUNS);

    /* Allocate matrices */
    A = aligned_alloc(64, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    B = aligned_alloc(64, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    C = aligned_alloc(64, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    if (!A || !B || !C) {
        printf("Memory allocation failed\n");
        return 1;
    }

    /* Initialize with random values */
    srand(42);
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }

    /* Initialize workers */
    for (int i = 0; i < NUM_WORKERS; i++) {
        atomic_store(&workers[i].has_work, 0);
        atomic_store(&workers[i].done, 0);
        atomic_store(&workers[i].shutdown, 0);
        workers[i].cpu_pin = i;
        pthread_create(&workers[i].thread, NULL, worker_fn, &workers[i]);
    }

    usleep(10000);  /* Let workers start */

    /* Warmup */
    printf("Warming up...\n");
    for (int i = 0; i < 3; i++) run_gemm();

    /* Benchmark */
    printf("Running benchmark...\n\n");
    double times[NUM_RUNS];
    double sum = 0, min = 1e9, max = 0;

    for (int i = 0; i < NUM_RUNS; i++) {
        times[i] = run_gemm();
        sum += times[i];
        if (times[i] < min) min = times[i];
        if (times[i] > max) max = times[i];
        printf("  Run %2d: %.1f ms\n", i + 1, times[i]);
    }

    double mean = sum / NUM_RUNS;
    double sq_sum = 0;
    for (int i = 0; i < NUM_RUNS; i++) {
        double diff = times[i] - mean;
        sq_sum += diff * diff;
    }
    double std = sqrt(sq_sum / (NUM_RUNS - 1));

    printf("\n================================================================\n");
    printf("RESULTS\n");
    printf("================================================================\n");
    printf("Mean: %.1f ms, Std: %.1f ms, Min: %.1f ms, Max: %.1f ms\n",
           mean, std, min, max);

    /* Shutdown */
    for (int i = 0; i < NUM_WORKERS; i++) {
        atomic_store(&workers[i].shutdown, 1);
    }
    for (int i = 0; i < NUM_WORKERS; i++) {
        pthread_join(workers[i].thread, NULL);
    }

    free(A);
    free(B);
    free(C);

    return 0;
}
