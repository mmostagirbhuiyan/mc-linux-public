/*
 * FINAL Dispatch Comparison: Spin-pool vs MC (SPSC)
 * Clean working version
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdatomic.h>
#include <unistd.h>
#include <pthread.h>
#include <mach/mach_time.h>

#define NUM_WORKERS 8
#define RING_CAPACITY 4096
#define NUM_TRIALS 10

static mach_timebase_info_data_t timebase_info;
static inline uint64_t get_time_ns(void) {
    if (timebase_info.denom == 0) mach_timebase_info(&timebase_info);
    return mach_absolute_time() * timebase_info.numer / timebase_info.denom;
}

static volatile uint64_t sink = 0;
static void do_work(int iterations) {
    uint64_t x = 1;
    for (int i = 0; i < iterations; i++)
        x = x * 6364136223846793005ULL + 1442695040888963407ULL;
    sink = x;
}

/* ============================================================================
 * SPIN-POOL
 * ============================================================================ */

typedef struct {
    pthread_t thread;
    _Atomic int has_work;
    _Atomic int shutdown;
    int work_size;
} spin_worker_t;

static spin_worker_t spin_workers[NUM_WORKERS];
static atomic_int spin_done = 0;

static void *spin_worker_fn(void *arg) {
    spin_worker_t *w = (spin_worker_t *)arg;
    while (!atomic_load(&w->shutdown)) {
        if (atomic_load(&w->has_work)) {
            do_work(w->work_size);
            atomic_store(&w->has_work, 0);
            atomic_fetch_add(&spin_done, 1);
        }
    }
    return NULL;
}

static void spin_init(void) {
    for (int i = 0; i < NUM_WORKERS; i++) {
        atomic_store(&spin_workers[i].has_work, 0);
        atomic_store(&spin_workers[i].shutdown, 0);
        pthread_create(&spin_workers[i].thread, NULL, spin_worker_fn, &spin_workers[i]);
    }
    usleep(5000);
}

static void spin_shutdown(void) {
    for (int i = 0; i < NUM_WORKERS; i++) atomic_store(&spin_workers[i].shutdown, 1);
    for (int i = 0; i < NUM_WORKERS; i++) pthread_join(spin_workers[i].thread, NULL);
}

static double run_spin(int num_tasks, int work_size) {
    atomic_store(&spin_done, 0);
    uint64_t start = get_time_ns();

    for (int t = 0; t < num_tasks; t++) {
        int w = t % NUM_WORKERS;
        while (atomic_load(&spin_workers[w].has_work)) {}
        spin_workers[w].work_size = work_size;
        atomic_store(&spin_workers[w].has_work, 1);
    }

    while (atomic_load(&spin_done) < num_tasks) {}

    return (get_time_ns() - start) / 1e6;
}

/* ============================================================================
 * MC (SPSC Ring Buffer)
 * ============================================================================ */

typedef struct {
    int *buffer;
    size_t mask;
    atomic_uint_fast64_t head;
    atomic_uint_fast64_t tail;
} mc_ring_t;

typedef struct {
    mc_ring_t ring;
    pthread_t thread;
    atomic_bool running;
    atomic_int completed;
} mc_worker_t;

static mc_worker_t mc_workers[NUM_WORKERS];

static void *mc_worker_fn(void *arg) {
    mc_worker_t *w = (mc_worker_t *)arg;
    while (atomic_load(&w->running)) {
        uint64_t head = atomic_load(&w->ring.head);
        uint64_t tail = atomic_load(&w->ring.tail);
        if (head < tail) {
            int work_size = w->ring.buffer[head & w->ring.mask];
            atomic_store(&w->ring.head, head + 1);
            do_work(work_size);
            atomic_fetch_add(&w->completed, 1);
        }
    }
    return NULL;
}

static void mc_init(void) {
    for (int i = 0; i < NUM_WORKERS; i++) {
        mc_workers[i].ring.mask = RING_CAPACITY - 1;
        mc_workers[i].ring.buffer = malloc(RING_CAPACITY * sizeof(int));
        atomic_init(&mc_workers[i].ring.head, 0);
        atomic_init(&mc_workers[i].ring.tail, 0);
        atomic_init(&mc_workers[i].running, true);
        atomic_init(&mc_workers[i].completed, 0);
        pthread_create(&mc_workers[i].thread, NULL, mc_worker_fn, &mc_workers[i]);
    }
    usleep(5000);
}

static void mc_shutdown(void) {
    for (int i = 0; i < NUM_WORKERS; i++) atomic_store(&mc_workers[i].running, false);
    for (int i = 0; i < NUM_WORKERS; i++) {
        pthread_join(mc_workers[i].thread, NULL);
        free(mc_workers[i].ring.buffer);
    }
}

static int mc_total_done(void) {
    int total = 0;
    for (int i = 0; i < NUM_WORKERS; i++)
        total += atomic_load(&mc_workers[i].completed);
    return total;
}

static double run_mc(int num_tasks, int work_size) {
    for (int i = 0; i < NUM_WORKERS; i++) {
        atomic_store(&mc_workers[i].completed, 0);
        atomic_store(&mc_workers[i].ring.head, 0);
        atomic_store(&mc_workers[i].ring.tail, 0);
    }

    uint64_t start = get_time_ns();

    for (int t = 0; t < num_tasks; t++) {
        int w = t % NUM_WORKERS;
        mc_ring_t *ring = &mc_workers[w].ring;
        uint64_t tail = atomic_load(&ring->tail);
        ring->buffer[tail & ring->mask] = work_size;
        atomic_store(&ring->tail, tail + 1);
    }

    while (mc_total_done() < num_tasks) {}

    return (get_time_ns() - start) / 1e6;
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    printf("================================================================\n");
    printf("SPIN-POOL vs MC (SPSC) COMPARISON\n");
    printf("================================================================\n\n");
    printf("Workers: %d, Trials: %d\n\n", NUM_WORKERS, NUM_TRIALS);

    struct {
        const char *name;
        int iterations;
        int num_tasks;
    } tests[] = {
        {"Tiny (~100ns)",   100,    10000},
        {"Small (~1us)",    1000,   10000},
        {"Medium (~10us)",  10000,  5000},
        {"Large (~100us)",  100000, 1000},
    };
    int num_tests = sizeof(tests) / sizeof(tests[0]);

    double spin_times[10], mc_times[10];

    /* Test spin-pool */
    printf("Testing SPIN-POOL...\n");
    spin_init();
    run_spin(500, 100);  /* warmup */
    for (int t = 0; t < num_tests; t++) {
        double total = 0;
        for (int r = 0; r < NUM_TRIALS; r++)
            total += run_spin(tests[t].num_tasks, tests[t].iterations);
        spin_times[t] = total / NUM_TRIALS;
        printf("  %s: %.2f ms\n", tests[t].name, spin_times[t]);
    }
    spin_shutdown();

    /* Test MC */
    printf("\nTesting MC (SPSC)...\n");
    mc_init();
    run_mc(500, 100);  /* warmup */
    for (int t = 0; t < num_tests; t++) {
        double total = 0;
        for (int r = 0; r < NUM_TRIALS; r++)
            total += run_mc(tests[t].num_tasks, tests[t].iterations);
        mc_times[t] = total / NUM_TRIALS;
        printf("  %s: %.2f ms\n", tests[t].name, mc_times[t]);
    }
    mc_shutdown();

    /* Results */
    printf("\n================================================================\n");
    printf("RESULTS\n");
    printf("================================================================\n\n");

    printf("%-16s %8s %10s %10s %12s\n",
           "Task Size", "Tasks", "Spin(ms)", "MC(ms)", "MC Advantage");
    printf("------------------------------------------------------------------------\n");

    for (int t = 0; t < num_tests; t++) {
        double advantage = (spin_times[t] - mc_times[t]) / spin_times[t] * 100;
        printf("%-16s %8d %10.2f %10.2f %+11.1f%%\n",
               tests[t].name, tests[t].num_tasks, spin_times[t], mc_times[t], advantage);
    }

    printf("\n================================================================\n");
    printf("INTERPRETATION\n");
    printf("================================================================\n");
    printf("Positive %% = MC's SPSC ring buffer is faster\n");
    printf("Negative %% = Spin-pool's atomic flags are faster\n");
    printf("\nMC advantage for small tasks comes from:\n");
    printf("1. Ring buffer allows batching (dispatch doesn't wait per-task)\n");
    printf("2. Less contention (producer updates tail, consumer updates head)\n");

    return 0;
}
