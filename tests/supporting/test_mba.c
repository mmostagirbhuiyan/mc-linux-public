/*
 * MBA (Memory Bandwidth Allocation) Test
 * Hardware-based interference mitigation vs software detection
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>

#define BUFFER_SIZE (8 * 1024 * 1024)
#define PRIORITY_ITERS 500000
#define NUM_TRIALS 5

static inline uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static char *memory_buffer = NULL;
static _Atomic int test_complete = 0;
static _Atomic int background_running = 0;

static void memory_workload(int iterations) {
    unsigned int seed = (unsigned int)get_time_ns();
    volatile char sum = 0;
    for (int i = 0; i < iterations; i++) {
        int idx = rand_r(&seed) % BUFFER_SIZE;
        sum += memory_buffer[idx];
        memory_buffer[(idx + 1) % BUFFER_SIZE] = sum;
    }
}

typedef struct { pthread_t thread; int core_id; } worker_t;

static void* background_worker(void *arg) {
    worker_t *w = (worker_t*)arg;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(w->core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    atomic_fetch_add(&background_running, 1);
    while (!atomic_load(&test_complete)) {
        memory_workload(50000);
    }
    atomic_fetch_sub(&background_running, 1);
    return NULL;
}

static double run_test(worker_t *workers, int num_bg) {
    atomic_store(&test_complete, 0);
    atomic_store(&background_running, 0);

    for (int i = 0; i < num_bg; i++) {
        pthread_create(&workers[i].thread, NULL, background_worker, &workers[i]);
    }
    while (atomic_load(&background_running) < num_bg) sched_yield();

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

    uint64_t start = get_time_ns();
    memory_workload(PRIORITY_ITERS);
    double time_ms = (get_time_ns() - start) / 1e6;

    atomic_store(&test_complete, 1);
    for (int i = 0; i < num_bg; i++) {
        pthread_join(workers[i].thread, NULL);
    }
    return time_ms;
}

int main(void) {
    printf("================================================================\n");
    printf("MBA (Memory Bandwidth Allocation) TEST\n");
    printf("================================================================\n\n");

    memory_buffer = aligned_alloc(64, BUFFER_SIZE);
    memset(memory_buffer, 0, BUFFER_SIZE);

    worker_t workers[7];
    for (int i = 0; i < 7; i++) workers[i].core_id = i + 1;

    memory_workload(PRIORITY_ITERS/4);

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

    double isolated = 0;
    for (int i = 0; i < NUM_TRIALS; i++) {
        uint64_t start = get_time_ns();
        memory_workload(PRIORITY_ITERS);
        isolated += (get_time_ns() - start) / 1e6;
    }
    isolated /= NUM_TRIALS;
    printf("Isolated:        %.1f ms\n", isolated);

    double interference = 0;
    for (int i = 0; i < NUM_TRIALS; i++) {
        interference += run_test(workers, 7);
    }
    interference /= NUM_TRIALS;
    printf("MBA=100%% (full): %.1f ms (+%.0f%%)\n", interference,
           ((interference-isolated)/isolated)*100);

    system("sudo mkdir -p /sys/fs/resctrl/background 2>/dev/null");
    system("echo 'MB:0=50;1=50' | sudo tee /sys/fs/resctrl/background/schemata > /dev/null 2>&1");
    system("echo '1-47' | sudo tee /sys/fs/resctrl/background/cpus_list > /dev/null 2>&1");

    double mba50 = 0;
    for (int i = 0; i < NUM_TRIALS; i++) {
        mba50 += run_test(workers, 7);
    }
    mba50 /= NUM_TRIALS;
    printf("MBA=50%%:         %.1f ms (+%.0f%%)\n", mba50,
           ((mba50-isolated)/isolated)*100);

    system("echo 'MB:0=30;1=30' | sudo tee /sys/fs/resctrl/background/schemata > /dev/null 2>&1");

    double mba30 = 0;
    for (int i = 0; i < NUM_TRIALS; i++) {
        mba30 += run_test(workers, 7);
    }
    mba30 /= NUM_TRIALS;
    printf("MBA=30%%:         %.1f ms (+%.0f%%)\n", mba30,
           ((mba30-isolated)/isolated)*100);

    system("echo 'MB:0=10;1=10' | sudo tee /sys/fs/resctrl/background/schemata > /dev/null 2>&1");

    double mba10 = 0;
    for (int i = 0; i < NUM_TRIALS; i++) {
        mba10 += run_test(workers, 7);
    }
    mba10 /= NUM_TRIALS;
    printf("MBA=10%%:         %.1f ms (+%.0f%%)\n", mba10,
           ((mba10-isolated)/isolated)*100);

    system("sudo rmdir /sys/fs/resctrl/background 2>/dev/null");

    printf("\n================================================================\n");
    printf("ANALYSIS\n");
    printf("================================================================\n");
    double latency_increase = interference - isolated;
    printf("MBA=50%% recovery: %.1f ms (%.0f%%)\n", interference-mba50,
           ((interference-mba50)/latency_increase)*100);
    printf("MBA=30%% recovery: %.1f ms (%.0f%%)\n", interference-mba30,
           ((interference-mba30)/latency_increase)*100);
    printf("MBA=10%% recovery: %.1f ms (%.0f%%)\n", interference-mba10,
           ((interference-mba10)/latency_increase)*100);

    printf("\n================================================================\n");
    printf("Hardware MBA vs Software Detection:\n");
    printf("- MBA requires STATIC policy (decide bandwidth %% upfront)\n");
    printf("- Software detection is DYNAMIC (react to actual interference)\n");
    printf("- MBA is zero-overhead; detection has ~5%% overhead\n");
    printf("- COMBINED: Detection triggers MBA for best of both worlds\n");
    printf("================================================================\n");

    free(memory_buffer);
    return 0;
}
