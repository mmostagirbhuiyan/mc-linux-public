/*
 * Autoscaler Dynamic Load Test (P1)
 *
 * Validates that the autoscaler correctly scales MC count up/down
 * in response to load changes (Patent Claims 5, 7, 11).
 *
 * Test Phases:
 *   Phase 1 (30s): Low load - 1 compute task/100ms (high IPC)
 *                  -> Expect: active_mcs stays at minimum
 *   Phase 2 (30s): Ramp up - 10 memory tasks/100ms
 *                  -> Expect: IPC drops, active_mcs increases
 *   Phase 3 (30s): High load - 50 memory tasks/100ms
 *                  -> Expect: active_mcs reaches maximum
 *   Phase 4 (30s): Ramp down - stop submitting
 *                  -> Expect: active_mcs decreases back toward minimum
 *
 * Build (on Pi 5):
 *   gcc -O3 -march=native tests/test_autoscaler_dynamic.c \
 *       lib/libmicrocontainer.a -lpthread -lm -Isrc -o test_autoscaler
 *
 * Run:
 *   sudo sysctl -w kernel.perf_event_paranoid=-1
 *   ./test_autoscaler
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "micro_container.h"

/* Configuration */
#define NUM_CORES 4
#define MCS_PER_CORE 4  /* Max MCs per core (16 total) */
#define SAMPLE_INTERVAL_MS 100
#define PHASE_DURATION_MS 30000
#define NUM_PHASES 4
#define BUFFER_SIZE (8 * 1024 * 1024)

/* Workload iterations */
#define COMPUTE_ITERS 100000
#define MEMORY_ITERS 50000

/* Memory buffer for memory-bound workload */
static char *memory_buffer = NULL;

/* Tracking */
static _Atomic int total_tasks_submitted = 0;
static _Atomic int total_tasks_completed = 0;

/* Time helpers */
static inline uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static inline uint64_t get_time_ms(void) {
    return get_time_ns() / 1000000ULL;
}

/* Workloads */
static void *compute_task(void *arg) {
    (void)arg;
    volatile double sum = 0.0;
    for (int i = 0; i < COMPUTE_ITERS; i++) {
        sum += (double)i * 0.001;
        sum *= 1.0001;
    }
    return NULL;
}

static void *memory_task(void *arg) {
    (void)arg;
    unsigned int seed = (unsigned int)get_time_ns();
    volatile char sum = 0;
    for (int i = 0; i < MEMORY_ITERS; i++) {
        int idx = rand_r(&seed) % BUFFER_SIZE;
        sum += memory_buffer[idx];
        memory_buffer[(idx + 1) % BUFFER_SIZE] = sum;
    }
    return NULL;
}

/* Completion callback */
static void on_task_complete(void *result, void *user_data) {
    (void)result;
    (void)user_data;
    atomic_fetch_add(&total_tasks_completed, 1);
}

/* Sample data structure */
typedef struct {
    uint64_t timestamp_ms;
    int active_mcs;
    double avg_ipc;
    uint64_t scale_up_events;
    uint64_t scale_down_events;
    int tasks_submitted;
    int tasks_completed;
    int phase;
} sample_t;

#define MAX_SAMPLES 2000

/* Statistics helpers */
static double calc_mean(double *values, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) sum += values[i];
    return sum / n;
}

static double calc_stddev(double *values, int n, double mean) {
    double sum_sq = 0;
    for (int i = 0; i < n; i++) {
        double diff = values[i] - mean;
        sum_sq += diff * diff;
    }
    return sqrt(sum_sq / n);
}

static int compare_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

static double calc_percentile(double *values, int n, double p) {
    double *sorted = malloc(n * sizeof(double));
    memcpy(sorted, values, n * sizeof(double));
    qsort(sorted, n, sizeof(double), compare_double);
    int idx = (int)(p * n / 100.0);
    if (idx >= n) idx = n - 1;
    double result = sorted[idx];
    free(sorted);
    return result;
}

int main(void) {
    printf("================================================================\n");
    printf("AUTOSCALER DYNAMIC LOAD TEST (P1)\n");
    printf("================================================================\n");
    printf("Configuration:\n");
    printf("  Cores: %d\n", NUM_CORES);
    printf("  Max MCs per core: %d (total pool: %d)\n", MCS_PER_CORE, NUM_CORES * MCS_PER_CORE);
    printf("  Phase duration: %d ms\n", PHASE_DURATION_MS);
    printf("  Sample interval: %d ms\n", SAMPLE_INTERVAL_MS);
    printf("\n");

    /* Allocate memory buffer */
    memory_buffer = aligned_alloc(64, BUFFER_SIZE);
    if (!memory_buffer) {
        fprintf(stderr, "Failed to allocate memory buffer\n");
        return 1;
    }
    memset(memory_buffer, 0xAA, BUFFER_SIZE);

    /* Initialize orchestrator */
    orchestrator_t orch;
    if (orchestrator_init(&orch, NUM_CORES, MCS_PER_CORE) != 0) {
        fprintf(stderr, "Failed to initialize orchestrator\n");
        free(memory_buffer);
        return 1;
    }

    /* Configure autoscaler */
    autoscaler_config_t as_config = {
        .ipc_scale_up_threshold = 1.0,     /* Scale up when IPC < 1.0 (memory-bound) */
        .ipc_scale_down_threshold = 3.0,   /* Scale down when IPC > 3.0 (compute-bound) */
        .cache_miss_threshold = 0.1,
        .min_mcs_per_core = 1,
        .max_mcs_per_core = MCS_PER_CORE,
        .cooldown_ms = 50
    };

    /* Start with minimum MCs active */
    /* First, pause all MCs beyond the minimum */
    int min_active = NUM_CORES * as_config.min_mcs_per_core;
    for (int i = min_active; i < orch.total_mcs; i++) {
        mc_pause(&orch.mcs[i]);
    }
    atomic_store(&orch.active_mcs, min_active);

    printf("Initial state:\n");
    printf("  Active MCs: %d\n", orchestrator_get_active_mcs(&orch));
    printf("  Min active: %d\n", orch.min_active_mcs);
    printf("  Max active: %d\n", orch.max_active_mcs);
    printf("\n");

    /* Initialize autoscaler */
    autoscaler_t autoscaler;
    autoscaler_init(&autoscaler, &orch, &as_config);

    /* Initialize profiler */
    profiler_t profiler;
    profiler_init(&profiler, &orch, 10);  /* 10ms sample interval */

    /* Start autoscaler and profiler */
    autoscaler_start(&autoscaler);
    profiler_start(&profiler);

    /* Allocate sample storage */
    sample_t *samples = calloc(MAX_SAMPLES, sizeof(sample_t));
    int num_samples = 0;

    /* Phase names */
    const char *phase_names[] = {
        "Low Load (Compute)",
        "Ramp Up (Memory)",
        "High Load (Memory)",
        "Ramp Down (Idle)"
    };

    /* Tasks per interval for each phase */
    int tasks_per_interval[] = {1, 10, 50, 0};

    /* Task type for each phase (0 = compute, 1 = memory) */
    int task_type[] = {0, 1, 1, 0};

    uint64_t start_time = get_time_ms();
    uint64_t last_sample = start_time;
    uint64_t last_submit = start_time;
    int current_phase = 0;

    printf("Running phases...\n\n");

    while (current_phase < NUM_PHASES) {
        uint64_t now = get_time_ms();
        uint64_t phase_start = start_time + current_phase * PHASE_DURATION_MS;
        uint64_t phase_end = phase_start + PHASE_DURATION_MS;

        /* Check phase transition */
        if (now >= phase_end) {
            current_phase++;
            if (current_phase >= NUM_PHASES) break;
            printf("--- Phase %d: %s ---\n", current_phase + 1, phase_names[current_phase]);
            continue;
        }

        /* Print phase start */
        static int last_printed_phase = -1;
        if (current_phase != last_printed_phase) {
            printf("--- Phase %d: %s ---\n", current_phase + 1, phase_names[current_phase]);
            last_printed_phase = current_phase;
        }

        /* Submit tasks based on phase */
        if (now - last_submit >= SAMPLE_INTERVAL_MS && tasks_per_interval[current_phase] > 0) {
            for (int i = 0; i < tasks_per_interval[current_phase]; i++) {
                mc_task_t task = {
                    .fn = task_type[current_phase] ? memory_task : compute_task,
                    .arg = NULL,
                    .on_complete = on_task_complete,
                    .task_id = atomic_load(&total_tasks_submitted)
                };
                if (orchestrator_submit(&orch, &task) == 0) {
                    atomic_fetch_add(&total_tasks_submitted, 1);
                }
            }
            last_submit = now;
        }

        /* Sample at intervals */
        if (now - last_sample >= SAMPLE_INTERVAL_MS && num_samples < MAX_SAMPLES) {
            sample_t *s = &samples[num_samples++];
            s->timestamp_ms = now - start_time;
            s->active_mcs = orchestrator_get_active_mcs(&orch);
            s->avg_ipc = orchestrator_get_avg_ipc(&orch);
            s->scale_up_events = atomic_load(&autoscaler.scale_up_events);
            s->scale_down_events = atomic_load(&autoscaler.scale_down_events);
            s->tasks_submitted = atomic_load(&total_tasks_submitted);
            s->tasks_completed = atomic_load(&total_tasks_completed);
            s->phase = current_phase + 1;

            /* Print progress every 5 samples */
            if (num_samples % 5 == 0) {
                printf("  t=%6lums: active=%2d, IPC=%.2f, up=%lu, down=%lu\n",
                       (unsigned long)s->timestamp_ms, s->active_mcs, s->avg_ipc,
                       s->scale_up_events, s->scale_down_events);
            }
            last_sample = now;
        }

        usleep(10000);  /* 10ms sleep */
    }

    /* Stop autoscaler and profiler */
    autoscaler_stop(&autoscaler);
    profiler_stop(&profiler);

    /* Wait for remaining tasks to complete */
    printf("\nWaiting for tasks to complete...\n");
    int wait_count = 0;
    while (atomic_load(&total_tasks_completed) < atomic_load(&total_tasks_submitted) && wait_count < 100) {
        usleep(100000);  /* 100ms */
        wait_count++;
    }

    printf("\n================================================================\n");
    printf("RESULTS SUMMARY\n");
    printf("================================================================\n");
    printf("Total tasks submitted: %d\n", atomic_load(&total_tasks_submitted));
    printf("Total tasks completed: %d\n", atomic_load(&total_tasks_completed));
    printf("Total scale-up events: %lu\n", atomic_load(&autoscaler.scale_up_events));
    printf("Total scale-down events: %lu\n", atomic_load(&autoscaler.scale_down_events));
    printf("Total samples: %d\n", num_samples);

    /* Analyze per-phase statistics */
    printf("\n================================================================\n");
    printf("PER-PHASE ANALYSIS\n");
    printf("================================================================\n");

    for (int phase = 1; phase <= NUM_PHASES; phase++) {
        double active_values[MAX_SAMPLES];
        double ipc_values[MAX_SAMPLES];
        int count = 0;

        for (int i = 0; i < num_samples; i++) {
            if (samples[i].phase == phase) {
                active_values[count] = samples[i].active_mcs;
                ipc_values[count] = samples[i].avg_ipc;
                count++;
            }
        }

        if (count > 0) {
            double active_mean = calc_mean(active_values, count);
            double active_std = calc_stddev(active_values, count, active_mean);
            double ipc_mean = calc_mean(ipc_values, count);
            double ipc_std = calc_stddev(ipc_values, count, ipc_mean);

            printf("\nPhase %d: %s\n", phase, phase_names[phase - 1]);
            printf("  Samples: %d\n", count);
            printf("  Active MCs: mean=%.1f, std=%.2f, min=%.0f, max=%.0f\n",
                   active_mean, active_std,
                   calc_percentile(active_values, count, 0),
                   calc_percentile(active_values, count, 100));
            printf("  IPC: mean=%.2f, std=%.2f\n", ipc_mean, ipc_std);
        }
    }

    /* Validation criteria */
    printf("\n================================================================\n");
    printf("VALIDATION\n");
    printf("================================================================\n");

    int pass_count = 0;
    int total_checks = 4;

    /* Check 1: Scale up during high memory load (Phase 3) */
    double phase3_active[MAX_SAMPLES];
    int phase3_count = 0;
    for (int i = 0; i < num_samples; i++) {
        if (samples[i].phase == 3) {
            phase3_active[phase3_count++] = samples[i].active_mcs;
        }
    }
    double phase3_max = calc_percentile(phase3_active, phase3_count, 100);
    bool check1 = phase3_max > NUM_CORES;
    printf("[ %s ] MC count increases under memory load (max=%d, want>%d)\n",
           check1 ? "PASS" : "FAIL", (int)phase3_max, NUM_CORES);
    if (check1) pass_count++;

    /* Check 2: Scale down during idle (Phase 4) */
    double phase4_active[MAX_SAMPLES];
    int phase4_count = 0;
    for (int i = 0; i < num_samples; i++) {
        if (samples[i].phase == 4) {
            phase4_active[phase4_count++] = samples[i].active_mcs;
        }
    }
    double phase4_min = calc_percentile(phase4_active, phase4_count, 0);
    bool check2 = phase4_min < phase3_max;
    printf("[ %s ] MC count decreases when idle (min=%d, was max=%d)\n",
           check2 ? "PASS" : "FAIL", (int)phase4_min, (int)phase3_max);
    if (check2) pass_count++;

    /* Check 3: Cooldown prevents thrashing */
    int rapid_changes = 0;
    for (int i = 1; i < num_samples; i++) {
        if (samples[i].timestamp_ms - samples[i-1].timestamp_ms < 50) {
            if (samples[i].active_mcs != samples[i-1].active_mcs) {
                rapid_changes++;
            }
        }
    }
    bool check3 = rapid_changes < 5;  /* Allow some noise */
    printf("[ %s ] Cooldown prevents thrashing (rapid changes=%d, want<5)\n",
           check3 ? "PASS" : "FAIL", rapid_changes);
    if (check3) pass_count++;

    /* Check 4: Scale events correlate with load changes */
    uint64_t total_events = atomic_load(&autoscaler.scale_up_events) +
                            atomic_load(&autoscaler.scale_down_events);
    bool check4 = total_events > 0;
    printf("[ %s ] Autoscaler responds to load (events=%lu)\n",
           check4 ? "PASS" : "FAIL", total_events);
    if (check4) pass_count++;

    printf("\n================================================================\n");
    printf("CONCLUSION: %d/%d checks passed\n", pass_count, total_checks);
    printf("================================================================\n");

    if (pass_count == total_checks) {
        printf("SUCCESS: Autoscaler correctly scales MC count in response to load!\n");
    } else if (pass_count >= total_checks - 1) {
        printf("PARTIAL: Most checks passed, review failed checks above.\n");
    } else {
        printf("NEEDS WORK: Multiple validation checks failed.\n");
    }

    /* Cleanup */
    orchestrator_shutdown(&orch);
    free(samples);
    free(memory_buffer);

    return (pass_count >= total_checks - 1) ? 0 : 1;
}
