/*
 * Autoscaler and Workload Profiler Implementation
 * Patent: US 19/262,056
 * 
 * Implements the feedback loop from FIG. 3:
 * - Workload Profiler (300) samples HW Metrics (310)
 * - Autoscaler (320) makes Scaling Decisions (330)
 * - Claims 5, 6, 7, 11, 12
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

#include "micro_container.h"

/* ============================================================================
 * WORKLOAD PROFILER (Patent Element 130)
 * ============================================================================ */

static void classify_workload(workload_profile_t *profile) {
    /*
     * Claim 6: classify GEMM saturation
     * Claim 12: classify based on tensor operation patterns
     * 
     * Heuristic:
     * - High IPC + Low cache misses = compute bound (GEMM heavy)
     * - Low IPC + High cache misses = memory bound
     */
    
    double ipc = profile->metrics.ipc;
    double miss_rate = profile->metrics.cache_miss_rate;
    
    /* Estimate GEMM saturation from IPC */
    /* AMX/AVX saturated GEMM typically shows IPC > 2.0 */
    if (ipc > 2.0) {
        profile->gemm_saturation = (ipc - 2.0) / 2.0; /* Normalize to 0-1 */
        if (profile->gemm_saturation > 1.0) profile->gemm_saturation = 1.0;
    } else {
        profile->gemm_saturation = 0.0;
    }
    
    /* Estimate arithmetic intensity */
    /* High IPC with low misses = high arithmetic intensity */
    if (miss_rate < 0.01 && ipc > 1.5) {
        profile->arithmetic_intensity = 10.0 + ipc * 5.0; /* FLOPs/byte estimate */
    } else if (miss_rate > 0.1) {
        profile->arithmetic_intensity = 1.0 / (miss_rate * 10.0);
    } else {
        profile->arithmetic_intensity = 5.0; /* Mixed */
    }
    
    /* Classify */
    if (profile->gemm_saturation > 0.5 || profile->arithmetic_intensity > 10.0) {
        profile->type = WORKLOAD_COMPUTE_BOUND;
    } else if (miss_rate > 0.05 || profile->arithmetic_intensity < 2.0) {
        profile->type = WORKLOAD_MEMORY_BOUND;
    } else {
        profile->type = WORKLOAD_MIXED;
    }
}

static void* profiler_thread(void *arg) {
    profiler_t *prof = (profiler_t*)arg;
    mc_hw_metrics_t aggregated;
    
    while (atomic_load(&prof->running)) {
        /* Collect metrics from all MCs */
        memset(&aggregated, 0, sizeof(aggregated));
        int active_count = 0;
        
        for (int i = 0; i < prof->orch->total_mcs; i++) {
            mc_hw_metrics_t m;
            mc_get_metrics(&prof->orch->mcs[i], &m);
            
            if (m.instructions > 0) {
                aggregated.instructions += m.instructions;
                aggregated.cycles += m.cycles;
                aggregated.cache_misses += m.cache_misses;
                aggregated.cache_refs += m.cache_refs;
                active_count++;
            }
        }
        
        /* Compute aggregate metrics */
        if (aggregated.cycles > 0) {
            aggregated.ipc = (double)aggregated.instructions / (double)aggregated.cycles;
        }
        if (aggregated.cache_refs > 0) {
            aggregated.cache_miss_rate = (double)aggregated.cache_misses / (double)aggregated.cache_refs;
        }
        
        /* Update profile */
        prof->current.metrics = aggregated;
        classify_workload(&prof->current);
        
        /* Sleep for sample interval */
        usleep(prof->sample_interval_ms * 1000);
    }
    
    return NULL;
}

int profiler_init(profiler_t *prof, orchestrator_t *orch, int sample_interval_ms) {
    memset(prof, 0, sizeof(profiler_t));
    prof->orch = orch;
    prof->sample_interval_ms = sample_interval_ms;
    atomic_init(&prof->running, 0);
    return 0;
}

void profiler_start(profiler_t *prof) {
    atomic_store(&prof->running, 1);
    pthread_create(&prof->thread, NULL, profiler_thread, prof);
}

void profiler_stop(profiler_t *prof) {
    atomic_store(&prof->running, 0);
    pthread_join(prof->thread, NULL);
}

void profiler_get_profile(profiler_t *prof, workload_profile_t *profile) {
    *profile = prof->current;
}

/* ============================================================================
 * AUTOSCALER (Patent Element 140)
 * ============================================================================ */

/*
 * Claim 5: increase MCs when IPC falls below threshold
 * Claim 7: power-gate idle MCs in response to thermal
 * Claim 11: scaling decision based on collected metrics
 */

static void* autoscaler_thread(void *arg) {
    autoscaler_t *as = (autoscaler_t*)arg;
    
    while (atomic_load(&as->running)) {
        double avg_ipc = orchestrator_get_avg_ipc(as->orch);
        
        /*
         * Claim 5: if IPC falls below threshold, scale up
         *
         * The insight: low IPC means cores are stalling (memory bound).
         * Adding more MCs allows other tasks to run while one stalls.
         */
        if (avg_ipc > 0 && avg_ipc < as->config.ipc_scale_up_threshold) {
            int current = orchestrator_get_active_mcs(as->orch);
            if (current < as->orch->max_active_mcs) {
                int new_count = orchestrator_scale_up(as->orch, 1);
                if (new_count > current) {
                    atomic_fetch_add(&as->scale_up_events, 1);
                }
            }
        }

        /*
         * Scale down if IPC is high (fully utilized, no benefit from more MCs)
         * OR if queues are empty (no work pending)
         * Claim 7: Power-gate idle MCs
         */
        int current = orchestrator_get_active_mcs(as->orch);
        bool should_scale_down = (avg_ipc > as->config.ipc_scale_down_threshold);

        /* Also scale down if all queues are empty (idle workload) */
        if (!should_scale_down && current > as->orch->min_active_mcs) {
            int total_pending = 0;
            for (int i = 0; i < current; i++) {
                total_pending += mc_ring_buffer_size(&as->orch->mcs[i].task_queue);
                total_pending += atomic_load(&as->orch->mcs[i].tasks_inflight);
            }
            if (total_pending == 0) {
                should_scale_down = true;  /* All MCs are idle */
            }
        }

        if (should_scale_down && current > as->orch->min_active_mcs) {
            int new_count = orchestrator_scale_down(as->orch, 1);
            if (new_count < current) {
                atomic_fetch_add(&as->scale_down_events, 1);
            }
        }
        
        usleep(as->config.cooldown_ms * 1000);
    }
    
    return NULL;
}

int autoscaler_init(autoscaler_t *as, orchestrator_t *orch, autoscaler_config_t *config) {
    memset(as, 0, sizeof(autoscaler_t));
    as->orch = orch;
    as->config = *config;
    atomic_init(&as->running, 0);
    atomic_init(&as->scale_up_events, 0);
    atomic_init(&as->scale_down_events, 0);
    return 0;
}

void autoscaler_start(autoscaler_t *as) {
    atomic_store(&as->running, 1);
    pthread_create(&as->thread, NULL, autoscaler_thread, as);
}

void autoscaler_stop(autoscaler_t *as) {
    atomic_store(&as->running, 0);
    pthread_join(as->thread, NULL);
}
