/*
 * Micro-Container Architecture Header
 * Patent: US 19/262,056
 * * Structures map directly to patent elements:
 * - mc_t: Micro-Container (110, 200)
 * - mc_ring_buffer_t: Communication Layer (150)
 * - mc_perf_counter_t: Hardware Metrics (310)
 * - orchestrator_t: Orchestrator Engine (120)
 * - autoscaler_t: Autoscaler (140)
 */

#ifndef MICRO_CONTAINER_H
#define MICRO_CONTAINER_H

#include <pthread.h>
#include <stdalign.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * TASK DEFINITION
 * ============================================================================
 */

typedef void *(*mc_task_fn)(void *arg);
typedef void (*mc_completion_fn)(void *result, void *user_data);

typedef struct {
  mc_task_fn fn;                /* Function to execute */
  void *arg;                    /* Argument to function */
  void *result;                 /* Result storage */
  mc_completion_fn on_complete; /* Completion callback */
  void *user_data;              /* User context */
  uint64_t task_id;             /* Unique task ID */
  uint64_t submit_time_ns;      /* Submission timestamp */
} mc_task_t;

/* ============================================================================
 * HARDWARE METRICS (Patent Element 310)
 * ============================================================================
 */

typedef struct {
  uint64_t instructions;
  uint64_t cycles;
  uint64_t cache_misses;
  uint64_t cache_refs;
  double ipc; /* Instructions per cycle */
  double cache_miss_rate;
  bool multiplexed; /* True if PMU scaling occurred */
} mc_hw_metrics_t;

typedef struct {
  int fd_instructions;
  int fd_cycles;
  int fd_cache_misses;
  int fd_cache_refs;
  int cpu;
} mc_perf_counter_t;

/* ============================================================================
 * LOCK-FREE RING BUFFER (Patent Element 150, Claim 3)
 * ============================================================================
 */

typedef struct {
  mc_task_t *buffer;
  size_t capacity;
  size_t mask;

  /* Separate cache lines for head/tail to prevent false sharing */
  alignas(64) atomic_uint_fast64_t head;
  alignas(64) atomic_uint_fast64_t tail;
} mc_ring_buffer_t;

int mc_ring_buffer_init(mc_ring_buffer_t *rb, size_t capacity);
bool mc_ring_buffer_push(mc_ring_buffer_t *rb, const mc_task_t *task);
bool mc_ring_buffer_pop(mc_ring_buffer_t *rb, mc_task_t *task);
size_t mc_ring_buffer_size(mc_ring_buffer_t *rb);
void mc_ring_buffer_destroy(mc_ring_buffer_t *rb);

/* ============================================================================
 * MICRO-CONTAINER (Patent Element 110, 200)
 * ============================================================================
 */

typedef enum {
  MC_STATE_IDLE,
  MC_STATE_RUNNING,
  MC_STATE_PAUSED,
  MC_STATE_STOPPED
} mc_state_t;

typedef struct {
  int id;
  int cpu_id; /* Pinned CPU core */
  mc_state_t state;

  /* Task Queue (Patent Element 210) */
  mc_ring_buffer_t task_queue;

  /* MPSC Protection (Reviewer Fix) */
  pthread_mutex_t submit_lock;

  /* Memory Slice (Patent Element 220) */
  void *memory;
  size_t memory_size;

  /* Performance monitoring */
  mc_perf_counter_t perf;
  mc_hw_metrics_t metrics;

  /* Worker thread */
  pthread_t thread;
  atomic_bool running;
  atomic_uint_fast64_t tasks_completed;
  atomic_int tasks_inflight; /* For Load Balancing */
  bool perf_enabled;         /* Flag to avoid invalid ioctls */
} mc_t;

int mc_init(mc_t *mc, int mc_id, int cpu_id, size_t memory_bytes);
int mc_submit_task(mc_t *mc, mc_task_t *task);
void mc_get_metrics(mc_t *mc, mc_hw_metrics_t *metrics);
void mc_shutdown(mc_t *mc);

/* Dynamic Scaling: Power-gate / Un-gate MCs (Patent Claim 7) */
void mc_pause(mc_t *mc);   /* Power-gate: stop polling, preserve thread */
void mc_resume(mc_t *mc);  /* Un-gate: resume polling */

/* ============================================================================
 * PERF COUNTER API
 * ============================================================================
 */

int mc_perf_counter_init(mc_perf_counter_t *counter, int cpu);
int mc_perf_counter_start(mc_perf_counter_t *counter);
void mc_perf_counter_stop(mc_perf_counter_t *counter);
void mc_perf_counter_read(mc_perf_counter_t *counter, mc_hw_metrics_t *metrics);
void mc_perf_counter_destroy(mc_perf_counter_t *counter);

/* ============================================================================
 * ORCHESTRATOR ENGINE (Patent Element 120)
 * ============================================================================
 */

typedef struct {
  mc_t *mcs;
  int num_cores;
  int mcs_per_core;
  int total_mcs;

  atomic_int next_mc; /* Round-robin counter */
  bool running;

  /* Dynamic Scaling (Patent Claim 5, 7) */
  atomic_int active_mcs;  /* Currently active MCs (for routing) */
  int min_active_mcs;     /* Lower bound (floor) */
  int max_active_mcs;     /* Upper bound (ceiling, same as total_mcs) */
} orchestrator_t;

int orchestrator_init(orchestrator_t *orch, int num_cores, int mcs_per_core);
int orchestrator_submit(orchestrator_t *orch, mc_task_t *task);
int orchestrator_submit_to_core(orchestrator_t *orch, int core,
                                mc_task_t *task);
void orchestrator_get_metrics(orchestrator_t *orch, mc_hw_metrics_t *metrics,
                              int mc_idx);
double orchestrator_get_avg_ipc(orchestrator_t *orch);
void orchestrator_shutdown(orchestrator_t *orch);

/* Dynamic Scaling API (Patent Claims 5, 7, 11) */
int orchestrator_scale_up(orchestrator_t *orch, int count);
int orchestrator_scale_down(orchestrator_t *orch, int count);
int orchestrator_get_active_mcs(orchestrator_t *orch);

typedef struct {
  int active_mcs;
  int paused_mcs;
  int min_mcs;
  int max_mcs;
  uint64_t scale_up_events;   /* Populated from autoscaler if available */
  uint64_t scale_down_events; /* Populated from autoscaler if available */
} mc_scaling_state_t;

void orchestrator_get_scaling_state(orchestrator_t *orch, mc_scaling_state_t *state);

/* ============================================================================
 * AUTOSCALER (Patent Element 140, Claims 5, 7, 11)
 * ============================================================================
 */

typedef struct {
  double ipc_scale_up_threshold;   /* Scale up when IPC below this */
  double ipc_scale_down_threshold; /* Scale down when IPC above this */
  double cache_miss_threshold;     /* React to cache pressure */
  int min_mcs_per_core;
  int max_mcs_per_core;
  int cooldown_ms;
} autoscaler_config_t;

typedef struct {
  orchestrator_t *orch;
  autoscaler_config_t config;

  pthread_t thread;
  atomic_bool running;

  /* Stats */
  atomic_uint_fast64_t scale_up_events;
  atomic_uint_fast64_t scale_down_events;
} autoscaler_t;

int autoscaler_init(autoscaler_t *as, orchestrator_t *orch,
                    autoscaler_config_t *config);
void autoscaler_start(autoscaler_t *as);
void autoscaler_stop(autoscaler_t *as);

/* ============================================================================
 * WORKLOAD PROFILER (Patent Element 130, Claims 6, 12)
 * ============================================================================
 */

typedef enum {
  WORKLOAD_COMPUTE_BOUND,
  WORKLOAD_MEMORY_BOUND,
  WORKLOAD_MIXED
} workload_type_t;

typedef struct {
  mc_hw_metrics_t metrics;
  workload_type_t type;
  double gemm_saturation; /* Claim 6 */
  double arithmetic_intensity;
} workload_profile_t;

typedef struct {
  orchestrator_t *orch;
  workload_profile_t current;

  pthread_t thread;
  atomic_bool running;
  int sample_interval_ms;
} profiler_t;

int profiler_init(profiler_t *prof, orchestrator_t *orch,
                  int sample_interval_ms);
void profiler_start(profiler_t *prof);
void profiler_stop(profiler_t *prof);
void profiler_get_profile(profiler_t *prof, workload_profile_t *profile);

#ifdef __cplusplus
}
#endif

#endif /* MICRO_CONTAINER_H */