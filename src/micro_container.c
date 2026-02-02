/*
 * Micro-Container Implementation for Linux
 * Patent: US 19/262,056
 * * This implements the actual patent architecture:
 * - Real hardware counter access via perf_event_open (Grouped + Scaled + Thread
 * Private)
 * - Real core affinity via pthread_setaffinity_np
 * - Real memory isolation via mmap with alignment
 * - Lock-free ring buffers (SPSC corrected)
 */

#define _GNU_SOURCE
#include <errno.h>
#include <linux/perf_event.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "micro_container.h"

/* ============================================================================
 * HARDWARE PERFORMANCE COUNTERS (Patent Element 310)
 * ============================================================================
 */

/* Format for PERF_FORMAT_GROUP | PERF_FORMAT_TOTAL_TIME_ENABLED |
 * PERF_FORMAT_TOTAL_TIME_RUNNING read */
struct read_format {
  uint64_t nr;
  uint64_t time_enabled;
  uint64_t time_running;
  struct {
    uint64_t value;
  } values[4]; /* Instructions, Cycles, Misses, Refs */
};

static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                            int cpu, int group_fd, unsigned long flags) {
  return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

int mc_perf_counter_init(mc_perf_counter_t *counter, int cpu) {
  struct perf_event_attr pe;

  memset(&pe, 0, sizeof(pe));
  pe.size = sizeof(pe);
  pe.disabled = 1;
  pe.exclude_kernel = 1;
  pe.exclude_hv = 1;
  pe.inherit = 0; /* Prevent counting child threads */

  /* Critical: Read total time enabled/running to detect multiplexing */
  pe.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_TOTAL_TIME_ENABLED |
                   PERF_FORMAT_TOTAL_TIME_RUNNING;

  /* * PATCH APPLIED: Use cpu=-1 to track THIS THREAD only.
   * Since the thread is pinned, we don't need to bind the perf event to the CPU
   * explicitly. This prevents counting other threads (process-wide) running on
   * the same core.
   */

  /* 1. Leader: Instructions */
  pe.type = PERF_TYPE_HARDWARE;
  pe.config = PERF_COUNT_HW_INSTRUCTIONS;
  /* cpu = -1 means "measure this thread on whatever CPU it runs on" */
  counter->fd_instructions =
      perf_event_open(&pe, 0, -1, -1, PERF_FLAG_FD_CLOEXEC);
  if (counter->fd_instructions == -1) {
    return -1;
  }

  /* 2. Follower: Cycles */
  pe.config = PERF_COUNT_HW_CPU_CYCLES;
  counter->fd_cycles = perf_event_open(&pe, 0, -1, counter->fd_instructions,
                                       PERF_FLAG_FD_CLOEXEC);
  if (counter->fd_cycles == -1) {
    close(counter->fd_instructions);
    return -1;
  }

  /* 3. Follower: Cache Misses */
  pe.config = PERF_COUNT_HW_CACHE_MISSES;
  counter->fd_cache_misses = perf_event_open(
      &pe, 0, -1, counter->fd_instructions, PERF_FLAG_FD_CLOEXEC);
  if (counter->fd_cache_misses == -1) {
    close(counter->fd_instructions);
    close(counter->fd_cycles);
    return -1;
  }

  /* 4. Follower: Cache References */
  pe.config = PERF_COUNT_HW_CACHE_REFERENCES;
  counter->fd_cache_refs = perf_event_open(&pe, 0, -1, counter->fd_instructions,
                                           PERF_FLAG_FD_CLOEXEC);
  if (counter->fd_cache_refs == -1) {
    close(counter->fd_instructions);
    close(counter->fd_cycles);
    close(counter->fd_cache_misses);
    return -1;
  }

  counter->cpu = cpu;
  return 0;
}

/* Returns 0 on success, -1 on failure */
int mc_perf_counter_start(mc_perf_counter_t *counter) {
  if (ioctl(counter->fd_instructions, PERF_EVENT_IOC_RESET,
            PERF_IOC_FLAG_GROUP) == -1)
    return -1;
  if (ioctl(counter->fd_instructions, PERF_EVENT_IOC_ENABLE,
            PERF_IOC_FLAG_GROUP) == -1)
    return -1;
  return 0;
}

void mc_perf_counter_stop(mc_perf_counter_t *counter) {
  ioctl(counter->fd_instructions, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
}

void mc_perf_counter_read(mc_perf_counter_t *counter,
                          mc_hw_metrics_t *metrics) {
  struct read_format rf;

  ssize_t n = read(counter->fd_instructions, &rf, sizeof(rf));

  if (n != sizeof(rf) || rf.nr != 4) {
    memset(metrics, 0, sizeof(*metrics));
    return;
  }

  /* Multiplexing Detection & Scaling */
  double scale = 1.0;
  metrics->multiplexed = false;

  if (rf.time_running < rf.time_enabled && rf.time_running > 0) {
    scale = (double)rf.time_enabled / (double)rf.time_running;
    metrics->multiplexed = true;
  }

  /* Map values and apply scale */
  metrics->instructions = (uint64_t)(rf.values[0].value * scale);
  metrics->cycles = (uint64_t)(rf.values[1].value * scale);
  metrics->cache_misses = (uint64_t)(rf.values[2].value * scale);
  metrics->cache_refs = (uint64_t)(rf.values[3].value * scale);

  if (metrics->cycles > 0) {
    metrics->ipc = (double)metrics->instructions / (double)metrics->cycles;
  } else {
    metrics->ipc = 0.0;
  }

  if (metrics->cache_refs > 0) {
    metrics->cache_miss_rate =
        (double)metrics->cache_misses / (double)metrics->cache_refs;
  } else {
    metrics->cache_miss_rate = 0.0;
  }
}

void mc_perf_counter_destroy(mc_perf_counter_t *counter) {
  if (counter->fd_cache_refs >= 0)
    close(counter->fd_cache_refs);
  if (counter->fd_cache_misses >= 0)
    close(counter->fd_cache_misses);
  if (counter->fd_cycles >= 0)
    close(counter->fd_cycles);
  if (counter->fd_instructions >= 0)
    close(counter->fd_instructions);
}

/* ============================================================================
 * LOCK-FREE RING BUFFER (Patent Element 150, Claim 3)
 * ============================================================================
 */

#define CACHE_LINE_SIZE 64

int mc_ring_buffer_init(mc_ring_buffer_t *rb, size_t capacity) {
  if (capacity == 0 || (capacity & (capacity - 1)) != 0) {
    return -1;
  }

  rb->capacity = capacity;
  rb->mask = capacity - 1;

  size_t buffer_size = capacity * sizeof(mc_task_t);
  buffer_size = (buffer_size + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1);

  rb->buffer = aligned_alloc(CACHE_LINE_SIZE, buffer_size);
  if (!rb->buffer) {
    return -1;
  }
  memset(rb->buffer, 0, buffer_size);

  atomic_init(&rb->head, 0);
  atomic_init(&rb->tail, 0);

  return 0;
}

bool mc_ring_buffer_push(mc_ring_buffer_t *rb, const mc_task_t *task) {
  uint64_t tail = atomic_load_explicit(&rb->tail, memory_order_relaxed);
  uint64_t head = atomic_load_explicit(&rb->head, memory_order_acquire);

  if (tail - head >= rb->capacity) {
    return false;
  }

  rb->buffer[tail & rb->mask] = *task;
  atomic_store_explicit(&rb->tail, tail + 1, memory_order_release);

  return true;
}

bool mc_ring_buffer_pop(mc_ring_buffer_t *rb, mc_task_t *task) {
  uint64_t head = atomic_load_explicit(&rb->head, memory_order_relaxed);
  uint64_t tail = atomic_load_explicit(&rb->tail, memory_order_acquire);

  if (head == tail) {
    return false;
  }

  *task = rb->buffer[head & rb->mask];
  atomic_store_explicit(&rb->head, head + 1, memory_order_release);

  return true;
}

size_t mc_ring_buffer_size(mc_ring_buffer_t *rb) {
  uint64_t head = atomic_load_explicit(&rb->head, memory_order_acquire);
  uint64_t tail = atomic_load_explicit(&rb->tail, memory_order_acquire);
  int64_t diff = (int64_t)(tail - head);
  return (diff < 0) ? 0 : (size_t)diff;
}

void mc_ring_buffer_destroy(mc_ring_buffer_t *rb) {
  if (rb->buffer) {
    free(rb->buffer);
    rb->buffer = NULL;
  }
}

/* ============================================================================
 * MICRO-CONTAINER (Patent Element 110, 200)
 * ============================================================================
 */

static void *mc_worker_thread(void *arg) {
  mc_t *mc = (mc_t *)arg;
  mc_task_t task;

  /* Pin to assigned CPU core */
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(mc->cpu_id, &cpuset);

  if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
    perror("pthread_setaffinity_np");
  }

  /* Initialize perf counters for this CPU (Now correctly measuring THIS THREAD)
   */
  mc->perf_enabled = (mc_perf_counter_init(&mc->perf, mc->cpu_id) == 0);

  while (atomic_load(&mc->running)) {
    /* Check for power-gated state (Patent Claim 7) */
    if (mc->state == MC_STATE_PAUSED) {
      usleep(1000);  /* Low-power sleep when paused */
      continue;
    }

    if (mc_ring_buffer_pop(&mc->task_queue, &task)) {

      /* Start measuring - check return code */
      bool measuring = false;
      if (mc->perf_enabled) {
        if (mc_perf_counter_start(&mc->perf) == 0) {
          measuring = true;
        }
      }

      /* Execute task */
      if (task.fn) {
        task.result = task.fn(task.arg);
      }

      /* Stop and read if we started successfully */
      if (measuring) {
        mc_perf_counter_stop(&mc->perf);
        /* Lockless metrics update */
        mc_perf_counter_read(&mc->perf, &mc->metrics);
      }

      atomic_fetch_add(&mc->tasks_completed, 1);
      atomic_fetch_sub(&mc->tasks_inflight, 1); /* Decrement inflight */

      if (task.on_complete) {
        task.on_complete(task.result, task.user_data);
      }
    } else {
#if defined(__aarch64__) || defined(__arm__)
      __asm__ __volatile__("yield" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
      __asm__ __volatile__("pause" ::: "memory");
#else
      sched_yield();
#endif
    }
  }

  if (mc->perf_enabled) {
    mc_perf_counter_destroy(&mc->perf);
  }
  return NULL;
}

int mc_init(mc_t *mc, int mc_id, int cpu_id, size_t memory_bytes) {
  memset(mc, 0, sizeof(mc_t));

  mc->id = mc_id;
  mc->cpu_id = cpu_id;
  mc->state = MC_STATE_IDLE;
  pthread_mutex_init(&mc->submit_lock, NULL);

  if (mc_ring_buffer_init(&mc->task_queue, 256) != 0) {
    return -1;
  }

  mc->memory_size = memory_bytes;
  mc->memory = mmap(NULL, memory_bytes, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (mc->memory == MAP_FAILED) {
    mc_ring_buffer_destroy(&mc->task_queue);
    return -1;
  }

  memset(mc->memory, 0, memory_bytes);

  atomic_init(&mc->running, 1);
  atomic_init(&mc->tasks_completed, 0);
  atomic_init(&mc->tasks_inflight, 0);

  if (pthread_create(&mc->thread, NULL, mc_worker_thread, mc) != 0) {
    munmap(mc->memory, mc->memory_size);
    mc_ring_buffer_destroy(&mc->task_queue);
    return -1;
  }

  mc->state = MC_STATE_RUNNING;
  return 0;
}

int mc_submit_task(mc_t *mc, mc_task_t *task) {
  if (!atomic_load(&mc->running)) {
    return -1;
  }

  /* MPSC Safety: Lock the submission side */
  pthread_mutex_lock(&mc->submit_lock);

  if (!mc_ring_buffer_push(&mc->task_queue, task)) {
    pthread_mutex_unlock(&mc->submit_lock);
    return -1; /* Queue full */
  }

  atomic_fetch_add(&mc->tasks_inflight, 1);
  pthread_mutex_unlock(&mc->submit_lock);

  return 0;
}

void mc_get_metrics(mc_t *mc, mc_hw_metrics_t *metrics) {
  /* Simple copy - racy but non-fatal for telemetry */
  *metrics = mc->metrics;
}

void mc_shutdown(mc_t *mc) {
  atomic_store(&mc->running, 0);
  pthread_join(mc->thread, NULL);

  pthread_mutex_destroy(&mc->submit_lock);
  mc_ring_buffer_destroy(&mc->task_queue);

  if (mc->memory && mc->memory != MAP_FAILED) {
    munmap(mc->memory, mc->memory_size);
  }

  mc->state = MC_STATE_STOPPED;
}

/* Dynamic Scaling: Power-gate / Un-gate MCs (Patent Claim 7) */
void mc_pause(mc_t *mc) {
  mc->state = MC_STATE_PAUSED;
}

void mc_resume(mc_t *mc) {
  mc->state = MC_STATE_RUNNING;
}

/* ============================================================================
 * ORCHESTRATOR ENGINE (Patent Element 120)
 * ============================================================================
 */

int orchestrator_init(orchestrator_t *orch, int num_cores, int mcs_per_core) {
  memset(orch, 0, sizeof(orchestrator_t));

  orch->num_cores = num_cores;
  orch->mcs_per_core = mcs_per_core;
  orch->total_mcs = num_cores * mcs_per_core;

  orch->mcs = calloc(orch->total_mcs, sizeof(mc_t));
  if (!orch->mcs) {
    return -1;
  }

  int mc_id = 0;
  for (int core = 0; core < num_cores; core++) {
    for (int i = 0; i < mcs_per_core; i++) {
      if (mc_init(&orch->mcs[mc_id], mc_id, core, 2 * 1024 * 1024) != 0) {
        for (int j = 0; j < mc_id; j++) {
          mc_shutdown(&orch->mcs[j]);
        }
        free(orch->mcs);
        return -1;
      }
      mc_id++;
    }
  }

  atomic_init(&orch->next_mc, 0);
  orch->running = true;

  /* Initialize dynamic scaling fields (Patent Claims 5, 7) */
  atomic_init(&orch->active_mcs, orch->total_mcs);  /* All active by default */
  orch->min_active_mcs = num_cores;                 /* At least 1 MC per core */
  orch->max_active_mcs = orch->total_mcs;           /* Max is total pool size */

  return 0;
}

int orchestrator_submit(orchestrator_t *orch, mc_task_t *task) {
  /* Route only to active MCs (Patent Claim 5) */
  int active = atomic_load(&orch->active_mcs);
  if (active <= 0) active = 1;  /* Safety: at least 1 MC */
  int mc_idx = atomic_fetch_add(&orch->next_mc, 1) % active;
  return mc_submit_task(&orch->mcs[mc_idx], task);
}

int orchestrator_submit_to_core(orchestrator_t *orch, int core,
                                mc_task_t *task) {
  if (core < 0 || core >= orch->num_cores) {
    return -1;
  }

  /* Improved Load Balancing: Use tasks_inflight + queue_size */
  int base = core * orch->mcs_per_core;
  int best = base;

  /* Calculate load = queue_size + inflight (approximate) */
  size_t min_load = mc_ring_buffer_size(&orch->mcs[base].task_queue) +
                    atomic_load(&orch->mcs[base].tasks_inflight);

  for (int i = 1; i < orch->mcs_per_core; i++) {
    size_t qsize = mc_ring_buffer_size(&orch->mcs[base + i].task_queue);
    int inflight = atomic_load(&orch->mcs[base + i].tasks_inflight);
    size_t load = qsize + inflight;

    if (load < min_load) {
      min_load = load;
      best = base + i;
    }
  }

  return mc_submit_task(&orch->mcs[best], task);
}

void orchestrator_get_metrics(orchestrator_t *orch, mc_hw_metrics_t *metrics,
                              int mc_idx) {
  if (mc_idx >= 0 && mc_idx < orch->total_mcs) {
    mc_get_metrics(&orch->mcs[mc_idx], metrics);
  }
}

double orchestrator_get_avg_ipc(orchestrator_t *orch) {
  double total_ipc = 0.0;
  int count = 0;

  for (int i = 0; i < orch->total_mcs; i++) {
    if (orch->mcs[i].metrics.ipc > 0) {
      total_ipc += orch->mcs[i].metrics.ipc;
      count++;
    }
  }

  return count > 0 ? total_ipc / count : 0.0;
}

void orchestrator_shutdown(orchestrator_t *orch) {
  orch->running = false;

  for (int i = 0; i < orch->total_mcs; i++) {
    mc_shutdown(&orch->mcs[i]);
  }

  free(orch->mcs);
  orch->mcs = NULL;
}

/* ============================================================================
 * DYNAMIC SCALING API (Patent Claims 5, 7, 11)
 * ============================================================================ */

int orchestrator_scale_up(orchestrator_t *orch, int count) {
  int current = atomic_load(&orch->active_mcs);
  int activated = 0;

  for (int i = current; i < orch->max_active_mcs && activated < count; i++) {
    if (orch->mcs[i].state == MC_STATE_PAUSED) {
      mc_resume(&orch->mcs[i]);
      activated++;
    }
  }

  if (activated > 0) {
    atomic_fetch_add(&orch->active_mcs, activated);
  }

  return atomic_load(&orch->active_mcs);
}

int orchestrator_scale_down(orchestrator_t *orch, int count) {
  int current = atomic_load(&orch->active_mcs);
  int deactivated = 0;

  /* Scale down from the end, only pause MCs with empty queues */
  for (int i = current - 1; i >= orch->min_active_mcs && deactivated < count; i--) {
    if (orch->mcs[i].state == MC_STATE_RUNNING) {
      /* Only pause if queue is empty to avoid task loss */
      size_t qsize = mc_ring_buffer_size(&orch->mcs[i].task_queue);
      int inflight = atomic_load(&orch->mcs[i].tasks_inflight);
      if (qsize == 0 && inflight == 0) {
        mc_pause(&orch->mcs[i]);
        deactivated++;
      }
    }
  }

  if (deactivated > 0) {
    atomic_fetch_sub(&orch->active_mcs, deactivated);
  }

  return atomic_load(&orch->active_mcs);
}

int orchestrator_get_active_mcs(orchestrator_t *orch) {
  return atomic_load(&orch->active_mcs);
}

void orchestrator_get_scaling_state(orchestrator_t *orch, mc_scaling_state_t *state) {
  int active = atomic_load(&orch->active_mcs);
  state->active_mcs = active;
  state->paused_mcs = orch->total_mcs - active;
  state->min_mcs = orch->min_active_mcs;
  state->max_mcs = orch->max_active_mcs;
  /* Note: scale events are tracked by autoscaler, not orchestrator */
  state->scale_up_events = 0;
  state->scale_down_events = 0;
}