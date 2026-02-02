# Micro-Container Interference Detection System

Reference implementation for the paper: **"Per-Task Interference Detection Without Dedicated Cores: A Micro-Containerized Architecture with Hardware Telemetry Integration"**

---

## Overview

This repository contains the complete implementation and experimental validation code for an interrupt-driven interference detection system that identifies cache contention at per-task granularity using hardware performance counters.

**Key Results:**
- 100% detection accuracy for cache-intensive interference
- 1-5% overhead versus raw pthreads
- 84-97% latency recovery through software throttling
- 67-69% p99 tail latency reduction for Redis/Memcached

---

## Quick Start

### Prerequisites

**Raspberry Pi 5 (primary platform):**
```bash
sudo apt install build-essential libhiredis-dev libmemcached-dev
```

**Apple M4 Pro (cross-platform):**
```bash
brew install hiredis libmemcached
```

### Build and Run

```bash
# Build library
make clean && make

# Enable performance counters
sudo sysctl -w kernel.perf_event_paranoid=-1

# Run core interference detection test
cd tests/validated
gcc -O3 -march=native test_interference_detection.c \
    ../../lib/libmicrocontainer.a -lpthread -lm -I../../src -o test_interference
./test_interference
```

**Expected output:** 100% detection rate (30/30 trials), IPC degradation -27% to -43%

---

## Repository Structure

```
├── src/                          # Core implementation
│   ├── micro_container.h         # API definitions
│   ├── micro_container.c         # MC, orchestrator, ring buffer
│   └── autoscaler.c              # Autoscaler and profiler
├── tests/
│   ├── validated/                # 14 tests cited in paper
│   │   ├── test_interference_detection.c
│   │   ├── test_redis_workload.c
│   │   └── ...
│   └── supporting/               # 11 infrastructure tests
│       ├── test_basic.c
│       ├── test_autoscaler_dynamic.c
│       └── ...
└── README.md
```

---

## Reproducing Paper Results

All validated tests map directly to paper tables and claims. See [`tests/README.md`](tests/README.md) for complete test-to-paper mapping.

### Key Tests

| Paper Table | Test File | Expected Result |
|-------------|-----------|-----------------|
| Table 15 | test_redis_workload.c | 97% recovery, 67% p99 reduction |
| Table 19 | test_interrupt_sweep_v2.c | 24-473μs detection range |
| Table ~detection | test_interference_detection.c | 100% detection (30/30 trials) |
| Table ~dispatch | test_dispatch_final.c | 30-51% throughput advantage |

### Build Instructions

```bash
# Build all validated tests
cd tests/validated
for test in test_interference_detection test_qos_protection test_sustained_load; do
  gcc -O3 -march=native -std=c11 ${test}.c \
      ../../lib/libmicrocontainer.a -lpthread -lm -I../../src -o $test
done

# Tests requiring additional libraries
gcc -O3 -march=native test_redis_workload.c \
    ../../lib/libmicrocontainer.a -lpthread -lm -lhiredis -I../../src -o test_redis

gcc -O3 -march=native test_memcached_workload.c \
    ../../lib/libmicrocontainer.a -lpthread -lm -lmemcached -I../../src -o test_memcached
```

---

## Platform Requirements

**Validated platforms:**
- ARM Cortex-A76 (Raspberry Pi 5) - Primary validation
- Intel Xeon Platinum 8275CL (AWS c5.metal) - Cross-platform validation
- Apple M4 Pro (14 cores) - Cross-platform validation

**Dependencies:**
- Linux kernel with perf_event support
- GCC with C11 support
- libhiredis-dev, libmemcached-dev (for Redis/Memcached tests)
- Python 3 with tflite-runtime (for TFLite test)

---

## Troubleshooting

**Permission denied for perf counters:**
```bash
sudo sysctl -w kernel.perf_event_paranoid=-1
```

**Missing libraries:**
```bash
sudo apt install libhiredis-dev libmemcached-dev
```

**High result variance:**
```bash
# Disable CPU frequency scaling
sudo cpupower frequency-set -g performance
```

---

## System Architecture

The system consists of:
1. **Orchestrator** - Manages micro-container pool and task distribution
2. **Micro-Containers (MCs)** - Lightweight execution contexts pinned to CPU cores
3. **SPSC Ring Buffers** - Lock-free task queues per worker
4. **PMU Integration** - Hardware performance counter sampling via perf_event_open()
5. **Autoscaler** - Dynamic MC scaling based on IPC feedback

See paper for detailed architecture description and design rationale.

---

## Citation

If you use this code or build upon this work, please cite:

```bibtex
@article{bhuiyan2026microcontainer,
  title={Per-Task Interference Detection Without Dedicated Cores:
         A Micro-Containerized Architecture with Hardware Telemetry Integration},
  author={Bhuiyan, M. Mostagir},
  journal={Under Review},
  year={2026}
}
```

---

## Contact

For questions about reproducing results or implementation details, please open an issue on this repository.
