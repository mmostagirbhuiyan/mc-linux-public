# Test Suite for Paper Reproducibility

This directory contains tests that validate claims in the paper. All tests are reproducible with build instructions below.

---

## Validated Tests (validated/)

Tests cited in the paper with quantitative results. **These reproduce paper claims.**

| Test | Paper Citation | Key Result | Platform |
|------|----------------|------------|----------|
| test_redis_workload.c | Table 15, Line 802 | 97% recovery, 67% p99 reduction (1847→612μs) | Pi 5 |
| test_memcached_workload.c | Table 15, Line 802 | 97% recovery, 69% p99 reduction (81.3→45.9μs) | Pi 5 |
| test_interrupt_sweep_v2.c | Table 19, Line 394 | 24-473μs detection @ 2.4-21% overhead sweep | Pi 5 |
| test_polling_vs_interrupt_v2.c | Table ~pollint, Line 747 | Interrupt 2-13× faster at same overhead | Pi 5 |
| test_caladan_comparison.c | Table ~corecost, Line 886 | Analytical: 25% core cost vs 1-5% | Pi 5 |
| test_interference_detection.c | Table ~detection, Line 674 | 100% detection (30/30), -27 to -43% IPC drop | Pi 5 |
| test_qos_protection.c | Line 555 | 95.6% recovery @ 2.1% overhead | Pi 5 |
| test_sustained_load.c | Table ~sustained, Line 775 | 200/200 detection, zero degradation | Pi 5 |
| test_dispatch_final.c | Table ~dispatch, Line 649 | MC 30-51% faster for sub-μs tasks | M4 |
| test_combined_detection_mba.c | Line 432 | Throttle comparison (sched_yield/usleep/hard_pause) | Pi 5 |
| test_fair_comparison.c | Line 943 | MC ≈ pthreads parity validation | Pi 5 |
| test_sampling_interval_sweep.c | Table ~sampling, Line 739 | 2ms optimal: 83% recovery @ 0.2% overhead | Pi 5 |
| test_inference_workload.c | Table ~inference, Line 503 | 92% recovery on 11-layer GEMM inference | Pi 5 |
| test_tflite_inference.c | Table ~tflite, Line 531 | 96-98% recovery on TFLite MobileNetV2 | Pi 5 |

---

## Supporting Tests (supporting/)

Infrastructure and validation tests not explicitly cited but support paper claims.

| Test | Purpose | Paper Reference |
|------|---------|-----------------|
| test_basic.c | Unit test - MC creation, task execution, perf counters | Core infrastructure |
| test_baseline_gemm.c | Benchmark baseline for all GEMM-based tests | Foundation for all interference tests |
| test_compute_interferer.c | Validates CPU-bound interferers = noise floor | Line 717 detection scope |
| test_classification_fixed.c | 100% workload classification (6× IPC separation) | Line 1074 workload differentiation |
| test_detection_from_onset.c | True reaction time (clarifies latency semantics) | Table ~onset, Line 697 |
| test_homogeneous_interference.c | 87% interferer ID, 62% RR throttle recovery | Line 1074 interferer identification |
| test_adaptive_fixed.c | Classification works, 0% throughput improvement | Establishes scope (detection ≠ scheduling) |
| test_mba.c | Intel MBA: 0% recovery validates cache vs bandwidth | Table ~intel, Line 854 |
| test_scaling_mac.c | Apple M4 scaling efficiency (1-14 workers) | Cross-platform validation |
| test_fair_comparison_mac.c | macOS parity validation (MC ≈ pthreads) | Line 943 cross-platform |
| test_autoscaler_dynamic.c | Autoscaler 4-phase validation (W2 requirement) | Line 437 autoscaler validation |

---

## Build Instructions

### Prerequisites
- Raspberry Pi 5 (ARM Cortex-A76) or Apple M4 for most tests
- AWS c5.metal (Intel Xeon) for Intel-specific tests
- Libraries: `libhiredis-dev`, `libmemcached-dev`, `tflite-runtime` (for TFLite test)

### Build Library
```bash
cd /path/to/mc-linux
make clean && make
```

### Build Validated Tests
```bash
cd tests/validated

# Most tests
gcc -O3 -march=native -std=c11 test_interference_detection.c \
    ../../lib/libmicrocontainer.a -lpthread -lm -I../../src -o test_interference

# Redis/Memcached tests (need libraries)
gcc -O3 -march=native test_redis_workload.c \
    ../../lib/libmicrocontainer.a -lpthread -lm -lhiredis -I../../src -o test_redis

# TFLite test (needs tflite-runtime Python package)
gcc -O3 -march=native test_tflite_inference.c \
    ../../lib/libmicrocontainer.a -lpthread -lm -I../../src -o test_tflite
```

### Run Tests
```bash
# Enable perf counters
sudo sysctl -w kernel.perf_event_paranoid=-1

# Run validated test
./test_interference

# Expected output: 100% detection (30/30 trials)
```

---

## Archived Tests

See `/archive/` for non-validated tests:
- `exploratory-2026-02/` - Superseded versions (v1 when v2 exists), debug utilities, flawed methodology
  - **IMPORTANT:** `FLAWED-DO-NOT-USE-test_comprehensive.c` has broken methodology (spinning workers during measurement). Results are invalid.
- `paper-investigation-2026-02/` - Failed experimental angles documented in CLAUDE.md
  - test_telemetry_overhead.c - "Zero-cost observability" (FAILED: 3-22% overhead)
  - test_adaptive_scheduling.c - "Adaptive scheduling improvement" (FAILED: 0% benefit, 47% task loss bug)
  - test_anomaly_detection.c - "Anomaly detection" (FAILED: prior art exists)
  - test_workload_classification.c - Original buggy version (FAILED: prefetcher defeated it)
- `investigation/` - Very early architecture exploration (pre-publication)

---

## Quick Reference: Paper Table → Test File Mapping

| Paper Location | Test File | Directory |
|----------------|-----------|-----------|
| Abstract Line 66 | test_redis_workload.c, test_memcached_workload.c | validated/ |
| Table 15 (Production workloads) | test_redis_workload.c, test_memcached_workload.c | validated/ |
| Table 19 (Interrupt sweep) | test_interrupt_sweep_v2.c | validated/ |
| Table ~pollint (Polling vs interrupt) | test_polling_vs_interrupt_v2.c | validated/ |
| Table ~corecost (Core cost) | test_caladan_comparison.c | validated/ |
| Table ~detection (Detection accuracy) | test_interference_detection.c | validated/ |
| Table ~onset (True reaction time) | test_detection_from_onset.c | supporting/ |
| Table ~dispatch (Dispatch overhead) | test_dispatch_final.c | validated/ |
| Table ~sampling (Interval sweep) | test_sampling_interval_sweep.c | validated/ |
| Table ~inference (ML inference) | test_inference_workload.c | validated/ |
| Table ~tflite (TFLite) | test_tflite_inference.c | validated/ |
| Table ~sustained (Sustained load) | test_sustained_load.c | validated/ |
| Line 437 (Autoscaler) | test_autoscaler_dynamic.c | supporting/ |
| Line 1074 (Interferer ID) | test_homogeneous_interference.c | supporting/ |

---

## Reproducibility Notes

### Platform-Specific Tests
- **Pi 5 only:** Most detection/interference tests (require ARM perf counters)
- **M4 only:** test_dispatch_final.c, test_scaling_mac.c, test_fair_comparison_mac.c
- **Intel only:** test_mba.c (requires Intel MBA support)

### Expected Variance
Results should match paper within ±5% due to:
- Hardware variability (cache, frequency scaling)
- OS scheduler noise
- Random number generation

### Debugging Failed Reproductions
1. Check `sudo sysctl kernel.perf_event_paranoid=-1` is set
2. Verify CPU frequency scaling disabled: `sudo cpupower frequency-set -g performance`
3. Ensure no other processes running: `top` should show idle CPUs
4. Check library rebuild: `make clean && make`

---

## Contact

For reproducibility issues, please open an issue on this repository.
