#!/usr/bin/env python3
"""
Simple TFLite inference runner - outputs latencies per line.
Usage: python3 run_inference.py <model.tflite> <num_inferences>
"""

import sys
import time
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    print("ERROR: tflite-runtime not installed", file=sys.stderr)
    sys.exit(1)

def main():
    if len(sys.argv) < 3:
        print("Usage: run_inference.py <model.tflite> <num_inferences>", file=sys.stderr)
        sys.exit(1)

    model_path = sys.argv[1]
    num_inferences = int(sys.argv[2])

    # Load model
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input details
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    # Prepare input
    if input_dtype == np.float32:
        input_data = np.random.rand(*input_shape).astype(np.float32)
    else:
        input_data = np.random.randint(0, 256, size=input_shape, dtype=input_dtype)

    # Warmup
    for _ in range(5):
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

    # Run inference loop
    for _ in range(num_inferences):
        start = time.perf_counter_ns()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        end = time.perf_counter_ns()
        latency_us = (end - start) / 1000.0
        print(f"INF:{latency_us:.1f}", flush=True)

    print("DONE", flush=True)

if __name__ == "__main__":
    main()
