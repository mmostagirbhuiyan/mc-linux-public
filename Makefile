# Micro-Container Library for Linux
# Patent: US 19/262,056
#
# Build targets:
#   make all              - Build library and basic test
#   make test_comprehensive - Build comprehensive benchmark (for paper)
#   make test_tbb         - Build TBB comparison (requires libtbb-dev)
#   make benchmark        - Build all benchmarks
#   make run_benchmark    - Run full benchmark suite

CC = gcc
CXX = g++
CFLAGS = -Wall -Wextra -O3 -march=native -std=c11 -D_GNU_SOURCE
CXXFLAGS = -Wall -Wextra -O3 -march=native -std=c++17
LDFLAGS = -lpthread -lm

# OpenMP support (comment out if not available)
# CFLAGS += -fopenmp
# LDFLAGS += -fopenmp

SRC_DIR = src
BUILD_DIR = build
LIB_DIR = lib

SOURCES = $(SRC_DIR)/micro_container.c $(SRC_DIR)/autoscaler.c
OBJECTS = $(BUILD_DIR)/micro_container.o $(BUILD_DIR)/autoscaler.o
LIBRARY = $(LIB_DIR)/libmicrocontainer.a
SHARED = $(LIB_DIR)/libmicrocontainer.so

.PHONY: all clean test shared static benchmark run_benchmark

all: static shared test_basic

static: $(LIBRARY)

shared: $(SHARED)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(LIB_DIR):
	mkdir -p $(LIB_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c $(SRC_DIR)/micro_container.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -fPIC -c $< -o $@

$(LIBRARY): $(OBJECTS) | $(LIB_DIR)
	ar rcs $@ $(OBJECTS)

$(SHARED): $(OBJECTS) | $(LIB_DIR)
	$(CC) -shared -o $@ $(OBJECTS) $(LDFLAGS)

# Test programs - link statically to avoid LD_LIBRARY_PATH issues with sudo
test_basic: tests/test_basic.c $(LIBRARY)
	$(CC) $(CFLAGS) -I$(SRC_DIR) $< $(LIBRARY) $(LDFLAGS) -o $@

test_gemm: tests/test_gemm.c $(LIBRARY)
	$(CC) $(CFLAGS) -I$(SRC_DIR) $< $(LIBRARY) $(LDFLAGS) -o $@

test_honest: tests/test_honest.c $(LIBRARY)
	$(CC) $(CFLAGS) -I$(SRC_DIR) $< $(LIBRARY) $(LDFLAGS) -o $@

# Comprehensive benchmark (THE ONE FOR THE PAPER)
# Link statically to avoid LD_LIBRARY_PATH issues with sudo
test_comprehensive: tests/test_comprehensive.c $(LIBRARY)
	$(CC) $(CFLAGS) -I$(SRC_DIR) $< $(LIBRARY) $(LDFLAGS) -o $@
	@echo ""
	@echo "=================================================="
	@echo "Comprehensive benchmark built successfully!"
	@echo "Run with: sudo ./test_comprehensive"
	@echo "Quick test: sudo ./test_comprehensive --quick"
	@echo "=================================================="

# Stagger hypothesis test (validates thundering herd avoidance theory)
test_stagger: tests/test_stagger_hypothesis.c $(LIBRARY)
	$(CC) $(CFLAGS) -I$(SRC_DIR) $< $(LIBRARY) $(LDFLAGS) -o $@
	@echo ""
	@echo "=================================================="
	@echo "Stagger hypothesis test built successfully!"
	@echo "Run with: ./test_stagger"
	@echo "=================================================="

# TBB benchmark (requires libtbb-dev)
test_tbb: tests/test_tbb.cpp
	$(CXX) $(CXXFLAGS) $< -ltbb -lpthread -o $@
	@echo ""
	@echo "=================================================="
	@echo "TBB benchmark built successfully!"
	@echo "Run with: ./test_tbb"
	@echo "=================================================="

# Build all benchmarks (TBB is optional)
benchmark: test_comprehensive test_honest test_gemm test_stagger
	@if command -v g++ >/dev/null 2>&1 && pkg-config --exists tbb 2>/dev/null; then \
		$(MAKE) test_tbb; \
	else \
		echo "Note: TBB not available, skipping test_tbb (install libtbb-dev)"; \
	fi

test: test_basic test_gemm
	@echo "Running basic test..."
	./test_basic
	@echo ""
	@echo "Running GEMM test..."
	./test_gemm

# Run full benchmark suite (for paper data collection)
run_benchmark: benchmark
	@echo ""
	@echo "=================================================="
	@echo "RUNNING FULL BENCHMARK SUITE"
	@echo "=================================================="
	@echo ""
	@echo "--- Comprehensive Benchmark (MC vs Pthreads vs OpenMP) ---"
	sudo ./test_comprehensive
	@echo ""
	@echo "--- TBB Benchmark ---"
	./test_tbb
	@echo ""
	@echo "--- Honest Comparison ---"
	sudo ./test_honest

# Quick validation run
run_quick: benchmark
	@echo "Running quick validation..."
	sudo ./test_comprehensive --quick
	./test_tbb --quick

# Install
PREFIX ?= /usr/local

install: $(LIBRARY) $(SHARED)
	install -d $(PREFIX)/lib
	install -d $(PREFIX)/include
	install -m 644 $(LIBRARY) $(PREFIX)/lib/
	install -m 755 $(SHARED) $(PREFIX)/lib/
	install -m 644 $(SRC_DIR)/micro_container.h $(PREFIX)/include/

clean:
	rm -rf $(BUILD_DIR) $(LIB_DIR) test_basic test_gemm test_honest test_comprehensive test_tbb test_stagger

# Debug info
info:
	@echo "Sources: $(SOURCES)"
	@echo "Objects: $(OBJECTS)"
	@echo "Library: $(LIBRARY)"
	@echo ""
	@echo "Build comprehensive benchmark: make test_comprehensive"
	@echo "Build TBB benchmark:           make test_tbb"
	@echo "Build all benchmarks:          make benchmark"
	@echo "Run full suite:                make run_benchmark"
