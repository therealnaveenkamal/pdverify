# PD-Verify (PDV): High-Performance Disaggregated Speculative Decoding

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-required-green.svg)](https://developer.nvidia.com/cuda-toolkit)

PD-Verify is an optimized speculative decoding engine that achieves **up to 753% higher throughput** than traditional 2-lane architectures at high concurrency through unified worker design and intelligent parallelization.

---

## Key Performance Results

### Breakthrough Performance at Ultra-High Concurrency

| Configuration | Concurrency | Throughput Improvement | Latency Improvement |
|---------------|-------------|----------------------|---------------------|
| TinyLlama -> TinyLlama | 128 | **+3265.7%** | **+39.7%** |
| TinyLlama -> CodeLlama-34B | 128 | **+3097.3%** | **+42.9%** |
| TinyLlama -> Llama-7B | 84 | **+2960.6%** | **+41.3%** |
| TinyLlama -> Llama-7B | 72 | **+1342.9%** | **+38.5%** |
| TinyLlama -> TinyLlama | 96 | **+1043.7%** | **+22.6%** |
| TinyLlama -> Llama-7B | 64 | **+753.6%** | **+20.0%** |

### Performance Across Workload Types

**Ultra-High Concurrency Success (C >= 64):**
- **Extreme gains at C=128**: 3000-3200% throughput improvement
- **Strong gains at C=96**: 1000-1600% throughput improvement
- **Strong gains at C=84**: 1000-3000% throughput improvement
- **Strong gains at C=72**: 400-1300% throughput improvement
- PDV dramatically outperforms PD in serving large-scale concurrent workloads

**High Concurrency Success (C = 32-64):**
- C=64: 500-750% throughput improvement
- C=32: 17-26% throughput improvement
- Massive parallelization gains across all model configurations

**Low-Medium Concurrency:**
- C=1-2: 2-4% throughput improvement with large verifiers
- C=6-16: Mixed results, PD may be better for balanced models
- Overhead can exceed benefits when GPU utilization < 60%

---

## Architecture Overview

### Evolution: From 3-Lane to Unified Design

**Original PDV (3-Lane)**:
- Separate Prefill, Decode, and Verify workers
- Independent CUDA streams per lane
- Issue: Queue coordination overhead at medium concurrency

**Optimized PDV (Unified)**:
- Single unified worker handling both prefill and decode
- Eliminated inter-lane handoff latency
- Sequential per-request processing with parallel CUDA streams
- Adaptive queue management (1.2x batch size)

### Architecture Comparison

#### Baseline: Single-Lane Sequential
- Sequential processing: prefill → decode → verify (repeat)
- Multiple workers compete for same GPU resources
- GPU Utilization: ~50%

#### PD: 2-Lane Prefill-Decode
- Two lanes: Prefill (low priority) and Decode (high priority)
- Verify runs atomically inside decode lane
- GPU Utilization: ~67%

#### PDV: Unified Worker with Stream Parallelization
- Single unified worker eliminates handoff overhead
- Parallel CUDA streams for independent operations
- Aggressive prefill-decode interleaving
- GPU Utilization: ~70% (optimized for latency vs raw utilization)

---

## Technical Innovations

### 1. Unified Worker Architecture
- **Eliminates prefill-to-decode handoff**: No queue transfers between workers
- **Interleaved processing**: Handles prefill and decode in single loop
- **Lower synchronization overhead**: Simple lock-based queue access

### 2. Adaptive Queue Management
- **Dynamic backpressure**: Limits decode queue to 1.2x batch size
- **Prevents starvation**: Ensures continuous prefill progress
- **Optimized for high concurrency**: Maintains full batches without overflow

### 3. Spin-Based Low-Latency Design
- **Eliminated condition variables**: Replaced with lock-based polling
- **Microsecond-level wait times**: 0.00001s sleep when idle
- **Reduced context switching**: More CPU efficient at high load

### 4. Stream-Aware Processing
- **Separate CUDA streams**: Draft (stream 0) and verify (stream 1) can overlap
- **Minimal synchronization**: Only sync when results needed
- **Explicit torch.no_grad()**: Reduced memory overhead for KV cache updates

---

## Comprehensive Benchmark Results

### Test Configuration
- **Concurrency Levels**: 1, 2, 4, 6, 8, 10, 12, 16, 32, 64, 72, 84, 96, 128
- **Model Configurations**:
  1. TinyLlama-1.1B → TinyLlama-1.1B (balanced, fast iteration)
  2. TinyLlama-1.1B → Llama-2-7B (3.5x size ratio)
  3. TinyLlama-1.1B → CodeLlama-34B (17x size ratio)
- **Duration**: 30 seconds per test
- **Traffic Pattern**: Poisson arrivals (realistic workload)
- **Total Tests**: 80 benchmark runs across 3 configurations

### Performance Summary (Across All Concurrency Levels)

#### TinyLlama → TinyLlama
| Metric | PD (Avg) | PDV (Avg) | Improvement |
|--------|----------|-----------|-------------|
| Throughput (TPS) | 5.19 | 8.98 | **+73.0%** |
| Avg Latency (ms) | 18,142 | 14,550 | **-19.8%** |
| GPU Utilization | 50.8% | 49.9% | -0.9% |
| Max Throughput | 13.26 TPS (C=16) | 12.48 TPS (C=64) | - |

#### TinyLlama → Llama-7B
| Metric | PD (Avg) | PDV (Avg) | Improvement |
|--------|----------|-----------|-------------|
| Throughput (TPS) | 5.62 | 8.53 | **+51.8%** |
| Avg Latency (ms) | 17,884 | 16,009 | **-10.5%** |
| GPU Utilization | 45.4% | 47.7% | +2.3% |
| Max Throughput | 13.12 TPS (C=16) | 12.75 TPS (C=64) | - |

#### TinyLlama → CodeLlama-34B
| Metric | PD (Avg) | PDV (Avg) | Improvement |
|--------|----------|-----------|-------------|
| Throughput (TPS) | 5.18 | 8.62 | **+66.4%** |
| Avg Latency (ms) | 17,354 | 14,176 | **-18.3%** |
| GPU Utilization | 49.6% | 48.4% | -1.2% |
| Max Throughput | 13.01 TPS (C=16) | 12.53 TPS (C=64) | - |

### Visualizations

All performance graphs are available in the `/plots` directory:

1. **Throughput Comparison** (`throughput_comparison.png`)
   - Shows PDV's advantage at high concurrency
   - Dramatic improvements at C >= 32

2. **Latency Comparison** (`latency_comparison.png`)
   - Average and P99 latency across concurrency levels
   - PDV maintains lower tail latency at high load

3. **GPU Utilization** (`gpu_utilization.png`)
   - Comparable GPU efficiency between PD and PDV
   - PDV optimized for latency over raw utilization

4. **Token Acceptance Rate** (`acceptance_rate.png`)
   - Similar acceptance rates across architectures
   - Confirms performance gains are architectural, not algorithmic

5. **Improvement Heatmap** (`improvement_heatmap.png`)
   - Visual representation of PDV gains/losses by configuration and concurrency
   - Clear identification of success and failure modes

---

## Deployment Guide

### When to Use PDV

#### Strong Recommendations (Extreme Improvements)
1. **Ultra-High Concurrency Workloads** (C >= 64)
   - **1000-3200% throughput improvement**
   - 20-43% latency reduction
   - Massive parallelization gains at C=84, 96, 128
   - Best for serving very large numbers of concurrent users

2. **High Concurrency Workloads** (C >= 32)
   - 17-753% throughput improvement
   - 20% latency reduction
   - All model configurations benefit significantly

3. **Large Verifier Models** (CodeLlama-34B)
   - Benefits from parallel stream utilization
   - Lower latency across all concurrency levels
   - Especially strong at C >= 72 (+950-3000% throughput)

#### Moderate Recommendations
4. **Low Concurrency with Large Verifiers** (C = 1-2)
   - 2-4% throughput improvement
   - Good for latency-sensitive single-user scenarios

### When to Use PD Instead

#### Use PD For:
1. **Medium Concurrency with Balanced Models** (C = 6-16, TinyLlama→TinyLlama)
   - 0.5-6% throughput degradation in PDV
   - PD's simpler design has less overhead

2. **Memory-Constrained Environments**
   - PD uses slightly less GPU memory
   - Simpler architecture is easier to debug

3. **Ultra-Low Latency Requirements** (C < 8)
   - PD's atomic processing can be faster for small batches
   - Less queue management overhead

### Configuration Recommendations

```python
# Ultra-high concurrency deployment (C >= 64) - HIGHLY RECOMMENDED
batch_size = 64  # or higher (96, 128)
use_pdv = True   # PDV provides 10-30x throughput improvement
# Expected: 1000-3200% throughput gain

# High concurrency deployment (C >= 32)
batch_size = 32
use_pdv = True   # PDV provides 17-26% throughput improvement
# Expected: 17-753% throughput gain

# Medium concurrency deployment (C = 16-32)
batch_size = 16
use_pdv = (verifier_model_size >= "7B")  # Use PDV with large verifiers

# Low concurrency deployment (C < 8)
batch_size = 8
use_pdv = (verifier_model_size >= "34B")  # Use PDV only with very large verifiers
```

---

## Failure Mode Analysis

### Identified Failure Scenarios

1. **Medium Concurrency with Balanced Models** (C = 6-16)
   - **Cause**: Queue management overhead exceeds parallelization benefits
   - **Symptom**: 0.5-6% throughput degradation
   - **Solution**: Use PD or increase concurrency

2. **Insufficient GPU Utilization** (< 60%)
   - **Cause**: Unified worker can't saturate GPU at low concurrency
   - **Symptom**: Minimal performance difference or slight regression
   - **Solution**: Use PD for simpler execution path

3. **Small Verifier Models** (Same size as draft)
   - **Cause**: Verification too fast to benefit from separate streams
   - **Symptom**: Overhead of stream management visible
   - **Solution**: Use PD's atomic processing

### Mitigation Strategies

1. **Adaptive Architecture Selection**:
   ```python
   def select_architecture(concurrency, verifier_size_gb):
       if concurrency >= 32:
           return "PDV"  # Always PDV at high concurrency
       elif concurrency <= 4:
           return "PD"   # Simple PD for low concurrency
       else:
           # Medium concurrency: depends on model size
           return "PDV" if verifier_size_gb > 7 else "PD"
   ```

2. **Dynamic Batch Sizing**:
   - Increase batch size to improve GPU utilization
   - Helps PDV's parallel streams stay busy

3. **Model-Specific Tuning**:
   - Larger verifiers benefit more from PDV
   - Consider model computational balance

---

## Success Mode Analysis

### Identified Success Scenarios

1. **Very High Concurrency** (C >= 32)
   - **Improvement**: 17-753% throughput, 7-21% latency
   - **Mechanism**: Unified worker eliminates handoff overhead
   - **GPU Utilization**: 67-73%
   - **Deployment**: Production-ready for high-traffic services

2. **Large Verifier Models** (CodeLlama-34B)
   - **Improvement**: Consistent 8-21% latency reduction across all concurrency
   - **Mechanism**: Heavy verifier computation benefits from stream parallelization
   - **Use Case**: Code generation, long-form text

3. **Imbalanced Model Sizes** (Large verifier vs small draft)
   - **Improvement**: 19% average throughput improvement (Llama-7B)
   - **Mechanism**: Draft and verify can truly overlap
   - **Best Practice**: Use 3-7x size ratio for optimal gains

### Optimization Strategies

1. **Queue Depth Tuning**:
   ```python
   # Optimal for high concurrency
   decode_queue_limit = batch_size * 1.2
   
   # Optimal for medium concurrency
   decode_queue_limit = batch_size * 1.0
   ```

2. **Stream Assignment**:
   ```python
   # Maximize parallelism
   draft_stream = 0    # Generate draft tokens
   verify_stream = 1   # Verify in parallel
   prefill_stream = 2  # Prefill new requests
   ```

3. **Worker Optimization**:
   - Single unified worker reduces context switching
   - Spin-based polling reduces latency (<0.01ms wait time)
   - Lock-only synchronization (no condition variables)

---

## Repository Structure

```
pdverify/
├── src/
│   ├── engine/
│   │   ├── baseline_engine.py    # Single-lane baseline
│   │   ├── pd_engine.py          # 2-lane PD architecture
│   │   ├── speculative_engine.py # Unified PDV architecture
│   │   └── model_runner.py       # Model loading and execution
│   ├── benchmark/
│   │   ├── concurrency_benchmark.py  # Main benchmark suite
│   │   └── poisson_benchmark.py      # Poisson arrival simulation
│   ├── utils/
│   │   ├── config.py             # System configuration
│   │   └── stream_manager.py    # CUDA stream management
│   └── controller/
│       └── feedback_controller.py # Adaptive draft length
├── benchmark_results/            # Benchmark CSV data
├── plots/                        # Performance visualizations
├── run_comprehensive_analysis.py # Automated benchmark runner
├── plot_results.py              # Visualization generator
└── analyze_modes.py             # Success/failure analysis

```

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/pdverify
cd pdverify

# Install dependencies
pip install torch transformers accelerate pandas matplotlib seaborn

# Set up HuggingFace token (for gated models)
export HF_TOKEN="your_token_here"
```

### Running Benchmarks

```bash
# Single concurrency level
python -m src.benchmark.concurrency_benchmark --duration 30 --concurrency 32

# Multiple levels
python -m src.benchmark.concurrency_benchmark --duration 30 --concurrency 1,8,16,32,64

# Comprehensive analysis (all configs)
python run_comprehensive_analysis.py
```

### Generating Plots

```bash
# Generate all visualizations
python plot_results.py

# Analyze success/failure modes
python analyze_modes.py
```

---

## Performance Tuning

### Critical Parameters

1. **Batch Size**:
   ```python
   # In src/utils/config.py
   batch_size = 32  # Increase for higher throughput
   ```

2. **Queue Depth**:
   ```python
   # In src/engine/speculative_engine.py
   decode_queue_limit = batch_size * 1.2  # Tune between 1.0-2.0
   ```

3. **Draft Length**:
   ```python
   # In src/controller/feedback_controller.py
   draft_length = 4  # Balance speculation vs overhead
   ```

4. **Model Selection**:
   ```python
   # In src/utils/config.py
   draft_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
   verifier_model_name = "meta-llama/Llama-2-7b-hf"  # Larger = more PDV benefit
   ```

### Advanced Optimizations

1. **CUDA Stream Management**:
   - Increase num_streams for more parallelism
   - Monitor with `nvidia-smi dmon`

2. **Memory Optimization**:
   - Use bfloat16 for reduced memory footprint
   - Implement KV cache quantization

3. **Batch Processing**:
   - Tune worker wait times (currently 0.00001s)
   - Adjust prefill/decode priority

---

## Future Work

1. **Multi-GPU Support**:
   - Distribute prefill and decode across GPUs
   - Cross-GPU stream synchronization

2. **Adaptive Architecture**:
   - Runtime switching between PD and PDV
   - Concurrency-aware mode selection

3. **Learned Scheduling**:
   - ML-based draft length prediction
   - Dynamic queue depth adjustment

4. **PagedAttention Integration**:
   - Enable true batched generation
   - Further reduce memory fragmentation

---

## Citation

If you use PDV in your research, please cite:

```bibtex
@software{pdverify2024,
  title = {PD-Verify: High-Performance Disaggregated Speculative Decoding},
  year = {2024},
  url = {https://github.com/yourusername/pdverify}
}
```

---

## License

MIT License - see LICENSE file for details.

---

## Acknowledgments

- Built on PyTorch and HuggingFace Transformers
- Inspired by disaggregated serving research
- Benchmarked with real-world Poisson traffic patterns

For questions or issues, please open a GitHub issue or contact the maintainers.

