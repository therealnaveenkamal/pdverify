# Verify-PD: Disaggregated Serving for Speculative Decoding

Verify-PD improves tail latency (p95/p99) in LLM inference by isolating the verification step of speculative decoding into a dedicated medium-priority execution lane.

## Architecture

The system implements a **three-lane scheduler** with preemptive priorities:

1. **Prefill Lane** (Lowest Priority): Processes initial prompts for both draft and verifier models
2. **Decode Lane** (Highest Priority): Generates single tokens using the draft model - latency-critical
3. **Verify Lane** (Medium Priority): Validates L draft tokens using the verifier model - compute-intensive but non-blocking

An **Acceptance-Aware Feedback Controller** dynamically adjusts the draft length (L) based on:
- Token acceptance ratio
- Verify lane queue depth
- Decode lane p95 latency

## Requirements

### Hardware
- **Development**: CPU-compatible (runs without GPU)
- **Production**: NVIDIA A100 or H100 GPU (via Modal.com, Vast.ai, etc.)

### Software
- Python 3.9+
- PyTorch 2.0+
- vLLM 0.2.7+

## Installation

```bash
# Clone and navigate to project
cd pdverify

# Install dependencies
pip install -r requirements.txt

# For GPU support, ensure CUDA is properly configured
# pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### CPU Development Mode

```bash
# Run with CPU (for testing logic without GPU)
python main.py --device cpu --draft-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --verifier-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --test-mode
```

### GPU Production Mode

```bash
# Confirm a GPU is visible
nvidia-smi

# Run on GPU with actual models
python main.py --device cuda --draft-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --verifier-model meta-llama/Llama-2-7b-hf
```

If you pass `--device cuda` on a machine without a GPU, the program now logs a warning and automatically falls back to CPU execution.

## Running Experiments

Compare baseline vs Verify-PD:

```bash
# Run comparison benchmark
python run_experiment.py --dataset sharegpt --num-requests 100 --output results/
```

This will generate:
- Latency distributions (p50, p95, p99)
- Throughput comparisons
- Queue depth analysis
- Acceptance rate statistics

## Configuration

Edit `src/utils/config.py` to adjust:
- Model paths
- Controller parameters (min/max draft length)
- Lane priorities
- Metrics collection intervals

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_scheduler.py -v
```

## Project Structure

```
pdverify/
├── src/
│   ├── scheduler/          # Three-lane scheduler
│   ├── engine/             # Speculative decoding engine
│   ├── controller/         # Feedback controller
│   ├── metrics/            # Performance tracking
│   ├── benchmark/          # Testing tools
│   └── utils/              # Utilities
├── tests/                  # Unit tests
├── main.py                 # Main entry point
└── run_experiment.py       # Experiment runner
```

## GPU Deployment

### Using Modal.com

1. Install Modal CLI: `pip install modal`
2. Set up account: `modal token new`
3. Deploy: `modal deploy modal_deploy.py`

### Using Vast.ai

1. Rent an A100/H100 instance
2. SSH into instance
3. Clone repo and run experiments

## Performance Expectations

Verify-PD aims to achieve:
- **30-50% reduction** in p95/p99 decode latency under mixed workloads
- **Stable latency** even when verify queue has backlog
- **Equivalent throughput** to baseline speculative decoding

## References

- Leviathan et al. (2023): "Fast Inference from Transformers via Speculative Decoding"
- Zhong et al. (2024): "DistServe: Disaggregating Prefill and Decoding"

## Authors

- Naveenraj Kamalakannan (nk3940)
- Megh Panandikar (mp6545)
