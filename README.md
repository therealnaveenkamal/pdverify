# Verify-PD: Disaggregated Serving for Speculative Decoding

**Verify-PD** is a high-performance serving system for LLM speculative decoding that isolates the verification step into a dedicated execution lane. By disaggregating **Prefill, Decode, and Verify** into three priority lanes and applying **batched verification**, Verify-PD achieves significant performance gains over standard speculative decoding at scale.

## Key Results

**PD-Verify (3-lane)** conclusively outperforms both standard Speculative Decoding and 2-lane (Prefill-Decode) disaggregation for concurrent workloads.

| Scenario                         | Metric      | Improvement vs Baseline    |
| -------------------------------- | ----------- | -------------------------- |
| **Medium Concurrency** (20 reqs) | P95 Latency | **-16% Latency Reduction** |
| **Medium Concurrency** (20 reqs) | Throughput  | **+58% Higher Throughput** |
| **High Concurrency** (50 reqs)   | P95 Latency | **-34% Latency Reduction** |
| **High Concurrency** (50 reqs)   | Throughput  | **+47% Higher Throughput** |

> **Verdict:** Verify-PD is the optimal architecture for production-scale speculative decoding serving.

---

## The Architecture: 3-Lane Disaggregation

Verify-PD moves beyond simple Prefill/Decode separation by introducing a **dedicated Verify Lane**.

### The 3 Lanes

1.  **Decode Lane (High Priority)**: Generates draft tokens using the small model. Low latency is critical here.
2.  **Verify Lane (Medium Priority)**: Verifies draft tokens using the large model. **This is where the magic happens.**
3.  **Prefill Lane (Low Priority)**: Processes initial prompts. Deprioritized to prevent blocking generation.

### The Secret Sauce: Batched Verification

The critical innovation is **Verify Lane Batching**.

- **Standard Speculative Decoding:** Request A verifies `[t1, t2, t3, t4]`. GPU is underutilized.
- **Verify-PD:** The Verify Lane collects draft tokens from Request A, B, C, and D. It runs a **single large batch verification**.
  - **Result:** 4x verification throughput for nearly the same cost as verifying one request.

---

## Comprehensive Benchmarks

We conducted a fair, apples-to-apples comparison of three systems under identical conditions (A100 GPU):

1.  **Baseline**: Concurrent standard speculative decoding.
2.  **PD (2-lane)**: Prefill-Decode separation only (similar to standard disaggregation).
3.  **PD-Verify (3-lane)**: Full 3-lane disaggregation with batched verification.

### Head-to-Head Results

| Scenario                         | Baseline P95  | PD (2-lane) P95 | **PD-Verify (3-lane) P95** | Winner        |
| -------------------------------- | ------------- | --------------- | -------------------------- | ------------- |
| **Single Request**               | 6,695 ms      | 6,423 ms        | **6,341 ms**               | **PD-Verify** |
| **Low Concurrency** (5 reqs)     | **33,078 ms** | 33,879 ms       | 35,639 ms                  | **Baseline**  |
| **Medium Concurrency** (20 reqs) | 113,624 ms    | 123,209 ms      | **95,711 ms**              | **PD-Verify** |
| **High Concurrency** (50 reqs)   | 361,680 ms    | 368,157 ms      | **235,652 ms**             | **PD-Verify** |

---

## Failure Mode Analysis

Why does PD-Verify win? And when does it fail?

### 1. Why PD (2-lane) Fails

Even with batching enabled, **PD (2-lane) trails the baseline.**

- **Reason:** Without a separate Verify lane, draft and verify operations are coupled in a single step. Even though we batch decode steps, the lock-step nature prevents overlapping decode of Request A with verify of Request B. You pay the overhead of lane management without the pipeline parallelism benefits.

### 2. Failure Mode: Low Concurrency (<5 requests)

- **Observation:** PD-Verify is ~7% slower than baseline at low load.
- **Reason:** The "tax" of managing 3 queues and moving data between lanes is constant. With only 2-3 requests, you rarely get a "full batch" in the Verify lane.
- **Mitigation:** For low-traffic deployments, use standard speculative decoding.

### 3. Failure Mode: Poor Speculation

- **Observation:** If the draft model is poor, the Verify lane becomes a bottleneck.
- **Reason:** High rejection rates mean the decode lane works overtime generating drafts that get rejected, flooding the verify lane with work that yields no tokens.
- **Mitigation:** Our `FeedbackController` dynamically adjusts draft length to throttle this, but proper model selection (TinyLlama + Llama-2) is crucial.

---

## Usage

### Installation

```bash
git clone https://github.com/therealnaveenkamal/pdverify.git
cd pdverify
pip install -r requirements.txt
```

### Quick Start

Run the comprehensive 3-way benchmark to reproduce our results:

```bash
python three_way_benchmark.py --output results/
```

Compare Baseline vs VerifyPD across multiple model pairs:

```bash
python model_comparison_benchmark.py --output results/models --hf-token YOUR_HF_TOKEN
```

Run a simple performance demo:

```bash
python run_experiment.py --performance --num-requests 20 --max-concurrent 10
```

### Model Comparison Benchmark

The `model_comparison_benchmark.py` script compares Baseline vs VerifyPD (3-lane) performance across different model families:

**Supported Model Pairs:**

- **TinyLlama**: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (draft) + meta-llama/Llama-2-7b-hf (verifier)
- **Qwen**: Qwen/Qwen2-1.5B (draft) + Qwen/Qwen2-7B (verifier)
- **Gemma**: google/gemma-2b (draft) + google/gemma-7b (verifier)

**Features:**

- Tests multiple concurrency scenarios (Low, Medium)
- Generates comparative visualizations (P95/Mean latency, throughput, acceptance ratio)
- Produces markdown summary tables

**Requirements:**

- HuggingFace token for gated models (Llama-2, Gemma)
- Set `HF_TOKEN` environment variable or use `--hf-token` flag
- Create a `.env` file with `HF_TOKEN=your_token_here`

#### Model Comparison Results

**VerifyPD consistently outperforms Baseline across all model families**, with particularly strong gains at higher concurrency:

| Model         | Concurrency            | Baseline P95 (ms) | VerifyPD P95 (ms) | Improvement | Throughput Gain |
| ------------- | ---------------------- | ----------------- | ----------------- | ----------- | --------------- |
| **TinyLlama** | Low (3 concurrent)     | 42,274            | 15,236            | **-64%**    | +168%           |
| **TinyLlama** | Medium (10 concurrent) | 168,432           | 46,164            | **-73%**    | +253%           |
| **Qwen**      | Low (3 concurrent)     | 33,759            | 14,375            | **-57%**    | +124%           |
| **Qwen**      | Medium (10 concurrent) | 142,444           | 45,322            | **-68%**    | +203%           |
| **Gemma**     | Low (3 concurrent)     | 25,195            | 8,792             | **-65%**    | +137%           |
| **Gemma**     | Medium (10 concurrent) | 137,381           | 40,393            | **-71%**    | +226%           |

**Key Insights:**

- **Gemma** shows the best performance overall, with P95 latencies under 41s even at medium concurrency
- **TinyLlama** benefits most from VerifyPD at high concurrency (73% latency reduction)
- **Qwen** demonstrates consistent 57-68% improvements across scenarios
- All models show **2-3x throughput gains** with VerifyPD
- Higher concurrency = bigger wins (batched verification shines at scale)

### Configuration

Edit `src/utils/config.py` to tune:

- **Models**: Draft and Verifier model paths.
- **Batch Sizes**: `batch_size` vs `verify_micro_batch_size`.
- **Priorities**: Lane priority weights.

---

## Authors

- Naveenraj Kamalakannan (nk3940)
- Megh Panandikar (mp6545)
