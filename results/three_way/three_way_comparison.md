# Three-Way Comparison: Baseline vs PD vs PD-Verify

**Fair apples-to-apples comparison of all three systems.**

## Systems Compared

1. **Baseline** - Standard speculative decoding with concurrent workers
2. **PD (2-lane)** - Prefill-Decode disaggregation
3. **PD-Verify (3-lane)** - Full disaggregation with separate Verify lane

## Test Configuration

- Models: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (draft) + meta-llama/Llama-2-7b-hf (verifier)
- Device: cuda
- Max tokens per request: 100

## Overall Summary

- Total scenarios tested: 4

**Head-to-Head Results:**

- PD (2-lane) vs Baseline: 1 wins, 3 losses
- PD-Verify (3-lane) vs Baseline: 3 wins, 1 losses
- PD-Verify (3-lane) vs PD (2-lane): 3 wins, 1 losses

## Detailed Results

### Single Request

**Description:** Measure per-request latency without concurrency

**Configuration:**
- Requests: 1
- Max concurrent: 1
- Arrival rate: 1.0 req/s

| Metric | Baseline | PD (2-lane) | PD-Verify (3-lane) |
|--------|----------|-------------|--------------------|
| Latency P95 (ms) | 6695.0 | 6422.6 | 6341.1 |
| Latency P99 (ms) | 6695.0 | 6422.6 | 6341.1 |
| Throughput (req/s) | 0.15 | 0.16 | 0.16 |
| Acceptance Rate | 55.6% | 55.6% | 55.6% |

### Low Concurrency

**Description:** Test basic concurrent handling (2-5 requests)

**Configuration:**
- Requests: 5
- Max concurrent: 3
- Arrival rate: 2.0 req/s

| Metric | Baseline | PD (2-lane) | PD-Verify (3-lane) |
|--------|----------|-------------|--------------------|
| Latency P95 (ms) | 33077.5 | 33879.4 | 35639.2 |
| Latency P99 (ms) | 33970.0 | 34794.2 | 36553.1 |
| Throughput (req/s) | 0.15 | 0.14 | 0.14 |
| Acceptance Rate | 67.0% | 67.0% | 46.5% |

### Medium Concurrency

**Description:** Test performance under typical load

**Configuration:**
- Requests: 20
- Max concurrent: 10
- Arrival rate: 5.0 req/s

| Metric | Baseline | PD (2-lane) | PD-Verify (3-lane) |
|--------|----------|-------------|--------------------|
| Latency P95 (ms) | 149678.9 | 155439.1 | 95711.2 |
| Latency P99 (ms) | 151842.8 | 160070.4 | 96491.0 |
| Throughput (req/s) | 0.13 | 0.12 | 0.21 |
| Acceptance Rate | 53.6% | 53.6% | 62.7% |

### High Concurrency

**Description:** Stress test and scalability

**Configuration:**
- Requests: 50
- Max concurrent: 25
- Arrival rate: 10.0 req/s

| Metric | Baseline | PD (2-lane) | PD-Verify (3-lane) |
|--------|----------|-------------|--------------------|
| Latency P95 (ms) | 361679.7 | 368157.2 | 235652.3 |
| Latency P99 (ms) | 363769.1 | 378743.6 | 245246.3 |
| Throughput (req/s) | 0.14 | 0.13 | 0.20 |
| Acceptance Rate | 54.2% | 54.2% | 59.3% |

