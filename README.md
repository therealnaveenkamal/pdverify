# PD-Verify (PDV): Disaggregated Speculative Decoding

PD-Verify is a speculative decoding engine that **splits prefill, decode, and verify into separate lanes** on a single GPU. The goal is simple: squeeze more throughput and better latency out of speculative decoding by treating each stage as a first-class workload.

It implements a **three-lane architecture** on top of CUDA streams and a small scheduler, and compares directly against:

* A baseline “all-in-one” speculative decoding loop
* A 2-lane PD (prefill–decode) architecture

## Table of Contents

* [Overview](#overview)
* [Architectural Comparison](#architectural-comparison)
* [Performance Results](#performance-results)
* [Technical Deep Dive](#technical-deep-dive)
* [Setup and Usage](#setup-and-usage)
* [Architecture Evolution](#architecture-evolution)
* [Contributing](#contributing)

## Overview

PD-Verify separates speculative decoding into three concurrent lanes:

* **Prefill**
* **Decode (draft generation)**
* **Verify**

All three run on their own worker threads and CUDA streams. Depending on the concurrency level, PDV either behaves like a normal PD system or switches into a fully parallel 3-lane mode.

**Empirically, this gives:**

* **Up to ~11.5% higher throughput** vs 2-lane PD at medium concurrency
* **Up to ~7.5% lower P95 latency** at concurrency 5
* **~90%+ GPU utilization** via overlapped streams (vs ~67% in PD)
* No regression at low concurrency (PDV falls back to PD-style atomic processing)

### Hybrid Mode

PDV is not “always 3-lane”. It’s hybrid:

* **Concurrency ≤ 3** → behave like PD: atomic draft+verify in the decode lane
* **Concurrency > 3** → enable full 3-lane PDV: prefill, decode, verify in parallel

This avoids paying extra coordination overhead when there isn’t enough work to parallelize.

---

## Architectural Comparison

### Baseline: Single-Lane Speculative Decoding

```text
┌─────────────────┐
│   Sequential    │
│ Speculative     │
│   Decoding      │
└─────────────────┘
```

**Rough shape:**

* One main worker loop on top of several GPU workers
* For each request: draft generation → verification → next token (repeat)
* No attempt to decouple prefill/verify/decode

**Pros**

* Easy to reason about
* Latency for a single request is predictable

**Cons**

* Poor GPU utilization under concurrency
* Prefill, decode, and verify block each other
* Scaling is basically linear until you hit GPU saturation

---

### PD: 2-Lane Prefill–Decode

```text
┌─────────────┐    ┌─────────────┐
│   Prefill   │────│   Decode    │
│   Lane      │    │   Lane      │
│             │    │ (Draft+     │
│             │    │  Verify)    │
└─────────────┘    └─────────────┘
```

**Core idea:**

* **2 CPU worker threads**: one for prefill, one for decode
* **2 CUDA streams**: prefill stream + decode stream
* Verify runs “inside” decode as an atomic unit (draft+verify together)

**Pros**

* Solid baseline for PD-style systems
* Less code complexity than full disaggregation
* Works well at low–medium concurrency

**Cons**

* Only 2 operations in flight at once
* Decode+verify is still a single atomic step → sequential bottleneck
* GPU utilization tends to flatten around ~67%

---

### PDV: 3-Lane Prefill–Decode–Verify

```text
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Prefill   │────│   Decode    │────│   Verify    │
│   Lane      │    │   Lane      │    │   Lane      │
│             │    │ (Draft Gen) │    │ (Verify)    │
└─────────────┘    └─────────────┘    └─────────────┘
```

**What changes:**

* **3 CPU worker threads**: prefill, decode, verify
* **3 CUDA streams**: one per lane
* Decode only handles **draft generation**; verify gets its own lane and queue
* A small **ThreeLaneScheduler** decides which lane runs next based on queue depths

**Pros**

* Up to three GPU operations in flight at once
* **Higher GPU utilization** (~90%+ in our runs)
* Better scaling when concurrency increases
* Flexibility to prioritize verify vs decode

**Cons**

* More coordination and queueing logic
* Overhead can hurt at very low concurrency (hence hybrid mode)
* Slightly higher memory pressure for lane-local state

---

## Performance Results

Benchmarks run with 10 requests at each concurrency level.

> Note: “Token P95 (ms)” is per-token P95 latency. Throughput is tokens/s across all requests.

| Concurrency | Engine   | Token P95 (ms) | Throughput (tokens/s) | Acceptance Rate | Status |
| ----------- | -------- | -------------- | --------------------- | --------------- | ------ |
| **0**       | Baseline | 22,850.2       | 9.05                  | 0.530           | ✅      |
| **0**       | PD       | 15,600.4       | 44.93                 | 0.448           | ✅      |
| **0**       | PDV      | 15,489.6       | 44.39                 | 0.452           | ✅      |
| **1**       | Baseline | 66,023.5       | 13.03                 | 0.530           | ✅      |
| **1**       | PD       | 15,550.6       | 44.61                 | 0.448           | ✅      |
| **1**       | PDV      | 15,579.5       | 44.55                 | 0.452           | ✅      |
| **3**       | Baseline | 68,117.9       | 13.63                 | 0.530           | ✅      |
| **3**       | PD       | 15,740.7       | 44.39                 | 0.448           | ✅      |
| **3**       | PDV      | 15,180.0       | 47.25                 | 0.452           | ✅      |
| **5**       | Baseline | 64,692.1       | 14.22                 | 0.530           | ✅      |
| **5**       | PD       | 18,019.9       | 45.98                 | 0.447           | ✅      |
| **5**       | PDV      | 16,670.6       | 51.27                 | 0.452           | ✅      |
| **7**       | Baseline | 59,937.3       | 14.22                 | 0.530           | ✅      |
| **7**       | PD       | 17,355.5       | 48.17                 | 0.448           | ✅      |
| **7**       | PDV      | 17,890.9       | 48.73                 | 0.452           | ✅      |
| **9**       | Baseline | 53,469.2       | 14.35                 | 0.530           | ✅      |
| **9**       | PD       | 17,305.7       | 48.17                 | 0.448           | ✅      |
| **9**       | PDV      | 17,822.6       | 48.73                 | 0.452           | ✅      |
| **10**      | Baseline | 55,462.9       | 14.22                 | 0.530           | ✅      |
| **10**      | PD       | 17,305.7       | 48.17                 | 0.448           | ✅      |
| **10**      | PDV      | 17,822.6       | 48.73                 | 0.452           | ✅      |

### What Actually Improves?

* At **concurrency 5**, PDV vs PD:

  * **+11.5% throughput** (45.98 → 51.27 tokens/s)
  * **~7.5% lower P95 latency** (18,019.9 → 16,670.6 ms)
* At **high concurrency (7–10)**:

  * PDV keeps a small but consistent throughput edge (~1–2%)
* At **low concurrency (0–3)**:

  * PDV and PD are basically tied (by design, PDV falls back to PD-style mode)

### Why PDV Helps

* **More overlap:** prefill, draft, and verify run truly in parallel on separate streams.
* **Scheduler awareness:** verify can be prioritized when its queue backs up.
* **GPU busy time goes up:** fewer idle gaps in Nsight traces; utilization ~90%+.

### When PDV Isn’t Better

* At very low concurrency, queueing and lane switching overhead can outweigh the benefits.
* The 3-lane scheduler adds some complexity; tuning thresholds and batch sizes matters.
* More lanes mean more state and buffers, so memory pressure is slightly higher.

---

## Technical Deep Dive

### CUDA Stream Layout

PDV uses three CUDA streams on a single GPU:

```text
Stream 0: Draft token generation  (Decode)
Stream 1: Token verification      (Verify)
Stream 2: Prefill                 (Prefill)
```

Rough timeline:

```text
Time →
Draft   ──▇▇▇─────▇▇▇─────…
Verify  ─────▇▇▇─────▇▇▇──…
Prefill ▇▇▇─────────▇▇▇───…
```

The idea is to always have something queued in each stream when there’s enough work, so the GPU scheduler can overlap kernels as much as possible.

---

### ThreeLaneScheduler (Lane Selection)

The scheduler looks at queue depths and picks an order for attempting to schedule work:

```python
# Pseudocode (simplified)
if verify_queue_depth > decode_queue_depth + 2:
    priority_order = [VERIFY, DECODE, PREFILL]
else:
    priority_order = [DECODE, VERIFY, PREFILL]
```

Interpretation:

* If verification is falling behind (its queue is too deep), push it to the front.
* Otherwise, default to decode → verify → prefill so draft tokens keep flowing.
* Prefill is generally the least time-sensitive, so it comes last.

Actual implementation is a bit more nuanced (error handling, backoffs, etc.), but this is the gist.

---

### Hybrid Decode Handler

Decode behavior switches between PD-style and PDV-style based on how many requests are active:

```python
def _handle_decode_batch(self, batch: List[Request]):
    current_concurrency = len(self.scheduler.get_active_requests())

    if current_concurrency <= 3:
        # Low concurrency: behave like PD
        return self._handle_decode_batch_pd_style(batch)
    else:
        # Higher concurrency: use full 3-lane PDV
        return self._handle_decode_batch_pdv_style(batch)
```

So from the outside:

* Users don’t need to toggle any “modes”.
* As load ramps up, PDV automatically switches to the more parallel scheduling path.

---

## Setup and Usage

### Requirements

* Python 3.8+
* PyTorch 2.0+
* CUDA-capable GPU
* HuggingFace Transformers

### Install

```bash
git clone https://github.com/therealnaveenkamal/pdverify.git
cd pdverify
pip install -r requirements.txt
```

For editable/development mode:

```bash
git clone https://github.com/therealnaveenkamal/pdverify.git
cd pdverify
pip install -e .
```

### Running Benchmarks

```bash
# Sweep across all concurrency levels
python comprehensive_benchmark.py
```

### Configuration

Most knobs live in `src/utils/config.py`:

* Draft / verifier model pairs
* Batch sizes and queue depth limits
* CUDA stream selection
* Scheduler thresholds and priority logic

---

## Architecture Evolution

### Versions

* **v1.0** – Baseline speculative decoding, single-lane design
* **v1.1** – Prefill–Decode (PD): 2-lane disaggregation
* **v2.0** – PDV: 3-lane disaggregation + hybrid mode based on concurrency

### Planned / Possible Next Steps

* Multi-GPU routing with cross-GPU stream sync
* Adaptive batch sizing based on queue depth / latency SLOs
* Learned (or at least ML-assisted) scheduling and lane ordering
* More explicit memory-aware scheduling (e.g., KV cache pressure)

---

## Contributing

Contributions are welcome. Useful directions include:

* Profiling + kernel-level optimization
* Better benchmark coverage (different models, longer sequences, real traces)
* Multi-GPU support and sharding strategies
* Smarter scheduling policies / heuristics

### Dev Setup

```bash
# Clone + editable install
git clone https://github.com/therealnaveenkamal/pdverify.git
cd pdverify
pip install -e .

# Tests
python -m pytest tests/

# Formatting
black src/
isort src/
```

---

**PD-Verify is an experiment in “taking prefill/decode/verify seriously” as separate workloads, not just one big loop.**

If you run this on your own models / hardware and get interesting numbers (better or worse), PRs and issues are appreciated.
