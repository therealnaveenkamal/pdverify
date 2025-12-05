# Deepsearch Findings and Recommended Enhancements

## Runtime readiness
- **Explicit CUDA fallback**: `main.py` now warns and falls back to CPU when `--device cuda` is requested without an available GPU so users understand why GPU execution isn't used.
- **GPU validation before runs**: Ensure `nvidia-smi` succeeds before starting production runs; failures currently cause CPU fallback rather than hard errors.

## Engine and scheduling gaps
- **Hard-coded generation limit**: `SpeculativeEngine.process_request` stops after 100 tokens; make this configurable so benchmarks can align with dataset prompt lengths and compare baselines fairly.
- **Sequential execution**: Prefill, decode, and verify run serially in `process_request` even though a three-lane scheduler exists. Adding asynchronous workers (or background threads/tasks) per lane would let multiple requests overlap and better reflect the intended disaggregated design.
- **Batch sizing**: Verify micro-batching and decode batching are fixed in `SchedulerConfig`. Consider dynamic sizing tied to latency targets or GPU utilization to improve tail behavior under load.

## Model handling
- **No paged KV cache**: `ModelRunner.prefill` returns token IDs but does not populate a KV cache; integrating PagedAttention or vLLM’s paged cache would reduce verifier latency and GPU memory pressure.
- **Greedy-only decoding**: `generate_draft_tokens` uses argmax sampling. Support for temperature, top-p, or multinomial sampling would align draft generation with modern inference setups and improve acceptance distributions.
- **Verification shortcut**: `verify_tokens` recomputes logits over concatenated inputs and checks token-by-token but does not reuse the draft model’s logits. Sharing cached logits or using speculative sampling APIs would cut duplicate compute and better approximate production behavior.

## Metrics and observability
- **Metrics sink is stubbed**: Although `MetricsConfig` enables metrics, there is no metrics emission. Add a metrics reporter (stdout, JSONL, or Prometheus) that records acceptance ratio, queue depths, and latency percentiles per request/window.
- **Controller visibility**: Expose controller decisions (e.g., when `current_draft_length` changes and why) via structured logs or metrics to help tune thresholds in `ControllerConfig`.

## Testing and benchmarking
- **Add CPU smoke tests**: Unit tests should cover the new CUDA fallback path and ensure CPU execution still runs end-to-end with tiny models or mocks.
- **Parameterized benchmarks**: `run_experiment.py` should accept generation length, sampling strategy, and controller bounds so GPU/CPU runs can be compared consistently across hardware.
