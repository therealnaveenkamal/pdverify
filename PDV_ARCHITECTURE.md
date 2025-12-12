# PDV Architecture - Unified Worker Design

## High-Level Architecture Diagram

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#4c6ef5','secondaryColor':'#51cf66','tertiaryColor':'#ff6b6b'}}}%%

graph LR
    subgraph Input["Client Requests"]
        Clients[Multiple Clients]
    end
    
    subgraph PDV["PDV Engine"]
        PQ[Prefill<br/>Queue]
        DQ[Decode<br/>Queue<br/>⚠️ 1.2x limit]
        
        UW[Unified Worker<br/>Single Thread]
        
        S0[Stream 0<br/>Draft]
        S1[Stream 1<br/>Verify]
        S2[Stream 2<br/>Prefill]
    end
    
    subgraph Output["Results"]
        Results[Completed<br/>Responses]
    end
    
    Clients -->|submit| PQ
    PQ --> UW
    UW --> DQ
    DQ --> UW
    
    UW -.->|Parallel| S0
    UW -.->|Parallel| S1
    UW -.->|Parallel| S2
    
    S0 --> Results
    S1 --> Results
    
    style PQ fill:#51cf66,stroke:#2b8a3e,color:#000,stroke-width:3px
    style DQ fill:#ffa94d,stroke:#e8590c,color:#000,stroke-width:3px
    style UW fill:#ff6b6b,stroke:#c92a2a,color:#fff,stroke-width:4px
    style S0 fill:#845ef7,stroke:#5f3dc4,color:#fff,stroke-width:2px
    style S1 fill:#845ef7,stroke:#5f3dc4,color:#fff,stroke-width:2px
    style S2 fill:#845ef7,stroke:#5f3dc4,color:#fff,stroke-width:2px
    style Clients fill:#e7f5ff,stroke:#1971c2,color:#000
    style Results fill:#d3f9d8,stroke:#37b24d,color:#000
```

### Key Components
- **Prefill Queue**: Incoming requests waiting for initialization
- **Decode Queue**: Active requests with 1.2x batch_size backpressure
- **Unified Worker**: Single thread eliminates handoff overhead
- **3 CUDA Streams**: Parallel execution (Draft, Verify, Prefill)

## Simplified Worker Loop

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#4c6ef5'}}}%%

flowchart TD
    Start([Unified Worker Loop])
    
    Prefill[Process Prefill<br/>if queue available]
    Batch[Collect Decode Batch<br/>up to batch_size]
    
    subgraph Parallel["Parallel Execution per Request"]
        Draft[Generate Drafts<br/>Stream 0]
        Verify[Verify Tokens<br/>Stream 1]
        Update[Update KV Caches]
    end
    
    Complete{Done?}
    Requeue[Back to Decode Queue]
    Done[Complete & Return]
    
    Start --> Prefill
    Prefill --> Batch
    Batch --> Draft
    Draft --> Verify
    Verify --> Update
    Update --> Complete
    Complete -->|No| Requeue
    Complete -->|Yes| Done
    Requeue --> Start
    Done --> Start
    
    style Start fill:#51cf66,stroke:#2b8a3e,color:#fff,stroke-width:3px
    style Prefill fill:#74c0fc,stroke:#1971c2,color:#000,stroke-width:2px
    style Batch fill:#ffa94d,stroke:#e8590c,color:#000,stroke-width:2px
    style Draft fill:#845ef7,stroke:#5f3dc4,color:#fff,stroke-width:2px
    style Verify fill:#845ef7,stroke:#5f3dc4,color:#fff,stroke-width:2px
    style Update fill:#845ef7,stroke:#5f3dc4,color:#fff,stroke-width:2px
    style Done fill:#20c997,stroke:#087f5b,color:#fff,stroke-width:3px
```

## How Parallel Streams Work

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#845ef7'}}}%%

gantt
    title PDV Parallel Execution Timeline
    dateFormat X
    axisFormat %L
    
    section Stream 0 (Draft)
    Req 1 Draft    :s0r1, 0, 2
    Req 2 Draft    :s0r2, 2, 4
    Req 3 Draft    :s0r3, 4, 6
    
    section Stream 1 (Verify)
    Req 1 Verify   :s1r1, 2, 5
    Req 2 Verify   :s1r2, 4, 7
    
    section Stream 2 (Prefill)
    New Req Prefill :s2r1, 0, 3
    New Req Prefill :s2r2, 3, 6
```

**Key Insight**: Draft, Verify, and Prefill operations overlap in time, maximizing GPU utilization.

## Request Lifecycle (Simplified)

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#4c6ef5'}}}%%

stateDiagram-v2
    [*] --> Prefill: Submit Request
    Prefill --> Decode: Init Complete
    Decode --> Draft: Batch Ready
    Draft --> Verify: Tokens Generated
    Verify --> Decode: Continue (if not done)
    Verify --> [*]: Complete & Return
    
    note right of Prefill
        Stream 2
        Init KV caches
    end note
    
    note right of Draft
        Stream 0
        Generate K tokens
    end note
    
    note right of Verify
        Stream 1
        Accept/reject tokens
    end note
```

## PD vs PDV: Side-by-Side Comparison

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#ffa94d'}}}%%

graph TB
    subgraph PD["PD: 2 Workers, Queue Handoff"]
        PD_PW[Prefill Worker]
        PD_DW[Decode Worker]
        PD_PW -->|Queue Transfer<br/>❌ Overhead| PD_DW
        PD_Result[Throughput at C=128:<br/>0.4 TPS]
    end
    
    subgraph PDV["PDV: Unified Worker"]
        PDV_UW[Unified Worker<br/>✅ No handoff]
        PDV_S[3 Parallel<br/>CUDA Streams]
        PDV_UW --> PDV_S
        PDV_Result[Throughput at C=128:<br/>13.4 TPS]
    end
    
    PDV_Result -.->|33x faster| PD_Result
    
    style PD fill:#ffe0e0,stroke:#ff6b6b,color:#000,stroke-width:3px
    style PDV fill:#d3f9d8,stroke:#51cf66,color:#000,stroke-width:3px
    style PD_PW fill:#ffc9c9,stroke:#fa5252,color:#000
    style PD_DW fill:#ffc9c9,stroke:#fa5252,color:#000
    style PDV_UW fill:#96f2d7,stroke:#20c997,color:#000,stroke-width:3px
    style PDV_S fill:#c0eb75,stroke:#82c91e,color:#000
    style PD_Result fill:#fff3bf,stroke:#fab005,color:#000,stroke-width:2px
    style PDV_Result fill:#d0ebff,stroke:#1c7ed6,color:#000,stroke-width:2px
```

## Key Architectural Differences

### PD Architecture Issues
1. **Queue Handoff**: Requests must transfer from prefill worker to decode worker
2. **Lock Contention**: Two workers competing for shared resources
3. **Context Switching**: Multiple threads require OS scheduling
4. **Overhead Accumulation**: Each handoff adds latency

### PDV Architecture Advantages
1. **Unified Worker**: Single thread eliminates handoff
2. **Spin-Based Polling**: 0.00001s wait time, no context switching
3. **Adaptive Backpressure**: 1.2x batch_size prevents queue overflow
4. **Stream Parallelization**: Draft, verify, prefill run concurrently
5. **Lock-Optimized**: Minimal critical sections with simple locks

## Performance Summary

### Scaling Behavior at Ultra-High Concurrency

| Concurrency | PD TPS | PDV TPS | Improvement |
|-------------|--------|---------|-------------|
| C = 32 | 10.0 | 13.0 | **+30%** |
| C = 64 | 1.5 | 12.0 | **+700%** |
| C = 96 | 0.5 | 8.0 | **+1500%** |
| C = 128 | 0.4 | 13.4 | **+3250%** |

**Key Insight**: PDV maintains stable throughput while PD degrades at ultra-high concurrency.

**See full performance graphs in `/plots/throughput_comparison.png`**

---

## Technical Implementation Details

### Unified Worker Pseudo-Code

```python
class PDVLiteEngine:
    def _unified_worker_loop(self):
        while self.is_running:
            # Phase 1: Handle prefill (with backpressure)
            if len(prefill_queue) > 0 and len(decode_queue) < 1.2 * batch_size:
                request = prefill_queue.pop()
                prefill(request, stream_id=2)  # Stream 2
                decode_queue.append(request)
            
            # Phase 2: Handle decode batch
            if len(decode_queue) > 0:
                batch = decode_queue.pop_batch(batch_size)
                
                for request in batch:
                    # Parallel execution via CUDA streams
                    draft_tokens = generate_draft(request, stream_id=0)     # Stream 0
                    accepted = verify_tokens(request, draft_tokens, stream_id=1)  # Stream 1
                    
                    update_kv_caches(request, accepted)
                    
                    if request.is_complete():
                        complete_request(request)
                    else:
                        decode_queue.append(request)
            else:
                # Spin-based polling (microsecond-level wait)
                sleep(0.00001)
```

### CUDA Stream Management

```python
# Stream initialization
stream_manager = StreamManager(num_streams=3)
stream_0 = stream_manager.get_stream(0)  # Draft generation
stream_1 = stream_manager.get_stream(1)  # Verification
stream_2 = stream_manager.get_stream(2)  # Prefill

# Parallel execution
with torch.cuda.stream(stream_0):
    draft_output = draft_model(input_ids, past_kv=kv_cache_draft)

with torch.cuda.stream(stream_1):
    verify_output = verify_model(draft_tokens, past_kv=kv_cache_verify)

# Streams execute in parallel, synchronize only when needed
stream_0.synchronize()  # Wait for draft
stream_1.synchronize()  # Wait for verify
```

---

## Design Principles

1. **Simplicity Over Complexity**: Remove coordination overhead by unifying workers
2. **Parallelism Where It Matters**: Use CUDA streams for GPU-level parallelism
3. **Adaptive Control**: Dynamic queue management prevents overflow
4. **Low-Latency Polling**: Spin-based design for minimal context switching
5. **Scalability First**: Architecture designed for ultra-high concurrency (C >= 64)

---

**Last Updated**: December 2024
**Repository**: https://github.com/therealnaveenkamal/pdverify

