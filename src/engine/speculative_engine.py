"""
GPU stream-based speculative decoding engine with unified 2-lane architecture.
Eliminates the verify lane bottleneck by integrating verification into the decode loop
with parallel CUDA streams for maximum throughput.
"""

import time
import threading
from typing import Optional, List, Dict, Callable, Any
import logging
import torch
from collections import deque

from ..scheduler import Request
from ..controller import FeedbackController
from .model_runner import ModelRunner
from ..utils.config import VerifyPDConfig
from .pd_engine import PDEngine

logger = logging.getLogger(__name__)


class PDVLiteEngine:
    """
    Unified 2-lane speculative decoding engine with integrated draft+verify parallelism.
    Eliminates verify lane bottleneck by processing draft generation and verification
    in parallel streams within the decode loop.

    ARCHITECTURE: 2-lane with parallel draft/verify streams
    - Prefill Lane: Tokenization and KV cache initialization
    - Decode Lane: Parallel draft generation + verification on separate CUDA streams
    """

    def __init__(self, config: VerifyPDConfig):
        """
        Initialize unified GPU stream-based speculative engine.

        Args:
            config: System configuration
        """
        self.config = config

        # Initialize components
        from ..utils.stream_manager import StreamManager

        # Simple 2-lane scheduler (prefill + decode)
        from ..scheduler import LaneType
        from ..scheduler.lane import Lane
        self.prefill_lane = Lane(LaneType.PREFILL, max_size=config.scheduler.max_queue_size)
        self.decode_lane = Lane(LaneType.DECODE, max_size=config.scheduler.max_queue_size)

        self.controller = FeedbackController(config.controller)

        # Initialize stream manager for parallel draft+verify
        self.stream_manager = StreamManager(
            device=config.hardware.device,
            num_streams=3  # Stream 0: draft, Stream 1: verify, Stream 2: prefill
        )

        self.model_runner = ModelRunner(
            model_config=config.model,
            hardware_config=config.hardware,
            stream_manager=self.stream_manager
        )

        # Engine state
        self.is_running = False
        self.prefill_worker_thread = None
        self.decode_worker_thread = None
        self.total_tokens_generated = 0
        self.total_tokens_accepted = 0

        # Request tracking
        self.request_callbacks: Dict[str, Callable] = {}
        self._lock = threading.Lock()
        self._queue_condition = threading.Condition(self._lock)

        # Configurable parameters
        self.max_new_tokens = config.model.max_new_tokens

        logger.info("UnifiedSpeculativeEngine initialized with integrated draft+verify parallelism")

    def start(self):
        """Start the engine and worker threads."""
        logger.info("Starting UnifiedSpeculativeEngine...")
        self.model_runner.load_models()
        self.is_running = True

        # Start worker threads for 2-lane architecture
        self.prefill_worker_thread = threading.Thread(target=self._prefill_worker_loop, daemon=True)
        self.decode_worker_thread = threading.Thread(target=self._decode_worker_loop, daemon=True)

        self.prefill_worker_thread.start()
        self.decode_worker_thread.start()

        logger.info("UnifiedSpeculativeEngine started with 2-lane parallel draft+verify architecture")

    def stop(self):
        """Stop the engine and worker threads."""
        logger.info("Stopping UnifiedSpeculativeEngine...")
        self.is_running = False

        # Wake up workers
        with self._queue_condition:
            self._queue_condition.notify_all()

        if self.prefill_worker_thread:
            self.prefill_worker_thread.join(timeout=2.0)
        if self.decode_worker_thread:
            self.decode_worker_thread.join(timeout=2.0)

        self.stream_manager.cleanup()
        self.model_runner.cleanup()
        logger.info("UnifiedSpeculativeEngine stopped")

    def submit_request_async(self, request: Request, callback: Optional[Callable] = None):
        """
        Submit a request asynchronously.

        Args:
            request: Request to process
            callback: Optional callback function(request, result_text) called on completion
        """
        if not self.is_running:
            raise RuntimeError("Engine is not running. Call start() first.")

        # Store callback
        if callback:
            with self._lock:
                self.request_callbacks[request.request_id] = callback

        # Submit to prefill lane
        from ..scheduler import LaneType
        request.stage = LaneType.PREFILL

        # Initialize request state
        request.tokens_generated = 0
        request.tokens_accepted = 0
        request.generated_tokens = []

        success = self.prefill_lane.add_request(request)
        if success:
            logger.debug(f"Submitted request {request.request_id} to prefill lane")
            # Notify prefill worker
            with self._queue_condition:
                self._queue_condition.notify()
        else:
            logger.warning(f"Failed to submit request {request.request_id} - queue full")

    def _prefill_worker_loop(self):
        """Prefill worker loop processing requests from prefill lane."""
        from ..scheduler import LaneType
        logger.info("Prefill worker loop started")

        while self.is_running:
            try:
                # Get next batch from prefill lane
                batch = self.prefill_lane.get_batch(self.config.scheduler.batch_size)

                if not batch:
                    with self._queue_condition:
                        self._queue_condition.wait(timeout=0.01)
                    continue

                # Intelligent backpressure: limit downstream queue depth
                downstream_depth = len(self.decode_lane)
                max_downstream = int(self.config.scheduler.batch_size * 1.5)  # Balanced limit

                if downstream_depth >= max_downstream:
                    # Re-queue and wait to prevent flooding
                    for req in reversed(batch):
                        self.prefill_lane.add_request(req)
                    time.sleep(0.001)
                    continue

                # Process prefill batch
                self._handle_prefill_batch(batch)

            except Exception as e:
                logger.error(f"Error in prefill worker loop: {e}", exc_info=True)
                time.sleep(0.1)

        logger.info("Prefill worker loop stopped")

    def _decode_worker_loop(self):
        """Unified decode worker loop with parallel draft+verify processing."""
        logger.info("Unified decode worker loop started - integrated draft+verify parallelism")

        while self.is_running:
            try:
                # Get batch from decode lane
                batch = self.decode_lane.get_batch(self.config.scheduler.batch_size)

                if not batch:
                    with self._queue_condition:
                        self._queue_condition.wait(timeout=0.01)
                    continue

                logger.debug(f"Processing unified decode batch of {len(batch)} requests")
                # Process batch with integrated draft+verify parallelism
                self._handle_unified_decode_batch(batch)

            except Exception as e:
                logger.error(f"Error in unified decode worker loop: {e}", exc_info=True)
                time.sleep(0.1)

        logger.info("Unified decode worker loop stopped")

    def _handle_prefill_batch(self, batch: List[Request]):
        """Handle prefill and submit to decode lane."""
        # Use CUDA stream 2 for prefill
        stream = self.stream_manager.get_stream(2)

        for req in batch:
            req.prefill_start_time = time.time()
            input_ids, draft_kv, verifier_kv, verifier_logits, draft_logits = self.model_runner.prefill(
                req.prompt,
                stream_id=2
            )
            req.prefill_end_time = time.time()

            # Store state in request
            req._input_ids = input_ids
            req.kv_cache_draft = draft_kv
            req.kv_cache_verifier = verifier_kv
            req.last_verifier_logits = verifier_logits
            req.last_draft_logits = draft_logits

            # Submit to decode lane
            from ..scheduler import LaneType
            req.stage = LaneType.DECODE
            self.decode_lane.add_request(req)

            logger.debug(f"Prefill complete for {req.request_id}, submitted to decode lane")

    def _handle_unified_decode_batch(self, batch: List[Request]):
        """TRUE PDV PIPELINING: Parallel draft+verify across multiple requests simultaneously."""
        import torch
        from ..scheduler import LaneType

        # Get draft length
        draft_length = self.controller.get_draft_length()

        # Use parallel streams for true pipelining
        stream_draft = self.stream_manager.get_stream(0)  # Draft generation
        stream_verify = self.stream_manager.get_stream(1)  # Verification

        # Initialize timing
        for req in batch:
            if req.decode_start_time == 0.0:
                req.decode_start_time = time.time()

        completed_requests = []
        continuing_requests = []

        try:
            # TRUE PDV PIPELINE: Process ALL requests in parallel across both streams
            # This creates a true assembly line where GPU is always busy

            # PHASE 1: Launch ALL draft generations concurrently
            draft_events = []
            for req in batch:
                # Sample d1
                d1 = 0
                if req.last_draft_logits is not None:
                    d1 = torch.argmax(req.last_draft_logits).item()
                    if self.model_runner.draft_vocab_size and d1 >= self.model_runner.draft_vocab_size:
                        d1 = self.model_runner.draft_vocab_size - 1

                d1_tensor = torch.tensor([[d1]], device=self.model_runner.device)
                req.draft_tokens = [d1]

                # Launch draft generation on stream 0
                if draft_length > 1:
                    gen_ids, _ = self.model_runner.generate_draft_tokens(
                        d1_tensor,
                        num_tokens=draft_length-1,
                        past_key_values=req.kv_cache_draft,
                        stream_id=0
                    )
                    req.draft_tokens.extend(gen_ids[0])

                # Record when draft completes
                event = torch.cuda.Event(blocking=False)
                event.record(stream_draft)
                draft_events.append((req, event))

            # PHASE 2: Launch ALL verifications concurrently (pipelined with drafts)
            verify_events = []
            for req, draft_event in draft_events:
                # Wait for this request's draft to complete, then start verification
                stream_verify.wait_event(draft_event)

                full_drafts = req.draft_tokens
                verify_input = torch.tensor([full_drafts], device=self.model_runner.device)
                prev_logits = req.last_verifier_logits.unsqueeze(0) if req.last_verifier_logits is not None else None

                # Launch verification on stream 1
                accepted_batch, num_accepted_batch, _, _ = self.model_runner.verify_tokens(
                    verify_input,
                    [full_drafts],
                    past_key_values=req.kv_cache_verifier,
                    previous_verifier_logits=prev_logits,
                    stream_id=1
                )

                req._accepted_tokens = accepted_batch[0]
                req._num_accepted = num_accepted_batch[0]

                # Record when verification completes
                event = torch.cuda.Event(blocking=False)
                event.record(stream_verify)
                verify_events.append((req, event))

            # PHASE 3: Synchronize and process results
            for req, verify_event in verify_events:
                verify_event.synchronize()  # Wait for verification to complete

                accepted_tokens = req._accepted_tokens
                num_accepted = req._num_accepted

                # Update KV caches
                if accepted_tokens:
                    acc_tensor = torch.tensor([accepted_tokens], device=self.model_runner.device)

                    with self.model_runner._draft_lock:
                        out_d = self.model_runner.draft_model(acc_tensor, past_key_values=req.kv_cache_draft, use_cache=True)
                        req.kv_cache_draft = out_d.past_key_values
                        req.last_draft_logits = out_d.logits[:, -1, :]

                    with self.model_runner._verifier_lock:
                        out_v = self.model_runner.verifier_model(acc_tensor, past_key_values=req.kv_cache_verifier, use_cache=True)
                        req.kv_cache_verifier = out_v.past_key_values
                        req.last_verifier_logits = out_v.logits[:, -1, :]

                # Update metrics
                req.tokens_generated += len(req.draft_tokens)
                req.tokens_accepted += num_accepted
                req.generated_tokens.extend(accepted_tokens)

                # Record timestamps
                current_time = time.time()
                if not hasattr(req, 'token_timestamps'):
                    req.token_timestamps = []
                for _ in accepted_tokens:
                    req.token_timestamps.append(current_time)

                # Check completion
                is_complete = False
                if accepted_tokens:
                    new_ids = torch.tensor([accepted_tokens], device=self.model_runner.device, dtype=torch.long)
                    req._input_ids = torch.cat([req._input_ids, new_ids], dim=1)

                    if self.model_runner.tokenizer.eos_token_id in accepted_tokens:
                        is_complete = True

                if len(req.generated_tokens) >= self.max_new_tokens:
                    is_complete = True

                if is_complete:
                    completed_requests.append(req)
                else:
                    continuing_requests.append(req)

                # Clean up temporary attributes
                delattr(req, '_accepted_tokens')
                delattr(req, '_num_accepted')

        except Exception as e:
            logger.error(f"Error in true PDV pipelining: {e}", exc_info=True)
            import traceback
            logger.error(traceback.format_exc())
            # On error, mark remaining requests as completed with error
            for req in batch:
                if req not in completed_requests and req not in continuing_requests:
                    req.error = str(e)
                    completed_requests.append(req)

            # Update global metrics
            with self._lock:
                for req in batch:
                    self.total_tokens_generated += len(req.draft_tokens)
                    self.total_tokens_accepted += req.tokens_accepted

                # Update controller with batch averages
                total_ratio = 0.0
                total_requests = 0
                for req in batch:
                    if req.draft_tokens:
                        ratio = req.tokens_accepted / len(req.draft_tokens)
                        total_ratio += ratio
                        total_requests += 1

                if total_requests > 0:
                    avg_ratio = total_ratio / total_requests
                    self.controller.record_acceptance(avg_ratio)

        except Exception as e:
            logger.error(f"Error in unified decode batch: {e}", exc_info=True)

        # Update global metrics
        with self._lock:
            for req in batch:
                self.total_tokens_generated += len(req.draft_tokens)
                self.total_tokens_accepted += req.tokens_accepted

            # Update controller
            if batch:
                total_ratio = sum(req.tokens_accepted / len(req.draft_tokens) if req.draft_tokens else 0 for req in batch)
                avg_ratio = total_ratio / len(batch)
                self.controller.record_acceptance(avg_ratio)

        # Handle completions
        for req in completed_requests:
            self._handle_completion(req)

        # Re-queue continuing requests to decode lane
        for req in continuing_requests:
            req.stage = LaneType.DECODE
            self.decode_lane.add_request(req)



    def _handle_completion(self, request: Request):
        """Handle request completion."""
        request.decode_end_time = time.time()

        # Convert tokens to text
        if not request.error:
            try:
                request.result_text = self.model_runner.tokenizer.decode(
                    request.generated_tokens, skip_special_tokens=True
                )
            except Exception as e:
                logger.error(f"Error decoding tokens: {e}")
                request.error = f"Decoding error: {e}"

        self._trigger_callback(request, request.result_text)

    def _trigger_callback(self, request: Request, result_text: str):
        """Trigger completion callback."""
        with self._lock:
            callback = self.request_callbacks.pop(request.request_id, None)

        if callback:
            try:
                callback(request, result_text)
            except Exception as e:
                logger.error(f"Error in callback for {request.request_id}: {e}")

    def process_request(self, request: Request) -> str:
        """
        Process a request synchronously (blocking).
        
        Args:
            request: Request to process
            
        Returns:
            Generated text
        """
        import threading
        completion_event = threading.Event()
        result_container = {}
        
        def callback(req, res):
            result_container['result'] = res
            completion_event.set()
        
        self.submit_request_async(request, callback)
        completion_event.wait()
        
        return result_container.get('result', "")

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        lane_stats = {
            "prefill_depth": len(self.prefill_lane),
            "decode_depth": len(self.decode_lane)
        }

        controller_stats = {
            "current_draft_length": self.controller.get_draft_length()
        }

        overall_acceptance = 0.0
        if self.total_tokens_generated > 0:
            overall_acceptance = self.total_tokens_accepted / self.total_tokens_generated

        return {
            "overall_acceptance_rate": overall_acceptance,
            "total_tokens_generated": self.total_tokens_generated,
            "total_tokens_accepted": self.total_tokens_accepted,
            "lanes": lane_stats,
            "controller": controller_stats
        }

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class PDVLiteEngine:
    """
    PDV: Standalone implementation with AGGRESSIVE parallelization.
    Uses request-level pipelining: draft(N+1) || verify(N) for maximum throughput.
    """

    def __init__(self, config: VerifyPDConfig):
        self.config = config
        
        # Initialize model runner with 3 streams
        from ..utils.stream_manager import StreamManager
        self.stream_manager = StreamManager(
            device=config.hardware.device,
            num_streams=3  # Stream 0: draft, Stream 1: verify, Stream 2: prefill
        )
        
        self.model_runner = ModelRunner(
            model_config=config.model,
            hardware_config=config.hardware,
            stream_manager=self.stream_manager
        )
        
        self.controller = FeedbackController(config.controller)
        
        from collections import deque
        self.prefill_queue: deque = deque()
        self.decode_queue: deque = deque()
        self.request_callbacks: Dict[str, Callable] = {}
        self._lock = threading.Lock()
        
        # Metrics
        self.total_tokens_generated = 0
        self.total_tokens_accepted = 0
        
        self.is_running = False
        self.worker_thread = None
        self.max_new_tokens = config.model.max_new_tokens
        self.batch_size = config.scheduler.batch_size
        
        logger.info("PDVLiteEngine: Standalone with aggressive parallelization")

    def start(self):
        logger.info("Starting PDVLiteEngine...")
        self.model_runner.load_models()
        self.is_running = True
        
        self.worker_thread = threading.Thread(target=self._unified_worker_loop, daemon=True)
        self.worker_thread.start()
        
        logger.info("PDVLiteEngine started")

    def stop(self):
        logger.info("Stopping PDVLiteEngine...")
        self.is_running = False
        
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
        
        self.stream_manager.cleanup()
        self.model_runner.cleanup()
        logger.info("PDVLiteEngine stopped")

    def submit_request_async(self, request: Request, callback: Optional[Callable] = None):
        """Submit a request."""
        if not self.is_running:
            raise RuntimeError("Engine is not running. Call start() first.")

        if callback:
            with self._lock:
                self.request_callbacks[request.request_id] = callback

        request.tokens_generated = 0
        request.tokens_accepted = 0
        request.generated_tokens = []
        
        with self._lock:
            self.prefill_queue.append(request)

    def _unified_worker_loop(self):
        logger.info("Unified worker loop started")
        import time
        while self.is_running:
            with self._lock:
                decode_queue_len = len(self.decode_queue)
                prefill_queue_len = len(self.prefill_queue)
            
            if decode_queue_len < self.batch_size:
                prefill_req = None
                with self._lock:
                    if prefill_queue_len > 0:
                        prefill_req = self.prefill_queue.popleft()

                if prefill_req:
                    try:
                        self._handle_prefill(prefill_req)
                    except Exception as e:
                        logger.error(f"Error in prefill: {e}", exc_info=True)
                        prefill_req.error = str(e)
                        self._complete_request(prefill_req)
            
            batch = []
            with self._lock:
                while len(self.decode_queue) > 0 and len(batch) < self.batch_size:
                    batch.append(self.decode_queue.popleft())
            
            if batch:
                try:
                    self._handle_decode_batch_aggressive(batch)
                except Exception as e:
                    logger.error(f"Error in decode batch: {e}", exc_info=True)
                    for req in batch:
                        req.error = str(e)
                        self._complete_request(req)
            elif decode_queue_len == 0 and prefill_queue_len == 0:
                time.sleep(0.00001)

        logger.info("Unified worker loop stopped")

    def _handle_prefill(self, request: Request):
        """Handle prefill."""
        request.prefill_start_time = time.time()
        input_ids, draft_kv, verifier_kv, verifier_logits, draft_logits = self.model_runner.prefill(
            request.prompt,
            stream_id=2
        )
        request.prefill_end_time = time.time()
        
        request._input_ids = input_ids
        request.kv_cache_draft = draft_kv
        request.kv_cache_verifier = verifier_kv
        request.last_verifier_logits = verifier_logits
        request.last_draft_logits = draft_logits
        
        with self._lock:
            self.decode_queue.append(request)

    def _handle_decode_batch_aggressive(self, batch: List[Request]):
        import torch
        draft_length = self.controller.get_draft_length()
        
        for req in batch:
            if req.decode_start_time == 0.0:
                req.decode_start_time = time.time()

        completed_requests = []
        continuing_requests = []

        try:
            for req in batch:
                d1 = 0
                if req.last_draft_logits is not None:
                    d1 = torch.argmax(req.last_draft_logits).item()
                    if self.model_runner.draft_vocab_size and d1 >= self.model_runner.draft_vocab_size:
                        d1 = self.model_runner.draft_vocab_size - 1

                d1_tensor = torch.tensor([[d1]], device=self.model_runner.device)
                curr_draft_tokens = [d1]

                generated_ids = []
                if draft_length > 1:
                    gen_ids, _ = self.model_runner.generate_draft_tokens(
                        d1_tensor,
                        num_tokens=draft_length-1,
                        past_key_values=req.kv_cache_draft,
                        stream_id=1
                    )
                    generated_ids = gen_ids[0]

                full_drafts = curr_draft_tokens + generated_ids

                verify_input = torch.tensor([full_drafts], device=self.model_runner.device)
                prev_logits = req.last_verifier_logits.unsqueeze(0) if req.last_verifier_logits is not None else None

                accepted_batch, num_accepted_batch, _, _ = self.model_runner.verify_tokens(
                    verify_input,
                    [full_drafts],
                    past_key_values=req.kv_cache_verifier,
                    previous_verifier_logits=prev_logits,
                    stream_id=1
                )

                accepted_tokens = accepted_batch[0]
                num_accepted = num_accepted_batch[0]

                if accepted_tokens:
                    acc_tensor = torch.tensor([accepted_tokens], device=self.model_runner.device)

                    with self.model_runner._draft_lock:
                        with torch.no_grad():
                            out_d = self.model_runner.draft_model(acc_tensor, past_key_values=req.kv_cache_draft, use_cache=True)
                            req.kv_cache_draft = out_d.past_key_values
                            req.last_draft_logits = out_d.logits[:, -1, :]
                    
                    with self.model_runner._verifier_lock:
                        with torch.no_grad():
                            out_v = self.model_runner.verifier_model(acc_tensor, past_key_values=req.kv_cache_verifier, use_cache=True)
                            req.kv_cache_verifier = out_v.past_key_values
                            req.last_verifier_logits = out_v.logits[:, -1, :]
                
                req.tokens_generated += len(full_drafts)
                req.tokens_accepted += num_accepted
                req.generated_tokens.extend(accepted_tokens)

                current_time = time.time()
                if not hasattr(req, 'token_timestamps'):
                    req.token_timestamps = []
                for _ in accepted_tokens:
                    req.token_timestamps.append(current_time)

                is_complete = False
                if accepted_tokens:
                    new_ids = torch.tensor([accepted_tokens], device=self.model_runner.device, dtype=torch.long)
                    req._input_ids = torch.cat([req._input_ids, new_ids], dim=1)
                    if self.model_runner.tokenizer.eos_token_id in accepted_tokens:
                        is_complete = True

                if len(req.generated_tokens) >= self.max_new_tokens:
                    is_complete = True

                if is_complete:
                    completed_requests.append(req)
                else:
                    continuing_requests.append(req)

        except Exception as e:
            logger.error(f"Error in decode: {e}", exc_info=True)
            for req in batch:
                if req not in completed_requests and req not in continuing_requests:
                    req.error = str(e)
                    completed_requests.append(req)

        # Update metrics
        with self._lock:
            for req in batch:
                self.total_tokens_generated += len(req.draft_tokens)
                self.total_tokens_accepted += req.tokens_accepted

        # Handle completions
        for req in completed_requests:
            self._complete_request(req)

        for req in continuing_requests:
            with self._lock:
                self.decode_queue.append(req)

    def _complete_request(self, request: Request):
        """Complete a request."""
        request.decode_end_time = time.time()
        
        if not request.error:
            try:
                request.result_text = self.model_runner.tokenizer.decode(
                    request.generated_tokens, skip_special_tokens=True
                )
            except Exception as e:
                logger.error(f"Error decoding tokens: {e}")
                request.error = f"Decoding error: {e}"

        with self._lock:
            callback = self.request_callbacks.pop(request.request_id, None)

        if callback:
            try:
                callback(request, request.result_text)
            except Exception as e:
                logger.error(f"Error in callback: {e}")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
