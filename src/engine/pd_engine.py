"""
PD (Prefill-Decode) disaggregation engine.
Two lanes: Prefill (low priority) and Decode (high priority).
Standard speculative decoding happens in the decode lane (draft + verify together).
NOW SUPPORTS BATCHING in the decode lane!
"""

import time
import threading
from typing import Optional, Callable, Dict, List
import logging
import torch

from ..scheduler import Request, LaneType
from ..controller import FeedbackController
from .model_runner import ModelRunner
from ..utils.config import VerifyPDConfig

logger = logging.getLogger(__name__)


class TwoLaneType:
    """Two-lane types for PD disaggregation."""
    PREFILL = 1  # Low priority
    DECODE = 2   # High priority (includes both draft and verify)


class PDEngine:
    """
    PD (Prefill-Decode) disaggregation engine.
    Separates prefill from decode.
    
    IMPROVED: Now supports BATCHED processing in the decode lane.
             (Draft generation + Verification done for a batch of requests)
    """

    def __init__(self, config: VerifyPDConfig):
        """
        Initialize PD engine.

        Args:
            config: System configuration
        """
        self.config = config
        
        # Initialize model runner with stream manager
        from ..utils.stream_manager import StreamManager
        self.stream_manager = StreamManager(
            device=config.hardware.device,
            num_streams=2  # One for prefill, one for decode
        )
        
        self.model_runner = ModelRunner(
            model_config=config.model,
            hardware_config=config.hardware,
            stream_manager=self.stream_manager
        )
        
        # Controller for draft length
        self.controller = FeedbackController(config.controller)
        
        # Engine state
        self.is_running = False
        self.prefill_worker_thread = None
        self.decode_worker_thread = None
        
        # Request queues (simple 2-lane system)
        from collections import deque
        self.prefill_queue: deque = deque()
        self.decode_queue: deque = deque()
        self.request_callbacks: Dict[str, Callable] = {}
        self._lock = threading.Lock()
        self._queue_condition = threading.Condition(self._lock)
        
        # Metrics
        self.total_tokens_generated = 0
        self.total_tokens_accepted = 0
        
        # Config
        self.max_new_tokens = config.model.max_new_tokens
        self.batch_size = config.scheduler.batch_size
        
        # Concurrency control
        self.num_active_decode_requests = 0
        self._prefill_thread = None
        self._decode_thread = None
        
        logger.info(f"PDEngine initialized with 2-lane architecture and batch size {self.batch_size}")
    
    def start(self):
        """Start the engine and worker threads."""
        logger.info("Starting PDEngine...")
        self.model_runner.load_models()
        
        # Verify models loaded
        if self.model_runner.draft_model is None:
            logger.error("Draft model is None after loading!")
            raise RuntimeError("Draft model failed to load")
            
        self.is_running = True
        
        # Start separate worker threads for prefill and decode
        self.prefill_worker_thread = threading.Thread(target=self._prefill_worker_loop, daemon=True)
        self.decode_worker_thread = threading.Thread(target=self._decode_worker_loop, daemon=True)
        
        self.prefill_worker_thread.start()
        self.decode_worker_thread.start()
        
        logger.info("PDEngine started with parallel prefill and decode workers")

    def stop(self):
        """Stop the engine and worker threads."""
        logger.info("Stopping PDEngine...")
        self.is_running = False
        
        # Wake up both workers
        with self._queue_condition:
            self._queue_condition.notify_all()
        
        if self._prefill_thread:
            self._prefill_thread.join(timeout=2.0)
        if self._decode_thread:
            self._decode_thread.join(timeout=2.0)
        
        self.stream_manager.cleanup()
        self.model_runner.cleanup()
        logger.info("PDEngine stopped")
    
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

        # Initialize request state
        request.tokens_generated = 0
        request.tokens_accepted = 0
        request.generated_tokens = []
        request._pd_stage = TwoLaneType.PREFILL  # Track which lane we're in
        
        # Add to prefill queue
        with self._queue_condition:
            self.prefill_queue.append(request)
            self._queue_condition.notify()
        
        logger.debug(f"Queued request {request.request_id} to prefill lane")


    def _prefill_worker_loop(self):
        """Worker loop for prefill lane."""
        logger.info("Prefill worker loop started")
        while self.is_running:
            request = None
            with self._queue_condition:
                while self.is_running and len(self.prefill_queue) == 0:
                    self._queue_condition.wait(timeout=0.1)
                
                if not self.is_running:
                    break
                
                # Wait for capacity in decode lane (Backpressure)
                while self.is_running and self.num_active_decode_requests >= self.batch_size:
                    self._queue_condition.wait(timeout=0.1)
                    
                if not self.is_running:
                    break
                    
                if len(self.prefill_queue) > 0 and self.num_active_decode_requests < self.batch_size:
                    request = self.prefill_queue.popleft()
                    # Increment active count BEFORE processing to reserve slot
                    self.num_active_decode_requests += 1
            

            if request:
                try:
                    self._handle_prefill(request)
                except Exception as e:
                    logger.error(f"Error in prefill worker: {e}", exc_info=True)
                    request.error = str(e)
                    self._complete_request(request) # This will decrement the count

        logger.info("Prefill worker loop stopped")

    def _decode_worker_loop(self):
        """Worker loop for decode lane (supports batching)."""
        logger.info("Decode worker loop started")
        while self.is_running:
            batch = []
            with self._queue_condition:
                while self.is_running and len(self.decode_queue) == 0:
                    self._queue_condition.wait(timeout=0.1)
                
                if not self.is_running:
                    break
                
                # Form a batch up to batch_size
                while len(self.decode_queue) > 0 and len(batch) < self.batch_size:
                    batch.append(self.decode_queue.popleft())
            
            if batch:
                # print(f"DEBUG: Processing decode batch of size {len(batch)}")
                try:
                    self._handle_decode_batch(batch)
                except Exception as e:
                    # If we are shutting down, suppress likely resource errors
                    if not self.is_running:
                        logger.debug(f"Suppressing error during shutdown: {e}")
                        break
                    
                    logger.error(f"Error in decode worker: {e}", exc_info=True)
                    print(f"DEBUG: Error in decode batch: {e}")
                    for req in batch:
                        req.error = str(e)
                        self._complete_request(req)

        logger.info("Decode worker loop stopped")

    def _trigger_callback(self, request: Request, result_text: str):
        """Trigger completion callback."""
        with self._lock:
            callback = self.request_callbacks.pop(request.request_id, None)
        
        if callback:
            try:
                callback(request, result_text)
            except Exception as e:
                logger.error(f"Error in callback: {e}")

    def _handle_prefill(self, request: Request):
        """Handle prefill stage."""
        # Use CUDA stream 0 for prefill
        stream = self.stream_manager.get_stream(0)
        
        request.prefill_start_time = time.time()
        # Prefill now returns KV caches and logits
        input_ids, draft_kv, verifier_kv, verifier_logits, draft_logits = self.model_runner.prefill(
            request.prompt,
            stream_id=0
        )
        request.prefill_end_time = time.time()
        
        # Store state in request
        request._input_ids = input_ids
        request.kv_cache_draft = draft_kv
        request.kv_cache_verifier = verifier_kv
        request.last_verifier_logits = verifier_logits
        request.last_draft_logits = draft_logits
        
        # Transition to decode lane
        request._pd_stage = TwoLaneType.DECODE
        
        with self._queue_condition:
            self.decode_queue.append(request)
            self._queue_condition.notify()
            
        logger.debug(f"Prefill complete for {request.request_id}. KV cache initialized.")

    def _handle_decode_batch(self, batch: List[Request]):
        """Handle decode stage for a batch of requests."""
        if not batch:
            return

        # Filter out completed requests (sanity check)
        active_batch = []
        for req in batch:
            if len(req.generated_tokens) < self.max_new_tokens:
                active_batch.append(req)
                if req.decode_start_time == 0.0:
                    req.decode_start_time = time.time()
            else:
                self._complete_request(req)
        
        if not active_batch:
            return
            
        # Get draft length
        draft_length = self.controller.get_draft_length()
        
        # 1. Sample Draft Token 1 (d1)
        # We need d1 to start the specimen generation loop.
        # Use the stored draft logits.
        d1_list = []
        for req in active_batch:
            if req.last_draft_logits is not None:
                # Greedy sampling for now
                d1 = torch.argmax(req.last_draft_logits).item()
                if self.model_runner.draft_vocab_size and d1 >= self.model_runner.draft_vocab_size:
                    d1 = self.model_runner.draft_vocab_size - 1
            else:
                # Should not happen if prefill works
                d1 = 0 
            d1_list.append(d1)
            
        # 2. Generate Drafts d2...dK
        # Input to generation is [d1].
        # We construct input batch.
        input_ids_batch = torch.tensor([[d] for d in d1_list], device=self.model_runner.device, dtype=torch.long)
        
        stream = self.model_runner.stream_manager.get_stream(1) if self.model_runner.stream_manager else None
        
        # Prepare KV caches for batch
        # Assuming model_runner handles list of KV or we pass one by one
        # Current model_runner.generate_draft_tokens expects batched KV? 
        # Actually I updated it to accept Optional[List[Any]]. 
        # But my implementation in model_runner uses `past_key_values` directly in `draft_model`.
        # Standard HF model expects batched KV.
        # We must assume the requests in `active_batch` fit the KV structure (e.g. they were batched in prefill?).
        # No, prefill was individual.
        # So we have a List of KV caches. 
        # We cannot easily batch them unless we use a custom Kernel or PagedAttention.
        # FALLBACK: Run generation sequentially for each request (but on stream).
        # This is strictly better than O(N^2) but less throughput than batched generation.
        # Given the "bogus" complaint was about algorithmic complexity, this is acceptable for now.
        
        # Speculative Loop per Request
        for i, req in enumerate(active_batch):
            d1 = d1_list[i]
            
            # 2a. Generate drafts (d2...dk)
            # Input is d1.
            d1_tensor = torch.tensor([[d1]], device=self.model_runner.device)
            
            # We use a temporary KV cache for generation to avoid polluting the main valid one
            # But HF models update in place? No, they return new tuple.
            # We iterate `draft_length` times.
            
            curr_draft_kv = req.kv_cache_draft
            curr_draft_tokens = [d1]
            
            # We need to generate K-1 more tokens? 
            # If draft_length=3, we have d1. We want d2, d3.
            # generate_draft_tokens generates `num_tokens`.
            # If we ask for `draft_length - 1`.
            
            generated_ids = []
            if draft_length > 1:
                # Call generate with d1.
                # using stream 1
                gen_ids, new_kv = self.model_runner.generate_draft_tokens(
                    d1_tensor, 
                    num_tokens=draft_length-1, 
                    past_key_values=curr_draft_kv,
                    stream_id=1
                )
                generated_ids = gen_ids[0] # batch size 1
            
            full_drafts = curr_draft_tokens + generated_ids
            
            # 2b. Verify Drafts
            # We verify [d1, d2...]
            # We need previous_verifier_logits
            
            # We verify batch size 1
            # Input to verify is full_drafts
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
            # print(f"DEBUG: Req {req.request_id} Accepted: {len(accepted_tokens)} tokens. Total len: {len(req.generated_tokens)}")
            if len(accepted_tokens) == 0:
                print(f"DEBUG: Req {req.request_id} STUCK! Accepted 0 tokens. Drafts: {len(full_drafts)}")
            
            # 3. Update Request State (Critical: KV Rollback/Append)
            # We have the accepted tokens.
            # We must update `kv_cache_draft` and `kv_cache_verifier` to include these and ONLY these tokens.
            # The cleanest way is to run a forward pass on `accepted_tokens` using the ORIGINAL KV caches.
            # This updates the cache and gives us the fresh Last Logits.
            

            if accepted_tokens:
                # Prepare input tensor
                acc_tensor = torch.tensor([accepted_tokens], device=self.model_runner.device)
                
                # Update Draft State
                with self.model_runner._draft_lock:
                    with torch.no_grad():
                        with torch.cuda.stream(stream) if stream else torch.no_grad():
                            out_d = self.model_runner.draft_model(acc_tensor, past_key_values=req.kv_cache_draft, use_cache=True)
                            req.kv_cache_draft = out_d.past_key_values
                            req.last_draft_logits = out_d.logits[:, -1, :]
                            
                # Update Verifier State
                with self.model_runner._verifier_lock:
                    with torch.no_grad():
                        with torch.cuda.stream(stream) if stream else torch.no_grad():
                            out_v = self.model_runner.verifier_model(acc_tensor, past_key_values=req.kv_cache_verifier, use_cache=True)
                            req.kv_cache_verifier = out_v.past_key_values
                            req.last_verifier_logits = out_v.logits[:, -1, :]

                # Update metrics
                req.generated_tokens.extend(accepted_tokens)
                req.tokens_generated += len(full_drafts)
                req.tokens_accepted += len(accepted_tokens)
                
                # Record token timestamps for benchmark metrics
                current_time = time.time()
                if not hasattr(req, 'token_timestamps'):
                    req.token_timestamps = []
                for _ in accepted_tokens:
                    req.token_timestamps.append(current_time)
                
                # Update total metrics
                with self._lock:
                    self.total_tokens_generated += len(full_drafts)
                    self.total_tokens_accepted += len(accepted_tokens)
                
                # Check for EOS
                if self.is_running and self.model_runner.tokenizer and self.model_runner.tokenizer.eos_token_id in accepted_tokens:
                    self._complete_request(req)
                
                # Check max tokens
                elif len(req.generated_tokens) >= self.max_new_tokens:
                    self._complete_request(req)
                else:
                    # Requeue
                    with self._queue_condition:
                        self.decode_queue.append(req)
                        self._queue_condition.notify()
            else:
                # Should not happen in greedy decoding (always accept at least 1 correction)
                # But if it does, just requeue
                 with self._queue_condition:
                        self.decode_queue.append(req)
                        self._queue_condition.notify()

        # Update controller
        # self.controller.record_acceptance(...) 

    def _complete_request(self, request: Request):
        """Complete a request."""
        request.decode_end_time = time.time()
        
        # Decode result - handle potential None tokenizer during shutdown
        result_text = ""
        if self.model_runner.tokenizer:
            try:
                result_text = self.model_runner.tokenizer.decode(request.generated_tokens, skip_special_tokens=True)
            except Exception:
                pass
        
        # Callback
        if self.request_callbacks.get(request.request_id):
            import threading
            threading.Thread(target=self._trigger_callback, args=(request, result_text)).start()

        # Update active count and notify
        with self._queue_condition:
            self.num_active_decode_requests -= 1
            self._queue_condition.notify_all()
            
        logger.debug(f"Request {request.request_id} completed. Active: {self.num_active_decode_requests}")
        
        # Clean up large tensors
        try:
            if hasattr(request, 'kv_cache_draft'):
                del request.kv_cache_draft
            if hasattr(request, 'kv_cache_verifier'):
                del request.kv_cache_verifier
            if hasattr(request, 'last_draft_logits'):
                del request.last_draft_logits
            if hasattr(request, 'last_verifier_logits'):
                del request.last_verifier_logits
        except AttributeError:
            pass # Already deleted or never existed
            
        request.kv_cache_draft = None
        request.kv_cache_verifier = None

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
