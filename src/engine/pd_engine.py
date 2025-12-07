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
        
        # Initialize model runner
        self.model_runner = ModelRunner(
            model_config=config.model,
            hardware_config=config.hardware
        )
        
        # Controller for draft length
        self.controller = FeedbackController(config.controller)
        
        # Engine state
        self.is_running = False
        self.worker_thread = None
        
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
        
        logger.info(f"PDEngine initialized with 2-lane architecture and batch size {self.batch_size}")
    
    def start(self):
        """Start the engine and worker thread."""
        logger.info("Starting PDEngine...")
        self.model_runner.load_models()
        self.is_running = True
        
        # Start single worker thread
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        logger.info("PDEngine started")

    def stop(self):
        """Stop the engine and worker thread."""
        logger.info("Stopping PDEngine...")
        self.is_running = False
        
        # Wake up worker
        with self._queue_condition:
            self._queue_condition.notify()
        
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
        
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

    def _worker_loop(self):
        """Worker loop that processes requests from both lanes with priority."""
        logger.info("Worker loop started")
        
        while self.is_running:
            requests_to_process = []
            lane_type = None
            
            # Get requests from queues (decode has priority)
            with self._queue_condition:
                while self.is_running and len(self.decode_queue) == 0 and len(self.prefill_queue) == 0:
                    self._queue_condition.wait(timeout=0.1)
                
                if not self.is_running:
                    break
                
                # Priority: Decode > Prefill
                # Collect batch for Decode
                if len(self.decode_queue) > 0:
                    lane_type = TwoLaneType.DECODE
                    count = 0
                    while len(self.decode_queue) > 0 and count < self.batch_size:
                        requests_to_process.append(self.decode_queue.popleft())
                        count += 1
                
                # If no decode work, check prefill (only process 1 prefill at a time typically to avoid blocking)
                # But we can batch if desired. For now, let's just do 1 prefill to keep it simple and responsive.
                elif len(self.prefill_queue) > 0:
                    lane_type = TwoLaneType.PREFILL
                    requests_to_process.append(self.prefill_queue.popleft())
            
            if not requests_to_process:
                continue
            
            # Process batch based on stage
            try:
                if lane_type == TwoLaneType.PREFILL:
                    # Prefill is usually memory intensive, process one by one or small batch
                    for req in requests_to_process:
                        self._handle_prefill(req)
                        
                elif lane_type == TwoLaneType.DECODE:
                    # Batch processing for decode!
                    self._handle_decode_batch(requests_to_process)
                    
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                for req in requests_to_process:
                    req.error = str(e)
                    req.completion_time = time.time()
                    self._trigger_callback(req, None)
        
        logger.info("Worker loop stopped")

    def _trigger_callback(self, request: Request, result: Optional[str]):
        """Helper to trigger callback."""
        with self._lock:
            callback = self.request_callbacks.pop(request.request_id, None)
        
        if callback:
            try:
                callback(request, result)
            except Exception as e:
                logger.error(f"Error in callback: {e}")

    def _handle_prefill(self, request: Request):
        """Handle prefill stage."""
        # Tokenize prompt
        input_ids = self.model_runner.tokenizer(
            request.prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.model.max_model_len
        )["input_ids"].to(self.model_runner.device)
        
        request._input_ids = input_ids
        
        # Transition to decode lane
        request._pd_stage = TwoLaneType.DECODE
        
        with self._queue_condition:
            self.decode_queue.append(request)
            self._queue_condition.notify()

    def _handle_decode_batch(self, batch: List[Request]):
        """Handle decode stage for a batch of requests."""
        if not batch:
            return

        # Filter out completed requests (sanity check)
        active_batch = []
        for req in batch:
            if len(req.generated_tokens) < self.max_new_tokens:
                active_batch.append(req)
            else:
                self._complete_request(req)
        
        if not active_batch:
            return
            
        # Get draft length
        draft_length = self.controller.get_draft_length()
        
        # Prepare inputs
        # We need to pad input_ids to matching length
        input_tensors = [req._input_ids for req in active_batch]
        
        # Pad sequence
        max_len = max([t.size(1) for t in input_tensors])
        padded_inputs = []
        pad_token_id = self.model_runner.tokenizer.pad_token_id or 0
        
        for t in input_tensors:
            pad_len = max_len - t.size(1)
            if pad_len > 0:
                # Pad left side
                padding = torch.full((1, pad_len), pad_token_id, device=t.device, dtype=t.dtype)
                padded_t = torch.cat([padding, t], dim=1)
                padded_inputs.append(padded_t)
            else:
                padded_inputs.append(t)
        
        batch_input_ids = torch.cat(padded_inputs, dim=0)
        
        # 1. Generate Draft Tokens (Batched)
        draft_tokens_batch = self.model_runner.generate_draft_tokens(batch_input_ids, draft_length)
        
        # 2. Verify Tokens (Batched)
        accepted_tokens_batch, num_accepted_list = self.model_runner.verify_tokens(batch_input_ids, draft_tokens_batch)
        
        # 3. Update Requests
        requeue_list = []
        
        for i, req in enumerate(active_batch):
            accepted_tokens = accepted_tokens_batch[i]
            num_accepted = num_accepted_list[i]
            
            # Update metrics
            req.tokens_generated += len(draft_tokens_batch[i])
            req.tokens_accepted += num_accepted
            
            with self._lock:
                self.total_tokens_generated += len(draft_tokens_batch[i])
                self.total_tokens_accepted += num_accepted
            
            # Append accepted tokens
            req.generated_tokens.extend(accepted_tokens)
            
            # Update input for next step
            if accepted_tokens:
                new_ids = torch.tensor([accepted_tokens], device=self.model_runner.device, dtype=torch.long)
                req._input_ids = torch.cat([req._input_ids, new_ids], dim=1)
                
                # Check for EOS
                if self.model_runner.tokenizer.eos_token_id in accepted_tokens:
                    self._complete_request(req)
                    continue
            
            # Check max tokens
            if len(req.generated_tokens) >= self.max_new_tokens:
                self._complete_request(req)
                continue
                
            requeue_list.append(req)
            
        # Update controller (average acceptance)
        if num_accepted_list:
            avg_acceptance = sum(num_accepted_list) / (len(num_accepted_list) * draft_length) if draft_length > 0 else 0
            self.controller.record_acceptance(avg_acceptance)
        
        # Requeue active requests
        if requeue_list:
            with self._queue_condition:
                for req in requeue_list:
                    self.decode_queue.append(req)
                self._queue_condition.notify()

    def _complete_request(self, request: Request):
        """Complete a request."""
        # Decode result
        result_text = self.model_runner.tokenizer.decode(request.generated_tokens, skip_special_tokens=True)
        request.completion_time = time.time()
        self._trigger_callback(request, result_text)

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
