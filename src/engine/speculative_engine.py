"""
GPU stream-based speculative decoding engine with disaggregated serving.
"""

import time
import threading
from typing import Optional, List, Dict, Callable, Any
import logging
import torch

from ..scheduler import Request
from ..controller import FeedbackController
from .model_runner import ModelRunner
from ..utils.config import VerifyPDConfig

logger = logging.getLogger(__name__)


class SpeculativeEngine:
    """
    GPU stream-based speculative decoding engine with disaggregated serving.
    Uses a three-lane scheduler and worker thread to interleave Prefill, Decode, and Verify steps.
    """

    def __init__(self, config: VerifyPDConfig):
        """
        Initialize GPU stream-based speculative engine.

        Args:
            config: System configuration
        """
        self.config = config

        # Initialize components
        from ..scheduler import ThreeLaneScheduler
        self.scheduler = ThreeLaneScheduler(config.scheduler)
        self.controller = FeedbackController(config.controller)
        self.model_runner = ModelRunner(
            model_config=config.model,
            hardware_config=config.hardware
        )

        # Engine state
        self.is_running = False
        self.worker_thread = None
        self.total_tokens_generated = 0
        self.total_tokens_accepted = 0

        # Request tracking
        self.request_callbacks: Dict[str, Callable] = {}
        self._lock = threading.Lock()

        # Configurable parameters
        self.max_new_tokens = config.model.max_new_tokens

        logger.info("GPU Stream SpeculativeEngine initialized with ThreeLaneScheduler")
    
    def start(self):
        """Start the engine and worker thread."""
        logger.info("Starting SpeculativeEngine...")
        self.model_runner.load_models()
        self.is_running = True
        
        # Start worker thread
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        logger.info("SpeculativeEngine started")

    def stop(self):
        """Stop the engine and worker thread."""
        logger.info("Stopping SpeculativeEngine...")
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
        self.model_runner.cleanup()
        logger.info("SpeculativeEngine stopped")
    
    def submit_request_async(self, request: Request, callback: Optional[Callable] = None):
        """
        Submit a request asynchronously to the scheduler.

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

        # Submit to Prefill lane
        from ..scheduler import LaneType
        request.stage = LaneType.PREFILL
        
        # Initialize request state if not already
        request.tokens_generated = 0
        request.tokens_accepted = 0
        request.generated_tokens = []
        
        success = self.scheduler.submit_request(request)
        if not success:
            logger.error(f"Failed to submit request {request.request_id} (queue full)")
            request.error = "Queue full"
            self._handle_completion(request)
        else:
            logger.debug(f"Submitted request {request.request_id} to scheduler")

    def _worker_loop(self):
        """Background worker loop processing requests from the scheduler."""
        logger.info("Worker loop started")
        
        while self.is_running:
            try:
                # Get next batch from scheduler
                batch = self.scheduler.get_next_batch()
                
                if not batch:
                    time.sleep(0.001)  # Sleep briefly to avoid busy wait
                    continue
                
                # Check consistency
                first_stage = batch[0].stage
                if any(r.stage != first_stage for r in batch):
                    logger.error("Batch contains mixed stages, falling back to sequential")
                    for request in batch:
                        self._process_single_request_step(request)
                    continue

                # Process batch based on stage
                from ..scheduler import LaneType
                if first_stage == LaneType.PREFILL:
                    self._handle_prefill_batch(batch)
                elif first_stage == LaneType.DECODE:
                    self._handle_decode_batch(batch)
                elif first_stage == LaneType.VERIFY:
                    self._handle_verify_batch(batch)
                else:
                    logger.error(f"Unknown stage {first_stage}")
                    for req in batch:
                        req.error = f"Unknown stage {first_stage}"
                        self._handle_completion(req)
                    
            except Exception as e:
                logger.error(f"Error in worker loop: {e}", exc_info=True)
                time.sleep(0.1)  # Backoff on error

    def _handle_prefill_batch(self, batch: List[Request]):
        """Handle batch prefill step."""
        from ..scheduler import LaneType
        
        prompts = [req.prompt for req in batch]
        
        # Tokenize batch
        inputs = self.model_runner.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.model.max_model_len
        )
        
        input_ids_batch = inputs["input_ids"].to(self.model_runner.device)
        
        # Assign back to requests
        for i, req in enumerate(batch):
            # Keep batch dim [1, L]
            req._input_ids = input_ids_batch[i:i+1]
            
            # Transition
            self.scheduler.transition_request(req, LaneType.DECODE)

    def _handle_decode_batch(self, batch: List[Request]):
        """Handle batch decode step."""
        from ..scheduler import LaneType
        
        # Prepare inputs
        # Pad input_ids to create a batch tensor
        max_len = max(req._input_ids.size(1) for req in batch)
        padded_inputs = []
        
        for req in batch:
            input_ids = req._input_ids
            curr_len = input_ids.size(1)
            if curr_len < max_len:
                pad_len = max_len - curr_len
                pad_token = self.model_runner.tokenizer.pad_token_id or 0
                padding = torch.full((1, pad_len), pad_token, device=input_ids.device, dtype=input_ids.dtype)
                padded_input = torch.cat([padding, input_ids], dim=1)
                padded_inputs.append(padded_input)
            else:
                padded_inputs.append(input_ids)
                
        batch_input_ids = torch.cat(padded_inputs, dim=0) # [B, max_len]
        
        # Get draft length
        draft_length = self.controller.get_draft_length()
        
        # Generate
        draft_tokens_batch = self.model_runner.generate_draft_tokens(
            batch_input_ids, num_tokens=draft_length
        )
        
        # Update requests
        for i, req in enumerate(batch):
            req.draft_tokens = draft_tokens_batch[i]
            self.scheduler.transition_request(req, LaneType.VERIFY)

    def _handle_verify_batch(self, batch: List[Request]):
        """Handle batch verify step."""
        from ..scheduler import LaneType
        
        # Prepare inputs
        max_len = max(req._input_ids.size(1) for req in batch)
        padded_inputs = []
        
        for req in batch:
            input_ids = req._input_ids
            curr_len = input_ids.size(1)
            if curr_len < max_len:
                pad_len = max_len - curr_len
                pad_token = self.model_runner.tokenizer.pad_token_id or 0
                padding = torch.full((1, pad_len), pad_token, device=input_ids.device, dtype=input_ids.dtype)
                padded_input = torch.cat([padding, input_ids], dim=1)
                padded_inputs.append(padded_input)
            else:
                padded_inputs.append(input_ids)
                
        batch_input_ids = torch.cat(padded_inputs, dim=0)
        
        draft_batch = [req.draft_tokens for req in batch]
        
        # Verify
        accepted_tokens_batch, num_accepted_list = self.model_runner.verify_tokens(
            batch_input_ids, draft_batch
        )
        
        # Process results
        for i, req in enumerate(batch):
            accepted_tokens = accepted_tokens_batch[i]
            num_accepted = num_accepted_list[i]
            
            # Update metrics
            req.tokens_generated += len(req.draft_tokens)
            req.tokens_accepted += num_accepted
            self.total_tokens_generated += len(req.draft_tokens)
            self.total_tokens_accepted += num_accepted
            
            # Feedback
            ratio = num_accepted / len(req.draft_tokens) if req.draft_tokens else 0.0
            self.controller.record_acceptance(ratio)
            
            # Append tokens
            req.generated_tokens.extend(accepted_tokens)
            
            if accepted_tokens:
                new_ids = torch.tensor([accepted_tokens], device=self.model_runner.device, dtype=torch.long)
                req._input_ids = torch.cat([req._input_ids, new_ids], dim=1)
                
                # Check EOS
                if self.model_runner.tokenizer.eos_token_id in accepted_tokens:
                    self._handle_completion(req)
                    continue
            
            # Check max tokens
            if len(req.generated_tokens) >= self.max_new_tokens:
                self._handle_completion(req)
                continue
                
            # Continue
            self.scheduler.transition_request(req, LaneType.DECODE)

    def _handle_completion(self, request: Request):
        """Handle request completion."""
        # Convert tokens to text
        if not request.error:
            try:
                request.result_text = self.model_runner.tokenizer.decode(
                    request.generated_tokens, skip_special_tokens=True
                )
            except Exception as e:
                logger.error(f"Error decoding tokens: {e}")
                request.error = f"Decoding error: {e}"
        
        # Mark complete in scheduler
        self.scheduler.complete_request(request)
        
        # Trigger callback
        with self._lock:
            callback = self.request_callbacks.pop(request.request_id, None)
            
        if callback:
            try:
                callback(request, request.result_text)
            except Exception as e:
                logger.error(f"Error in callback for {request.request_id}: {e}")
    
    def _process_single_request_step(self, request: Request):
        """Legacy fallback."""
        from ..scheduler import LaneType
        if request.stage == LaneType.PREFILL:
            self._handle_prefill_batch([request])
        elif request.stage == LaneType.DECODE:
            self._handle_decode_batch([request])
        elif request.stage == LaneType.VERIFY:
            self._handle_verify_batch([request])

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
