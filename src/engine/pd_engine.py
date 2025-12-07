"""
PD (Prefill-Decode) disaggregation engine.
Two lanes: Prefill (low priority) and Decode (high priority).
Standard speculative decoding happens in the decode lane (draft + verify together).
"""

import time
import threading
from typing import Optional, Callable, Dict
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
    Separates prefill from decode, but keeps draft+verify together in decode lane.
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
        
        logger.info("PDEngine initialized with 2-lane architecture")
    
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
            request = None
            
            # Get request from queues (decode has priority)
            with self._queue_condition:
                while self.is_running and len(self.decode_queue) == 0 and len(self.prefill_queue) == 0:
                    self._queue_condition.wait(timeout=0.1)
                
                if not self.is_running:
                    break
                
                # Priority: Decode > Prefill
                if len(self.decode_queue) > 0:
                    request = self.decode_queue.popleft()
                elif len(self.prefill_queue) > 0:
                    request = self.prefill_queue.popleft()
            
            if request is None:
                continue
            
            # Process request based on stage
            try:
                if request._pd_stage == TwoLaneType.PREFILL:
                    self._handle_prefill(request)
                elif request._pd_stage == TwoLaneType.DECODE:
                    self._handle_decode(request)
                    
            except Exception as e:
                logger.error(f"Error processing request {request.request_id}: {e}")
                request.error = str(e)
                request.completion_time = time.time()
                
                # Trigger callback with error
                with self._lock:
                    callback = self.request_callbacks.pop(request.request_id, None)
                
                if callback:
                    try:
                        callback(request, None)
                    except Exception as e:
                        logger.error(f"Error in error callback: {e}")
        
        logger.info("Worker loop stopped")

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

    def _handle_decode(self, request: Request):
        """Handle decode stage (draft + verify together)."""
        # Check if we're done
        if len(request.generated_tokens) >= self.max_new_tokens:
            self._complete_request(request)
            return
        
        # Get draft length
        draft_length = self.controller.get_draft_length()
        
        # Generate draft tokens
        draft_tokens = self._generate_draft(request._input_ids, draft_length)
        
        # Verify draft tokens (standard speculative decoding)
        accepted_tokens, num_accepted = self._verify_draft(request._input_ids, draft_tokens)
        
        # Update metrics
        request.tokens_generated += len(draft_tokens)
        request.tokens_accepted += num_accepted
        
        with self._lock:
            self.total_tokens_generated += len(draft_tokens)
            self.total_tokens_accepted += num_accepted
        
        # Update controller
        acceptance_ratio = num_accepted / len(draft_tokens) if draft_tokens else 0.0
        self.controller.record_acceptance(acceptance_ratio)
        
        # Append accepted tokens
        request.generated_tokens.extend(accepted_tokens)
        
        # Update input
        if accepted_tokens:
            new_ids = torch.tensor([accepted_tokens], device=self.model_runner.device, dtype=torch.long)
            request._input_ids = torch.cat([request._input_ids, new_ids], dim=1)
            
            # Check for EOS
            if self.model_runner.tokenizer.eos_token_id in accepted_tokens:
                self._complete_request(request)
                return
        
        # Check max tokens
        if len(request.generated_tokens) >= self.max_new_tokens:
            self._complete_request(request)
            return
        
        # Continue decoding
        with self._queue_condition:
            self.decode_queue.append(request)
            self._queue_condition.notify()

    def _generate_draft(self, input_ids: torch.Tensor, num_tokens: int):
        """Generate draft tokens using small model."""
        draft_tokens = []
        current_ids = input_ids
        
        with torch.no_grad():
            for _ in range(num_tokens):
                outputs = self.model_runner.draft_model(current_ids)
                logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1)
                token_id = next_token.item()
                
                # Clamp to vocab size
                if self.model_runner.draft_vocab_size is not None and token_id >= self.model_runner.draft_vocab_size:
                    token_id = self.model_runner.draft_vocab_size - 1
                
                draft_tokens.append(token_id)
                next_token_tensor = torch.tensor([[token_id]], device=current_ids.device, dtype=current_ids.dtype)
                current_ids = torch.cat([current_ids, next_token_tensor], dim=1)
        
        return draft_tokens

    def _verify_draft(self, input_ids: torch.Tensor, draft_tokens):
        """Verify draft tokens using large model."""
        draft_ids = torch.tensor([draft_tokens], device=self.model_runner.device, dtype=input_ids.dtype)
        full_input = torch.cat([input_ids, draft_ids], dim=1)
        
        accepted_tokens = []
        
        with torch.no_grad():
            outputs = self.model_runner.verifier_model(full_input)
            logits = outputs.logits
            
            for i, draft_token in enumerate(draft_tokens):
                logit_idx = input_ids.size(1) + i
                if logit_idx >= logits.size(1):
                    break
                
                predicted_token = torch.argmax(logits[:, logit_idx, :], dim=-1).item()
                
                # Clamp to vocab size
                if self.model_runner.verifier_vocab_size is not None and predicted_token >= self.model_runner.verifier_vocab_size:
                    predicted_token = self.model_runner.verifier_vocab_size - 1
                
                if predicted_token == draft_token:
                    accepted_tokens.append(draft_token)
                else:
                    accepted_tokens.append(predicted_token)
                    break
        
        return accepted_tokens, len(accepted_tokens)

    def _complete_request(self, request: Request):
        """Complete a request."""
        # Decode result
        result_text = self.model_runner.tokenizer.decode(request.generated_tokens, skip_special_tokens=True)
        request.completion_time = time.time()
        
        # Trigger callback
        with self._lock:
            callback = self.request_callbacks.pop(request.request_id, None)
        
        if callback:
            try:
                callback(request, result_text)
            except Exception as e:
                logger.error(f"Error in callback for {request.request_id}: {e}")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
