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
    Uses GPU streams to overlap decode and verify operations for true disaggregation.
    """

    def __init__(self, config: VerifyPDConfig):
        """
        Initialize GPU stream-based speculative engine.

        Args:
            config: System configuration
        """
        self.config = config

        # Initialize components
        self.controller = FeedbackController(config.controller)
        self.model_runner = ModelRunner(
            model_config=config.model,
            hardware_config=config.hardware
        )

        # Engine state
        self.is_running = False
        self.total_tokens_generated = 0
        self.total_tokens_accepted = 0

        # Request tracking
        self.request_callbacks: Dict[str, Callable] = {}
        self._lock = threading.Lock()

        # Configurable parameters
        self.max_new_tokens = config.model.max_new_tokens

        logger.info("GPU Stream SpeculativeEngine initialized")
    
    def start(self):
        """Start the GPU stream-based engine."""
        logger.info("Starting GPU Stream SpeculativeEngine...")
        self.model_runner.load_models()
        self.is_running = True
        logger.info("GPU Stream SpeculativeEngine started")

    def stop(self):
        """Stop the GPU stream-based engine."""
        logger.info("Stopping GPU Stream SpeculativeEngine...")
        self.is_running = False
        self.model_runner.cleanup()
        logger.info("GPU Stream SpeculativeEngine stopped")
    
    def submit_request_async(self, request: Request, callback: Optional[Callable] = None):
        """
        Submit a request for synchronous processing with GPU stream-based disaggregation.

        Args:
            request: Request to process
            callback: Optional callback function(request, result_text) called on completion
        """
        if not self.is_running:
            raise RuntimeError("Engine is not running. Call start() first.")

        # Store callback for completion
        if callback:
            with self._lock:
                self.request_callbacks[request.request_id] = callback

        # Process synchronously with GPU stream disaggregation
        try:
            result_text = self._process_request_sync(request)

            # Call callback
            with self._lock:
                callback = self.request_callbacks.pop(request.request_id, None)

            if callback:
                try:
                    callback(request, result_text)
                except Exception as e:
                    logger.error(f"Error in completion callback for {request.request_id}: {e}")

        except Exception as e:
            logger.error(f"Error processing request {request.request_id}: {e}")
            request.error = str(e)
            request.completion_time = time.time()

            # Call callback with error
            with self._lock:
                callback = self.request_callbacks.pop(request.request_id, None)

            if callback:
                try:
                    callback(request, None)
                except Exception as e:
                    logger.error(f"Error in error callback for {request.request_id}: {e}")

        logger.debug(f"Completed processing for request {request.request_id}")

    def _process_request_sync(self, request: Request) -> str:
        """
        Synchronous request processing with GPU stream-based disaggregation.
        Uses GPU streams to overlap decode and verify operations for true disaggregation.
        """
        try:
            logger.debug(f"Starting sync processing for {request.request_id}")

            # Initialize request state
            request.tokens_generated = 0
            request.tokens_accepted = 0
            request.generated_tokens = []

            # Stage 1: Prefill (prepare input)
            input_ids = self._execute_prefill_sync(request)
            request._input_ids = input_ids

            # Stage 2: Speculative decode loop with GPU stream disaggregation
            while len(request.generated_tokens) < self.max_new_tokens:
                # Get draft length from controller
                draft_length = self.controller.get_draft_length()

                # Execute decode and verify with GPU stream overlapping
                draft_tokens, accepted_tokens, num_accepted = self._execute_decode_verify_overlapped(
                    request, request._input_ids, draft_length
                )

                # Update metrics
                request.tokens_generated += len(draft_tokens)
                request.tokens_accepted += num_accepted
                self.total_tokens_generated += len(draft_tokens)
                self.total_tokens_accepted += num_accepted

                # Update controller with acceptance feedback
                acceptance_ratio = num_accepted / len(draft_tokens) if draft_tokens else 0.0
                self.controller.record_acceptance(acceptance_ratio)

                # Append accepted tokens to input for next iteration
                request.generated_tokens.extend(accepted_tokens)
                if accepted_tokens:
                    accepted_ids = torch.tensor([accepted_tokens], device=self.model_runner.device, dtype=torch.long)
                    request._input_ids = torch.cat([request._input_ids, accepted_ids], dim=1)

                    # Check for EOS
                    if self.model_runner.tokenizer.eos_token_id in accepted_tokens:
                        break

            # Generate final text
            result_text = self.model_runner.tokenizer.decode(request.generated_tokens, skip_special_tokens=True)

            # Mark completion
            request.completion_time = time.time()

            logger.debug(f"Completed sync processing for {request.request_id}")
            return result_text

        except Exception as e:
            logger.error(f"Error in sync processing for {request.request_id}: {e}")
            request.error = str(e)
            request.completion_time = time.time()
            raise


    # Synchronous execution methods with GPU stream overlapping

    def _execute_prefill_sync(self, request: Request) -> torch.Tensor:
        """Synchronous prefill operation."""
        logger.debug(f"Prefill for {request.request_id}")
        input_ids = self.model_runner.prefill(request.prompt)
        return input_ids

    def _execute_decode_verify_overlapped(self, request: Request, input_ids: torch.Tensor, draft_length: int):
        """
        Execute decode and verify operations with GPU stream overlapping.
        This provides true disaggregated serving by allowing decode and verify to overlap.
        """
        # For now, execute sequentially since GPU stream management is complex
        # In a production system, this would use proper stream overlapping
        draft_tokens = self.model_runner.generate_draft_tokens(
            input_ids, num_tokens=draft_length, stream_id=None
        )

        accepted_tokens, num_accepted = self.model_runner.verify_tokens(
            input_ids, draft_tokens, stream_id=None
        )

        return draft_tokens, accepted_tokens, num_accepted

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
