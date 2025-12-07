"""
GPU stream-based speculative decoding engine with disaggregated serving.
"""

import time
import threading
from typing import Optional, List, Dict, Callable, Any
import logging
import torch

from ..scheduler import Request, RequestState, LaneType, ThreeLaneScheduler
from ..controller import FeedbackController
from .model_runner import ModelRunner
from ..utils.config import VerifyPDConfig

logger = logging.getLogger(__name__)


class RequestContext:
    """Context for tracking request state across async processing stages."""

    def __init__(self, request: Request):
        """
        Initialize request context.

        Args:
            request: The request being processed
        """
        self.request = request
        self.state = RequestState.QUEUED

        # Input/output tracking
        self.input_ids: Optional[torch.Tensor] = None
        self.generated_tokens: List[int] = []
        self.draft_tokens: List[int] = []

        # Iteration tracking
        self.current_iteration = 0
        self.draft_length = 0

        # Timing
        self.start_time = time.time()

        # Error handling
        self.error: Optional[Exception] = None


class WorkerThread(threading.Thread):
    """Background worker thread that processes requests from scheduler."""

    def __init__(self, engine: 'SpeculativeEngine'):
        """
        Initialize worker thread.

        Args:
            engine: Reference to parent SpeculativeEngine
        """
        super().__init__(daemon=True, name="VerifyPD-Worker")
        self.engine = engine
        self.scheduler = engine.scheduler
        self.model_runner = engine.model_runner
        self.controller = engine.controller
        self.running = False

        # Active request contexts
        self.active_contexts: Dict[str, RequestContext] = {}
        self._lock = threading.Lock()

        logger.info("WorkerThread initialized")

    def run(self):
        """Main worker loop - pulls tasks from scheduler and executes stages."""
        logger.info("WorkerThread starting...")
        self.running = True

        while self.running:
            try:
                # Get next highest priority task from scheduler
                task = self.scheduler.get_next_task()

                if task is None:
                    # No work available, small sleep to avoid busy-wait
                    time.sleep(0.0001)  # Reduced from 0.001 for better responsiveness
                    continue

                # Execute appropriate stage
                if task.stage == LaneType.PREFILL:
                    self._execute_prefill(task)
                elif task.stage == LaneType.DECODE:
                    self._execute_decode(task)
                elif task.stage == LaneType.VERIFY:
                    self._execute_verify(task)
                else:
                    logger.error(f"Unknown stage {task.stage} for {task.request_id}")

            except Exception as e:
                logger.error(f"Worker thread error: {e}", exc_info=True)
                # Continue running despite errors

        logger.info("WorkerThread stopped")

    def stop(self):
        """Stop the worker thread gracefully."""
        logger.info("Stopping WorkerThread...")
        self.running = False

    def _execute_prefill(self, request: Request):
        """Execute prefill stage and transition to decode."""
        try:
            logger.debug(f"[PREFILL] Starting for {request.request_id}")

            # Get or create context
            with self._lock:
                if request.request_id not in self.active_contexts:
                    self.active_contexts[request.request_id] = RequestContext(request)
                context = self.active_contexts[request.request_id]

            # Update state
            context.state = RequestState.PREFILLING
            request.state = RequestState.PREFILLING
            request.prefill_start_time = time.time()

            # Execute prefill (tokenize prompt)
            input_ids = self.model_runner.prefill(request.prompt)
            context.input_ids = input_ids

            logger.debug(f"[PREFILL] Completed for {request.request_id}, input shape: {input_ids.shape}")

            # Transition to decode stage
            context.state = RequestState.DECODING
            request.state = RequestState.DECODING
            request.stage = LaneType.DECODE
            self.scheduler.submit_request(request)

        except Exception as e:
            logger.error(f"[PREFILL] Error for {request.request_id}: {e}", exc_info=True)
            self._handle_error(request, e)

    def _execute_decode(self, request: Request):
        """Execute decode iteration and transition to verify."""
        try:
            logger.debug(f"[DECODE] Starting for {request.request_id}")

            # Get context
            with self._lock:
                context = self.active_contexts.get(request.request_id)
                if not context:
                    raise RuntimeError(f"No context found for {request.request_id}")

            # Update state
            context.state = RequestState.DECODING
            request.state = RequestState.DECODING
            request.decode_start_time = time.time()

            # Get draft length from controller
            draft_length = self.controller.get_draft_length()
            context.draft_length = draft_length

            logger.debug(f"[DECODE] Generating {draft_length} draft tokens for {request.request_id}")

            # Generate draft tokens (use DECODE stream - stream_id=0)
            draft_tokens = self.model_runner.generate_draft_tokens(
                context.input_ids,
                num_tokens=draft_length,
                stream_id=0
            )
            context.draft_tokens = draft_tokens

            logger.debug(f"[DECODE] Generated {len(draft_tokens)} draft tokens for {request.request_id}")

            # Transition to verify stage
            context.state = RequestState.VERIFYING
            request.state = RequestState.VERIFYING
            request.stage = LaneType.VERIFY
            self.scheduler.submit_request(request)

        except Exception as e:
            logger.error(f"[DECODE] Error for {request.request_id}: {e}", exc_info=True)
            self._handle_error(request, e)

    def _execute_verify(self, request: Request):
        """Execute verification and transition back to decode or complete."""
        try:
            logger.debug(f"[VERIFY] Starting for {request.request_id}")

            # Get context
            with self._lock:
                context = self.active_contexts.get(request.request_id)
                if not context:
                    raise RuntimeError(f"No context found for {request.request_id}")

            # Update state
            context.state = RequestState.VERIFYING
            request.state = RequestState.VERIFYING
            request.verify_start_time = time.time()

            # Verify draft tokens (use VERIFY stream - stream_id=1)
            accepted_tokens, num_accepted = self.model_runner.verify_tokens(
                context.input_ids,
                context.draft_tokens,
                stream_id=1
            )

            logger.debug(f"[VERIFY] Accepted {num_accepted}/{len(context.draft_tokens)} tokens for {request.request_id}")

            # Update metrics
            request.tokens_generated += len(context.draft_tokens)
            request.tokens_accepted += num_accepted
            self.engine.total_tokens_generated += len(context.draft_tokens)
            self.engine.total_tokens_accepted += num_accepted

            # Update controller with acceptance feedback
            acceptance_ratio = num_accepted / len(context.draft_tokens) if context.draft_tokens else 0.0
            self.controller.record_acceptance(acceptance_ratio)

            # Update input_ids with accepted tokens
            context.generated_tokens.extend(accepted_tokens)
            request.generated_tokens.extend(accepted_tokens)

            if accepted_tokens:
                accepted_ids = torch.tensor([accepted_tokens], device=self.model_runner.device, dtype=torch.long)
                context.input_ids = torch.cat([context.input_ids, accepted_ids], dim=1)

            # Check completion conditions
            max_tokens_reached = len(context.generated_tokens) >= self.engine.max_new_tokens
            eos_generated = self.model_runner.tokenizer.eos_token_id in accepted_tokens if accepted_tokens else False

            if max_tokens_reached or eos_generated:
                # Request complete
                logger.debug(f"[VERIFY] Request {request.request_id} complete "
                           f"(tokens: {len(context.generated_tokens)}, eos: {eos_generated})")
                self._complete_request(request, context)
            else:
                # Continue with next decode iteration
                context.current_iteration += 1
                context.state = RequestState.DECODING
                request.state = RequestState.DECODING
                request.stage = LaneType.DECODE
                self.scheduler.submit_request(request)

        except Exception as e:
            logger.error(f"[VERIFY] Error for {request.request_id}: {e}", exc_info=True)
            self._handle_error(request, e)

    def _complete_request(self, request: Request, context: RequestContext):
        """Complete request and trigger callback."""
        try:
            logger.info(f"[COMPLETE] Request {request.request_id} finished with {len(context.generated_tokens)} tokens")

            # Update request state
            context.state = RequestState.COMPLETED
            request.state = RequestState.COMPLETED
            request.completed = True
            request.completion_time = time.time()

            # Generate final text
            result_text = self.model_runner.tokenizer.decode(context.generated_tokens, skip_special_tokens=True)
            request.result_text = result_text

            # Get callback
            with self.engine._lock:
                callback = self.engine.request_callbacks.pop(request.request_id, None)

            # Remove from active contexts
            with self._lock:
                self.active_contexts.pop(request.request_id, None)

            # Call callback
            if callback:
                try:
                    callback(request, result_text)
                except Exception as e:
                    logger.error(f"[COMPLETE] Callback error for {request.request_id}: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"[COMPLETE] Error completing {request.request_id}: {e}", exc_info=True)

    def _handle_error(self, request: Request, error: Exception):
        """Handle request error."""
        try:
            logger.error(f"[ERROR] Request {request.request_id} failed: {error}")

            # Update request state
            request.state = RequestState.ERROR
            request.error = str(error)
            request.completed = True
            request.completion_time = time.time()

            # Get callback
            with self.engine._lock:
                callback = self.engine.request_callbacks.pop(request.request_id, None)

            # Remove from active contexts
            with self._lock:
                context = self.active_contexts.pop(request.request_id, None)
                if context:
                    context.state = RequestState.ERROR
                    context.error = error

            # Call callback with None result
            if callback:
                try:
                    callback(request, None)
                except Exception as e:
                    logger.error(f"[ERROR] Callback error for {request.request_id}: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"[ERROR] Error handling error for {request.request_id}: {e}", exc_info=True)


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
        self.scheduler = ThreeLaneScheduler(config=config.scheduler)

        # Engine state
        self.is_running = False
        self.total_tokens_generated = 0
        self.total_tokens_accepted = 0

        # Request tracking
        self.request_callbacks: Dict[str, Callable] = {}
        self._lock = threading.Lock()

        # Configurable parameters
        self.max_new_tokens = config.model.max_new_tokens

        # Worker thread (created but not started yet)
        self.worker: Optional[WorkerThread] = None

        logger.info("GPU Stream SpeculativeEngine initialized")
    
    def start(self):
        """Start the GPU stream-based engine."""
        logger.info("Starting GPU Stream SpeculativeEngine...")
        self.model_runner.load_models()
        self.is_running = True

        # Create and start worker thread
        self.worker = WorkerThread(self)
        self.worker.start()

        logger.info("GPU Stream SpeculativeEngine started with async worker")

    def stop(self):
        """Stop the GPU stream-based engine."""
        logger.info("Stopping GPU Stream SpeculativeEngine...")
        self.is_running = False

        # Stop worker thread gracefully
        if self.worker:
            self.worker.stop()
            self.worker.join(timeout=5.0)
            if self.worker.is_alive():
                logger.warning("Worker thread did not stop within timeout")

        self.model_runner.cleanup()
        logger.info("GPU Stream SpeculativeEngine stopped")
    
    def submit_request_async(self, request: Request, callback: Optional[Callable] = None):
        """
        Submit a request for async processing with true disaggregated serving.
        Returns immediately - worker thread processes in background.

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

        # Initialize request state
        request.state = RequestState.QUEUED
        request.stage = LaneType.PREFILL

        # Submit to scheduler (PREFILL lane) - returns immediately!
        self.scheduler.submit_request(request)

        logger.debug(f"[ASYNC] Submitted {request.request_id} to scheduler (returns immediately)")


    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
