"""
Speculative decoding engine integrating three-lane scheduler.
"""

import time
from typing import Optional, List
import logging

from ..scheduler import ThreeLaneScheduler, Request, LaneType
from ..controller import FeedbackController
from .model_runner import ModelRunner
from ..utils.config import VerifyPDConfig
from ..utils.stream_manager import StreamManager

logger = logging.getLogger(__name__)


class SpeculativeEngine:
    """
    Core engine for Verify-PD speculative decoding.
    Integrates scheduler, controller, and model execution.
    """
    
    def __init__(self, config: VerifyPDConfig):
        """
        Initialize speculative engine.
        
        Args:
            config: System configuration
        """
        self.config = config
        
        # Initialize components
        self.scheduler = ThreeLaneScheduler(config.scheduler)
        self.controller = FeedbackController(config.controller)
        self.stream_manager = StreamManager(
            device=config.hardware.device,
            num_streams=config.hardware.num_streams
        )
        self.model_runner = ModelRunner(
            model_config=config.model,
            hardware_config=config.hardware,
            stream_manager=self.stream_manager
        )
        
        # Engine state
        self.is_running = False
        self.total_tokens_generated = 0
        self.total_tokens_accepted = 0
        
        logger.info("SpeculativeEngine initialized")
    
    def start(self):
        """Start the engine."""
        logger.info("Starting SpeculativeEngine...")
        self.model_runner.load_models()
        self.is_running = True
        logger.info("SpeculativeEngine started")
    
    def stop(self):
        """Stop the engine."""
        logger.info("Stopping SpeculativeEngine...")
        self.is_running = False
        self.model_runner.cleanup()
        self.stream_manager.cleanup()
        logger.info("SpeculativeEngine stopped")
    
    def process_request(self, request: Request) -> str:
        """
        Process a single request through all stages.
        
        Args:
            request: Request to process
            
        Returns:
            Generated text
        """
        # Stage 1: Prefill
        request.stage = LaneType.PREFILL
        request.prefill_start_time = time.time()
        self.scheduler.submit_request(request)
        
        input_ids = self._execute_prefill(request)
        
        # Stage 2: Decode loop
        generated_tokens = []
        max_new_tokens = 100  # Configurable
        
        while len(generated_tokens) < max_new_tokens:
            # Generate draft tokens
            request.stage = LaneType.DECODE
            request.decode_start_time = time.time()
            
            draft_length = self.controller.get_draft_length()
            draft_tokens = self._execute_decode(request, input_ids, draft_length)
            
            # Verify draft tokens
            request.stage = LaneType.VERIFY
            request.verify_start_time = time.time()
            request.draft_tokens = draft_tokens
            
            accepted_tokens, num_accepted = self._execute_verify(request, input_ids, draft_tokens)
            
            # Update metrics
            request.tokens_generated += len(draft_tokens)
            request.tokens_accepted += num_accepted
            self.total_tokens_generated += len(draft_tokens)
            self.total_tokens_accepted += num_accepted
            
            # Update controller
            acceptance_ratio = num_accepted / len(draft_tokens) if draft_tokens else 0.0
            self.controller.record_acceptance(acceptance_ratio)
            
            # Check for EOS or continue
            generated_tokens.extend(accepted_tokens)
            
            # Update input for next iteration
            new_token_ids = self.model_runner.tokenizer.encode(
                self.model_runner.tokenizer.decode(accepted_tokens),
                add_special_tokens=False
            )
            input_ids = self.model_runner.tokenizer.encode(
                request.prompt + self.model_runner.tokenizer.decode(generated_tokens),
                return_tensors="pt",
                add_special_tokens=False
            ).to(self.model_runner.device)
            
            # Check if we should continue
            if self.model_runner.tokenizer.eos_token_id in accepted_tokens:
                break
            
            # Update controller based on queue depth
            verify_queue_depth = self.scheduler.verify_lane.size()
            self.controller.update(verify_queue_depth)
        
        # Complete request
        self.scheduler.complete_request(request)
        
        # Decode tokens to text
        result_text = self.model_runner.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return result_text
    
    def _execute_prefill(self, request: Request):
        """Execute prefill stage."""
        logger.debug(f"Prefill for {request.request_id}")
        input_ids = self.model_runner.prefill(
            request.prompt,
            stream_id=LaneType.PREFILL
        )
        return input_ids
    
    def _execute_decode(self, request: Request, input_ids, draft_length: int) -> List[int]:
        """Execute decode stage."""
        decode_start = time.time()
        
        draft_tokens = self.model_runner.generate_draft_tokens(
            input_ids,
            num_tokens=draft_length,
            stream_id=LaneType.DECODE
        )
        
        decode_time_ms = (time.time() - decode_start) * 1000
        self.controller.record_decode_latency(decode_time_ms)
        
        logger.debug(f"Decode for {request.request_id}: {len(draft_tokens)} tokens in {decode_time_ms:.1f}ms")
        return draft_tokens
    
    def _execute_verify(self, request: Request, input_ids, draft_tokens: List[int]) -> tuple:
        """Execute verify stage."""
        logger.debug(f"Verify for {request.request_id}: {len(draft_tokens)} tokens")
        
        accepted_tokens, num_accepted = self.model_runner.verify_tokens(
            input_ids,
            draft_tokens,
            stream_id=LaneType.VERIFY
        )
        
        logger.debug(f"Accepted {num_accepted}/{len(draft_tokens)} tokens")
        return accepted_tokens, num_accepted
    
    def get_stats(self) -> dict:
        """Get engine statistics."""
        return {
            "scheduler": self.scheduler.get_scheduler_stats(),
            "controller": self.controller.get_stats(),
            "total_tokens_generated": self.total_tokens_generated,
            "total_tokens_accepted": self.total_tokens_accepted,
            "overall_acceptance_rate": (
                self.total_tokens_accepted / self.total_tokens_generated
                if self.total_tokens_generated > 0 else 0.0
            )
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
