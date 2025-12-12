"""
Baseline speculative decoding engine with concurrent processing support.
This provides a fair comparison with Verify-PD by supporting concurrent requests.
"""

import time
import threading
from typing import Optional, List, Dict, Callable
import logging
import torch
from collections import deque

from ..scheduler import Request
from ..utils.config import VerifyPDConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

logger = logging.getLogger(__name__)


class BaselineEngine:
    """
    Baseline speculative decoding engine with concurrent processing.
    No lane separation - just standard speculative decoding with concurrency support.
    """

    def __init__(self, config: VerifyPDConfig):
        """
        Initialize baseline engine.

        Args:
            config: System configuration
        """
        self.config = config
        
        # Initialize components
        from ..utils.stream_manager import StreamManager
        from .model_runner import ModelRunner
        
        # Device
        self.device = config.hardware.device
        
        # Stream manager (even for baseline, helpful for encapsulation)
        self.stream_manager = StreamManager(
            device=config.hardware.device,
            num_streams=1 # Single stream for baseline per worker technically, or just 1 global?
            # Baseline usually runs single stream per request context.
            # We will pass None to model_runner to let it handle default stream or CPU.
        )
        
        self.model_runner = ModelRunner(
            model_config=config.model,
            hardware_config=config.hardware,
            stream_manager=self.stream_manager
        )
        
        # Engine state
        self.is_running = False
        self.worker_thread = None  # Single unified worker (fair comparison with PD/PDV)
        self.batch_size = config.scheduler.batch_size
        
        # Request queue
        self.request_queue: deque = deque()
        self.request_callbacks: Dict[str, Callable] = {}
        self._lock = threading.Lock()
        
        # Metrics
        self.total_tokens_generated = 0
        self.total_tokens_accepted = 0
        
        # Config
        self.max_new_tokens = config.model.max_new_tokens
        self.draft_length = config.controller.initial_draft_length
        
        logger.info(f"BaselineEngine initialized with SINGLE THREAD (fair comparison), batch_size={self.batch_size}")
    
    def start(self):
        """Start the engine and unified worker thread."""
        logger.info("Starting BaselineEngine (single thread)...")
        self.model_runner.load_models()
        self.is_running = True
        
        # Start single unified worker thread (fair comparison with PD/PDV)
        self.worker_thread = threading.Thread(target=self._unified_worker_loop, daemon=True)
        self.worker_thread.start()
        
        logger.info("BaselineEngine started with single unified thread")

    def stop(self):
        """Stop the engine and worker thread."""
        logger.info("Stopping BaselineEngine...")
        self.is_running = False
        
        # Wait for worker to finish
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        
        # Clear remaining requests
        with self._lock:
            self.request_queue.clear()
        
        self.stream_manager.cleanup()
        self.model_runner.cleanup()
        logger.info("BaselineEngine stopped")
    
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
        
        # Add to queue (simple lock, no condition variable)
        with self._lock:
            self.request_queue.append(request)
        
        logger.debug(f"Queued request {request.request_id}")

    def process_request(self, request: Request) -> str:
        """Process a request synchronously (blocking)."""
        import threading
        completion_event = threading.Event()
        result_container = {}
        
        def callback(req, res):
            result_container['result'] = res
            completion_event.set()
        
        self.submit_request_async(request, callback)
        completion_event.wait()
        
        return result_container.get('result', "")

    def _unified_worker_loop(self):
        """
        Unified worker loop that processes requests sequentially.
        Uses single thread like PD/PDV for fair comparison.
        """
        logger.info("Unified worker loop started (single thread)")
        
        while self.is_running:
            # Get batch of requests
            batch = []
            with self._lock:
                while len(self.request_queue) > 0 and len(batch) < self.batch_size:
                    batch.append(self.request_queue.popleft())
            
            if not batch:
                # No requests, sleep briefly
                time.sleep(0.00001)
                continue
            
            # Check if still running before processing
            if not self.is_running:
                # Re-queue requests
                with self._lock:
                    for req in batch:
                        self.request_queue.append(req)
                break
            
            # Process each request in batch sequentially (like PD/PDV)
            for request in batch:
                try:
                    result_text = self._process_request(request)
                    request.completion_time = time.time()
                    
                    # Trigger callback
                    with self._lock:
                        callback = self.request_callbacks.pop(request.request_id, None)
                    
                    if callback:
                        try:
                            callback(request, result_text)
                        except Exception as e:
                            logger.error(f"Error in callback for {request.request_id}: {e}")
                            
                except Exception as e:
                    if not self.is_running:
                        break
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
        
        logger.info("Unified worker loop stopped")

    def _process_request(self, request: Request) -> str:
        """
        Process a single request using standard speculative decoding with KV Cache.
        """
        # 1. PREFILL
        # Use simple stream 0 (or None if CPU)
        request.prefill_start_time = time.time()
        input_ids, draft_kv, verifier_kv, verifier_logits, draft_logits = self.model_runner.prefill(
            request.prompt,
            stream_id=0
        )
        request.prefill_end_time = time.time()
        
        request.kv_cache_draft = draft_kv
        request.kv_cache_verifier = verifier_kv
        request.last_verifier_logits = verifier_logits
        request.last_draft_logits = draft_logits
        
        generated_tokens_list = []
        request.request_start_time = request.created_at if request.created_at > 0 else time.time()
        
        # 2. DECODE LOOP
        request.decode_start_time = time.time()
        while len(generated_tokens_list) < self.max_new_tokens:
            
            # 2a. Sample d1
            d1 = 0
            if request.last_draft_logits is not None:
                d1 = torch.argmax(request.last_draft_logits).item()
                if self.model_runner.draft_vocab_size and d1 >= self.model_runner.draft_vocab_size:
                    d1 = self.model_runner.draft_vocab_size - 1
            
            # 2b. Generate Drafts
            d1_tensor = torch.tensor([[d1]], device=self.model_runner.device)
            curr_draft_tokens = [d1]
            
            # Temp KV for generation (don't update main KV yet)
            temp_kv = request.kv_cache_draft
            generated_ids = []
            
            if self.draft_length > 1:
                gen_ids, temp_kv = self.model_runner.generate_draft_tokens(
                    d1_tensor,
                    num_tokens=self.draft_length - 1,
                    past_key_values=temp_kv,
                    stream_id=0
                )
                generated_ids = gen_ids[0]
            
            full_drafts = curr_draft_tokens + generated_ids
            
            # 2c. Verify
            verify_input = torch.tensor([full_drafts], device=self.model_runner.device)
            prev_logits = request.last_verifier_logits.unsqueeze(0) if request.last_verifier_logits is not None else None
            
            accepted_batch, num_accepted_batch, _, _ = self.model_runner.verify_tokens(
                verify_input,
                [full_drafts],
                past_key_values=request.kv_cache_verifier,
                previous_verifier_logits=prev_logits,
                stream_id=0
            )
            
            accepted_tokens = accepted_batch[0]
            num_accepted = num_accepted_batch[0]
            
            # Update metrics
            request.tokens_generated += len(full_drafts)
            request.tokens_accepted += num_accepted
            with self._lock:
                self.total_tokens_generated += len(full_drafts)
                self.total_tokens_accepted += num_accepted
                
            current_time = time.time()
            for _ in accepted_tokens:
                request.token_timestamps.append(current_time)
                
            generated_tokens_list.extend(accepted_tokens)
            
            # 2d. Update State (Critical for KV Cache)
            if accepted_tokens:
                acc_tensor = torch.tensor([accepted_tokens], device=self.model_runner.device)
                
                # Update Draft
                with self.model_runner._draft_lock:
                    with torch.no_grad():
                        out_d = self.model_runner.draft_model(acc_tensor, past_key_values=request.kv_cache_draft, use_cache=True)
                        request.kv_cache_draft = out_d.past_key_values
                        request.last_draft_logits = out_d.logits[:, -1, :]
                 
                # Update Verifier
                with self.model_runner._verifier_lock:
                    with torch.no_grad():
                        out_v = self.model_runner.verifier_model(acc_tensor, past_key_values=request.kv_cache_verifier, use_cache=True)
                        request.kv_cache_verifier = out_v.past_key_values
                        request.last_verifier_logits = out_v.logits[:, -1, :]
                        
                # Check EOS
                if self.model_runner.tokenizer.eos_token_id in accepted_tokens:
                    break
            else:
                # Should not happen in greedy decoding
                break
                
        request.decode_end_time = time.time()
        # Decode result
        result_text = self.model_runner.tokenizer.decode(generated_tokens_list, skip_special_tokens=True)
        
        # Cleanup
        del request.kv_cache_draft
        del request.kv_cache_verifier
        request.kv_cache_draft = None
        request.kv_cache_verifier = None
        
        return result_text

    def get_stats(self):
        """Get engine statistics."""
        return {
            "total_tokens_generated": self.total_tokens_generated,
            "total_tokens_accepted": self.total_tokens_accepted,
             "overall_acceptance_rate": (self.total_tokens_accepted / self.total_tokens_generated) if self.total_tokens_generated > 0 else 0
        }

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
