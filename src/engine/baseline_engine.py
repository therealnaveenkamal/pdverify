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
        self.worker_threads: List[threading.Thread] = []
        self.max_workers = config.scheduler.batch_size 
        
        # Request queue
        self.request_queue: deque = deque()
        self.request_callbacks: Dict[str, Callable] = {}
        self._lock = threading.Lock()
        self._queue_condition = threading.Condition(self._lock)
        
        # Metrics
        self.total_tokens_generated = 0
        self.total_tokens_accepted = 0
        
        # Config
        self.max_new_tokens = config.model.max_new_tokens
        self.draft_length = config.controller.initial_draft_length
        
        logger.info(f"BaselineEngine initialized with {self.max_workers} workers")
    
    def start(self):
        """Start the engine and worker threads."""
        logger.info("Starting BaselineEngine...")
        self.model_runner.load_models()
        self.is_running = True
        
        # Start worker threads
        for i in range(self.max_workers):
            thread = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            thread.start()
            self.worker_threads.append(thread)
        
        logger.info(f"BaselineEngine started with {self.max_workers} workers")

    def stop(self):
        """Stop the engine and worker threads."""
        logger.info("Stopping BaselineEngine...")
        self.is_running = False
        
        # Wake up all workers
        with self._queue_condition:
            self._queue_condition.notify_all()
        
        # Wait for workers to finish
        for thread in self.worker_threads:
            thread.join(timeout=2.0)
        
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
        
        # Add to queue
        with self._queue_condition:
            self.request_queue.append(request)
            self._queue_condition.notify()  # Wake up one worker
        
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

    def _worker_loop(self, worker_id: int):
        """Worker loop that processes requests from the queue."""
        logger.info(f"Worker {worker_id} started")
        
        while self.is_running:
            request = None
            
            # Get request from queue
            with self._queue_condition:
                while self.is_running and len(self.request_queue) == 0:
                    self._queue_condition.wait(timeout=0.1)
                
                if not self.is_running:
                    break
                
                if len(self.request_queue) > 0:
                    request = self.request_queue.popleft()
            
            if request is None:
                continue
            
            # Process request
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
        
        logger.info(f"Worker {worker_id} stopped")

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
        import time
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
