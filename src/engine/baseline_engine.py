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
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        
        # Models
        self.draft_model = None
        self.verifier_model = None
        self.tokenizer = None
        self.draft_vocab_size = None
        self.verifier_vocab_size = None
        
        # Device
        self.device = torch.device(
            config.hardware.device if torch.cuda.is_available() and config.hardware.device == "cuda" else "cpu"
        )
        
        # Engine state
        self.is_running = False
        self.worker_threads: List[threading.Thread] = []
        self.max_workers = config.scheduler.batch_size  # Use batch_size as max concurrent
        
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
        self._load_models()
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
        
        self._cleanup_models()
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
        Process a single request using standard speculative decoding.
        
        Args:
            request: Request to process
            
        Returns:
            Generated text
        """
        # Tokenize
        input_ids = self.tokenizer(
            request.prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.model.max_model_len
        )["input_ids"].to(self.device)
        
        generated_tokens = []
        
        with torch.no_grad():
            while len(generated_tokens) < self.max_new_tokens:
                # Generate draft tokens
                draft_tokens = self._generate_draft(input_ids, self.draft_length)
                
                # Verify draft tokens
                accepted_tokens, num_accepted = self._verify_draft(input_ids, draft_tokens)
                
                # Update metrics
                request.tokens_generated += len(draft_tokens)
                request.tokens_accepted += num_accepted
                
                with self._lock:
                    self.total_tokens_generated += len(draft_tokens)
                    self.total_tokens_accepted += num_accepted
                
                generated_tokens.extend(accepted_tokens)
                
                # Update input
                if accepted_tokens:
                    new_ids = torch.tensor([accepted_tokens], device=self.device, dtype=input_ids.dtype)
                    input_ids = torch.cat([input_ids, new_ids], dim=1)
                    
                    # Check for EOS
                    if self.tokenizer.eos_token_id in accepted_tokens:
                        break
        
        # Decode result
        result_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return result_text

    def _generate_draft(self, input_ids: torch.Tensor, num_tokens: int) -> List[int]:
        """Generate draft tokens using small model."""
        draft_tokens = []
        current_ids = input_ids
        
        for _ in range(num_tokens):
            outputs = self.draft_model(current_ids)
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1)
            token_id = next_token.item()
            
            # Clamp to vocab size
            if self.draft_vocab_size is not None and token_id >= self.draft_vocab_size:
                token_id = self.draft_vocab_size - 1
            
            draft_tokens.append(token_id)
            next_token_tensor = torch.tensor([[token_id]], device=self.device, dtype=input_ids.dtype)
            current_ids = torch.cat([current_ids, next_token_tensor], dim=1)
        
        return draft_tokens

    def _verify_draft(self, input_ids: torch.Tensor, draft_tokens: List[int]) -> tuple:
        """Verify draft tokens using large model."""
        draft_ids = torch.tensor([draft_tokens], device=self.device, dtype=input_ids.dtype)
        full_input = torch.cat([input_ids, draft_ids], dim=1)
        
        accepted_tokens = []
        outputs = self.verifier_model(full_input)
        logits = outputs.logits
        
        for i, draft_token in enumerate(draft_tokens):
            logit_idx = input_ids.size(1) + i
            if logit_idx >= logits.size(1):
                break
            
            predicted_token = torch.argmax(logits[:, logit_idx, :], dim=-1).item()
            
            # Clamp to vocab size
            if self.verifier_vocab_size is not None and predicted_token >= self.verifier_vocab_size:
                predicted_token = self.verifier_vocab_size - 1
            
            if predicted_token == draft_token:
                accepted_tokens.append(draft_token)
            else:
                accepted_tokens.append(predicted_token)
                break
        
        return accepted_tokens, len(accepted_tokens)

    def _load_models(self):
        """Load draft and verifier models."""
        logger.info("Loading models...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.draft_model_name,
            trust_remote_code=self.config.model.trust_remote_code,
            token=self.config.model.hf_token,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load draft model
        logger.info(f"Loading draft model: {self.config.model.draft_model_name}")
        # Prepare config to strip incompatible rope_scaling for Llama-family models on older transformers
        draft_config = AutoConfig.from_pretrained(
            self.config.model.draft_model_name,
            trust_remote_code=self.config.model.trust_remote_code,
            token=self.config.model.hf_token,
        )
        if hasattr(draft_config, "rope_scaling") and "llama" in self.config.model.draft_model_name.lower():
            draft_config.rope_scaling = None

        self.draft_model = AutoModelForCausalLM.from_pretrained(
            self.config.model.draft_model_name,
            config=draft_config,
            torch_dtype=self._get_dtype(),
            trust_remote_code=self.config.model.trust_remote_code,
            token=self.config.model.hf_token,
        ).to(self.device)
        self.draft_model.eval()
        
        # Load verifier model
        logger.info(f"Loading verifier model: {self.config.model.verifier_model_name}")
        verifier_config = AutoConfig.from_pretrained(
            self.config.model.verifier_model_name,
            trust_remote_code=self.config.model.trust_remote_code,
            token=self.config.model.hf_token,
        )
        if hasattr(verifier_config, "rope_scaling") and "llama" in self.config.model.verifier_model_name.lower():
            verifier_config.rope_scaling = None

        self.verifier_model = AutoModelForCausalLM.from_pretrained(
            self.config.model.verifier_model_name,
            config=verifier_config,
            torch_dtype=self._get_dtype(),
            trust_remote_code=self.config.model.trust_remote_code,
            token=self.config.model.hf_token,
        ).to(self.device)
        self.verifier_model.eval()
        
        # Store vocab sizes
        self.draft_vocab_size = self.draft_model.config.vocab_size
        self.verifier_vocab_size = self.verifier_model.config.vocab_size
        
        logger.info("Models loaded successfully")

    def _get_dtype(self) -> torch.dtype:
        """Get torch dtype from config."""
        dtype = self.config.model.dtype.lower()
        if dtype == "float16":
            return torch.float16
        if dtype == "bfloat16":
            return torch.bfloat16
        # "auto" or unknown: prefer half precision on GPU, fall back to fp32 on CPU
        if torch.cuda.is_available() and self.device.type == "cuda":
            return torch.float16
        return torch.float32

    def _cleanup_models(self):
        """Clean up model resources."""
        if self.draft_model is not None:
            del self.draft_model
            self.draft_model = None
        if self.verifier_model is not None:
            del self.verifier_model
            self.verifier_model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure all CUDA operations complete
        logger.info("Cleaned up models")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
