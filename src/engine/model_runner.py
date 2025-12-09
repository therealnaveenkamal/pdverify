"""
Model runner for executing draft and verifier models.
"""

import torch
import threading
from typing import List, Dict, Optional, Any
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils.config import ModelConfig, HardwareConfig
from ..utils.stream_manager import StreamManager

logger = logging.getLogger(__name__)


class ModelRunner:
    """Handles model loading and execution with CPU/CUDA support."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        hardware_config: HardwareConfig,
        stream_manager: Optional[StreamManager] = None
    ):
        """
        Initialize model runner.
        
        Args:
            model_config: Model configuration
            hardware_config: Hardware configuration
            stream_manager: CUDA stream manager
        """
        self.model_config = model_config
        self.hardware_config = hardware_config
        self.stream_manager = stream_manager
        
        self.device = torch.device(hardware_config.device if torch.cuda.is_available() 
                                   and hardware_config.device == "cuda" else "cpu")
        
        # Models and tokenizers
        self.draft_model = None
        self.verifier_model = None
        self.tokenizer = None
        self.draft_vocab_size = None
        self.verifier_vocab_size = None
        self.tokenizer_vocab_size = None
        
        # KV cache (simplified - in production would use PagedAttention)
        self.kv_cache: Dict[str, Any] = {}

        # Thread safety locks
        self._draft_lock = threading.Lock()
        self._verifier_lock = threading.Lock()

        logger.info(f"ModelRunner initialized on {self.device}")
    
    def load_models(self):
        """Load draft and verifier models."""
        logger.info("Loading models...")

        # Get token from config
        token = self.model_config.hf_token

        # Load tokenizer (shared between models)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.draft_model_name,
            trust_remote_code=self.model_config.trust_remote_code,
            token=token
        )

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load draft model
        logger.info(f"Loading draft model: {self.model_config.draft_model_name}")
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            self.model_config.draft_model_name,
            torch_dtype=self._get_dtype(),
            trust_remote_code=self.model_config.trust_remote_code,
            token=token
        ).to(self.device)
        self.draft_model.eval()

        # Load verifier model
        logger.info(f"Loading verifier model: {self.model_config.verifier_model_name}")
        self.verifier_model = AutoModelForCausalLM.from_pretrained(
            self.model_config.verifier_model_name,
            torch_dtype=self._get_dtype(),
            trust_remote_code=self.model_config.trust_remote_code,
            token=token
        ).to(self.device)
        self.verifier_model.eval()

        # Vocab sizes for clamping
        self.draft_vocab_size = self.draft_model.config.vocab_size
        self.verifier_vocab_size = self.verifier_model.config.vocab_size
        self.tokenizer_vocab_size = len(self.tokenizer)
        
        logger.info("Models loaded successfully")
    
    def _get_dtype(self) -> torch.dtype:
        """Get torch dtype from config."""
        dtype = self.model_config.dtype.lower()
        if dtype == "float16":
            return torch.float16
        if dtype == "bfloat16":
            return torch.bfloat16
        # "auto" or unknown: prefer half precision on GPU, fall back to fp32 on CPU
        if torch.cuda.is_available() and self.device.type == "cuda":
            return torch.float16
        return torch.float32
    
    def generate_draft_tokens(
        self,
        input_ids: torch.Tensor,
        num_tokens: int,
        stream_id: Optional[int] = None
    ) -> List[List[int]]:
        """
        Generate draft tokens using small model for a batch.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            num_tokens: Number of tokens to generate
            stream_id: CUDA stream ID if using streams
            
        Returns:
            List of List of generated token IDs (one list per sequence in batch)
        """
        batch_size = input_ids.shape[0]
        draft_tokens_batch = [[] for _ in range(batch_size)]
        current_ids = input_ids.to(torch.long)
        
        # Get CUDA stream if available
        stream = None
        if stream_id is not None and self.stream_manager:
            stream = self.stream_manager.get_stream(stream_id)
        
        # Use stream context if available
        with self._draft_lock:
            with torch.no_grad():
                ctx = torch.cuda.stream(stream) if stream is not None else(
                      torch.nullcontext() if hasattr(torch, "nullcontext") else torch.no_grad())
                
                with ctx:
                    for _ in range(num_tokens):
                        outputs = self.draft_model(current_ids)
                        logits = outputs.logits[:, -1, :] # [batch, vocab]
                        next_tokens = torch.argmax(logits, dim=-1) # [batch]
                        
                        # Process each sequence in batch
                        token_tensor_list = []
                        for i, token_id in enumerate(next_tokens):
                            tid = token_id.item()
                            if self.draft_vocab_size is not None and tid >= self.draft_vocab_size:
                                tid = self.draft_vocab_size - 1
                            draft_tokens_batch[i].append(tid)
                            token_tensor_list.append([tid])
                        
                        # Append to current_ids
                        new_tokens = torch.tensor(token_tensor_list, device=current_ids.device, dtype=current_ids.dtype)
                        current_ids = torch.cat([current_ids, new_tokens], dim=1)
                        
                    if stream is not None:
                        stream.synchronize()
        
        return draft_tokens_batch
    
    def verify_tokens(
        self,
        input_ids: torch.Tensor,
        draft_tokens_batch: List[List[int]],
        stream_id: Optional[int] = None
    ) -> tuple[List[List[int]], List[int]]:
        """
        Verify draft tokens using large model for a batch.
        
        Args:
            input_ids: Original input token IDs [batch_size, seq_len]
            draft_tokens_batch: List of draft token lists for each request
            stream_id: CUDA stream ID if using streams
            
        Returns:
            Tuple of (accepted_tokens_batch, num_accepted_list)
        """
        batch_size = input_ids.shape[0]
        if len(draft_tokens_batch) != batch_size:
            raise ValueError(f"Batch size mismatch: input {batch_size}, drafts {len(draft_tokens_batch)}")

        # Get CUDA stream if available
        stream = None
        if stream_id is not None and self.stream_manager:
            stream = self.stream_manager.get_stream(stream_id)

        # Create input with draft tokens (assuming equal length drafts for batching efficiency)
        # Note: In a real system we needs ragged batching, here we pad or assume same draft length
        max_draft_len = max(len(d) for d in draft_tokens_batch)
        
        # Pad drafts to max length for tensor creation
        padded_drafts = []
        for d in draft_tokens_batch:
            padded = d + [self.tokenizer.pad_token_id or 0] * (max_draft_len - len(d))
            padded_drafts.append(padded)
            
        draft_ids = torch.tensor(padded_drafts, device=self.device, dtype=torch.long)
        full_input = torch.cat([input_ids, draft_ids], dim=1)

        accepted_tokens_batch = []
        num_accepted_list = []

        # Use stream context if available
        with self._verifier_lock:
            with torch.no_grad():
                ctx = torch.cuda.stream(stream) if stream is not None else (
                      torch.nullcontext() if hasattr(torch, "nullcontext") else torch.no_grad())
                
                with ctx:
                    outputs = self.verifier_model(full_input)
                    logits = outputs.logits # [batch, seq_len, vocab]

                    for b in range(batch_size):
                        accepted_tokens = []
                        curr_draft_tokens = draft_tokens_batch[b]
                        
                        for i, draft_token in enumerate(curr_draft_tokens):
                            logit_idx = input_ids.size(1) + i
                            if logit_idx >= logits.size(1):
                                break
                                
                            pred_logits = logits[b, logit_idx, :]
                            predicted_token = torch.argmax(pred_logits, dim=-1).item()
                            
                            if self.verifier_vocab_size is not None and predicted_token >= self.verifier_vocab_size:
                                predicted_token = self.verifier_vocab_size - 1

                            if predicted_token == draft_token:
                                accepted_tokens.append(draft_token)
                            else:
                                accepted_tokens.append(predicted_token)
                                break
                        
                        accepted_tokens_batch.append(accepted_tokens)
                        num_accepted_list.append(len(accepted_tokens))

                    if stream is not None:
                        stream.synchronize()
        
        return accepted_tokens_batch, num_accepted_list
    
    def prefill(
        self,
        prompt: str,
        stream_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Prefill operation - encode prompt and prepare KV cache.

        Args:
            prompt: Input prompt
            stream_id: CUDA stream ID if using streams

        Returns:
            Input token IDs
        """
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_config.max_model_len
        )

        input_ids = inputs["input_ids"].to(self.device)

        # In production, this would populate PagedAttention KV cache
        # For now, we just return the input IDs
        return input_ids
    
    def cleanup(self):
        """Clean up resources."""
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
