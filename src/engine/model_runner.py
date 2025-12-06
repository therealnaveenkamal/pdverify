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
        
        # Load tokenizer (shared between models)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.draft_model_name,
            trust_remote_code=self.model_config.trust_remote_code
        )
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load draft model
        logger.info(f"Loading draft model: {self.model_config.draft_model_name}")
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            self.model_config.draft_model_name,
            torch_dtype=self._get_dtype(),
            trust_remote_code=self.model_config.trust_remote_code
        ).to(self.device)
        self.draft_model.eval()
        
        # Load verifier model
        logger.info(f"Loading verifier model: {self.model_config.verifier_model_name}")
        self.verifier_model = AutoModelForCausalLM.from_pretrained(
            self.model_config.verifier_model_name,
            torch_dtype=self._get_dtype(),
            trust_remote_code=self.model_config.trust_remote_code
        ).to(self.device)
        self.verifier_model.eval()

        # Vocab sizes for clamping
        self.draft_vocab_size = self.draft_model.config.vocab_size
        self.verifier_vocab_size = self.verifier_model.config.vocab_size
        self.tokenizer_vocab_size = len(self.tokenizer)
        
        logger.info("Models loaded successfully")
    
    def _get_dtype(self) -> torch.dtype:
        """Get torch dtype from config."""
        if self.model_config.dtype == "float16":
            return torch.float16
        elif self.model_config.dtype == "bfloat16":
            return torch.bfloat16
        else:
            return torch.float32
    
    def generate_draft_tokens(
        self,
        input_ids: torch.Tensor,
        num_tokens: int,
        stream_id: Optional[int] = None
    ) -> List[int]:
        """
        Generate draft tokens using small model.
        
        Args:
            input_ids: Input token IDs
            num_tokens: Number of tokens to generate
            stream_id: CUDA stream ID if using streams
            
        Returns:
            List of generated token IDs
        """
        draft_tokens = []
        current_ids = input_ids.to(torch.long)
        
        # Get CUDA stream if available
        stream = None
        if stream_id is not None and self.stream_manager:
            stream = self.stream_manager.get_stream(stream_id)
        
        # Use stream context if available
        with self._draft_lock:
            with torch.no_grad():
                if stream is not None:
                    with torch.cuda.stream(stream):
                        for _ in range(num_tokens):
                            outputs = self.draft_model(current_ids)
                            logits = outputs.logits[:, -1, :]
                            next_token = torch.argmax(logits, dim=-1)
                            token_id = next_token.item()
                            if self.draft_vocab_size is not None and token_id >= self.draft_vocab_size:
                                token_id = self.draft_vocab_size - 1
                            draft_tokens.append(token_id)
                            token_tensor = torch.tensor([[token_id]], device=current_ids.device, dtype=current_ids.dtype)
                            current_ids = torch.cat([current_ids, token_tensor], dim=1)
                    stream.synchronize()
                else:
                    for _ in range(num_tokens):
                        outputs = self.draft_model(current_ids)
                        logits = outputs.logits[:, -1, :]
                        next_token = torch.argmax(logits, dim=-1)
                        token_id = next_token.item()
                        if self.draft_vocab_size is not None and token_id >= self.draft_vocab_size:
                            token_id = self.draft_vocab_size - 1
                        draft_tokens.append(token_id)
                        token_tensor = torch.tensor([[token_id]], device=current_ids.device, dtype=current_ids.dtype)
                        current_ids = torch.cat([current_ids, token_tensor], dim=1)
        
        return draft_tokens
    
    def verify_tokens(
        self,
        input_ids: torch.Tensor,
        draft_tokens: List[int],
        stream_id: Optional[int] = None
    ) -> tuple[List[int], int]:
        """
        Verify draft tokens using large model.
        
        Args:
            input_ids: Original input token IDs
            draft_tokens: Draft tokens to verify
            stream_id: CUDA stream ID if using streams
            
        Returns:
            Tuple of (accepted_tokens, num_accepted)
        """
        # Get CUDA stream if available
        stream = None
        if stream_id is not None and self.stream_manager:
            stream = self.stream_manager.get_stream(stream_id)

        # Ensure integer type
        input_ids = input_ids.to(torch.long)

        # Create input with draft tokens
        draft_ids = torch.tensor([draft_tokens], device=self.device, dtype=torch.long)
        full_input = torch.cat([input_ids, draft_ids], dim=1)

        accepted_tokens = []

        # Use stream context if available
        with self._verifier_lock:
            with torch.no_grad():
                if stream is not None:
                    with torch.cuda.stream(stream):
                        outputs = self.verifier_model(full_input)
                        logits = outputs.logits

                        for i, draft_token in enumerate(draft_tokens):
                            logit_idx = input_ids.size(1) + i
                            if logit_idx >= logits.size(1):
                                break
                            predicted_token = torch.argmax(logits[:, logit_idx, :], dim=-1).item()
                            if self.verifier_vocab_size is not None and predicted_token >= self.verifier_vocab_size:
                                predicted_token = self.verifier_vocab_size - 1

                            if predicted_token == draft_token:
                                accepted_tokens.append(draft_token)
                            else:
                                accepted_tokens.append(predicted_token)
                                break
                    stream.synchronize()
                else:
                    outputs = self.verifier_model(full_input)
                    logits = outputs.logits

                    for i, draft_token in enumerate(draft_tokens):
                        logit_idx = input_ids.size(1) + i
                        if logit_idx >= logits.size(1):
                            break
                        predicted_token = torch.argmax(logits[:, logit_idx, :], dim=-1).item()
                        if self.verifier_vocab_size is not None and predicted_token >= self.verifier_vocab_size:
                            predicted_token = self.verifier_vocab_size - 1

                        if predicted_token == draft_token:
                            accepted_tokens.append(draft_token)
                        else:
                            accepted_tokens.append(predicted_token)
                            break
        
        return accepted_tokens, len(accepted_tokens)
    
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
        if self.verifier_model is not None:
            del self.verifier_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Cleaned up models")
