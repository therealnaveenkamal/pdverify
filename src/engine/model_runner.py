"""
Model runner for executing draft and verifier models.
"""

import torch
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
        
        # KV cache (simplified - in production would use PagedAttention)
        self.kv_cache: Dict[str, Any] = {}
        
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
        current_ids = input_ids
        
        with torch.no_grad():
            for _ in range(num_tokens):
                # Get logits from draft model
                outputs = self.draft_model(current_ids)
                logits = outputs.logits[:, -1, :]
                
                # Greedy sampling (can be made more sophisticated)
                next_token = torch.argmax(logits, dim=-1)
                draft_tokens.append(next_token.item())
                
                # Append for next iteration
                current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)
        
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
        # Create input with draft tokens
        draft_ids = torch.tensor([draft_tokens], device=self.device)
        full_input = torch.cat([input_ids, draft_ids], dim=1)
        
        accepted_tokens = []
        
        with torch.no_grad():
            # Forward pass through verifier
            outputs = self.verifier_model(full_input)
            logits = outputs.logits
            
            # Check each draft token
            for i, draft_token in enumerate(draft_tokens):
                # Get predicted token at this position
                predicted_token = torch.argmax(logits[:, input_ids.size(1) + i - 1, :], dim=-1).item()
                
                if predicted_token == draft_token:
                    # Token accepted
                    accepted_tokens.append(draft_token)
                else:
                    # Token rejected - use verifier's prediction and stop
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
