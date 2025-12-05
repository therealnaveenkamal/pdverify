"""
Baseline speculative decoding test (without Verify lane).
"""

import time
import logging
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils.config import ModelConfig

logger = logging.getLogger(__name__)


class BaselineSpeculativeDecoding:
    """
    Baseline speculative decoding without lane separation.
    Used for comparison with Verify-PD.
    """
    
    def __init__(self, model_config: ModelConfig, device: str = "cpu"):
        """
        Initialize baseline system.
        
        Args:
            model_config: Model configuration
            device: Device to use
        """
        self.model_config = model_config
        self.device = torch.device(device if torch.cuda.is_available() 
                                   and device == "cuda" else "cpu")
        
        self.draft_model = None
        self.verifier_model = None
        self.tokenizer = None
        
        self.total_tokens_generated = 0
        self.total_tokens_accepted = 0
    
    def load_models(self):
        """Load models."""
        logger.info("Loading baseline models...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.draft_model_name
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            self.model_config.draft_model_name
        ).to(self.device)
        self.draft_model.eval()
        
        self.verifier_model = AutoModelForCausalLM.from_pretrained(
            self.model_config.verifier_model_name
        ).to(self.device)
        self.verifier_model.eval()
        
        logger.info("Baseline models loaded")
    
    def generate(self, prompt: str, max_tokens: int = 100, draft_length: int = 4) -> tuple:
        """
        Generate text using baseline speculative decoding.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            draft_length: Number of draft tokens per iteration
            
        Returns:
            Tuple of (generated_text, latency_ms)
        """
        start_time = time.time()
        
        # Tokenize prompt
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        
        generated_tokens = []
        
        with torch.no_grad():
            while len(generated_tokens) < max_tokens:
                # Generate draft tokens
                draft_tokens = self._generate_draft(input_ids, draft_length)
                
                # Verify draft tokens (blocking operation in baseline)
                accepted_tokens, num_accepted = self._verify_draft(input_ids, draft_tokens)
                
                # Track statistics
                self.total_tokens_generated += len(draft_tokens)
                self.total_tokens_accepted += num_accepted
                
                generated_tokens.extend(accepted_tokens)
                
                # Update input
                new_ids = torch.tensor([accepted_tokens], device=self.device)
                input_ids = torch.cat([input_ids, new_ids], dim=1)
                
                # Check for EOS
                if self.tokenizer.eos_token_id in accepted_tokens:
                    break
        
        # Decode result
        result_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        latency_ms = (time.time() - start_time) * 1000
        
        return result_text, latency_ms
    
    def _generate_draft(self, input_ids: torch.Tensor, num_tokens: int) -> List[int]:
        """Generate draft tokens."""
        draft_tokens = []
        current_ids = input_ids
        
        for _ in range(num_tokens):
            outputs = self.draft_model(current_ids)
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1)
            draft_tokens.append(next_token.item())
            current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)
        
        return draft_tokens
    
    def _verify_draft(self, input_ids: torch.Tensor, draft_tokens: List[int]) -> tuple:
        """Verify draft tokens."""
        draft_ids = torch.tensor([draft_tokens], device=self.device)
        full_input = torch.cat([input_ids, draft_ids], dim=1)
        
        accepted_tokens = []
        outputs = self.verifier_model(full_input)
        logits = outputs.logits
        
        for i, draft_token in enumerate(draft_tokens):
            predicted_token = torch.argmax(logits[:, input_ids.size(1) + i - 1, :], dim=-1).item()
            
            if predicted_token == draft_token:
                accepted_tokens.append(draft_token)
            else:
                accepted_tokens.append(predicted_token)
                break
        
        return accepted_tokens, len(accepted_tokens)
    
    def get_acceptance_rate(self) -> float:
        """Get overall acceptance rate."""
        if self.total_tokens_generated == 0:
            return 0.0
        return self.total_tokens_accepted / self.total_tokens_generated
