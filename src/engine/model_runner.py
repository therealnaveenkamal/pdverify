"""
Model runner for executing draft and verifier models.
"""

import torch
import threading
from typing import List, Dict, Optional, Any
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

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
            token=token,
        )

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load draft model
        logger.info(f"Loading draft model: {self.model_config.draft_model_name}")
        # Prepare config to strip incompatible rope_scaling for Llama-family models on older transformers
        draft_config = AutoConfig.from_pretrained(
            self.model_config.draft_model_name,
            trust_remote_code=self.model_config.trust_remote_code,
            token=token,
        )
        if hasattr(draft_config, "rope_scaling") and "llama" in self.model_config.draft_model_name.lower():
            draft_config.rope_scaling = None

        self.draft_model = AutoModelForCausalLM.from_pretrained(
            self.model_config.draft_model_name,
            config=draft_config,
            torch_dtype=self._get_dtype(),
            trust_remote_code=self.model_config.trust_remote_code,
            token=token,
        ).to(self.device)
        self.draft_model.eval()

        # Load verifier model
        logger.info(f"Loading verifier model: {self.model_config.verifier_model_name}")
        verifier_config = AutoConfig.from_pretrained(
            self.model_config.verifier_model_name,
            trust_remote_code=self.model_config.trust_remote_code,
            token=token,
        )
        if hasattr(verifier_config, "rope_scaling") and "llama" in self.model_config.verifier_model_name.lower():
            verifier_config.rope_scaling = None

        self.verifier_model = AutoModelForCausalLM.from_pretrained(
            self.model_config.verifier_model_name,
            config=verifier_config,
            torch_dtype=self._get_dtype(),
            trust_remote_code=self.model_config.trust_remote_code,
            token=token,
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
        past_key_values: Optional[List[Any]] = None,
        stream_id: Optional[int] = None
    ) -> tuple[List[List[int]], List[Any]]:
        """
        Generate draft tokens using small model for a batch.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len] (or new tokens only if past_key_values provided)
            num_tokens: Number of tokens to generate
            past_key_values: KV cache for the batch (list of KV caches per request)
            stream_id: CUDA stream ID if using streams
            
        Returns:
            Tuple of (generated_token_lists, new_past_key_values_list)
        """
        batch_size = input_ids.shape[0]
        draft_tokens_batch = [[] for _ in range(batch_size)]
        current_ids = input_ids.to(torch.long)
        
        # Get CUDA stream if available
        stream = None
        if stream_id is not None and self.stream_manager:
            stream = self.stream_manager.get_stream(stream_id)
        
        # Helper to collate/split KV cache for batching is complex. 
        # For simplicity in this demo, we assume we process the whole batch together.
        # But `past_key_values` from Hugging Face is usually (n_layers, 2, batch, num_heads, seq_len, head_dim).
        # If we have a list of individual KVs, we'd need to collate them.
        # IMPROVEMENT: For this implementation, we will assume standard HF output and that the
        # engine handles batch construction such that we get a single batched `past_key_values` object
        # OR we handle list of KVs.
        # Given the engine structure, let's look at `_handle_decode_batch`. It constructs a batch.
        # Re-batching KVs is expensive. 
        # SIMPLIFICATION: We will assume we pass a LIST of KV caches (one per request) and we might need
        # to process them carefully. However, standard HF `generate` doesn't easily support "list of KVs" for a batch
        # unless we collate them.
        # COLLATING KVs: expensive.
        # ALTERNATIVE: Run iteratively or assume fixed batch slotting (vLLM style).
        # DECISION: For this "simulated" environment, we will run the batch loop explicitly if KVs are separate,
        # OR we try to collate.
        # Let's try to pass the `past_key_values` as a Batched object if the engine manages it.
        # But the engine has `Deque` of requests.
        # Let's assume for now we don't have PagedAttention. We must concatenate KVs.
        # This is SLOW. But accurate for correctness.
        
        # ACTUALLY: Let's assume `past_key_values` passed here is ALREADY a collated batch KV 
        # corresponding directly to `input_ids` batch.
        # The Engine will be responsible for Collating KVs from the requests into a batch KV.
        
        # Wait, if we use `past_key_values`, `input_ids` should only be the NEW tokens.
        
        next_kv_cache = past_key_values

        # Use stream context if available
        with self._draft_lock:
            with torch.no_grad():
                ctx = torch.cuda.stream(stream) if stream is not None else(
                      torch.nullcontext() if hasattr(torch, "nullcontext") else torch.no_grad())
                
                with ctx:
                    for step in range(num_tokens):
                        # Prepare inputs
                        # If first step and we have KV, input is just last token? 
                        # Caller responsible for passing correct input_ids (full if no KV, new if KV).
                        # Actually standard loop:
                        
                        outputs = self.draft_model(current_ids, past_key_values=next_kv_cache, use_cache=True)
                        next_kv_cache = outputs.past_key_values
                        
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
                        
                        # Update current_ids for next step (it becomes the input)
                        new_tokens = torch.tensor(token_tensor_list, device=current_ids.device, dtype=current_ids.dtype)
                        current_ids = new_tokens # Next input is just the generated token
                        
                    if stream is not None:
                        stream.synchronize()
        
        return draft_tokens_batch, next_kv_cache
    
    def verify_tokens(
        self,
        input_ids: torch.Tensor,
        draft_tokens_batch: List[List[int]],
        past_key_values: Optional[Any] = None,
        previous_verifier_logits: Optional[torch.Tensor] = None,
        stream_id: Optional[int] = None
    ) -> tuple[List[List[int]], List[int], Any, Optional[torch.Tensor]]:
        """
        Verify draft tokens using large model for a batch.
        
        Args:
            input_ids: Original input token IDs [batch_size, seq_len] (or new tokens only if past_key_values provided)
            draft_tokens_batch: List of draft token lists for each request
            past_key_values: KV cache for the batch
            previous_verifier_logits: Logits from the previous step [batch, vocab] to verify the first draft token
            stream_id: CUDA stream ID if using streams
            
        Returns:
            Tuple of (accepted_tokens_batch, num_accepted_list, new_past_key_values, new_verifier_logits)
        """
        """
        Verify draft tokens using large model for a batch.
        
        Args:
            input_ids: Original input token IDs [batch_size, seq_len] (or new tokens only if past_key_values provided)
            draft_tokens_batch: List of draft token lists for each request
            past_key_values: KV cache for the batch
            stream_id: CUDA stream ID if using streams
            
        Returns:
            Tuple of (accepted_tokens_batch, num_accepted_list, new_past_key_values)
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

        # Prepare input with draft tokens
        # If we have past_key_values, input_ids should be just the new draft tokens (plus maybe the last verifier token?)
        # NO: standard verification:
        # We have verified up to step T.
        # We need to compute logits for T+1...T+K
        # The input to the model should be [token_T, draft_1, ... draft_K]
        # BUT if we cached up to T, we just pass [draft_1 ... draft_K]??
        # WAIT. The model outputs logits for the NEXT token.
        # So to verify draft_1 (which is at T+1), we needed logits from token_T.
        # If KV cache has state up to T (inclusive, meaning processed T), then calling model with NO input
        # would give bad results. We need to pass the last token to get the next logit?
        # NO, 'past_key_values' usually implies we are ready to predict T+1.
        # So we pass [draft_1, ... draft_K]. The model uses KV up to T, processes draft_1, outputs logit for draft_2.
        # AND it outputs logit for draft_1 (at the first position).
        # So we need to evaluate: Does P(draft_1 | T) == draft_1?
        # To get P(draft_1 | T), we needed the output from processing T.
        # If KV cache stores processing of T, then the "next token" logic is already computed?
        # Typically KV cache stores keys/values. To get logits for T+1, we usually just need to run forward on... what?
        # If we ran forward on T last time, we got logits for T+1. We should have stored them?
        # OR we re-compute the last token?
        # Optimization: We usually return logits alongside KV.
        # For simplicity: We will assume we re-process the LAST verified token + drafts.
        # So input is [last_accepted_token, draft_1, ... draft_K].
        # But `input_ids` passed here is usually just the drafts?
        # Let's adjust `input_ids` to be [draft_tokens].
        # AND we need to make sure the KV cache matches.
        
        # Prepare inputs
        # We need logits for checking d1. These must come from `previous_verifier_logits`.
        # `previous_verifier_logits` should be [batch, vocab] corresponding to the last confirmed token.
        
        # We run the model on `[d1, ... dn]`.
        # It produces logits for `[d2, ... d(n+1)]`.
        
        # So:
        # Check d1 using `previous_verifier_logits`.
        # Check d2 using `outputs.logits[:, 0, :]`.
        # ...
        
        accepted_tokens_batch = []
        num_accepted_list = []
        new_kv_cache = None
        new_verifier_logits = None

        # Use stream context if available
        with self._verifier_lock:
            with torch.no_grad():
                ctx = torch.cuda.stream(stream) if stream is not None else (
                      torch.nullcontext() if hasattr(torch, "nullcontext") else torch.no_grad())
                
                with ctx:
                    outputs = self.verifier_model(full_input, past_key_values=past_key_values, use_cache=True)
                    logits = outputs.logits # [batch, seq_len, vocab]
                    new_kv_cache = outputs.past_key_values
                    
                    # Store last logits for next step (corresponding to the last draft token processed)
                    # Use the last logit from the sequence
                    new_verifier_logits = logits[:, -1, :]

                    for b in range(batch_size):
                        accepted_tokens = []
                        curr_draft_tokens = draft_tokens_batch[b]
                        
                        # 1. Verify first token using previous_logits
                        if previous_verifier_logits is not None:
                            prev_logits = previous_verifier_logits[b]
                            first_token_pred = torch.argmax(prev_logits).item()
                            if self.verifier_vocab_size is not None and first_token_pred >= self.verifier_vocab_size:
                                first_token_pred = self.verifier_vocab_size - 1
                                
                            if len(curr_draft_tokens) > 0:
                                if first_token_pred == curr_draft_tokens[0]:
                                    accepted_tokens.append(curr_draft_tokens[0])
                                else:
                                    # First token rejected.
                                    # We accept the correction (the predicted token)
                                    accepted_tokens.append(first_token_pred)
                                    accepted_tokens_batch.append(accepted_tokens)
                                    num_accepted_list.append(1) # Only 1 accepted (the correction)
                                    continue # Stop processing this sequence
                        
                        # 2. Verify remaining tokens
                        # We iterate starting from d1 (index 0).
                        # Logits at index i correspond to prediction for d(i+2).
                        # Wait. 
                        # Input: [d1, d2, d3]
                        # Logit[0] -> prediction for d2.
                        # We verify d2 against Logit[0].
                        
                        # We already verified d1 above.
                        # Now loop for d2...dn.
                        
                        # We need to be careful if d1 was rejected, we already continued.
                        # If d1 was accepted, we are here.
                        
                        start_idx = 0
                        # If we verified d1, we might want to check d2.
                        # curr_draft_tokens[0] is d1. accepted.
                        # We want to check curr_draft_tokens[1] (d2).
                        # It should match prediction from input[0] (d1).
                        # input[0] maps to logits[0].
                        
                        # So loop i from 0 to len-2?
                        # i corresponds to input token index.
                        # input[i] predicts draft[i+1].
                        
                        for i in range(len(curr_draft_tokens) - 1):
                            # We are checking draft[i+1]
                            draft_token_to_check = curr_draft_tokens[i+1]
                            
                            pred_logits = logits[b, i, :]
                            predicted_token = torch.argmax(pred_logits, dim=-1).item()
                            
                            if self.verifier_vocab_size is not None and predicted_token >= self.verifier_vocab_size:
                                predicted_token = self.verifier_vocab_size - 1

                            if predicted_token == draft_token_to_check:
                                accepted_tokens.append(draft_token_to_check)
                            else:
                                accepted_tokens.append(predicted_token)
                                break # Stop at first mismatch
                        
                        accepted_tokens_batch.append(accepted_tokens)
                        num_accepted_list.append(len(accepted_tokens))

                    if stream is not None:
                        stream.synchronize()
        
        return accepted_tokens_batch, num_accepted_list, new_kv_cache, new_verifier_logits
    
    def prefill(
        self,
        prompt: str,
        stream_id: Optional[int] = None
    ) -> tuple[torch.Tensor, Any, Any, Any]:
        """
        Prefill operation - encode prompt and prepare KV cache.

        Args:
            prompt: Input prompt
            stream_id: CUDA stream ID if using streams

        Returns:
            Tuple(Input token IDs, Draft KV Cache, Verifier KV Cache, Verifier Logits)
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
        
        # Get CUDA stream
        stream = None
        if stream_id is not None and self.stream_manager:
            stream = self.stream_manager.get_stream(stream_id)

        # Compute KV Caches for both models (Draft and Verifier)
        # We need to run both models on the prompt to initialize their caches.
        
        draft_logits = None
        # 1. Run Draft Model Prefill
        with self._draft_lock:
            with torch.no_grad():
                ctx = torch.cuda.stream(stream) if stream is not None else(
                      torch.nullcontext() if hasattr(torch, "nullcontext") else torch.no_grad())
                with ctx:
                    draft_out = self.draft_model(input_ids, use_cache=True)
                    draft_kv = draft_out.past_key_values
                    draft_logits = draft_out.logits[:, -1, :]
        
        # 2. Run Verifier Model Prefill
        verifier_logits = None
        with self._verifier_lock:
             with torch.no_grad():
                ctx = torch.cuda.stream(stream) if stream is not None else(
                      torch.nullcontext() if hasattr(torch, "nullcontext") else torch.no_grad())
                with ctx:
                    verifier_out = self.verifier_model(input_ids, use_cache=True)
                    verifier_kv = verifier_out.past_key_values
                    # Return the logits of the LAST token for next step verification
                    verifier_logits = verifier_out.logits[:, -1, :] 
                    
        if stream is not None:
            stream.synchronize()

        return input_ids, draft_kv, verifier_kv, verifier_logits, draft_logits
    
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
