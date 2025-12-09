"""
Configuration management for Verify-PD system.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for draft and verifier models."""
    draft_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    verifier_model_name: str = "meta-llama/Llama-2-7b-hf"
    max_model_len: int = 2048
    max_new_tokens: int = 100  # Maximum tokens to generate per request
    dtype: str = "auto"  # auto, float16, bfloat16
    trust_remote_code: bool = False
    hf_token: Optional[str] = None  # HuggingFace access token for gated models


@dataclass
class SchedulerConfig:
    """Configuration for three-lane scheduler."""
    # Lane priorities (lower number = higher priority)
    decode_priority: int = 1  # Highest priority
    verify_priority: int = 2  # Medium priority
    prefill_priority: int = 3  # Lowest priority
    
    # Queue management
    max_queue_size: int = 1000
    batch_size: int = 8
    
    # Micro-batching for verify lane
    verify_micro_batch_size: int = 4


@dataclass
class ControllerConfig:
    """Configuration for acceptance-aware feedback controller."""
    # Draft length bounds
    min_draft_length: int = 2
    max_draft_length: int = 8
    initial_draft_length: int = 4
    
    # Controller thresholds
    target_acceptance_ratio: float = 0.6  # Target 60% acceptance
    max_verify_queue_depth: int = 10
    target_decode_p95_ms: float = 50.0  # Target 50ms p95 latency
    
    # Adjustment rates
    length_increase_threshold: float = 0.7  # Increase L if acceptance > 70%
    length_decrease_threshold: float = 0.4  # Decrease L if acceptance < 40%
    
    # Update frequency
    update_interval_requests: int = 10  # Update L every N requests


@dataclass
class MetricsConfig:
    """Configuration for metrics collection."""
    enable_metrics: bool = True
    log_interval_seconds: int = 5
    percentiles: list = field(default_factory=lambda: [0.5, 0.95, 0.99])
    save_to_file: bool = True
    output_dir: str = "results/"


@dataclass
class HardwareConfig:
    """Configuration for hardware execution."""
    device: str = "cuda"  # cuda or cpu
    num_streams: int = 3  # One for each lane
    enable_cuda_graphs: bool = True
    gpu_memory_utilization: float = 0.9


@dataclass
class VerifyPDConfig:
    """Main configuration for Verify-PD system."""
    model: ModelConfig = field(default_factory=ModelConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    
    # General settings
    seed: int = 42
    test_mode: bool = False  # Use smaller models for testing


def get_default_config() -> VerifyPDConfig:
    """Get default configuration."""
    return VerifyPDConfig()


def get_cpu_config() -> VerifyPDConfig:
    """Get CPU-compatible configuration for development."""
    config = VerifyPDConfig()
    config.hardware.device = "cpu"
    config.hardware.enable_cuda_graphs = False
    config.model.dtype = "float32"
    config.scheduler.batch_size = 2  # Smaller batches for CPU
    return config


def get_performance_config() -> VerifyPDConfig:
    """Get configuration optimized to demonstrate Verify-PD performance benefits."""
    config = get_default_config()
    # Fast draft model + accurate verifier for optimal speculation
    config.model.draft_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    config.model.verifier_model_name = "meta-llama/Llama-2-7b-hf"
    config.model.max_new_tokens = 100  # Enough tokens for speculation to shine
    config.scheduler.batch_size = 4
    config.scheduler.verify_micro_batch_size = 4  # Allow parallel verification
    return config


def get_test_config() -> VerifyPDConfig:
    """Get test configuration with small models."""
    config = get_cpu_config()
    config.test_mode = True
    config.model.draft_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    config.model.verifier_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    config.model.max_model_len = 512
    config.model.max_new_tokens = 20  # Fewer tokens for fast iteration
    return config


def get_fast_config() -> VerifyPDConfig:
    """Get fast iteration configuration with small draft + large verifier models."""
    config = get_default_config()
    # Use fast draft model + accurate verifier model for true speculation benefit
    config.model.draft_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Fast draft
    config.model.verifier_model_name = "meta-llama/Llama-2-7b-hf"  # Accurate verifier
    config.model.max_new_tokens = 50  # More tokens to see speculation benefit
    config.scheduler.batch_size = 1  # Small batches for fast iteration
    config.scheduler.verify_micro_batch_size = 1
    return config
