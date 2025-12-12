"""
CUDA stream management with CPU fallback.
"""

import torch
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class StreamManager:
    """Manages CUDA streams for each lane with CPU fallback."""
    
    def __init__(self, device: str = "cuda", num_streams: int = 3):
        """
        Initialize stream manager.
        
        Args:
            device: Device to use (cuda or cpu)
            num_streams: Number of streams to create
        """
        self.device = device
        self.num_streams = num_streams
        self.streams: List[Optional[torch.cuda.Stream]] = []
        
        self._initialize_streams()
    
    def _initialize_streams(self):
        """Initialize CUDA streams or CPU placeholders."""
        if self.device == "cuda" and torch.cuda.is_available():
            logger.info(f"Initializing {self.num_streams} CUDA streams")
            for i in range(self.num_streams):
                self.streams.append(torch.cuda.Stream())
        else:
            logger.info(f"Running on CPU - stream management disabled")
            # Use None as placeholder for CPU execution
            self.streams = [None] * self.num_streams
    
    def get_stream(self, lane_id: int) -> Optional[torch.cuda.Stream]:
        """
        Get stream for a specific lane.
        
        Args:
            lane_id: Lane identifier (0=decode, 1=verify, 2=prefill)
            
        Returns:
            CUDA stream or None for CPU
        """
        if lane_id >= self.num_streams or lane_id >= len(self.streams):
            logger.warning(f"Lane ID {lane_id} exceeds available streams (count={len(self.streams)})")
            return None
        return self.streams[lane_id]
    
    def synchronize(self, lane_id: Optional[int] = None):
        """
        Synchronize streams.
        
        Args:
            lane_id: Specific lane to sync, or None to sync all
        """
        if self.device != "cuda" or not torch.cuda.is_available():
            return
        
        if lane_id is not None:
            stream = self.get_stream(lane_id)
            if stream is not None:
                stream.synchronize()
        else:
            # Synchronize all streams
            for stream in self.streams:
                if stream is not None:
                    stream.synchronize()
    
    def is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        return self.device == "cuda" and torch.cuda.is_available()
    
    def cleanup(self):
        """Clean up streams."""
        if self.is_cuda_available():
            self.synchronize()
        self.streams.clear()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
