"""
Lane abstraction for priority queues.
"""

from collections import deque
from typing import Any, Optional, List
from enum import IntEnum
import logging

logger = logging.getLogger(__name__)


class LaneType(IntEnum):
    """Lane types with priorities (lower value = higher priority)."""
    DECODE = 1   # Highest priority - single token generation
    VERIFY = 2   # Medium priority - verify L draft tokens
    PREFILL = 3  # Lowest priority - initial prompt processing


class Request:
    """Represents a request in the system."""
    
    def __init__(
        self,
        request_id: str,
        prompt: str,
        stage: LaneType,
        draft_tokens: Optional[list] = None,
        created_at: Optional[float] = None
    ):
        """
        Initialize request.
        
        Args:
            request_id: Unique identifier
            prompt: Input prompt or partial sequence
            stage: Current processing stage
            draft_tokens: Draft tokens to verify (for verify stage)
            created_at: Timestamp when request was created
        """
        self.request_id = request_id
        self.prompt = prompt
        self.stage = stage
        self.draft_tokens = draft_tokens or []
        self.created_at = created_at or 0.0
        
        # Tracking metrics
        self.tokens_generated = 0
        self.tokens_accepted = 0
        self.prefill_start_time = None
        self.decode_start_time = None
        self.verify_start_time = None
        self.completion_time = None
        
        # Token timestamp tracking for per-token latency metrics
        self.token_timestamps: List[float] = []
        self.request_start_time = created_at if created_at else 0.0
        
        # Async processing state
        self.completed = False
        self.error = None
        self.result_text = None
        self.generated_tokens = []
        self._input_ids = None
    
    def get_acceptance_ratio(self) -> float:
        """Calculate acceptance ratio for this request."""
        if self.tokens_generated == 0:
            return 0.0
        return self.tokens_accepted / self.tokens_generated
    
    def __repr__(self) -> str:
        return f"Request(id={self.request_id}, stage={self.stage.name}, tokens={self.tokens_generated})"


class Lane:
    """Priority queue for a specific processing stage."""
    
    def __init__(self, lane_type: LaneType, max_size: int = 1000):
        """
        Initialize lane.
        
        Args:
            lane_type: Type/priority of this lane
            max_size: Maximum queue size
        """
        self.lane_type = lane_type
        self.max_size = max_size
        self.queue: deque[Request] = deque()
        
        # Statistics
        self.total_processed = 0
        self.total_rejected = 0  # Rejected due to full queue
    
    def add_request(self, request: Request) -> bool:
        """
        Add request to lane queue.
        
        Args:
            request: Request to add
            
        Returns:
            True if added successfully, False if queue is full
        """
        if len(self.queue) >= self.max_size:
            logger.warning(f"{self.lane_type.name} lane queue full, rejecting request {request.request_id}")
            self.total_rejected += 1
            return False
        
        self.queue.append(request)
        logger.debug(f"Added {request.request_id} to {self.lane_type.name} lane (size={len(self.queue)})")
        return True
    
    def get_next_request(self) -> Optional[Request]:
        """
        Get next request from queue.
        
        Returns:
            Next request or None if queue is empty
        """
        if not self.queue:
            return None
        
        request = self.queue.popleft()
        self.total_processed += 1
        return request
    
    def get_batch(self, batch_size: int) -> list[Request]:
        """
        Get a batch of requests.
        
        Args:
            batch_size: Maximum batch size
            
        Returns:
            List of requests
        """
        batch = []
        for _ in range(min(batch_size, len(self.queue))):
            request = self.get_next_request()
            if request:
                batch.append(request)
        return batch

    def get_all_requests(self) -> List[Request]:
        """Get all requests currently in this lane."""
        return list(self.queue)

    def peek(self) -> Optional[Request]:
        """Peek at next request without removing it."""
        return self.queue[0] if self.queue else None
    
    def size(self) -> int:
        """Get current queue size."""
        return len(self.queue)
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self.queue) == 0
    
    def is_full(self) -> bool:
        """Check if queue is full."""
        return len(self.queue) >= self.max_size
    
    def clear(self):
        """Clear all requests from queue."""
        self.queue.clear()
    
    def get_stats(self) -> dict:
        """Get lane statistics."""
        return {
            "lane_type": self.lane_type.name,
            "current_size": len(self.queue),
            "max_size": self.max_size,
            "total_processed": self.total_processed,
            "total_rejected": self.total_rejected,
            "utilization": len(self.queue) / self.max_size if self.max_size > 0 else 0.0
        }
    
    def __len__(self) -> int:
        return len(self.queue)
    
    def __repr__(self) -> str:
        return f"Lane({self.lane_type.name}, size={len(self.queue)}/{self.max_size})"
