"""
Three-lane scheduler for Verify-PD system.
"""

import time
import threading
from typing import Optional, List, Dict
import logging

from .lane import Lane, LaneType, Request
from ..utils.config import SchedulerConfig

logger = logging.getLogger(__name__)


class ThreeLaneScheduler:
    """
    Scheduler implementing three-lane priority system:
    - Decode Lane (highest priority)
    - Verify Lane (medium priority) 
    - Prefill Lane (lowest priority)
    """
    
    def __init__(self, config: SchedulerConfig):
        """
        Initialize three-lane scheduler.
        
        Args:
            config: Scheduler configuration
        """
        self.config = config
        
        # Initialize lanes with priorities
        self.decode_lane = Lane(LaneType.DECODE, max_size=config.max_queue_size)
        self.verify_lane = Lane(LaneType.VERIFY, max_size=config.max_queue_size)
        self.prefill_lane = Lane(LaneType.PREFILL, max_size=config.max_queue_size)
        
        # Lane mapping for easy access
        self.lanes: Dict[LaneType, Lane] = {
            LaneType.DECODE: self.decode_lane,
            LaneType.VERIFY: self.verify_lane,
            LaneType.PREFILL: self.prefill_lane
        }
        
        # Active requests being processed
        self.active_requests: Dict[str, Request] = {}
        
        # Statistics
        self.total_requests_submitted = 0
        self.total_requests_completed = 0
        self.scheduler_start_time = time.time()

        # Thread safety - use RLock for reentrant locking
        self._lock = threading.RLock()
    
    def submit_request(self, request: Request) -> bool:
        """
        Submit a request to appropriate lane based on its stage.

        Args:
            request: Request to submit

        Returns:
            True if submitted successfully
        """
        with self._lock:
            lane = self.lanes[request.stage]
            success = lane.add_request(request)

            if success:
                self.total_requests_submitted += 1
                logger.debug(f"Submitted {request.request_id} to {request.stage.name} lane")

            return success

    def get_next_task(self) -> Optional[Request]:
        """
        Get next single task from highest priority non-empty lane.
        Used by worker thread for stage-based async processing.
        Priority: DECODE (1) > PREFILL (2) > VERIFY (3)
        This allows new requests to preempt verification!

        Returns:
            Single request from highest priority lane, or None if no work
        """
        with self._lock:
            # Check lanes in priority order (lower number = higher priority)
            priority_order = [LaneType.DECODE, LaneType.PREFILL, LaneType.VERIFY]

            for lane_type in priority_order:
                lane = self.lanes[lane_type]
                if not lane.is_empty():
                    request = lane.get_next_request()
                    if request:
                        logger.debug(f"[SCHEDULER] Dispatched {request.request_id} from {lane_type.name} lane")
                        return request

            return None

    def get_next_batch(self) -> Optional[List[Request]]:
        """
        Get next batch of requests following priority order.
        Priority: DECODE > PREFILL > VERIFY

        Returns:
            Batch of requests from highest priority non-empty lane
        """
        with self._lock:
            # Check lanes in priority order
            priority_order = [LaneType.DECODE, LaneType.PREFILL, LaneType.VERIFY]

            for lane_type in priority_order:
                    lane = self.lanes[lane_type]
                    if not lane.is_empty():
                        # Get batch size based on lane type
                        if lane_type == LaneType.VERIFY:
                            batch_size = self.config.verify_micro_batch_size
                        else:
                            batch_size = self.config.batch_size

                        batch = lane.get_batch(batch_size)
                        if batch:
                            logger.debug(f"Scheduled batch of {len(batch)} from {lane_type.name} lane")
                            return batch

            return None
    
    def transition_request(self, request: Request, new_stage: LaneType) -> bool:
        """
        Transition a request to a new processing stage.

        Args:
            request: Request to transition
            new_stage: New lane/stage for the request

        Returns:
            True if transition successful
        """
        with self._lock:
            # Note: In the current design, requests are removed from lanes when get_next_batch is called,
            # so we don't need to explicitly remove them here. Just update the stage and submit.

            request.stage = new_stage
            return self.submit_request(request)
    
    def complete_request(self, request: Request):
        """
        Mark a request as completed.

        Args:
            request: Request that completed
        """
        with self._lock:
            request.completion_time = time.time()
            self.total_requests_completed += 1

            if request.request_id in self.active_requests:
                del self.active_requests[request.request_id]

            logger.info(f"Completed request {request.request_id} "
                       f"(accepted {request.tokens_accepted}/{request.tokens_generated} tokens)")

    def get_active_requests(self) -> List[Request]:
        """
        Get all currently active requests across all lanes.

        Returns:
            List of active requests
        """
        with self._lock:
            active = []
            for lane in self.lanes.values():
                active.extend(lane.get_all_requests())
            return active
    
    def get_lane_stats(self) -> Dict[str, dict]:
        """Get statistics for all lanes."""
        return {
            lane_type.name: lane.get_stats()
            for lane_type, lane in self.lanes.items()
        }
    
    def get_scheduler_stats(self) -> dict:
        """Get overall scheduler statistics."""
        uptime = time.time() - self.scheduler_start_time
        
        return {
            "uptime_seconds": uptime,
            "total_submitted": self.total_requests_submitted,
            "total_completed": self.total_requests_completed,
            "pending_requests": self.total_requests_submitted - self.total_requests_completed,
            "active_requests": len(self.active_requests),
            "lanes": self.get_lane_stats()
        }
    
    def has_pending_work(self) -> bool:
        """Check if there are any pending requests in any lane."""
        return any(not lane.is_empty() for lane in self.lanes.values())
    
    def clear_all(self):
        """Clear all lanes (for testing/reset)."""
        for lane in self.lanes.values():
            lane.clear()
        self.active_requests.clear()
        logger.info("Cleared all lanes")
    
    def __repr__(self) -> str:
        return (f"ThreeLaneScheduler("
                f"decode={len(self.decode_lane)}, "
                f"verify={len(self.verify_lane)}, "
                f"prefill={len(self.prefill_lane)})")
