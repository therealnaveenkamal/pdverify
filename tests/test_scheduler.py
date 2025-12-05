"""
Unit tests for three-lane scheduler.
"""

import pytest
from src.scheduler import Lane, LaneType, Request, ThreeLaneScheduler
from src.utils import SchedulerConfig


def test_lane_creation():
    """Test lane creation and initialization."""
    lane = Lane(LaneType.DECODE, max_size=10)
    assert lane.lane_type == LaneType.DECODE
    assert lane.size() == 0
    assert lane.is_empty()
    assert not lane.is_full()


def test_request_creation():
    """Test request creation."""
    request = Request(
        request_id="test_1",
        prompt="Hello world",
        stage=LaneType.PREFILL
    )
    assert request.request_id == "test_1"
    assert request.stage == LaneType.PREFILL
    assert request.tokens_generated == 0


def test_lane_add_request():
    """Test adding requests to lane."""
    lane = Lane(LaneType.DECODE, max_size=3)
    
    req1 = Request("req1", "prompt1", LaneType.DECODE)
    req2 = Request("req2", "prompt2", LaneType.DECODE)
    
    assert lane.add_request(req1)
    assert lane.add_request(req2)
    assert lane.size() == 2


def test_lane_full_rejection():
    """Test that full lane rejects requests."""
    lane = Lane(LaneType.DECODE, max_size=2)
    
    req1 = Request("req1", "prompt1", LaneType.DECODE)
    req2 = Request("req2", "prompt2", LaneType.DECODE)
    req3 = Request("req3", "prompt3", LaneType.DECODE)
    
    assert lane.add_request(req1)
    assert lane.add_request(req2)
    assert not lane.add_request(req3)  # Should be rejected
    assert lane.total_rejected == 1


def test_lane_get_next_request():
    """Test getting next request from lane."""
    lane = Lane(LaneType.DECODE, max_size=10)
    
    req1 = Request("req1", "prompt1", LaneType.DECODE)
    req2 = Request("req2", "prompt2", LaneType.DECODE)
    
    lane.add_request(req1)
    lane.add_request(req2)
    
    next_req = lane.get_next_request()
    assert next_req.request_id == "req1"
    assert lane.size() == 1


def test_lane_batch():
    """Test getting batch of requests."""
    lane = Lane(LaneType.DECODE, max_size=10)
    
    for i in range(5):
        lane.add_request(Request(f"req{i}", f"prompt{i}", LaneType.DECODE))
    
    batch = lane.get_batch(3)
    assert len(batch) == 3
    assert lane.size() == 2


def test_scheduler_creation():
    """Test scheduler creation."""
    config = SchedulerConfig()
    scheduler = ThreeLaneScheduler(config)
    
    assert scheduler.decode_lane.lane_type == LaneType.DECODE
    assert scheduler.verify_lane.lane_type == LaneType.VERIFY
    assert scheduler.prefill_lane.lane_type == LaneType.PREFILL


def test_scheduler_submit():
    """Test submitting requests to scheduler."""
    config = SchedulerConfig()
    scheduler = ThreeLaneScheduler(config)
    
    decode_req = Request("req1", "prompt1", LaneType.DECODE)
    verify_req = Request("req2", "prompt2", LaneType.VERIFY)
    prefill_req = Request("req3", "prompt3", LaneType.PREFILL)
    
    assert scheduler.submit_request(decode_req)
    assert scheduler.submit_request(verify_req)
    assert scheduler.submit_request(prefill_req)
    
    assert scheduler.decode_lane.size() == 1
    assert scheduler.verify_lane.size() == 1
    assert scheduler.prefill_lane.size() == 1


def test_scheduler_priority():
    """Test scheduler respects priority order."""
    config = SchedulerConfig()
    scheduler = ThreeLaneScheduler(config)
    
    # Add requests to all lanes
    scheduler.submit_request(Request("prefill1", "p1", LaneType.PREFILL))
    scheduler.submit_request(Request("verify1", "v1", LaneType.VERIFY))
    scheduler.submit_request(Request("decode1", "d1", LaneType.DECODE))
    
    # Should get decode first (highest priority)
    batch = scheduler.get_next_batch()
    assert batch[0].request_id == "decode1"
    
    # Then verify (medium priority)
    batch = scheduler.get_next_batch()
    assert batch[0].request_id == "verify1"
    
    # Finally prefill (lowest priority)
    batch = scheduler.get_next_batch()
    assert batch[0].request_id == "prefill1"


def test_scheduler_transition():
    """Test transitioning request between lanes."""
    config = SchedulerConfig()
    scheduler = ThreeLaneScheduler(config)
    
    request = Request("req1", "prompt1", LaneType.PREFILL)
    scheduler.submit_request(request)
    
    # Get request from prefill
    batch = scheduler.get_next_batch()
    req = batch[0]
    
    # Transition to decode
    assert scheduler.transition_request(req, LaneType.DECODE)
    assert scheduler.decode_lane.size() == 1
    assert req.stage == LaneType.DECODE


def test_acceptance_ratio():
    """Test acceptance ratio calculation."""
    request = Request("req1", "prompt1", LaneType.DECODE)
    request.tokens_generated = 10
    request.tokens_accepted = 7
    
    assert request.get_acceptance_ratio() == 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
