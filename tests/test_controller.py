"""
Unit tests for feedback controller.
"""

import pytest
from src.controller import FeedbackController
from src.utils import ControllerConfig


def test_controller_creation():
    """Test controller creation."""
    config = ControllerConfig()
    controller = FeedbackController(config)
    
    assert controller.get_draft_length() == config.initial_draft_length


def test_record_acceptance():
    """Test recording acceptance ratios."""
    config = ControllerConfig()
    controller = FeedbackController(config)
    
    controller.record_acceptance(0.8)
    controller.record_acceptance(0.7)
    controller.record_acceptance(0.6)
    
    assert len(controller.acceptance_ratios) == 3
    assert controller._get_average_acceptance() == pytest.approx(0.7, abs=0.01)


def test_record_latency():
    """Test recording decode latencies."""
    config = ControllerConfig()
    controller = FeedbackController(config)
    
    controller.record_decode_latency(10.0)
    controller.record_decode_latency(20.0)
    controller.record_decode_latency(100.0)  # High latency
    
    assert len(controller.decode_latencies) == 3
    p95 = controller._get_decode_p95()
    assert p95 > 50.0  # Should be high due to outlier


def test_draft_length_decrease():
    """Test that draft length decreases under load."""
    config = ControllerConfig()
    config.update_interval_requests = 1  # Update immediately
    config.max_verify_queue_depth = 5
    
    controller = FeedbackController(config)
    initial_length = controller.get_draft_length()
    
    # Simulate high queue depth
    controller.record_acceptance(0.5)
    new_length = controller.update(verify_queue_depth=10)  # Above threshold
    
    assert new_length is not None
    assert controller.get_draft_length() < initial_length


def test_draft_length_increase():
    """Test that draft length increases with good conditions."""
    config = ControllerConfig()
    config.update_interval_requests = 1
    config.initial_draft_length = 3
    
    controller = FeedbackController(config)
    initial_length = controller.get_draft_length()
    
    # Simulate good conditions
    controller.record_acceptance(0.8)  # High acceptance
    controller.record_decode_latency(10.0)  # Low latency
    new_length = controller.update(verify_queue_depth=0)  # Empty queue
    
    assert new_length is not None
    assert controller.get_draft_length() > initial_length


def test_draft_length_bounds():
    """Test that draft length stays within bounds."""
    config = ControllerConfig()
    config.min_draft_length = 2
    config.max_draft_length = 8
    config.update_interval_requests = 1
    
    controller = FeedbackController(config)
    
    # Try to decrease below minimum
    controller.current_draft_length = 2
    controller.record_acceptance(0.1)  # Very low acceptance
    controller.update(verify_queue_depth=20)
    assert controller.get_draft_length() >= config.min_draft_length
    
    # Try to increase above maximum
    controller.current_draft_length = 8
    controller.record_acceptance(0.9)  # Very high acceptance
    controller.record_decode_latency(1.0)  # Very low latency
    controller.update(verify_queue_depth=0)
    assert controller.get_draft_length() <= config.max_draft_length


def test_update_interval():
    """Test that controller only updates at intervals."""
    config = ControllerConfig()
    config.update_interval_requests = 5
    
    controller = FeedbackController(config)
    
    # Record 4 acceptances (below threshold)
    for _ in range(4):
        controller.record_acceptance(0.5)
        result = controller.update(verify_queue_depth=0)
        assert result is None  # Should not update yet
    
    # Record 1 more (reaches threshold) with high queue depth to trigger change
    controller.record_acceptance(0.3)  # Low acceptance
    controller.record_decode_latency(100.0)  # High latency
    result = controller.update(verify_queue_depth=15)  # High queue
    # Should update now (may or may not change length depending on conditions)


def test_get_stats():
    """Test getting controller statistics."""
    config = ControllerConfig()
    controller = FeedbackController(config)
    
    controller.record_acceptance(0.7)
    controller.record_decode_latency(25.0)
    
    stats = controller.get_stats()
    
    assert "current_draft_length" in stats
    assert "average_acceptance" in stats
    assert "decode_p95_ms" in stats
    assert stats["samples_collected"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
