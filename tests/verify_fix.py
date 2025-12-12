
import sys
from unittest.mock import MagicMock

# Mock dependencies to avoid loading models
sys.modules['src.scheduler'] = MagicMock()
sys.modules['src.controller'] = MagicMock()
sys.modules['src.engine.model_runner'] = MagicMock()

# Now import the class under test
from src.engine.speculative_engine import SpeculativeEngine

def test_fix():
    print("Testing SpeculativeEngine for UnboundLocalError...")
    
    # Setup mock config - just a plain mock
    config = MagicMock()
    config.model.max_new_tokens = 10
    config.scheduler = MagicMock()
    config.controller = MagicMock()
    config.hardware.device = "cpu"
    
    # Instantiate engine
    engine = SpeculativeEngine(config)
    
    # Mock internal components
    engine.model_runner = MagicMock()
    engine.stream_manager = MagicMock()
    engine.controller = MagicMock()
    engine.scheduler = MagicMock()
    engine.scheduler.get_active_requests.return_value = [] # concurrency 0
    
    # Create dummy request
    req = MagicMock()
    req._input_ids.size.return_value = 5 # length
    # Important: decode_start_time = 0.0 triggers the line that was failing
    req.decode_start_time = 0.0
    req.last_draft_logits = None
    req.kv_cache_draft = None
    req.kv_cache_verifier = None
    req.last_verifier_logits = None
    
    # Mock model runner draft vocab size for line 311
    engine.model_runner.draft_vocab_size = 100
    
    batch = [req]
    
    try:
        # Call the problematic method
        # We need it to run far enough to hit req.decode_start_time = time.time()
        # That happens almost immediately.
        # Then it continues.
        engine._handle_decode_batch_pd_style(batch)
    except UnboundLocalError as e:
        print(f"❌ FAILED: UnboundLocalError detected: {e}")
        sys.exit(1)
    except Exception as e:
        # We expect other errors because our mocks are thin (e.g. torch operations)
        # But if we get past the UnboundLocalError, we are good.
        print(f"✅ PASSED: calling _handle_decode_batch_pd_style did not raise UnboundLocalError. (Got expected {type(e).__name__})")

if __name__ == "__main__":
    test_fix()
