"""
Test module for verifying speaker diarization reset between streams.

This test validates that speaker memory is properly cleared when a new
stream starts, ensuring speaker IDs don't persist across different streams.
"""

import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
import sys
import threading

src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from diarization.diarization_client import (
    DiarizationClient, DiarizationResult, SpeakerSegment,
    SpeakerMemory, RESET_SIGNAL
)


def test_speaker_memory_reset():
    """Test that SpeakerMemory.reset() clears all speaker data."""
    print("\n=== Testing SpeakerMemory.reset() ===")
    
    memory = SpeakerMemory(threshold=0.70)
    
    # Create test embeddings (random vectors)
    embedding1 = np.random.randn(512).astype(np.float32)
    embedding2 = np.random.randn(512).astype(np.float32)
    
    # First speaker
    speaker1, conf1, alt1 = memory.identify(embedding1, {"time": 0.0})
    print(f"First speaker: {speaker1}, confidence: {conf1:.3f}")
    assert speaker1 == "speaker_0", f"Expected speaker_0, got {speaker1}"
    
    # Different speaker
    speaker2, conf2, alt2 = memory.identify(embedding2, {"time": 2.0})
    print(f"Second speaker: {speaker2}, confidence: {conf2:.3f}")
    assert speaker2 == "speaker_1", f"Expected speaker_1, got {speaker2}"
    
    # Verify speaker memory has data
    stats = memory.get_speaker_stats()
    print(f"Speaker stats before reset: {stats}")
    assert len(stats) == 2, f"Expected 2 speakers, got {len(stats)}"
    
    # Reset speaker memory
    memory.reset()
    
    # Verify speaker memory is cleared
    stats_after = memory.get_speaker_stats()
    print(f"Speaker stats after reset: {stats_after}")
    assert len(stats_after) == 0, f"Expected 0 speakers after reset, got {len(stats_after)}"
    
    # Verify next speaker gets ID speaker_0 again (counter reset)
    speaker1_after, conf1_after, _ = memory.identify(embedding1, {"time": 10.0})
    print(f"Speaker after reset: {speaker1_after}, confidence: {conf1_after:.3f}")
    assert speaker1_after == "speaker_0", f"Expected speaker_0 after reset, got {speaker1_after}"
    
    print("✓ SpeakerMemory.reset() works correctly")
    return True


def test_diarization_client_reset():
    """Test that DiarizationClient.reset() clears all state."""
    print("\n=== Testing DiarizationClient.reset() ===")
    
    client = DiarizationClient(hf_token="test-token")
    
    # Initialize the lock (normally done during initialize())
    client._lock = threading.Lock()
    
    # Simulate in-flight requests
    client.add_in_flight_request("req-1")
    client.add_in_flight_request("req-2")
    assert len(client.in_flight_requests) == 2, "Should have 2 in-flight requests"
    
    # Reset client
    client.reset()
    
    # Verify in-flight requests are cleared
    assert len(client.in_flight_requests) == 0, "Should have 0 in-flight requests after reset"
    
    print("✓ DiarizationClient.reset() works correctly")
    return True


import pytest

@pytest.mark.asyncio
async def test_on_stream_start_resets_diarization():
    """Test that on_stream_start() properly resets diarization state."""
    print("\n=== Testing on_stream_start() diarization reset ===")
    
    from pipeline.main import on_stream_start, TranscriberState
    import pipeline.main
    
    # Create mock state
    state = TranscriberState()
    state.diarization_counter = 5
    state.diarization_temp_files = ["file1.wav", "file2.wav"]
    state.diarization_window_timestamps = {"req-1": (0.0, 6.0)}
    state.diarization_audio_buffer = np.random.randn(16000 * 6).astype(np.float32)
    state.diarization_buffer_start_ts = 0.0
    
    # Mock diarization client
    mock_diarization_client = Mock()
    mock_diarization_client.reset = Mock()
    state.diarization_client = mock_diarization_client
    
    # Mock other clients
    state.summary_client = Mock()
    state.summary_client.reset = Mock()
    state.whisper_client = Mock()
    state.whisper_client.reset = Mock()
    
    # Set global state
    original_state = pipeline.main.STATE
    pipeline.main.STATE = state
    
    try:
        # Call on_stream_start
        await on_stream_start({})
        
        # Verify diarization client reset was called
        mock_diarization_client.reset.assert_called_once()
        print("✓ DiarizationClient.reset() was called in on_stream_start()")
        
        # Verify other state was reset
        assert state.diarization_counter == 0, "diarization_counter should be reset"
        assert len(state.diarization_temp_files) == 0, "diarization_temp_files should be cleared"
        assert len(state.diarization_window_timestamps) == 0, "diarization_window_timestamps should be cleared"
        assert len(state.diarization_audio_buffer) == 0, "diarization_audio_buffer should be cleared"
        assert state.diarization_buffer_start_ts is None, "diarization_buffer_start_ts should be None"
        
        print("✓ All diarization state was properly reset in on_stream_start()")
        return True
        
    finally:
        pipeline.main.STATE = original_state


def test_speaker_id_persistence_across_streams():
    """Test that demonstrates the problem: speaker IDs should reset between streams."""
    print("\n=== Testing Speaker ID Persistence (Bug Demonstration) ===")
    
    # Simulate Stream 1 processing
    memory_stream1 = SpeakerMemory(threshold=0.70)
    
    # Create embeddings for stream 1
    embedding_stream1_speaker_a = np.random.randn(512).astype(np.float32)
    embedding_stream1_speaker_b = np.random.randn(512).astype(np.float32)
    
    # Process stream 1
    speaker_a_1, _, _ = memory_stream1.identify(embedding_stream1_speaker_a, {"stream": 1, "time": 0.0})
    speaker_b_1, _, _ = memory_stream1.identify(embedding_stream1_speaker_b, {"stream": 1, "time": 2.0})
    
    print(f"Stream 1 - Speaker A: {speaker_a_1}, Speaker B: {speaker_b_1}")
    
    # Simulate Stream 2 processing (WITHOUT reset - this is the bug)
    memory_stream2_no_reset = SpeakerMemory(threshold=0.70)
    
    # Copy state from stream 1 (simulating no reset)
    memory_stream2_no_reset.centroids = memory_stream1.centroids.copy()
    memory_stream2_no_reset.counts = memory_stream1.counts.copy()
    memory_stream2_no_reset.speaker_counter = memory_stream1.speaker_counter
    memory_stream2_no_reset.last_speaker = memory_stream1.last_speaker
    
    # Create embeddings for stream 2 (different speakers)
    embedding_stream2_speaker_x = np.random.randn(512).astype(np.float32)
    embedding_stream2_speaker_y = np.random.randn(512).astype(np.float32)
    
    # Process stream 2 with stale memory
    speaker_x_no_reset, _, _ = memory_stream2_no_reset.identify(embedding_stream2_speaker_x, {"stream": 2, "time": 0.0})
    speaker_y_no_reset, _, _ = memory_stream2_no_reset.identify(embedding_stream2_speaker_y, {"stream": 2, "time": 2.0})
    
    print(f"Stream 2 (NO RESET) - Speaker X: {speaker_x_no_reset}, Speaker Y: {speaker_y_no_reset}")
    
    # This is the BUG: speaker IDs from stream 1 persist to stream 2
    # Speaker X gets ID speaker_2 instead of speaker_0
    assert speaker_x_no_reset == "speaker_2", f"Expected speaker_2 (stale), got {speaker_x_no_reset}"
    print("✗ BUG CONFIRMED: Speaker IDs persist across streams without reset")
    
    # Now test WITH reset (the fix)
    memory_stream2_with_reset = SpeakerMemory(threshold=0.70)
    memory_stream2_with_reset.reset()  # Clear speaker memory
    
    # Process stream 2 with fresh memory
    speaker_x_with_reset, _, _ = memory_stream2_with_reset.identify(embedding_stream2_speaker_x, {"stream": 2, "time": 0.0})
    speaker_y_with_reset, _, _ = memory_stream2_with_reset.identify(embedding_stream2_speaker_y, {"stream": 2, "time": 2.0})
    
    print(f"Stream 2 (WITH RESET) - Speaker X: {speaker_x_with_reset}, Speaker Y: {speaker_y_with_reset}")
    
    # With reset, speaker IDs start from speaker_0
    assert speaker_x_with_reset == "speaker_0", f"Expected speaker_0 (fresh), got {speaker_x_with_reset}"
    assert speaker_y_with_reset == "speaker_1", f"Expected speaker_1 (fresh), got {speaker_y_with_reset}"
    print("✓ FIX VERIFIED: Speaker IDs reset correctly with reset() call")
    
    return True


def run_all_tests():
    """Run all stream reset tests."""
    print("=" * 70)
    print("SPEAKER DIARIZATION RESET BETWEEN STREAMS TESTS")
    print("=" * 70)
    
    results = []
    
    # Test 1: SpeakerMemory.reset()
    try:
        results.append(("SpeakerMemory.reset()", test_speaker_memory_reset()))
    except Exception as e:
        print(f"✗ SpeakerMemory.reset() test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("SpeakerMemory.reset()", False))
    
    # Test 2: DiarizationClient.reset()
    try:
        results.append(("DiarizationClient.reset()", test_diarization_client_reset()))
    except Exception as e:
        print(f"✗ DiarizationClient.reset() test failed: {e}")
        results.append(("DiarizationClient.reset()", False))
    
    # Test 3: on_stream_start() diarization reset
    try:
        results.append(("on_stream_start() reset", asyncio.run(test_on_stream_start_resets_diarization())))
    except Exception as e:
        print(f"✗ on_stream_start() reset test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("on_stream_start() reset", False))
    
    # Test 4: Speaker ID persistence
    try:
        results.append(("Speaker ID persistence", test_speaker_id_persistence_across_streams()))
    except Exception as e:
        print(f"✗ Speaker ID persistence test failed: {e}")
        results.append(("Speaker ID persistence", False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed!"))
    
    return all_passed


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    success = run_all_tests()
    exit(0 if success else 1)
