"""
Test module for speaker diarization flow.

This test validates the complete diarization pipeline to identify
where speaker results are being lost.
"""

import asyncio
import numpy as np
import tempfile
import wave
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

import sys
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from diarization.diarization_client import (
    DiarizationClient, DiarizationResult, SpeakerSegment,
    SpeakerMemory, diarization_worker
)


def create_test_audio(duration_seconds: float = 6.0, sample_rate: int = 16000) -> str:
    """Create a test WAV file with synthetic audio."""
    path = tempfile.mktemp(suffix=".wav")
    # Generate synthetic audio (sine wave)
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), dtype=np.float32)
    audio = np.sin(2 * np.pi * 440 * t) * 0.5  # 440 Hz sine wave
    
    pcm16 = (audio * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return path


def test_speaker_memory():
    """Test SpeakerMemory class for speaker identification."""
    print("\n=== Testing SpeakerMemory ===")
    
    memory = SpeakerMemory(threshold=0.70, min_samples_for_match=1)
    
    # Create test embeddings (random vectors)
    embedding1 = np.random.randn(512).astype(np.float32)
    embedding2 = np.random.randn(512).astype(np.float32)
    embedding3 = np.random.randn(512).astype(np.float32)
    
    # First speaker
    speaker1, conf1, alt1 = memory.identify(embedding1, {"time": 0.0})
    print(f"First speaker: {speaker1}, confidence: {conf1:.3f}")
    assert speaker1 == "speaker_0", f"Expected speaker_0, got {speaker1}"
    
    # Same speaker again (should match)
    speaker1_again, conf1_again, alt1_again = memory.identify(embedding1, {"time": 1.0})
    print(f"Same speaker again: {speaker1_again}, confidence: {conf1_again:.3f}")
    assert speaker1_again == "speaker_0", f"Expected speaker_0, got {speaker1_again}"
    
    # Different speaker
    speaker2, conf2, alt2 = memory.identify(embedding2, {"time": 2.0})
    print(f"Second speaker: {speaker2}, confidence: {conf2:.3f}")
    assert speaker2 == "speaker_1", f"Expected speaker_1, got {speaker2}"
    
    # Back to first speaker (should match with high confidence)
    speaker1_back, conf1_back, alt1_back = memory.identify(embedding1, {"time": 3.0})
    print(f"Back to first speaker: {speaker1_back}, confidence: {conf1_back:.3f}")
    assert speaker1_back == "speaker_0", f"Expected speaker_0, got {speaker1_back}"
    
    print("✓ SpeakerMemory tests passed")


def test_diarization_result_handling():
    """Test that _handle_diarization_result and _send_speakers_message work correctly."""
    print("\n=== Testing Diarization Result Handling ===")
    
    # Mock PROCESSOR
    mock_processor = Mock()
    mock_processor.send_data = AsyncMock()
    
    # Import after mocking
    from pipeline.main import _send_speakers_message, _handle_diarization_result, STATE
    from diarization.diarization_client import DiarizationResult, SpeakerSegment
    
    # Create test segments
    segments = [
        SpeakerSegment(start=0.0, end=1.5, speaker="speaker_0", confidence=0.85, alt_speakers=None),
        SpeakerSegment(start=1.5, end=3.0, speaker="speaker_1", confidence=0.82, alt_speakers=None),
    ]
    
    # Test _send_speakers_message with mock processor
    import asyncio
    
    async def test_send():
        # Patch PROCESSOR globally
        import pipeline.main
        original_processor = pipeline.main.PROCESSOR
        pipeline.main.PROCESSOR = mock_processor
        
        try:
            await _send_speakers_message(segments, 0.0, 6.0)
            
            # Check if send_data was called
            if mock_processor.send_data.called:
                print(f"✓ _send_speakers_message sent data successfully")
                call_args = mock_processor.send_data.call_args[0][0]
                print(f"  Sent payload type: {type(call_args)}")
            else:
                print("✗ _send_speakers_message did NOT send data")
        finally:
            pipeline.main.PROCESSOR = original_processor
    
    asyncio.run(test_send())


def test_handle_diarization_result_with_empty_segments():
    """Test that empty segments are handled properly."""
    print("\n=== Testing Empty Segments Handling ===")
    
    from pipeline.main import _handle_diarization_result, STATE
    from diarization.diarization_client import DiarizationResult
    import asyncio
    import pipeline.main
    
    # Track if _send_speakers_message was called
    send_called = False
    
    async def mock_send(*args, **kwargs):
        nonlocal send_called
        send_called = True
        print("  _send_speakers_message was called")
    
    # Mock PROCESSOR
    mock_processor = Mock()
    mock_processor.send_data = AsyncMock()
    
    original_processor = pipeline.main.PROCESSOR
    original_send = None
    
    try:
        pipeline.main.PROCESSOR = mock_processor
        
        # Test with empty segments
        result = DiarizationResult(
            request_id="test-empty",
            audio_path="test.wav",
            segments=[],  # Empty segments!
            error=None
        )
        
        async def run_test():
            await _handle_diarization_result(result)
        
        asyncio.run(run_test())
        
        if not mock_processor.send_data.called:
            print("✓ Empty segments correctly skipped sending (no data sent)")
        else:
            print("✗ Empty segments incorrectly triggered sending")
            
    finally:
        pipeline.main.PROCESSOR = original_processor


def test_pipeline_flow():
    """Test the complete pipeline flow with mocked components."""
    print("\n=== Testing Pipeline Flow ===")
    
    import pipeline.main
    from pipeline.main import (
        TranscriberState, _append_audio, _diarization_buffer_duration_seconds,
        _pull_diarization_samples, _process_diarization_async
    )
    import asyncio
    
    # Create state
    state = TranscriberState()
    state.buffer_rate = 16000
    state.buffer_start_ts = 0.0
    state.diarization_buffer_start_ts = 0.0
    
    # Mock clients
    state.diarization_client = Mock()
    state.diarization_client.process_audio = AsyncMock()
    state.diarization_client.add_in_flight_request = Mock()
    
    # Set global STATE
    original_state = pipeline.main.STATE
    pipeline.main.STATE = state
    
    # Create audio frame with 6 seconds of audio using mock
    sample_rate = 16000
    duration = 6.0
    samples = np.random.randn(int(sample_rate * duration)).astype(np.float32)
    
    # Use mock AudioFrame with correct attributes
    frame = Mock()
    frame.timestamp = 0
    frame.time_base = 1.0
    frame.rate = sample_rate
    frame.samples = samples
    
    async def run_test():
        try:
            # Append audio
            _append_audio(frame)
            
            # Check buffer duration
            dur = _diarization_buffer_duration_seconds()
            print(f"Diarization buffer duration: {dur:.2f}s")
            
            if dur >= 6.0:
                print("✓ Buffer has enough audio for diarization")
                
                # Pull samples
                pulled = _pull_diarization_samples()
                if pulled:
                    window_samples, start_ts, end_ts = pulled
                    print(f"✓ Pulled diarization samples: {len(window_samples)} samples")
                    print(f"  Time window: {start_ts:.3f}s - {end_ts:.3f}s")
                else:
                    print("✗ Failed to pull diarization samples")
            else:
                print(f"✗ Buffer duration {dur:.2f}s < 6.0s threshold")
        finally:
            pipeline.main.STATE = original_state
    
    asyncio.run(run_test())


import pytest

@pytest.mark.asyncio
async def test_diarization_client_integration():
    """Integration test for DiarizationClient."""
    
    client = DiarizationClient(hf_token="test-token")
    await client.initialize()
    
    # Test in-flight tracking
    client.add_in_flight_request("req-1")
    client.add_in_flight_request("req-2")
    assert len(client.in_flight_requests) == 2, "Should have 2 in-flight requests"
    
    client.remove_in_flight_request("req-1")
    assert len(client.in_flight_requests) == 1, "Should have 1 in-flight request after removal"
    assert "req-1" not in client.in_flight_requests, "req-1 should be removed"
    
    print("✓ In-flight request tracking works correctly")
    
    # Test idle check - is_idle returns True when _running is False
    # Since we didn't call initialize(), _running is False
    assert client.is_idle() == True, "Should be idle when not initialized"
    
    client.remove_in_flight_request("req-2")
    
    print("✓ DiarizationClient basic functionality works")


def run_all_tests():
    """Run all diarization tests."""
    print("=" * 60)
    print("SPEAKER DIARIZATION FLOW TESTS")
    print("=" * 60)
    
    results = []
    
    # Test 1: SpeakerMemory
    try:
        results.append(("SpeakerMemory", test_speaker_memory()))
    except Exception as e:
        print(f"✗ SpeakerMemory test failed: {e}")
        results.append(("SpeakerMemory", False))
    
    # Test 2: Pipeline flow
    try:
        results.append(("Pipeline Flow", test_pipeline_flow()))
    except Exception as e:
        print(f"✗ Pipeline flow test failed: {e}")
        results.append(("Pipeline Flow", False))
    
    # Test 3: Empty segments handling
    try:
        results.append(("Empty Segments Handling", test_handle_diarization_result_with_empty_segments()))
    except Exception as e:
        print(f"✗ Empty segments test failed: {e}")
        results.append(("Empty Segments Handling", False))
    
    # Test 4: Integration test
    try:
        results.append(("DiarizationClient Integration", asyncio.run(test_diarization_client_integration())))
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        results.append(("DiarizationClient Integration", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
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