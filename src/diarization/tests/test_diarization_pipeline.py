"""
Integration test for speaker diarization pipeline flow.

This test validates the complete path from receiving diarization results
to sending them to the client via PROCESSOR.send_data().
"""

import asyncio
import numpy as np
import tempfile
import wave
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
import sys
import pytest

src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from diarization.diarization_client import (
    DiarizationClient, DiarizationResult, SpeakerSegment,
    SpeakerMemory
)
from pipeline.main import (
    _handle_diarization_result,
    _send_speakers_message,
    _build_segments_payload,
    TranscriberState
)


class MockAudioFrame:
    """Mock AudioFrame for testing."""
    def __init__(self, timestamp, time_base, rate, samples):
        self.timestamp = timestamp
        self.time_base = time_base
        self.rate = rate
        self.samples = samples


def create_test_audio(duration_seconds: float = 6.0, sample_rate: int = 16000) -> str:
    """Create a test WAV file with synthetic audio."""
    path = tempfile.mktemp(suffix=".wav")
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), dtype=np.float32)
    audio = np.sin(2 * np.pi * 440 * t) * 0.5
    pcm16 = (audio * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return path


class TestDiarizationPipeline:
    """Test class for complete diarization pipeline flow."""
    
    @pytest.fixture
    def mock_processor(self):
        """Create a mock PROCESSOR."""
        processor = Mock()
        processor.send_data = AsyncMock()
        return processor
    
    @pytest.fixture
    def mock_state(self, mock_processor):
        """Create a mock STATE with required attributes."""
        state = TranscriberState()
        state.buffer_rate = 16000
        state.buffer_start_ts = 0.0
        state.diarization_buffer_start_ts = 0.0
        state.diarization_audio_buffer = np.zeros((0,), dtype=np.float32)
        state.pending_temp_files = {}
        state.diarization_window_timestamps = {}
        state.diarization_client = Mock()
        state.diarization_client.remove_in_flight_request = Mock()
        return state
    
    def test_handle_diarization_result_with_segments(self, mock_processor, mock_state):
        """Test that _handle_diarization_result sends speakers message when segments exist."""
        import pipeline.main
        
        # Setup
        original_processor = pipeline.main.PROCESSOR
        original_state = pipeline.main.STATE
        
        pipeline.main.PROCESSOR = mock_processor
        pipeline.main.STATE = mock_state
        
        # Create test result with segments
        segments = [
            SpeakerSegment(start=0.0, end=1.5, speaker="speaker_0", confidence=0.85, alt_speakers={"speaker_1": 0.45}),
            SpeakerSegment(start=1.5, end=3.0, speaker="speaker_1", confidence=0.82, alt_speakers=None),
        ]
        
        result = DiarizationResult(
            request_id="test-request-123",
            audio_path="test_audio.wav",
            segments=segments,
            error=None
        )
        
        try:
            # Execute
            async def run_test():
                await _handle_diarization_result(result)
            
            asyncio.run(run_test())
            
            # Verify
            assert mock_processor.send_data.called, "send_data should have been called"
            
            # Check the sent payload
            sent_payload = mock_processor.send_data.call_args[0][0]
            assert isinstance(sent_payload, str), "Payload should be a JSON string"
            
            import json
            payload = json.loads(sent_payload)
            
            assert payload["type"] == "speakers", f"Expected type 'speakers', got {payload['type']}"
            assert len(payload["segments"]) == 2, f"Expected 2 segments, got {len(payload['segments'])}"
            
            # Verify segment structure
            seg1 = payload["segments"][0]
            assert "start_ms" in seg1, "Segment should have start_ms"
            assert "end_ms" in seg1, "Segment should have end_ms"
            assert "speaker" in seg1, "Segment should have speaker"
            assert "confidence" in seg1, "Segment should have confidence"
            
            # Verify timing is adjusted correctly (window_start_ts + segment.start)
            assert seg1["start_ms"] == int((0.0 + 0.0) * 1000), "Start ms should be 0"
            assert seg1["end_ms"] == int((0.0 + 1.5) * 1000), "End ms should be 1500"
            
            # Verify diarization client was notified
            mock_state.diarization_client.remove_in_flight_request.assert_called_once_with("test-request-123")
            
            print("✓ _handle_diarization_result correctly sends speakers message with segments")
            
        finally:
            pipeline.main.PROCESSOR = original_processor
            pipeline.main.STATE = original_state
    
    def test_handle_diarization_result_with_empty_segments(self, mock_processor, mock_state):
        """Test that _handle_diarization_result does NOT send when segments are empty."""
        import pipeline.main
        
        original_processor = pipeline.main.PROCESSOR
        original_state = pipeline.main.STATE
        
        pipeline.main.PROCESSOR = mock_processor
        pipeline.main.STATE = mock_state
        
        try:
            # Create result with empty segments
            result = DiarizationResult(
                request_id="test-empty",
                audio_path="test.wav",
                segments=[],  # Empty!
                error=None
            )
            
            async def run_test():
                await _handle_diarization_result(result)
            
            asyncio.run(run_test())
            
            # Verify send_data was NOT called
            assert not mock_processor.send_data.called, "send_data should NOT be called with empty segments"
            
            print("✓ _handle_diarization_result correctly skips sending when segments are empty")
            
        finally:
            pipeline.main.PROCESSOR = original_processor
            pipeline.main.STATE = original_state
    
    def test_handle_diarization_result_with_error(self, mock_processor, mock_state):
        """Test that _handle_diarization_result handles errors correctly."""
        import pipeline.main
        
        original_processor = pipeline.main.PROCESSOR
        original_state = pipeline.main.STATE
        
        pipeline.main.PROCESSOR = mock_processor
        pipeline.main.STATE = mock_state
        
        try:
            # Create result with error
            result = DiarizationResult(
                request_id="test-error",
                audio_path="test.wav",
                segments=[],
                error="Diarization pipeline failed: model not loaded"
            )
            
            async def run_test():
                await _handle_diarization_result(result)
            
            asyncio.run(run_test())
            
            # Verify send_data was NOT called
            assert not mock_processor.send_data.called, "send_data should NOT be called with error"
            
            print("✓ _handle_diarization_result correctly handles errors")
            
        finally:
            pipeline.main.PROCESSOR = original_processor
            pipeline.main.STATE = original_state
    
    def test_send_speakers_message_structure(self, mock_processor):
        """Test that _send_speakers_message creates correct payload structure."""
        import pipeline.main
        
        original_processor = pipeline.main.PROCESSOR
        pipeline.main.PROCESSOR = mock_processor
        
        try:
            segments = [
                SpeakerSegment(start=0.5, end=2.0, speaker="speaker_A", confidence=0.92, alt_speakers=None),
                SpeakerSegment(start=2.0, end=4.5, speaker="speaker_B", confidence=0.88, alt_speakers={"speaker_A": 0.35}),
            ]
            
            async def run_test():
                await _send_speakers_message(segments, 10.0, 16.0)  # 6-second window starting at 10s
            
            asyncio.run(run_test())
            
            # Verify structure
            assert mock_processor.send_data.called
            
            import json
            payload = json.loads(mock_processor.send_data.call_args[0][0])
            
            # Check required fields
            assert payload["type"] == "speakers"
            assert "timestamp_utc" in payload
            assert "timing" in payload
            assert "segments" in payload
            
            # Check timing
            assert payload["timing"]["media_window_start_ms"] == 10000  # 10s * 1000
            assert payload["timing"]["media_window_end_ms"] == 16000    # 16s * 1000
            
            # Check segments
            assert len(payload["segments"]) == 2
            
            # First segment should be adjusted by window start (10s)
            assert payload["segments"][0]["start_ms"] == int((10.0 + 0.5) * 1000)  # 10500
            assert payload["segments"][0]["end_ms"] == int((10.0 + 2.0) * 1000)     # 12000
            assert payload["segments"][0]["speaker"] == "speaker_A"
            
            print("✓ _send_speakers_message creates correct payload structure")
            
        finally:
            pipeline.main.PROCESSOR = original_processor
    
    def test_build_segments_payload(self):
        """Test that _build_segments_payload creates correct segment structure."""
        from pipeline.main import _build_segments_payload
        from transcription.whisper_client import TranscriptionSegment, WordTimestamp
        
        # Create mock transcription segments
        segments = [
            TranscriptionSegment(
                id="seg-001",
                start=0.5,
                end=2.0,
                text="Hello world",
                speaker=None,
                words=[
                    WordTimestamp(start=0.5, end=1.0, text="Hello"),
                    WordTimestamp(start=1.0, end=2.0, text="world")
                ]
            ),
            TranscriptionSegment(
                id="seg-002",
                start=2.0,
                end=4.0,
                text="This is a test",
                speaker="speaker_0",
                words=[
                    WordTimestamp(start=2.0, end=2.5, text="This"),
                    WordTimestamp(start=2.5, end=3.0, text="is"),
                    WordTimestamp(start=3.0, end=4.0, text="a test")
                ]
            )
        ]
        
        window_start_ts = 10.0
        
        result = _build_segments_payload(segments, window_start_ts)
        
        assert len(result) == 2
        
        # First segment (no speaker)
        assert result[0]["id"] == f"{int((10.0 + 0.5) * 1000)}-{hash('Hello world') & 0xFFFFFFFF:08x}"
        assert result[0]["start_ms"] == int((10.0 + 0.5) * 1000)
        assert result[0]["end_ms"] == int((10.0 + 2.0) * 1000)
        assert result[0]["text"] == "Hello world"
        assert "speaker" not in result[0]  # No speaker
        assert len(result[0]["words"]) == 2
        
        # Second segment (with speaker)
        assert result[1]["speaker"] == "speaker_0"
        assert result[1]["text"] == "This is a test"
        assert len(result[1]["words"]) == 3
        
        print("✓ _build_segments_payload creates correct structure")
    
    def test_pipeline_integration_flow(self, mock_processor):
        """Test complete pipeline flow from result to sending."""
        import pipeline.main
        
        original_processor = pipeline.main.PROCESSOR
        pipeline.main.PROCESSOR = mock_processor
        
        try:
            # Create state
            state = TranscriberState()
            state.buffer_rate = 16000
            state.buffer_start_ts = 0.0
            state.diarization_buffer_start_ts = 0.0
            state.diarization_audio_buffer = np.zeros((0,), dtype=np.float32)
            state.pending_temp_files = {}
            state.diarization_window_timestamps = {}
            state.diarization_client = Mock()
            state.diarization_client.remove_in_flight_request = Mock()
            
            pipeline.main.STATE = state
            
            # Create realistic diarization result
            segments = [
                SpeakerSegment(start=0.0, end=1.2, speaker="speaker_0", confidence=0.88, alt_speakers=None),
                SpeakerSegment(start=1.2, end=2.5, speaker="speaker_1", confidence=0.85, alt_speakers=None),
                SpeakerSegment(start=2.5, end=4.0, speaker="speaker_0", confidence=0.90, alt_speakers=None),
                SpeakerSegment(start=4.0, end=5.5, speaker="speaker_1", confidence=0.87, alt_speakers=None),
            ]
            
            result = DiarizationResult(
                request_id="integration-test-001",
                audio_path="integration_test.wav",
                segments=segments,
                error=None
            )
            
            async def run_full_flow():
                # Step 1: Handle the diarization result
                await _handle_diarization_result(result)
                
                # Step 2: Verify send_data was called
                assert mock_processor.send_data.called, "Full pipeline should send data"
                
                # Step 3: Parse and validate the payload
                import json
                payload = json.loads(mock_processor.send_data.call_args[0][0])
                
                # Validate complete structure
                assert payload["type"] == "speakers"
                assert len(payload["segments"]) == 4
                
                # Verify all speakers are identified
                speakers = {s["speaker"] for s in payload["segments"]}
                assert speakers == {"speaker_0", "speaker_1"}
                
                # Verify timing is stream-relative
                for seg in payload["segments"]:
                    assert seg["start_ms"] >= 0
                    assert seg["end_ms"] > seg["start_ms"]
                    assert "confidence" in seg
                
                return True
            
            result = asyncio.run(run_full_flow())
            assert result
            
            print("✓ Complete pipeline integration test passed")
            
        finally:
            pipeline.main.PROCESSOR = original_processor
            pipeline.main.STATE = None


def run_tests():
    """Run all pipeline tests."""
    import pytest
    import sys
    
    print("=" * 70)
    print("DIARIZATION PIPELINE INTEGRATION TESTS")
    print("=" * 70)
    
    # Run pytest programmatically
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x"
    ])
    
    return exit_code == 0


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    success = run_tests()
    sys.exit(0 if success else 1)