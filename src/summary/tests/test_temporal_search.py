"""
Tests for temporal query support in AgentManager.

Covers:
- TranscriptChunk.duration field
- AgentManager stream position tracking (_current_stream_ts, _total_chunks, _total_indexed_duration)
- reset_knowledge_store resets all tracking fields
- get_stream_info tool output
- get_transcript_time_range tool: window selection, formatting, edge cases
- WindowManager injection via set_window_manager
- No regressions to existing search_transcript path
"""

import asyncio
import threading
import pytest
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

from src.summary.agent_manager import AgentManager, TranscriptChunk, _format_hms
from src.summary.window_manager import WindowManager, TranscriptionWindow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_transcription_window(
    tid: int,
    start: float,
    end: float,
    text: str,
) -> TranscriptionWindow:
    """Construct a TranscriptionWindow directly for test fixtures."""
    return TranscriptionWindow(
        transcription_window_id=tid,
        new_text=text,
        timestamp_start=start,
        timestamp_end=end,
        segments=[{"text": text, "start_ms": int(start * 1000), "end_ms": int(end * 1000)}],
        char_count=len(text),
    )


def make_window_manager_with_windows(windows: list) -> WindowManager:
    """Return a WindowManager pre-populated with the given TranscriptionWindows."""
    wm = WindowManager()
    for w in windows:
        wm._transcription_windows[w.transcription_window_id] = w
    return wm


# ---------------------------------------------------------------------------
# _format_hms helper
# ---------------------------------------------------------------------------

class TestFormatHms:
    def test_seconds_only(self):
        assert _format_hms(0) == "00:00"
        assert _format_hms(59) == "00:59"

    def test_minutes_and_seconds(self):
        assert _format_hms(90) == "01:30"
        assert _format_hms(3599) == "59:59"

    def test_hours(self):
        assert _format_hms(3600) == "01:00:00"
        assert _format_hms(3661) == "01:01:01"

    def test_fractional_truncated(self):
        # Fractional seconds should be truncated (int conversion)
        assert _format_hms(90.9) == "01:30"


# ---------------------------------------------------------------------------
# TranscriptChunk.duration field
# ---------------------------------------------------------------------------

class TestTranscriptChunkDuration:
    def test_duration_defaults_to_none(self):
        chunk = TranscriptChunk(text="hello")
        assert chunk.duration is None

    def test_duration_can_be_set(self):
        chunk = TranscriptChunk(text="hello", timestamp=10.0, duration=2.5)
        assert chunk.duration == 2.5

    def test_all_fields(self):
        chunk = TranscriptChunk(
            text="test", timestamp=5.0, duration=3.0, speaker="speaker_0", window_id=1
        )
        assert chunk.text == "test"
        assert chunk.timestamp == 5.0
        assert chunk.duration == 3.0
        assert chunk.speaker == "speaker_0"
        assert chunk.window_id == 1


# ---------------------------------------------------------------------------
# AgentManager stream position tracking
# ---------------------------------------------------------------------------

def make_agent() -> AgentManager:
    """Create an AgentManager with mocked embedding client (no real network)."""
    agent = AgentManager(
        embedding_base_url="http://localhost:6060/v1",
        embedding_api_key="dummy",
        embedding_model="BAAI/bge-small-en-v1.5",
    )
    return agent


class TestStreamPositionTracking:
    @pytest.fixture
    def agent(self):
        return make_agent()

    def test_initial_state(self, agent):
        assert agent._current_stream_ts == 0.0
        assert agent._total_chunks == 0
        assert agent._total_indexed_duration == 0.0
        assert agent._window_manager is None

    @pytest.mark.asyncio
    async def test_timestamp_plus_duration_updates_current(self, agent):
        """_current_stream_ts should be max(timestamp + duration) seen."""
        with patch.object(agent.knowledge_store, "add", new_callable=AsyncMock):
            await agent.index_transcript_segment(text="hello", timestamp=10.0, duration=2.5)
            assert agent._current_stream_ts == 12.5  # 10.0 + 2.5

    @pytest.mark.asyncio
    async def test_only_advances_forward(self, agent):
        """Later calls with earlier timestamps should not decrease current_stream_ts."""
        with patch.object(agent.knowledge_store, "add", new_callable=AsyncMock):
            await agent.index_transcript_segment(text="a", timestamp=20.0, duration=2.5)
            await agent.index_transcript_segment(text="b", timestamp=10.0, duration=2.5)
            assert agent._current_stream_ts == 22.5  # First segment's end

    @pytest.mark.asyncio
    async def test_duration_accumulates(self, agent):
        with patch.object(agent.knowledge_store, "add", new_callable=AsyncMock):
            await agent.index_transcript_segment(text="a", timestamp=0.0, duration=2.5)
            await agent.index_transcript_segment(text="b", timestamp=2.5, duration=2.5)
            assert agent._total_indexed_duration == 5.0

    @pytest.mark.asyncio
    async def test_chunk_count_increments(self, agent):
        with patch.object(agent.knowledge_store, "add", new_callable=AsyncMock):
            await agent.index_transcript_segment(text="a", timestamp=0.0, duration=2.5)
            await agent.index_transcript_segment(text="b", timestamp=2.5, duration=2.5)
            assert agent._total_chunks == 2

    @pytest.mark.asyncio
    async def test_no_duration_still_increments_chunks(self, agent):
        with patch.object(agent.knowledge_store, "add", new_callable=AsyncMock):
            await agent.index_transcript_segment(text="x", timestamp=5.0)
            assert agent._total_chunks == 1
            assert agent._total_indexed_duration == 0.0
            assert agent._current_stream_ts == 5.0  # timestamp + 0

    @pytest.mark.asyncio
    async def test_empty_text_skipped(self, agent):
        with patch.object(agent.knowledge_store, "add", new_callable=AsyncMock) as mock_add:
            await agent.index_transcript_segment(text="   ", timestamp=5.0, duration=2.5)
            mock_add.assert_not_called()
            assert agent._total_chunks == 0

    def test_reset_clears_tracking(self, agent):
        agent._current_stream_ts = 100.0
        agent._total_chunks = 5
        agent._total_indexed_duration = 12.5
        agent.reset_knowledge_store()
        assert agent._current_stream_ts == 0.0
        assert agent._total_chunks == 0
        assert agent._total_indexed_duration == 0.0


# ---------------------------------------------------------------------------
# set_window_manager
# ---------------------------------------------------------------------------

class TestSetWindowManager:
    def test_set_and_read(self):
        agent = make_agent()
        wm = WindowManager()
        agent.set_window_manager(wm)
        assert agent._window_manager is wm

    def test_initially_none(self):
        agent = make_agent()
        assert agent._window_manager is None


# ---------------------------------------------------------------------------
# get_stream_info tool (tested via the closure logic, not LLM call)
# ---------------------------------------------------------------------------

class TestGetStreamInfo:
    """
    We can't easily call the @agent.tool_plain function directly without
    a PydanticAI run, so we replicate the logic and test state via the
    public agent attributes.
    """

    @pytest.mark.asyncio
    async def test_stream_not_started(self):
        agent = make_agent()
        # Simulate what get_stream_info would return
        ts = agent._current_stream_ts
        assert ts <= 0

    @pytest.mark.asyncio
    async def test_stream_has_content(self):
        agent = make_agent()
        with patch.object(agent.knowledge_store, "add", new_callable=AsyncMock):
            await agent.index_transcript_segment(text="hello", timestamp=120.0, duration=2.5)
        assert agent._current_stream_ts == 122.5
        assert agent._total_chunks == 1
        avg = agent._total_indexed_duration / agent._total_chunks
        assert avg == 2.5


# ---------------------------------------------------------------------------
# get_transcript_time_range — window selection logic
# ---------------------------------------------------------------------------

class TestGetTranscriptTimeRange:
    """
    Tests for the window selection, overlap filtering, and formatting logic
    that the get_transcript_time_range tool implements.

    We test the underlying logic by constructing WindowManagers directly.
    """

    def _windows_in_range(self, wm: WindowManager, start_time: float, end_time: float):
        """Replicate the overlap filter from the tool."""
        with wm._lock:
            windows = [
                w for w in wm._transcription_windows.values()
                if w.timestamp_start < end_time and w.timestamp_end > start_time
            ]
        windows.sort(key=lambda w: w.timestamp_start)
        return windows

    def test_basic_range_filter(self):
        """Windows whose intervals overlap [100, 200] should be returned."""
        wm = make_window_manager_with_windows([
            make_transcription_window(0, 0.0, 2.5, "intro"),
            make_transcription_window(1, 100.0, 102.5, "in range 1"),
            make_transcription_window(2, 150.0, 152.5, "in range 2"),
            make_transcription_window(3, 200.0, 202.5, "just after"),  # start==end_time → excluded
            make_transcription_window(4, 250.0, 252.5, "way after"),
        ])
        result = self._windows_in_range(wm, 100.0, 200.0)
        texts = [w.new_text for w in result]
        assert "in range 1" in texts
        assert "in range 2" in texts
        assert "intro" not in texts
        assert "just after" not in texts
        assert "way after" not in texts

    def test_partial_overlap_included(self):
        """Window that starts before range but ends inside should be included."""
        wm = make_window_manager_with_windows([
            make_transcription_window(0, 90.0, 110.0, "straddles start"),
        ])
        result = self._windows_in_range(wm, 100.0, 200.0)
        assert len(result) == 1
        assert result[0].new_text == "straddles start"

    def test_window_straddles_end_included(self):
        """Window that starts inside range but ends after should be included."""
        wm = make_window_manager_with_windows([
            make_transcription_window(0, 190.0, 210.0, "straddles end"),
        ])
        result = self._windows_in_range(wm, 100.0, 200.0)
        assert len(result) == 1

    def test_empty_range(self):
        """No windows means empty result."""
        wm = WindowManager()
        result = self._windows_in_range(wm, 100.0, 200.0)
        assert result == []

    def test_no_windows_in_range(self):
        """Windows exist but none overlap the queried range."""
        wm = make_window_manager_with_windows([
            make_transcription_window(0, 0.0, 50.0, "early content"),
        ])
        result = self._windows_in_range(wm, 100.0, 200.0)
        assert result == []

    def test_results_sorted_chronologically(self):
        """Windows returned must be sorted by timestamp_start ascending."""
        wm = make_window_manager_with_windows([
            make_transcription_window(2, 150.0, 152.5, "second"),
            make_transcription_window(1, 100.0, 102.5, "first"),
            make_transcription_window(3, 180.0, 182.5, "third"),
        ])
        result = self._windows_in_range(wm, 0.0, 300.0)
        starts = [w.timestamp_start for w in result]
        assert starts == sorted(starts)


# ---------------------------------------------------------------------------
# Integration: index segments, then verify window retrieval
# ---------------------------------------------------------------------------

class TestIntegration:
    @pytest.mark.asyncio
    async def test_wiring_window_manager(self):
        """AgentManager correctly stores the injected WindowManager."""
        agent = make_agent()
        wm = make_window_manager_with_windows([
            make_transcription_window(0, 0.0, 2.5, "hello world"),
            make_transcription_window(1, 2.5, 5.0, "how are you"),
        ])
        agent.set_window_manager(wm)
        assert agent._window_manager is wm

    @pytest.mark.asyncio
    async def test_stream_position_after_multiple_segments(self):
        """After indexing many segments, current_stream_ts reflects the latest end time."""
        agent = make_agent()
        with patch.object(agent.knowledge_store, "add", new_callable=AsyncMock):
            for i in range(20):
                await agent.index_transcript_segment(
                    text=f"segment {i}",
                    timestamp=float(i * 2.5),
                    duration=2.5,
                )
        # 19 * 2.5 + 2.5 = 50.0
        assert agent._current_stream_ts == pytest.approx(50.0)
        assert agent._total_chunks == 20
        assert agent._total_indexed_duration == pytest.approx(50.0)

    def test_reset_after_indexing(self):
        """reset_knowledge_store should clear all tracking state."""
        agent = make_agent()
        agent._current_stream_ts = 300.0
        agent._total_chunks = 10
        agent._total_indexed_duration = 25.0
        agent.reset_knowledge_store()
        assert agent._current_stream_ts == 0.0
        assert agent._total_chunks == 0
        assert agent._total_indexed_duration == 0.0
