"""Tests for transcript_format plugin and chunked fast-LLM formatting."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.summary.transcript_format import TranscriptFormatPlugin
from src.summary.transcript_format.task import TranscriptFormatTask
from src.summary.window_manager import WindowManager


class TestTranscriptFormatTaskChunking:
    """Validate token-aware chunking for 16k fast model workflow."""

    def test_dedupe_transcript_removes_repeated_words_and_phrases(self):
        task = TranscriptFormatTask(llm_client=MagicMock())

        text = (
            "And this this is important because because "
            "they had to partner with they had to partner with international suppliers."
        )
        cleaned = task.dedupe_transcript(text)

        assert cleaned == "And this is important because they had to partner with international suppliers."

    def test_dedupe_preserves_timing_markers(self):
        task = TranscriptFormatTask(llm_client=MagicMock())

        text = "[00:10] because because this this works works [00:20] done done"
        cleaned = task._dedupe_preserving_timing_markers(text)

        assert cleaned == "[00:10] because this works [00:20] done"

    def test_dedupe_handles_punctuation_and_case_variants(self):
        task = TranscriptFormatTask(llm_client=MagicMock())

        text = (
            "But it... but it also understands what you're asking. "
            "what you're asking it for."
        )
        cleaned = task.dedupe_transcript(text)

        assert cleaned == "but it also understands what you're asking it for."

    def test_remove_boundary_overlap_handles_punctuation_variants(self):
        task = TranscriptFormatTask(llm_client=MagicMock())

        previous = "It's an amazing model."
        current = "It's an amazing model, and useful."

        fixed_prev, fixed_curr = task._remove_boundary_overlap(previous, current, max_overlap_words=10)

        assert fixed_curr == "and useful."
        assert fixed_prev == previous  # exact overlap — prev is unchanged

    @pytest.mark.asyncio
    async def test_format_text_applies_dedupe_preprocessing(self):
        task = TranscriptFormatTask(llm_client=MagicMock(), input_token_budget=12000)

        async def _fake_format_chunk(chunk_text, content_type, prior_tail, chunk_index, total_chunks):
            return {
                "text": chunk_text,
                "input_tokens": 10,
                "output_tokens": 5,
            }

        task._format_single_chunk = AsyncMock(side_effect=_fake_format_chunk)

        result = await task.format_text(
            raw_text="[00:10] because because we we test test",
            content_type="UNKNOWN",
            prior_formatted_tail="",
        )

        assert result["formatted_text"] == "[00:10] because we test"

    def test_split_into_chunks_for_long_input(self):
        task = TranscriptFormatTask(
            llm_client=MagicMock(),
            input_token_budget=300,  # max_chars floor still 2000
            chunk_overlap_chars=200,
        )

        sentence = "This is a sentence about the discussion and timeline. "
        long_text = sentence * 180  # > 2000 chars
        chunks = task.split_into_chunks(long_text)

        assert len(chunks) > 1
        assert all(chunk.strip() for chunk in chunks)
        # With overlap, chunks can exceed max_chars by overlap amount.
        assert all(len(chunk) <= 2205 for chunk in chunks)

    @pytest.mark.asyncio
    async def test_format_text_processes_multiple_chunks(self):
        task = TranscriptFormatTask(
            llm_client=MagicMock(),
            input_token_budget=300,
            chunk_overlap_chars=150,
        )

        sentence = "Detailed transcript sentence with context and continuity. "
        long_text = sentence * 180

        async def _fake_format_chunk(chunk_text, content_type, prior_tail, chunk_index, total_chunks):
            return {
                "heading": f"Heading {chunk_index}",
                "text": f"Formatted {chunk_index}",
                "input_tokens": 10,
                "output_tokens": 5,
            }

        task._format_single_chunk = AsyncMock(side_effect=_fake_format_chunk)

        result = await task.format_text(
            raw_text=long_text,
            content_type="NEWS_UPDATE",
            prior_formatted_tail="",
        )

        assert result["chunk_count"] > 1
        assert task._format_single_chunk.await_count == result["chunk_count"]
        assert result["formatted_text"].count("Formatted") == result["chunk_count"]


class TestTranscriptFormatPlugin:
    """Integration-ish behavior tests with window_manager timing and content-type transitions."""

    @pytest.mark.asyncio
    async def test_reformats_from_detected_window_forward(self):
        window_manager = WindowManager()
        window_0 = window_manager.add_summary_window(
            text="Window zero transcript text.",
            timestamp_start=0.0,
            timestamp_end=10.0,
            transcription_window_ids=[1],
        )
        window_1 = window_manager.add_summary_window(
            text="Window one transcript text.",
            timestamp_start=10.0,
            timestamp_end=20.0,
            transcription_window_ids=[2],
        )
        window_2 = window_manager.add_summary_window(
            text="Window two transcript text.",
            timestamp_start=20.0,
            timestamp_end=30.0,
            transcription_window_ids=[3],
        )

        llm_manager = MagicMock()
        llm_manager.rapid_llm_client = MagicMock()
        result_callback = AsyncMock()

        plugin = TranscriptFormatPlugin(
            window_manager=window_manager,
            llm_manager=llm_manager,
            result_callback=result_callback,
            summary_client=None,
        )

        async def _fake_format(raw_text, content_type, prior_formatted_tail=""):
            return {
                "heading": f"{content_type} heading",
                "formatted_text": f"[{content_type}] {raw_text}",
                "input_tokens": 20,
                "output_tokens": 10,
                "chunk_count": 1,
            }

        plugin._task.format_text = AsyncMock(side_effect=_fake_format)

        # Initial formatting via summary window events (defaults to UNKNOWN)
        await plugin.handle_summary_window(window_0)
        await plugin.handle_summary_window(window_1)
        await plugin.handle_summary_window(window_2)

        assert plugin._task.format_text.await_count == 3
        initial_calls = plugin._task.format_text.await_args_list
        assert initial_calls[0].kwargs["content_type"] == "UNKNOWN"
        assert initial_calls[1].kwargs["content_type"] == "UNKNOWN"
        assert initial_calls[2].kwargs["content_type"] == "UNKNOWN"

        # Content-type update at window_1 should reformat window_1 and window_2 only
        await plugin.handle_content_type_detected(
            content_type="NEWS_UPDATE",
            confidence=0.92,
            source="AUTO_DETECTED",
            reasoning="news-like narration",
            summary_window_id=window_1,
        )

        assert plugin._task.format_text.await_count == 5
        reformat_calls = plugin._task.format_text.await_args_list[3:]
        assert reformat_calls[0].kwargs["raw_text"] == "Window one transcript text."
        assert reformat_calls[0].kwargs["content_type"] == "NEWS_UPDATE"
        assert reformat_calls[1].kwargs["raw_text"] == "Window two transcript text."
        assert reformat_calls[1].kwargs["content_type"] == "NEWS_UPDATE"

        # Second type transition at window_2 should reformat only window_2
        await plugin.handle_content_type_detected(
            content_type="INTERVIEW",
            confidence=0.88,
            source="AUTO_DETECTED",
            reasoning="interview back-and-forth",
            summary_window_id=window_2,
        )

        assert plugin._task.format_text.await_count == 6
        last_call = plugin._task.format_text.await_args_list[-1]
        assert last_call.kwargs["raw_text"] == "Window two transcript text."
        assert last_call.kwargs["content_type"] == "INTERVIEW"

        # Final stored data should reflect mixed styles over time
        assert plugin._formatted_by_window[window_0]["content_type"] == "UNKNOWN"
        assert plugin._formatted_by_window[window_1]["content_type"] == "NEWS_UPDATE"
        assert plugin._formatted_by_window[window_2]["content_type"] == "INTERVIEW"

        # Plugin should emit payloads through callback
        assert result_callback.await_count >= 3
