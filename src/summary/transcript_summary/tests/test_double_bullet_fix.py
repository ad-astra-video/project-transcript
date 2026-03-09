"""
Test for the double bullet point fix in transcript_summary task.

Ensures that when the LLM returns items with leading bullet points,
they are stripped correctly and don't result in double bullet points
on the next cycle.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.summary.transcript_summary.task import TranscriptSummaryTask, TranscriptSummarySchema


class TestDoubleBulletPointFix:
    """Tests for the double bullet point issue fix."""

    @pytest.mark.asyncio
    async def test_strips_leading_bullets_from_key_points(self):
        """Verify that leading bullet points are stripped from key_points after parsing."""
        # Create a mock LLM client that returns key_points with leading bullets
        mock_llm_client = AsyncMock()
        task = TranscriptSummaryTask(llm_client=mock_llm_client)

        # Simulate LLM response with bullet points already in the items
        llm_response = {
            "sections": [
                {
                    "heading": "Budget Discussion",
                    "start_ms": 0,
                    "end_ms": 1000,
                    "content": "The team discussed Q3 budget allocation."
                }
            ],
            "key_points": [
                "- Budget approved at $2M",  # LLM included leading bullet
                "- Timeline set for Q3",      # LLM included leading bullet
                "Procurement review needed"   # No leading bullet
            ],
            "topics": [
                "- Budget Planning",  # LLM included leading bullet
                "Resource Allocation"  # No leading bullet
            ]
        }

        mock_llm_client.create_completion.return_value = (
            None,
            json.dumps(llm_response),
            100,  # input_tokens
            50,   # output_tokens
            None
        )

        result = await task.process(
            segments=[],
            current_summary=None,
            context_window_ms=600_000,
            max_segment_text_chars=500,
        )

        # Verify that items have had leading bullets stripped
        assert result["key_points"] == [
            "Budget approved at $2M",
            "Timeline set for Q3",
            "Procurement review needed"
        ]
        assert result["topics"] == [
            "Budget Planning",
            "Resource Allocation"
        ]

        # Verify no double bullets in key_points
        for kp in result["key_points"]:
            assert not kp.startswith("- "), f"Key point has double bullet: {kp}"

        # Verify no double bullets in topics
        for topic in result["topics"]:
            assert not topic.startswith("- "), f"Topic has double bullet: {topic}"

    @pytest.mark.asyncio
    async def test_no_double_bullets_on_next_cycle(self):
        """Verify that next cycle doesn't have double bullets even if this one did."""
        mock_llm_client = AsyncMock()
        task = TranscriptSummaryTask(llm_client=mock_llm_client)

        # First response with bullets
        first_response = {
            "sections": [],
            "key_points": ["- Point 1", "- Point 2"],
            "topics": ["- Topic A"]
        }

        # Second response with bullets
        second_response = {
            "sections": [],
            "key_points": ["- Point 1", "- Point 2", "- Point 3"],
            "topics": ["- Topic A", "- Topic B"]
        }

        mock_llm_client.create_completion.return_value = (
            None,
            json.dumps(first_response),
            100,
            50,
            None
        )

        first_result = await task.process(
            segments=[],
            current_summary=None,
            context_window_ms=600_000,
            max_segment_text_chars=500,
        )

        assert first_result["key_points"] == ["Point 1", "Point 2"]
        assert first_result["topics"] == ["Topic A"]

        # Now simulate the second call with the first result as current_summary
        mock_llm_client.create_completion.return_value = (
            None,
            json.dumps(second_response),
            100,
            50,
            None
        )

        second_result = await task.process(
            segments=[],
            current_summary=first_result,
            context_window_ms=600_000,
            max_segment_text_chars=500,
        )

        # Verify no double bullets in second result
        assert second_result["key_points"] == ["Point 1", "Point 2", "Point 3"]
        assert second_result["topics"] == ["Topic A", "Topic B"]

        for kp in second_result["key_points"]:
            assert not kp.startswith("- "), f"Double bullet in second cycle: {kp}"

    def test_build_user_content_no_double_bullets(self):
        """Verify that _build_user_content doesn't add extra bullets to existing items."""
        mock_llm_client = MagicMock()
        task = TranscriptSummaryTask(llm_client=mock_llm_client)

        current_summary = {
            "sections": [],
            "key_points": ["Budget approved", "Timeline set"],  # No leading bullets
            "topics": ["Budget", "Planning"]                     # No leading bullets
        }

        prompt = task._build_user_content(
            segments=[],
            current_summary=current_summary,
            context_window_ms=600_000,
            max_segment_text_chars=500,
        )

        # Verify that existing key_points and topics have exactly one bullet
        assert "- Budget approved\n" in prompt
        assert "- Timeline set\n" in prompt
        assert "- Budget\n" in prompt
        assert "- Planning\n" in prompt

        # Verify no double bullets
        assert "- - " not in prompt
