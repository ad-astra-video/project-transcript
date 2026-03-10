"""
Tests for the verification pass introduced in TranscriptSummaryTask.

Covers:
- Verification corrects a named entity in a changed section
- Verification is skipped when no sections changed (unchanged content)
- Verification is skipped when verification_llm_client is None
- Verification tokens are included in the process() return dict
- _get_changed_sections correctly identifies new and modified sections
- _build_verification_user_content includes the right transcript snippets
"""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.summary.transcript_summary.task import (
    TranscriptSummaryTask,
    VerificationSchema,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_client(response_dict: dict) -> MagicMock:
    """Return a mock LLMClient whose create_completion returns *response_dict*."""
    client = MagicMock()
    client.create_completion = AsyncMock(
        return_value=(
            None,                          # reasoning
            json.dumps(response_dict),     # content
            10,                            # input_tokens
            5,                             # output_tokens
            0,                             # reasoning_tokens
        )
    )
    return client


def _primary_response(sections=None, key_points=None, topics=None) -> dict:
    return {
        "sections": sections or [
            {
                "heading": "Project Kickoff",
                "start_ms": 0,
                "end_ms": 120_000,
                "content": "Alice presented the roadmap.",
            }
        ],
        "key_points": key_points or ["Alice owns the roadmap"],
        "topics": topics or ["project planning"],
    }


def _verification_response(sections=None, key_points=None, topics=None) -> dict:
    return {
        "sections": sections or [
            {
                "heading": "Project Kickoff",
                "start_ms": 0,
                "end_ms": 120_000,
                "content": "Bob presented the roadmap.",
            }
        ],
        "key_points": key_points or ["Bob owns the roadmap"],
        "topics": topics or ["project planning"],
    }


_SEGMENTS = [
    {
        "fast_summary_items": ["kickoff discussion"],
        "window_start_ms": 0,
        "window_end_ms": 120_000,
        "window_text": "Bob presented the roadmap to the team.",
    }
]


# ---------------------------------------------------------------------------
# _get_changed_sections
# ---------------------------------------------------------------------------

class TestGetChangedSections:
    def test_new_section_is_changed(self):
        prior = []
        new = [{"heading": "Topic A", "start_ms": 0, "end_ms": 60_000, "content": "intro"}]
        changed = TranscriptSummaryTask._get_changed_sections(prior, new)
        assert len(changed) == 1
        assert changed[0]["heading"] == "Topic A"

    def test_unchanged_section_is_excluded(self):
        section = {"heading": "Topic A", "start_ms": 0, "end_ms": 60_000, "content": "intro"}
        changed = TranscriptSummaryTask._get_changed_sections([section], [section])
        assert changed == []

    def test_modified_content_is_changed(self):
        prior = [{"heading": "Topic A", "start_ms": 0, "end_ms": 60_000, "content": "old text"}]
        new = [{"heading": "Topic A", "start_ms": 0, "end_ms": 60_000, "content": "new text"}]
        changed = TranscriptSummaryTask._get_changed_sections(prior, new)
        assert len(changed) == 1
        assert changed[0]["content"] == "new text"

    def test_section_outside_overlap_is_treated_as_new(self):
        prior = [{"heading": "Topic A", "start_ms": 0, "end_ms": 30_000, "content": "x"}]
        # Non-overlapping time range → no match → treated as new
        new = [{"heading": "Topic B", "start_ms": 60_000, "end_ms": 120_000, "content": "x"}]
        changed = TranscriptSummaryTask._get_changed_sections(prior, new)
        assert len(changed) == 1

    def test_mixed_sections(self):
        prior = [
            {"heading": "A", "start_ms": 0, "end_ms": 60_000, "content": "unchanged"},
            {"heading": "B", "start_ms": 60_000, "end_ms": 120_000, "content": "old"},
        ]
        new = [
            {"heading": "A", "start_ms": 0, "end_ms": 60_000, "content": "unchanged"},
            {"heading": "B", "start_ms": 60_000, "end_ms": 120_000, "content": "updated"},
            {"heading": "C", "start_ms": 120_000, "end_ms": 180_000, "content": "brand new"},
        ]
        changed = TranscriptSummaryTask._get_changed_sections(prior, new)
        headings = [s["heading"] for s in changed]
        assert "A" not in headings
        assert "B" in headings
        assert "C" in headings


# ---------------------------------------------------------------------------
# _build_verification_user_content
# ---------------------------------------------------------------------------

class TestBuildVerificationUserContent:
    def setup_method(self):
        self.task = TranscriptSummaryTask(llm_client=MagicMock())

    def test_includes_changed_section_heading(self):
        section = {"heading": "Q3 Planning", "start_ms": 0, "end_ms": 60_000, "content": "Alice reviewed goals."}
        content = self.task._build_verification_user_content(
            section=section,
            segments=_SEGMENTS,
            key_points=["Alice owns goals"],
            topics=["Q3 planning"],
        )
        assert "Q3 Planning" in content
        assert "Alice reviewed goals." in content

    def test_includes_overlapping_transcript_text(self):
        section = {"heading": "Kickoff", "start_ms": 0, "end_ms": 120_000, "content": "x"}
        content = self.task._build_verification_user_content(
            section=section,
            segments=_SEGMENTS,
            key_points=[],
            topics=[],
        )
        assert "Bob presented the roadmap" in content

    def test_excludes_non_overlapping_segment_text(self):
        section = {"heading": "Late topic", "start_ms": 600_000, "end_ms": 660_000, "content": "x"}
        content = self.task._build_verification_user_content(
            section=section,
            segments=_SEGMENTS,   # segments are at 0–120 s, no overlap
            key_points=[],
            topics=[],
        )
        assert "Bob presented the roadmap" not in content

    def test_includes_key_points_and_topics(self):
        content = self.task._build_verification_user_content(
            section={"heading": "X", "start_ms": 0, "end_ms": 10_000, "content": "y"},
            segments=[],
            key_points=["decision reached", "budget $1M"],
            topics=["finance", "strategy"],
        )
        assert "decision reached" in content
        assert "budget $1M" in content
        assert "finance" in content
        assert "strategy" in content


# ---------------------------------------------------------------------------
# _run_verification_pass
# ---------------------------------------------------------------------------

class TestRunVerificationPass:
    @pytest.mark.asyncio
    async def test_returns_none_when_client_is_none(self):
        task = TranscriptSummaryTask(llm_client=MagicMock(), verification_llm_client=None)
        result = await task._run_verification_pass(
            changed_sections=[{"heading": "X", "start_ms": 0, "end_ms": 10_000, "content": "y"}],
            segments=_SEGMENTS,
            key_points=[],
            topics=[],
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_changed_sections(self):
        v_client = _make_llm_client(_verification_response())
        task = TranscriptSummaryTask(
            llm_client=MagicMock(),
            verification_llm_client=v_client,
        )
        result = await task._run_verification_pass(
            changed_sections=[],
            segments=_SEGMENTS,
            key_points=[],
            topics=[],
        )
        assert result is None
        v_client.create_completion.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_corrected_content(self):
        v_response = _verification_response(
            sections=[
                {
                    "heading": "Project Kickoff",
                    "start_ms": 0,
                    "end_ms": 120_000,
                    "content": "Bob presented the roadmap.",
                }
            ],
            key_points=["Bob owns the roadmap"],
            topics=["project planning"],
        )
        v_client = _make_llm_client(v_response)
        task = TranscriptSummaryTask(
            llm_client=MagicMock(),
            verification_llm_client=v_client,
        )
        result = await task._run_verification_pass(
            changed_sections=[
                {"heading": "Project Kickoff", "start_ms": 0, "end_ms": 120_000, "content": "Alice presented the roadmap."}
            ],
            segments=_SEGMENTS,
            key_points=["Alice owns the roadmap"],
            topics=["project planning"],
        )
        assert result is not None
        v_sections, v_kp, v_topics, v_in, v_out = result
        assert v_sections[0]["content"] == "Bob presented the roadmap."
        assert v_kp == ["Bob owns the roadmap"]
        assert v_in == 10
        assert v_out == 5

    @pytest.mark.asyncio
    async def test_returns_none_on_api_failure(self):
        v_client = MagicMock()
        v_client.create_completion = AsyncMock(side_effect=RuntimeError("timeout"))
        task = TranscriptSummaryTask(
            llm_client=MagicMock(),
            verification_llm_client=v_client,
        )
        result = await task._run_verification_pass(
            changed_sections=[{"heading": "X", "start_ms": 0, "end_ms": 5_000, "content": "y"}],
            segments=_SEGMENTS,
            key_points=[],
            topics=[],
        )
        assert result is not None  # non-fatal — unchanged section should be returned
        v_sections, v_kp, v_topics, v_in, v_out = result
        assert v_sections[0]["content"] == "y"
        assert v_kp == []
        assert v_topics == []
        assert v_in == 0
        assert v_out == 0

    @pytest.mark.asyncio
    async def test_returns_none_on_invalid_json(self):
        v_client = MagicMock()
        v_client.create_completion = AsyncMock(
            return_value=(None, "not valid json", 1, 1, 0)
        )
        task = TranscriptSummaryTask(
            llm_client=MagicMock(),
            verification_llm_client=v_client,
        )
        result = await task._run_verification_pass(
            changed_sections=[{"heading": "X", "start_ms": 0, "end_ms": 5_000, "content": "y"}],
            segments=_SEGMENTS,
            key_points=[],
            topics=[],
        )
        assert result is not None
        v_sections, v_kp, v_topics, v_in, v_out = result
        assert v_sections[0]["content"] == "y"
        assert v_kp == []
        assert v_topics == []
        assert v_in == 1
        assert v_out == 1


# ---------------------------------------------------------------------------
# process() — end-to-end with verification wired in
# ---------------------------------------------------------------------------

class TestProcessWithVerification:
    @pytest.mark.asyncio
    async def test_entity_correction_applied_to_result(self):
        """Verification corrects a wrong name; the corrected content must be emitted."""
        primary_client = _make_llm_client(_primary_response())
        v_client = _make_llm_client(
            _verification_response(
                sections=[
                    {
                        "heading": "Project Kickoff",
                        "start_ms": 0,
                        "end_ms": 120_000,
                        "content": "Bob presented the roadmap.",
                    }
                ],
                key_points=["Bob owns the roadmap"],
                topics=["project planning"],
            )
        )

        task = TranscriptSummaryTask(
            llm_client=primary_client,
            verification_llm_client=v_client,
        )

        result = await task.process(
            segments=_SEGMENTS,
            current_summary=None,
            context_window_ms=600_000,
            max_segment_text_chars=4000,
        )

        sections = result["sections"]
        assert any("Bob" in s["content"] for s in sections), (
            "Expected verification to replace 'Alice' with 'Bob'"
        )
        assert result["key_points"] == ["Bob owns the roadmap"]

    @pytest.mark.asyncio
    async def test_verification_not_called_when_client_is_none(self):
        """When no verification client is set, only one LLM call is made."""
        primary_client = _make_llm_client(_primary_response())

        task = TranscriptSummaryTask(
            llm_client=primary_client,
            verification_llm_client=None,
        )

        result = await task.process(
            segments=_SEGMENTS,
            current_summary=None,
            context_window_ms=600_000,
            max_segment_text_chars=4000,
        )

        # Original (uncorrected) content from primary pass is emitted
        sections = result["sections"]
        assert any("Alice" in s["content"] for s in sections)
        assert result["verification_input_tokens"] == 0
        assert result["verification_output_tokens"] == 0

    @pytest.mark.asyncio
    async def test_verification_not_called_when_nothing_changed(self):
        """When all sections are unchanged, the verification client must not be called."""
        section = {
            "heading": "Project Kickoff",
            "start_ms": 0,
            "end_ms": 120_000,
            "content": "Alice presented the roadmap.",
        }
        primary_client = _make_llm_client(
            _primary_response(
                sections=[section],
                key_points=["Alice owns the roadmap"],
                topics=["project planning"],
            )
        )
        v_client = _make_llm_client(_verification_response())

        task = TranscriptSummaryTask(
            llm_client=primary_client,
            verification_llm_client=v_client,
        )

        # Feed the exact same section as prior_summary so nothing changed
        prior_summary = {
            "sections": [section],
            "key_points": ["Alice owns the roadmap"],
            "topics": ["project planning"],
        }

        await task.process(
            segments=_SEGMENTS,
            current_summary=prior_summary,
            context_window_ms=600_000,
            max_segment_text_chars=4000,
        )

        v_client.create_completion.assert_not_called()

    @pytest.mark.asyncio
    async def test_verification_tokens_in_result(self):
        """Verification token counts must appear in the process() return dict."""
        primary_client = _make_llm_client(_primary_response())
        v_client = _make_llm_client(_verification_response())

        task = TranscriptSummaryTask(
            llm_client=primary_client,
            verification_llm_client=v_client,
        )

        result = await task.process(
            segments=_SEGMENTS,
            current_summary=None,
            context_window_ms=600_000,
            max_segment_text_chars=4000,
        )

        assert "verification_input_tokens" in result
        assert "verification_output_tokens" in result
        # v_client mock returns (10, 5) tokens
        assert result["verification_input_tokens"] == 10
        assert result["verification_output_tokens"] == 5

    @pytest.mark.asyncio
    async def test_verification_failure_does_not_fail_process(self):
        """A failing verification pass must not raise — primary result is still returned."""
        primary_client = _make_llm_client(_primary_response())
        v_client = MagicMock()
        v_client.create_completion = AsyncMock(side_effect=RuntimeError("network error"))

        task = TranscriptSummaryTask(
            llm_client=primary_client,
            verification_llm_client=v_client,
        )

        result = await task.process(
            segments=_SEGMENTS,
            current_summary=None,
            context_window_ms=600_000,
            max_segment_text_chars=4000,
        )

        # Primary result still present; verification tokens are 0
        assert result["sections"]
        assert result["verification_input_tokens"] == 0
        assert result["verification_output_tokens"] == 0
