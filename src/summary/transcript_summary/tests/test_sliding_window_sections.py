import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.summary.transcript_summary import TranscriptSummaryPlugin
from src.summary.transcript_summary.task import TranscriptSummaryTask


class TestTranscriptSummaryTaskSlidingWindow:
    def test_build_user_content_filters_old_sections(self):
        task = TranscriptSummaryTask(llm_client=MagicMock())

        segments = [
            {
                "fast_summary_items": ["recent point"],
                "window_start_ms": 1_140_000,
                "window_end_ms": 1_200_000,
                "window_text": "x" * 300 + "TAIL",
            }
        ]

        current_summary = {
            "sections": [
                {
                    "heading": "Old Topic",
                    "start_ms": 0,
                    "end_ms": 300_000,
                    "content": "old content",
                },
                {
                    "heading": "Recent Topic",
                    "start_ms": 900_000,
                    "end_ms": 1_050_000,
                    "content": "recent content",
                },
            ],
            "key_points": ["kp1"],
            "topics": ["topic1"],
        }

        # 10-minute window from latest segment end (1,200,000 - 600,000 = 600,000)
        prompt = task._build_user_content(
            segments=segments,
            current_summary=current_summary,
            context_window_ms=600_000,
            max_segment_text_chars=100,
        )

        assert "Recent Topic" in prompt
        assert "Old Topic" not in prompt

        # Ensure segment transcription text is truncated to the tail
        assert "TAIL" in prompt
        assert ("x" * 200) not in prompt

    def test_build_user_content_pins_most_recent_section_when_all_below_cutoff(self):
        """Most-recent section must appear even when its end_ms is below the cutoff."""
        task = TranscriptSummaryTask(llm_client=MagicMock())

        current_summary = {
            "sections": [
                {"heading": "Early Topic", "start_ms": 0, "end_ms": 60_000, "content": "first"},
                {"heading": "Mid Topic", "start_ms": 60_000, "end_ms": 200_000, "content": "most recent"},
            ],
            "key_points": [],
            "topics": [],
        }

        # New segment at 20 min; 10-min window → cutoff = 1_200_000 - 600_000 = 600_000 ms.
        # Both existing sections end before 600_000, so both would normally be excluded.
        segments = [
            {"fast_summary_items": [], "window_start_ms": 900_000, "window_end_ms": 1_200_000, "window_text": ""}
        ]

        prompt = task._build_user_content(
            segments=segments,
            current_summary=current_summary,
            context_window_ms=600_000,
            max_segment_text_chars=4000,
        )

        # Most-recent section ("Mid Topic", end_ms=200_000) must be pinned in
        assert "Mid Topic" in prompt
        # Older section should remain excluded
        assert "Early Topic" not in prompt

    def test_build_user_content_includes_raw_ms_in_section_blocks(self):
        """Windowed section headings must embed start_ms/end_ms integers."""
        task = TranscriptSummaryTask(llm_client=MagicMock())

        current_summary = {
            "sections": [
                {"heading": "Budget", "start_ms": 480_000, "end_ms": 720_000, "content": "Approved."},
            ],
            "key_points": [],
            "topics": [],
        }
        segments = [
            {"fast_summary_items": [], "window_start_ms": 720_000, "window_end_ms": 780_000, "window_text": ""}
        ]

        prompt = task._build_user_content(
            segments=segments,
            current_summary=current_summary,
            context_window_ms=600_000,
            max_segment_text_chars=4000,
        )

        # Raw ms values must appear so the LLM can copy them directly
        assert "start_ms: 480000" in prompt
        assert "end_ms: 720000" in prompt

    def test_build_user_content_includes_raw_ms_in_segment_blocks(self):
        """New segment headings must embed window_start_ms/window_end_ms integers."""
        task = TranscriptSummaryTask(llm_client=MagicMock())

        segments = [
            {"fast_summary_items": ["point"], "window_start_ms": 720_000, "window_end_ms": 840_000, "window_text": "text"}
        ]

        prompt = task._build_user_content(
            segments=segments,
            current_summary={"sections": [], "key_points": [], "topics": []},
            context_window_ms=600_000,
            max_segment_text_chars=4000,
        )

        assert "window_start_ms: 720000" in prompt
        assert "window_end_ms: 840000" in prompt

    @pytest.mark.asyncio
    async def test_process_returns_sections_schema(self):
        mock_llm = MagicMock()
        response_payload = {
            "sections": [
                {
                    "heading": "Decisions",
                    "start_ms": 1000,
                    "end_ms": 5000,
                    "content": "Budget approved.",
                }
            ],
            "key_points": ["Budget approved"],
            "topics": ["Budget"],
        }

        mock_llm.create_completion = AsyncMock(
            return_value=(
                "",
                json.dumps(response_payload),
                123,
                45,
                0,
            )
        )

        task = TranscriptSummaryTask(llm_client=mock_llm)
        result = await task.process(
            segments=[
                {
                    "fast_summary_items": ["Budget approved"],
                    "window_start_ms": 0,
                    "window_end_ms": 10_000,
                    "window_text": "Team approved budget.",
                }
            ],
            current_summary={"sections": [], "key_points": [], "topics": []},
            context_window_ms=600_000,
            max_segment_text_chars=4000,
        )

        assert "sections" in result
        assert result["sections"][0]["heading"] == "Decisions"
        assert result["sections"][0]["start_ms"] == 1000
        assert result["key_points"] == ["Budget approved"]
        assert result["topics"] == ["Budget"]

    @pytest.mark.asyncio
    async def test_process_discards_degenerate_sections(self):
        """Sections with placeholder headings and empty content must be dropped."""
        from src.summary.transcript_summary.task import TranscriptSummaryTask
        mock_llm = MagicMock()
        response_payload = {
            "sections": [
                # degenerate — no content
                {"heading": "Untitled Topic", "start_ms": 0, "end_ms": 5000, "content": ""},
                # degenerate — heading is empty string
                {"heading": "", "start_ms": 0, "end_ms": 5000, "content": ""},
                # valid — untitled but has real content (keep it)
                {"heading": "Untitled Topic", "start_ms": 5000, "end_ms": 10000, "content": "Some real content here."},
                # valid
                {"heading": "Budget", "start_ms": 10000, "end_ms": 20000, "content": "Approved."},
            ],
            "key_points": [],
            "topics": [],
        }
        mock_llm.create_completion = AsyncMock(return_value=("", json.dumps(response_payload), 100, 50, 0))
        task = TranscriptSummaryTask(llm_client=mock_llm)
        result = await task.process(
            segments=[{"fast_summary_items": [], "window_start_ms": 0, "window_end_ms": 10_000, "window_text": ""}],
            current_summary={"sections": [], "key_points": [], "topics": []},
            context_window_ms=600_000,
            max_segment_text_chars=4000,
        )
        headings = [s["heading"] for s in result["sections"]]
        # Empty-content placeholder sections dropped; content-bearing section kept
        assert "Budget" in headings
        assert sum(1 for h in headings if h == "Untitled Topic") == 1
        # Two fully empty sections discarded (heading="Untitled Topic" + heading="")
        assert len(result["sections"]) == 2

    def test_section_model_clamps_negative_ms(self):
        """Negative start_ms / end_ms from the LLM must be clamped to 0."""
        from src.summary.transcript_summary.task import TranscriptSummarySection
        section = TranscriptSummarySection(
            heading="Test",
            start_ms=-6_200_000,
            end_ms=-5_000_000,
            content="content",
        )
        assert section.start_ms == 0
        assert section.end_ms == 0

    def test_section_model_fixes_inverted_ms(self):
        """end_ms < start_ms must be corrected to end_ms == start_ms."""
        from src.summary.transcript_summary.task import TranscriptSummarySection
        section = TranscriptSummarySection(
            heading="Test",
            start_ms=5000,
            end_ms=1000,
            content="content",
        )
        assert section.end_ms == section.start_ms == 5000


class TestTranscriptSummaryPluginSections:
    @pytest.fixture
    def plugin(self):
        window_manager = MagicMock()
        window_manager.get_window_text = MagicMock(return_value="Segment transcription text")

        llm_manager = MagicMock()
        llm_manager.reasoning_client = MagicMock()
        llm_manager.reasoning_llm_client = MagicMock()

        result_callback = AsyncMock()

        return TranscriptSummaryPlugin(
            window_manager=window_manager,
            llm_manager=llm_manager,
            result_callback=result_callback,
            summary_client=None,
        )

    def test_merge_sections_updates_existing_and_appends_new(self, plugin):
        existing = [
            {
                "heading": "Budget",
                "start_ms": 0,
                "end_ms": 120_000,
                "content": "Initial budget discussion.",
            }
        ]
        new_sections = [
            {
                "heading": "Budget",
                "start_ms": 0,
                "end_ms": 180_000,
                "content": "Budget discussion extended.",
            },
            {
                "heading": "Risks",
                "start_ms": 200_000,
                "end_ms": 260_000,
                "content": "Risk review started.",
            },
        ]

        merged = plugin._merge_sections(existing, new_sections)
        assert len(merged) == 2
        assert merged[0]["heading"] == "Budget"
        assert merged[0]["end_ms"] == 180_000
        assert merged[1]["heading"] == "Risks"

    def test_render_summary_markdown_includes_all_sections(self, plugin):
        sections = [
            {
                "heading": "Intro",
                "start_ms": 0,
                "end_ms": 30_000,
                "content": "Kickoff.",
            },
            {
                "heading": "Plan",
                "start_ms": 60_000,
                "end_ms": 120_000,
                "content": "Roadmap review.",
            },
        ]

        markdown = plugin._render_summary_markdown(sections)
        assert "## Intro [0:00:00 – 0:00:30]" in markdown
        assert "## Plan [0:01:00 – 0:02:00]" in markdown
        assert "Kickoff." in markdown
        assert "Roadmap review." in markdown

    def test_on_update_params_updates_window_and_text_limits(self, plugin):
        plugin.on_update_params(context_window_minutes=5, max_segment_text_chars=2500)
        assert plugin._context_window_minutes == 5
        assert plugin._context_window_ms == 300_000
        assert plugin._max_segment_text_chars == 2500

    @pytest.mark.asyncio
    async def test_process_uses_section_accumulation_and_markdown_output(self, plugin):
        plugin._task.process = AsyncMock(
            return_value={
                "sections": [
                    {
                        "heading": "Decisions",
                        "start_ms": 1000,
                        "end_ms": 60000,
                        "content": "Approved timeline.",
                    }
                ],
                "key_points": ["Timeline approved"],
                "topics": ["Planning"],
                "input_tokens": 50,
                "output_tokens": 25,
            }
        )

        await plugin.process(
            summary_window_id=1,
            fast_summary_items=["item1"],
            window_start_ms=0,
            window_end_ms=30_000,
        )
        await plugin.process(
            summary_window_id=2,
            fast_summary_items=["item2"],
            window_start_ms=30_000,
            window_end_ms=60_000,
        )

        plugin._task.process.assert_called_once()
        _, kwargs = plugin._task.process.call_args
        assert kwargs["context_window_ms"] == plugin._context_window_ms
        assert kwargs["max_segment_text_chars"] == plugin._max_segment_text_chars

        plugin._result_callback.assert_called_once()
        payload = plugin._result_callback.call_args[0][0]
        assert payload["summary"].startswith("## Decisions")
        assert payload["key_points"] == ["Timeline approved"]
        assert payload["topics"] == ["Planning"]
