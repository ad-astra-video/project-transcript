from unittest.mock import AsyncMock, MagicMock

import pytest

from src.summary.insights_distillation import InsightsDistillationPlugin


class TestInsightsDistillationEventSync:
    @pytest.fixture
    def plugin(self):
        window = MagicMock()
        window.get_result = MagicMock(
            side_effect=lambda name: {
                "context_summary": {
                    "summary_text": '{"insights": [{"insight_text": "Context insight"}]}'
                },
                "transcript_summary": {
                    "sections": [{"heading": "Plan", "start_ms": 0, "end_ms": 1000, "content": "..."}],
                    "key_points": ["KP"],
                    "topics": ["Topic"],
                    "summary": "## Plan",
                },
            }.get(name)
        )

        window_manager = MagicMock()
        window_manager.get_window = MagicMock(return_value=window)
        window_manager.store_plugin_result = MagicMock(return_value=True)

        llm_manager = MagicMock()
        llm_manager.reasoning_llm_client = MagicMock()

        plugin = InsightsDistillationPlugin(
            window_manager=window_manager,
            llm_manager=llm_manager,
            result_callback=AsyncMock(),
            summary_client=MagicMock(_notify_plugins=AsyncMock()),
        )
        plugin._task.process = AsyncMock(
            return_value={
                "insights": ["Improve handoff process"],
                "detailed_insights": [],
                "input_tokens": 100,
                "output_tokens": 30,
            }
        )
        return plugin

    @pytest.mark.asyncio
    async def test_waits_for_both_events_same_window(self, plugin):
        await plugin.handle_context_summary_complete(summary_window_id=2)
        assert plugin._task.process.await_count == 0

        await plugin.handle_transcript_summary_complete(summary_window_id=1)
        assert plugin._task.process.await_count == 0

        await plugin.handle_transcript_summary_complete(summary_window_id=2)

        assert plugin._task.process.await_count == 1
        assert plugin.get_latest_insights() == ["Improve handoff process"]

    @pytest.mark.asyncio
    async def test_reverse_event_order_also_processes(self, plugin):
        await plugin.handle_transcript_summary_complete(summary_window_id=4)
        assert plugin._task.process.await_count == 0

        await plugin.handle_context_summary_complete(summary_window_id=4)
        assert plugin._task.process.await_count == 1
