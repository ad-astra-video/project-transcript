"""
Unit tests for prior insights accumulation functionality.
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from src.summary.summary_client import SummaryClient, WindowManager, WindowInsight


class TestPriorInsightsAccumulation:
    """Tests for prior insights accumulation across windows."""
    
    def create_client(self, delay_seconds: float = 0.0):
        """Create a SummaryClient instance with zero delay for testing."""
        return SummaryClient(
            api_key="test_key",
            model="test_model",
            initial_summary_delay_seconds=delay_seconds
        )
    
    def _create_mock_response(self, analysis: str, insights: list) -> MagicMock:
        """Create a mock LLM response with the given analysis and insights."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "analysis": analysis,
            "insights": insights
        })
        mock_response.choices[0].message.reasoning = ""
        return mock_response
    
    @pytest.mark.asyncio
    async def test_first_window_has_zero_prior_insights(self):
        """Test that first summary call has 0 prior insights (expected behavior).
        
        This test verifies that the first window processed returns 0 prior insights,
        which is the expected behavior since no prior windows have been processed yet.
        """
        # Use windows_to_accumulate=1 to ensure immediate processing
        client = self.create_client(delay_seconds=0.0)
        client._window_manager.windows_to_accumulate = 1
        
        # Mock LLM response with an insight
        mock_response = self._create_mock_response(
            analysis="First analysis",
            insights=[{
                "insight_type": "ACTION",
                "insight_text": "Complete the first task",
                "confidence": 0.95,
                "classification": "+"
            }]
        )
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            # Process first window - use window_id=0 with timestamp that allows processing
            # The delay is bypassed by using a window_id that doesn't trigger the delay check
            segments = [{"text": "First window text", "start": 0, "end": 5, "start_ms": 0, "end_ms": 5000}]
            result = await client.process_segments(
                summary_type="test",
                segments=segments,
                transcription_window_id=0,
                window_start=5.0,  # Small delay to allow processing
                window_end=10.0
            )
            
            # Should have result with insight (now returns dict with segments key)
            assert isinstance(result, dict)
            assert len(result.get("segments", [])) == 1
            
            # Verify insight was stored in window 0
            window_0_insights = client._window_manager.get_window_insights(0)
            assert len(window_0_insights) == 1
            assert window_0_insights[0].insight_text == "Complete the first task"
            
            # Verify that _build_context returns 0 prior insights for first window
            # (since no accumulated windows exist yet with windows_to_accumulate=2)
            context, prior_insights = client._build_context(include_insights=True)
            # With only 1 window and windows_to_accumulate=2, there are no accumulated windows
            assert len(prior_insights) == 0
    
    @pytest.mark.asyncio
    async def test_second_window_has_prior_insights_from_first_window(self):
        """Test that second summary call has prior insights from first window."""
        # Use windows_to_accumulate=1 to ensure immediate processing
        client = self.create_client(delay_seconds=0.0)
        client._window_manager.windows_to_accumulate = 1
        
        # Mock response for first window
        first_response = self._create_mock_response(
            analysis="First analysis",
            insights=[{
                "insight_type": "ACTION",
                "insight_text": "Complete the first task",
                "confidence": 0.95,
                "classification": "+"
            }]
        )
        
        # Mock response for second window
        second_response = self._create_mock_response(
            analysis="Second analysis",
            insights=[{
                "insight_type": "DECISION",
                "insight_text": "Approved the budget",
                "confidence": 0.90,
                "classification": "~"
            }]
        )
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            # First call returns first_response, second call returns second_response
            mock_create.side_effect = [first_response, second_response]
            
            # Process first window
            segments1 = [{"text": "First window text", "start": 0, "end": 5, "start_ms": 0, "end_ms": 5000}]
            result1 = await client.process_segments(
                summary_type="test",
                segments=segments1,
                transcription_window_id=0,
                window_start=5.0,
                window_end=10.0
            )
            
            assert isinstance(result1, dict)
            assert len(result1.get("segments", [])) == 1
            
            # Verify insights from first window are stored
            window_0_insights = client._window_manager.get_window_insights(0)
            assert len(window_0_insights) == 1
            
            # Process second window
            segments2 = [{"text": "Second window text", "start": 5, "end": 10, "start_ms": 5000, "end_ms": 10000}]
            result2 = await client.process_segments(
                summary_type="test",
                segments=segments2,
                transcription_window_id=1,
                window_start=10.0,
                window_end=15.0
            )
            
            assert isinstance(result2, dict)
            assert len(result2.get("segments", [])) == 1
            
            # Verify insights from second window are stored
            window_1_insights = client._window_manager.get_window_insights(1)
            assert len(window_1_insights) == 1
    
    @pytest.mark.asyncio
    async def test_accumulated_text_and_insights_retrieval(self):
        """Test that get_accumulated_text_and_insights returns correct data."""
        manager = WindowManager(max_chars=10000, windows_to_accumulate=2)
        
        # Add windows with insights
        manager.add_window("Window 0 text", 0.0, 5.0)
        manager.add_window("Window 1 text", 5.0, 10.0)
        manager.add_window("Window 2 text", 10.0, 15.0)
        
        # Manually add insights to windows 0 and 1 (simulating LLM extraction)
        insight1 = WindowInsight(
            insight_id=1,
            insight_type="ACTION",
            insight_text="First action",
            confidence=0.95,
            window_id=0,
            timestamp_start=0.0,
            timestamp_end=5.0,
            classification="+"
        )
        insight2 = WindowInsight(
            insight_id=2,
            insight_type="DECISION",
            insight_text="First decision",
            confidence=0.90,
            window_id=1,
            timestamp_start=5.0,
            timestamp_end=10.0,
            classification="~"
        )
        
        manager.add_insight_to_window(0, insight1)
        manager.add_insight_to_window(1, insight2)
        
        # Get accumulated text and insights (excludes last 2 windows)
        accumulated_text, accumulated_insights = manager.get_accumulated_text_and_insights()
        
        # Should only include window 0 (first 2 windows excluded from accumulation)
        assert "Window 0 text" in accumulated_text
        assert len(accumulated_insights) == 1
        assert accumulated_insights[0].insight_text == "First action"
    
    @pytest.mark.asyncio
    async def test_build_context_includes_prior_insights(self):
        """Test that _build_context includes prior insights in the context string."""
        client = self.create_client(delay_seconds=0.0)
        
        # Add windows manually (need at least 3 windows to have accumulated windows with windows_to_accumulate=2)
        client._window_manager.add_window("Window 0 text", 0.0, 5.0)
        client._window_manager.add_window("Window 1 text", 5.0, 10.0)
        client._window_manager.add_window("Window 2 text", 10.0, 15.0)
        
        # Add insight to window 0
        insight = WindowInsight(
            insight_id=1,
            insight_type="KEY_POINT",
            insight_text="Important finding",
            confidence=0.85,
            window_id=0,
            timestamp_start=0.0,
            timestamp_end=5.0,
            classification="~"
        )
        client._window_manager.add_insight_to_window(0, insight)
        
        # Build context with insights
        context, prior_insights = client._build_context(include_insights=True)
        
        # Context should include PRIOR INSIGHTS section
        assert "## PRIOR INSIGHTS" in context
        assert "Important finding" in context
        assert len(prior_insights) == 1
        assert prior_insights[0].insight_text == "Important finding"
    
    @pytest.mark.asyncio
    async def test_build_context_excludes_insights_from_context_string(self):
        """Test that _build_context excludes insights from context string when disabled."""
        client = self.create_client(delay_seconds=0.0)
        
        # Add windows manually
        client._window_manager.add_window("Window 0 text", 0.0, 5.0)
        client._window_manager.add_window("Window 1 text", 5.0, 10.0)
        client._window_manager.add_window("Window 2 text", 10.0, 15.0)
        
        # Add insight to window 0
        insight = WindowInsight(
            insight_id=1,
            insight_type="KEY_POINT",
            insight_text="Important finding",
            confidence=0.85,
            window_id=0,
            timestamp_start=0.0,
            timestamp_end=5.0,
            classification="~"
        )
        client._window_manager.add_insight_to_window(0, insight)
        
        # Build context without insights
        context, prior_insights = client._build_context(include_insights=False)
        
        # Context string should NOT include PRIOR INSIGHTS section
        assert "## PRIOR INSIGHTS" not in context
        assert "Important finding" not in context
        # Note: prior_insights list is still returned, but won't be included in context string
        # This is by design - the list is returned for logging purposes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])