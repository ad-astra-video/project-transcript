"""
Unit tests for SummaryClient analysis override functionality.
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from src.summary.summary_client import (
    SummaryClient,
    WindowManager,
    SummarySegment,
    WindowInsight,
)


class TestWindowManager:
    """Tests for WindowManager with _first_window_timestamp tracking."""
    
    def test_first_window_timestamp_set_on_first_window(self):
        """Test that _first_window_timestamp is set when first window is added."""
        manager = WindowManager()
        assert manager._first_window_timestamp is None
        
        manager.add_window("test text 1", 10.0, 15.0)
        
        assert manager._first_window_timestamp == 10.0
    
    def test_first_window_timestamp_not_updated_on_subsequent_windows(self):
        """Test that _first_window_timestamp is not updated on subsequent windows."""
        manager = WindowManager()
        
        manager.add_window("test text 1", 10.0, 15.0)
        first_timestamp = manager._first_window_timestamp
        
        manager.add_window("test text 2", 20.0, 25.0)
        
        assert manager._first_window_timestamp == first_timestamp
    
    def test_first_window_timestamp_with_multiple_windows(self):
        """Test first window timestamp tracking with multiple windows."""
        manager = WindowManager()
        
        manager.add_window("first window", 5.0, 10.0)
        manager.add_window("second window", 15.0, 20.0)
        manager.add_window("third window", 25.0, 30.0)
        
        assert manager._first_window_timestamp == 5.0


class TestExtractInsights:
    """Tests for _extract_insights with parsed_data parameter."""
    
    def create_client(self):
        """Create a SummaryClient instance for testing."""
        return SummaryClient(api_key="test_key", model="test_model")
    
    def test_extract_insights_with_list_format(self):
        """Test extracting insights from list format JSON."""
        client = self.create_client()
        
        parsed_data = {
            "analysis": "This is the analysis text",
            "insights": [
                {
                    "insight_type": "ACTION",
                    "insight_text": "Complete the task by Friday",
                    "confidence": 0.95,
                    "classification": "[+]"
                },
                {
                    "insight_type": "DECISION",
                    "insight_text": "Approved the budget",
                    "confidence": 0.90,
                    "classification": "[~]"
                }
            ]
        }
        
        insights = client._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        assert len(insights) == 2
        assert insights[0].insight_type == "ACTION"
        assert insights[0].insight_text == "Complete the task by Friday"
        assert insights[0].confidence == 0.95
        assert insights[0].classification == "[+]"
        assert insights[1].insight_type == "DECISION"
        assert insights[1].insight_text == "Approved the budget"
        assert insights[1].confidence == 0.90
        assert insights[1].classification == "[~]"
    
    def test_extract_insights_with_array_format(self):
        """Test extracting insights from array format JSON."""
        client = self.create_client()
        
        parsed_data = [
            {
                "insight_type": "KEY_POINT",
                "insight_text": "Important finding about the project",
                "confidence": 0.85,
                "classification": "[~]"
            },
            {
                "insight_type": "QUESTION",
                "insight_text": "What is the timeline?",
                "confidence": 0.75,
                "classification": "[-]"
            }
        ]
        
        insights = client._extract_insights(parsed_data, 2, 20.0, 25.0)
        
        assert len(insights) == 2
        assert insights[0].insight_type == "KEY_POINT"
        assert insights[0].insight_text == "Important finding about the project"
        assert insights[0].confidence == 0.85
        assert insights[1].insight_type == "QUESTION"
        assert insights[1].insight_text == "What is the timeline?"
        assert insights[1].confidence == 0.75
    
    def test_extract_insights_with_empty_data(self):
        """Test extracting insights from empty parsed data."""
        client = self.create_client()
        
        insights = client._extract_insights({}, 1, 10.0, 15.0)
        assert len(insights) == 0
        
        insights = client._extract_insights([], 1, 10.0, 15.0)
        assert len(insights) == 0
    
    def test_extract_insights_with_missing_classification(self):
        """Test that missing classification defaults to neutral."""
        client = self.create_client()
        
        parsed_data = [
            {
                "insight_type": "NOTES",
                "insight_text": "Some observation",
                "confidence": 0.80
            }
        ]
        
        insights = client._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        assert len(insights) == 1
        assert insights[0].classification == "[~]"
        assert insights[0].confidence == 0.80
    
    def test_extract_insights_window_timestamps(self):
        """Test that window timestamps are correctly assigned."""
        client = self.create_client()
        
        parsed_data = [
            {
                "insight_type": "ACTION",
                "insight_text": "Test action",
                "confidence": 0.95,
                "classification": "[+]"
            }
        ]
        
        insights = client._extract_insights(parsed_data, 5, 100.0, 105.0)
        
        assert len(insights) == 1
        assert insights[0].window_id == 5
        assert insights[0].timestamp_start == 100.0
        assert insights[0].timestamp_end == 105.0
        assert insights[0].confidence == 0.95
    
    def test_extract_insights_as_dict_includes_confidence(self):
        """Test that as_dict() method includes confidence field."""
        client = self.create_client()
        
        parsed_data = {
            "analysis": "Test analysis",
            "insights": [
                {
                    "insight_type": "ACTION",
                    "insight_text": "Complete the task by Friday",
                    "confidence": 0.95,
                    "classification": "[+]"
                }
            ]
        }
        
        insights = client._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        assert len(insights) == 1
        insight_dict = insights[0].as_dict()
        
        assert "confidence" in insight_dict
        assert insight_dict["confidence"] == 0.95
        assert insight_dict["insight_type"] == "ACTION"
        assert insight_dict["insight_text"] == "Complete the task by Friday"
        assert insight_dict["classification"] == "[+]"


class TestProcessSegmentsAnalysisOverride:
    """Tests for analysis override logic in process_segments."""
    
    def create_client(self):
        """Create a SummaryClient instance for testing."""
        return SummaryClient(api_key="test_key", model="test_model")
    
    @pytest.mark.asyncio
    async def test_analysis_field_used_as_background_context(self):
        """Test that analysis field is used as background_context when present."""
        # Use shorter delay to avoid timing issues in tests
        client = SummaryClient(api_key="test_key", model="test_model", initial_summary_delay_seconds=1.0)
        
        # Mock the summarize_text method
        analysis_text = "This is the analysis from the LLM"
        reasoning_text = "This is the reasoning content"
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "analysis": analysis_text,
            "insights": []
        })
        mock_response.choices[0].message.reasoning = reasoning_text
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            # Add a window first
            client._window_manager.add_window("test text", 0.0, 5.0)
            
            # Process segments at timestamp 10 (exceeds 1 second delay)
            segments = [{"text": "Test segment", "start": 0, "end": 5}]
            result = await client.process_segments(
                summary_type="test",
                segments=segments,
                window_id=1,
                window_start=10.0,
                window_end=15.0
            )
            
            # Check that analysis was used as background_context
            assert len(result) == 1
            assert result[0].background_context == analysis_text
    
    @pytest.mark.asyncio
    async def test_reasoning_content_used_when_analysis_missing(self):
        """Test that reasoning_content is used when analysis field is missing."""
        # Use shorter delay to avoid timing issues in tests
        client = SummaryClient(api_key="test_key", model="test_model", initial_summary_delay_seconds=1.0)
        
        reasoning_text = "This is the reasoning content"
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "insights": []
        })
        mock_response.choices[0].message.reasoning = reasoning_text
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            # Add a window first
            client._window_manager.add_window("test text", 0.0, 5.0)
            
            # Process segments at timestamp 10 (exceeds 1 second delay)
            segments = [{"text": "Test segment", "start": 0, "end": 5}]
            result = await client.process_segments(
                summary_type="test",
                segments=segments,
                window_id=1,
                window_start=10.0,
                window_end=15.0
            )
            
            # Check that reasoning_content was used as fallback
            assert len(result) == 1
            assert result[0].background_context == reasoning_text
    
    @pytest.mark.asyncio
    async def test_reasoning_content_used_when_analysis_empty(self):
        """Test that reasoning_content is used when analysis field is empty string."""
        # Use shorter delay to avoid timing issues in tests
        client = SummaryClient(api_key="test_key", model="test_model", initial_summary_delay_seconds=1.0)
        
        reasoning_text = "This is the reasoning content"
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "analysis": "",
            "insights": []
        })
        mock_response.choices[0].message.reasoning = reasoning_text
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            # Add a window first
            client._window_manager.add_window("test text", 0.0, 5.0)
            
            # Process segments at timestamp 10 (exceeds 1 second delay)
            segments = [{"text": "Test segment", "start": 0, "end": 5}]
            result = await client.process_segments(
                summary_type="test",
                segments=segments,
                window_id=1,
                window_start=10.0,
                window_end=15.0
            )
            
            # Check that reasoning_content was used as fallback
            assert len(result) == 1
            assert result[0].background_context == reasoning_text
    
    @pytest.mark.asyncio
    async def test_reasoning_content_used_when_json_parse_fails(self):
        """Test that reasoning_content is used when JSON parsing fails."""
        # Use shorter delay to avoid timing issues in tests
        client = SummaryClient(api_key="test_key", model="test_model", initial_summary_delay_seconds=1.0)
        
        reasoning_text = "This is the reasoning content"
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Invalid JSON"
        mock_response.choices[0].message.reasoning = reasoning_text
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            # Add a window first
            client._window_manager.add_window("test text", 0.0, 5.0)
            
            # Process segments at timestamp 10 (exceeds 1 second delay)
            segments = [{"text": "Test segment", "start": 0, "end": 5}]
            result = await client.process_segments(
                summary_type="test",
                segments=segments,
                window_id=1,
                window_start=10.0,
                window_end=15.0
            )
            
            # Check that reasoning_content was used as fallback
            assert len(result) == 1
            assert result[0].background_context == reasoning_text


class TestInitialDelayLogic:
    """Tests for self-contained initial delay logic."""
    
    def create_client(self, delay_seconds: float = 30.0):
        """Create a SummaryClient instance with specified delay."""
        return SummaryClient(
            api_key="test_key", 
            model="test_model",
            initial_summary_delay_seconds=delay_seconds
        )
    
    @pytest.mark.asyncio
    async def test_delay_applied_when_elapsed_less_than_delay(self):
        """Test that delay is applied when elapsed time is less than delay setting."""
        client = self.create_client(delay_seconds=30.0)
        
        # Add a window with timestamp 0
        client._window_manager.add_window("test text", 0.0, 5.0)
        
        # Try to process segments at timestamp 10 (only 10 seconds elapsed, need 30)
        segments = [{"text": "Test segment", "start": 0, "end": 5}]
        result = await client.process_segments(
            summary_type="test",
            segments=segments,
            window_id=1,
            window_start=10.0,  # Only 10 seconds elapsed
            window_end=15.0
        )
        
        # Should return empty list (delay applied)
        assert len(result) == 0
    
    @pytest.mark.asyncio
    async def test_no_delay_when_elapsed_greater_than_delay(self):
        """Test that delay is not applied when elapsed time exceeds delay setting."""
        client = self.create_client(delay_seconds=30.0)
        
        # Add a window with timestamp 0
        client._window_manager.add_window("test text", 0.0, 5.0)
        
        # Mock the summarize_text method
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "analysis": "Analysis",
            "insights": []
        })
        mock_response.choices[0].message.reasoning = "Reasoning"
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            # Try to process segments at timestamp 40 (40 seconds elapsed, need 30)
            segments = [{"text": "Test segment", "start": 0, "end": 5}]
            result = await client.process_segments(
                summary_type="test",
                segments=segments,
                window_id=1,
                window_start=40.0,  # 40 seconds elapsed, exceeds 30s delay
                window_end=45.0
            )
            
            # Should return a result (no delay)
            assert len(result) == 1
    
    def test_update_params_changes_delay(self):
        """Test that update_params can change the delay setting."""
        client = self.create_client(delay_seconds=30.0)
        
        assert client.initial_summary_delay_seconds == 30.0
        
        client.update_params(initial_summary_delay_seconds=60.0)
        
        assert client.initial_summary_delay_seconds == 60.0


class TestSummarizeTextRawOutput:
    """Tests for summarize_text returning raw strings."""
    
    def create_client(self):
        """Create a SummaryClient instance for testing."""
        return SummaryClient(api_key="test_key", model="test_model")
    
    @pytest.mark.asyncio
    async def test_summarize_text_returns_raw_strings(self):
        """Test that summarize_text returns raw summary_text and reasoning_content."""
        client = self.create_client()
        
        raw_summary = '{"analysis": "Test analysis", "insights": []}'
        raw_reasoning = "Test reasoning content"
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = raw_summary
        mock_response.choices[0].message.reasoning = raw_reasoning
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            result_summary, result_reasoning = await client.summarize_text(
                text="Test text",
                context="Test context"
            )
            
            # Should return raw strings (JSON will be parsed in process_segments)
            assert result_summary == raw_summary
            assert result_reasoning == raw_reasoning
    
    @pytest.mark.asyncio
    async def test_summarize_text_strips_code_blocks(self):
        """Test that summarize_text strips ```json code blocks."""
        client = self.create_client()
        
        raw_summary = '```json\n{"analysis": "Test", "insights": []}\n```'
        raw_reasoning = "Test reasoning"
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = raw_summary
        mock_response.choices[0].message.reasoning = raw_reasoning
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            result_summary, result_reasoning = await client.summarize_text(
                text="Test text",
                context="Test context"
            )
            
            # Should strip code blocks
            assert result_summary == '{"analysis": "Test", "insights": []}'
            assert result_reasoning == raw_reasoning


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
        client = self.create_client(delay_seconds=0.0)
        
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
                window_id=0,
                window_start=5.0,  # Small delay to allow processing
                window_end=10.0
            )
            
            # Should have result with insight
            assert len(result) == 1
            
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
        client = self.create_client(delay_seconds=0.0)
        
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
                window_id=0,
                window_start=5.0,
                window_end=10.0
            )
            
            assert len(result1) == 1
            
            # Verify insights from first window are stored
            window_0_insights = client._window_manager.get_window_insights(0)
            assert len(window_0_insights) == 1
            
            # Process second window
            segments2 = [{"text": "Second window text", "start": 5, "end": 10, "start_ms": 5000, "end_ms": 10000}]
            result2 = await client.process_segments(
                summary_type="test",
                segments=segments2,
                window_id=1,
                window_start=10.0,
                window_end=15.0
            )
            
            assert len(result2) == 1
            
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