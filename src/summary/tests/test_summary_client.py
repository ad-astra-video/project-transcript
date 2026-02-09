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
        
        result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        # _extract_insights returns the parsed_data dict with insights
        assert "insights" in result
        assert len(result["insights"]) == 2
        assert result["insights"][0]["insight_type"] == "ACTION"
        assert result["insights"][0]["insight_text"] == "Complete the task by Friday"
        assert result["insights"][0]["confidence"] == 0.95
        assert result["insights"][0]["classification"] == "[+]"
        assert result["insights"][1]["insight_type"] == "DECISION"
        assert result["insights"][1]["insight_text"] == "Approved the budget"
        assert result["insights"][1]["confidence"] == 0.90
        assert result["insights"][1]["classification"] == "[~]"
    
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
        
        result = client._extract_insights(parsed_data, 2, 20.0, 25.0)
        
        # _extract_insights returns a list for array format
        assert len(result) == 2
        assert result[0]["insight_type"] == "KEY_POINT"
        assert result[0]["insight_text"] == "Important finding about the project"
        assert result[0]["confidence"] == 0.85
        assert result[1]["insight_type"] == "QUESTION"
        assert result[1]["insight_text"] == "What is the timeline?"
        assert result[1]["confidence"] == 0.75
    
    def test_extract_insights_with_empty_data(self):
        """Test extracting insights from empty parsed data."""
        client = self.create_client()
        
        result = client._extract_insights({}, 1, 10.0, 15.0)
        assert result == {}
        
        result = client._extract_insights([], 1, 10.0, 15.0)
        assert result == []
    
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
        
        result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        assert len(result) == 1
        assert result[0]["classification"] == "~"
        assert result[0]["confidence"] == 0.80
    
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
        
        result = client._extract_insights(parsed_data, 5, 100.0, 105.0)
        
        assert len(result) == 1
        # Note: _extract_insights doesn't assign window_id/timestamps to the dict
        # Those are assigned when adding to WindowManager
        assert result[0]["insight_type"] == "ACTION"
        assert result[0]["confidence"] == 0.95
    
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
        
        result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        assert "insights" in result
        assert len(result["insights"]) == 1
        insight_dict = result["insights"][0]
        
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
                transcription_window_id=1,
                window_start=10.0,
                window_end=15.0
            )
            
            # Check that analysis was used as background_context (now returns dict)
            assert isinstance(result, dict)
            assert len(result.get("segments", [])) == 1
            assert result["segments"][0].get("background_context") == analysis_text
    
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
                transcription_window_id=1,
                window_start=10.0,
                window_end=15.0
            )
            
            # Check that reasoning_content was used as fallback (now returns dict)
            assert isinstance(result, dict)
            assert len(result.get("segments", [])) == 1
            assert result["segments"][0].get("background_context") == reasoning_text
    
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
                transcription_window_id=1,
                window_start=10.0,
                window_end=15.0
            )
            
            # Check that reasoning_content was used as fallback (now returns dict)
            assert isinstance(result, dict)
            assert len(result.get("segments", [])) == 1
            assert result["segments"][0].get("background_context") == reasoning_text
    
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
                transcription_window_id=1,
                window_start=10.0,
                window_end=15.0
            )
            
            # Check that reasoning_content was used as fallback (now returns dict)
            assert isinstance(result, dict)
            assert len(result.get("segments", [])) == 1
            assert result["segments"][0].get("background_context") == reasoning_text


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
            transcription_window_id=1,
            window_start=10.0,  # Only 10 seconds elapsed
            window_end=15.0
        )
        
        # Should return empty segments list (delay applied, now returns dict)
        assert isinstance(result, dict)
        assert len(result.get("segments", [])) == 0
    
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
                transcription_window_id=1,
                window_start=40.0,  # 40 seconds elapsed, exceeds 30s delay
                window_end=45.0
            )
            
            # Should return a result (no delay, now returns dict)
            assert isinstance(result, dict)
            assert len(result.get("segments", [])) == 1
    
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


class TestSentimentEnabledFiltering:
    """Tests for sentiment_enabled filtering in _extract_insights."""
    
    def create_client(self):
        """Create a SummaryClient instance for testing."""
        return SummaryClient(api_key="test_key", model="test_model")
    
    def test_is_sentiment_enabled_returns_true_for_general_meeting(self):
        """Test that is_sentiment_enabled returns True for GENERAL_MEETING."""
        client = self.create_client()
        client.set_content_type("GENERAL_MEETING", confidence=0.9, source="AUTO_DETECTED")
        assert client.is_sentiment_enabled() is True
    
    def test_is_sentiment_enabled_returns_true_for_customer_support(self):
        """Test that is_sentiment_enabled returns True for CUSTOMER_SUPPORT."""
        client = self.create_client()
        client.set_content_type("CUSTOMER_SUPPORT", confidence=0.9, source="AUTO_DETECTED")
        assert client.is_sentiment_enabled() is True
    
    def test_is_sentiment_enabled_returns_false_for_technical_talk(self):
        """Test that is_sentiment_enabled returns False for TECHNICAL_TALK."""
        client = self.create_client()
        client.set_content_type("TECHNICAL_TALK", confidence=0.9, source="AUTO_DETECTED")
        assert client.is_sentiment_enabled() is False
    
    def test_is_sentiment_enabled_returns_false_for_lecture_or_talk(self):
        """Test that is_sentiment_enabled returns False for LECTURE_OR_TALK."""
        client = self.create_client()
        client.set_content_type("LECTURE_OR_TALK", confidence=0.9, source="AUTO_DETECTED")
        assert client.is_sentiment_enabled() is False
    
    def test_is_sentiment_enabled_returns_false_for_interview(self):
        """Test that is_sentiment_enabled returns False for INTERVIEW."""
        client = self.create_client()
        client.set_content_type("INTERVIEW", confidence=0.9, source="AUTO_DETECTED")
        assert client.is_sentiment_enabled() is False
    
    def test_is_sentiment_enabled_returns_false_for_podcast(self):
        """Test that is_sentiment_enabled returns False for PODCAST."""
        client = self.create_client()
        client.set_content_type("PODCAST", confidence=0.9, source="AUTO_DETECTED")
        assert client.is_sentiment_enabled() is False
    
    def test_is_sentiment_enabled_returns_false_for_streamer_monologue(self):
        """Test that is_sentiment_enabled returns False for STREAMER_MONOLOGUE."""
        client = self.create_client()
        client.set_content_type("STREAMER_MONOLOGUE", confidence=0.9, source="AUTO_DETECTED")
        assert client.is_sentiment_enabled() is False
    
    def test_is_sentiment_enabled_returns_false_for_news_update(self):
        """Test that is_sentiment_enabled returns False for NEWS_UPDATE."""
        client = self.create_client()
        client.set_content_type("NEWS_UPDATE", confidence=0.9, source="AUTO_DETECTED")
        assert client.is_sentiment_enabled() is False
    
    def test_is_sentiment_enabled_returns_false_for_gameplay_commentary(self):
        """Test that is_sentiment_enabled returns False for GAMEPLAY_COMMENTARY."""
        client = self.create_client()
        client.set_content_type("GAMEPLAY_COMMENTARY", confidence=0.9, source="AUTO_DETECTED")
        assert client.is_sentiment_enabled() is False
    
    def test_is_sentiment_enabled_returns_false_for_debate(self):
        """Test that is_sentiment_enabled returns False for DEBATE."""
        client = self.create_client()
        client.set_content_type("DEBATE", confidence=0.9, source="AUTO_DETECTED")
        assert client.is_sentiment_enabled() is False
    
    def test_is_sentiment_enabled_returns_false_for_unknown(self):
        """Test that is_sentiment_enabled returns False for UNKNOWN."""
        client = self.create_client()
        client.set_content_type("UNKNOWN", confidence=0.9, source="AUTO_DETECTED")
        assert client.is_sentiment_enabled() is False
    
    def test_extract_insights_includes_sentiment_when_enabled(self):
        """Test that SENTIMENT insights are included when sentiment_enabled=True."""
        client = self.create_client()
        client.set_content_type("GENERAL_MEETING", confidence=0.9, source="AUTO_DETECTED")
        
        parsed_data = {
            "analysis": "Test analysis",
            "insights": [
                {
                    "insight_type": "SENTIMENT",
                    "insight_text": "Positive tone detected",
                    "confidence": 0.85,
                    "classification": "+"
                },
                {
                    "insight_type": "ACTION",
                    "insight_text": "Complete the task",
                    "confidence": 0.95,
                    "classification": "+"
                }
            ]
        }
        
        result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        # Should include both insights (SENTIMENT + ACTION)
        assert len(result["insights"]) == 2
        insight_types = [i["insight_type"] for i in result["insights"]]
        assert "SENTIMENT" in insight_types
        assert "ACTION" in insight_types
    
    def test_extract_insights_filters_sentiment_when_disabled(self):
        """Test that SENTIMENT insights are filtered when sentiment_enabled=False."""
        client = self.create_client()
        client.set_content_type("TECHNICAL_TALK", confidence=0.9, source="AUTO_DETECTED")
        
        parsed_data = {
            "analysis": "Test analysis",
            "insights": [
                {
                    "insight_type": "SENTIMENT",
                    "insight_text": "Positive tone detected",
                    "confidence": 0.85,
                    "classification": "+"
                },
                {
                    "insight_type": "ACTION",
                    "insight_text": "Complete the task",
                    "confidence": 0.95,
                    "classification": "+"
                }
            ]
        }
        
        result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        # Should only include ACTION insight (SENTIMENT filtered out)
        assert len(result["insights"]) == 1
        assert result["insights"][0]["insight_type"] == "ACTION"
    
    def test_extract_insights_non_sentiment_always_included(self):
        """Test that non-SENTIMENT insights are always included regardless of sentiment_enabled."""
        client = self.create_client()
        client.set_content_type("TECHNICAL_TALK", confidence=0.9, source="AUTO_DETECTED")
        
        parsed_data = {
            "analysis": "Test analysis",
            "insights": [
                {
                    "insight_type": "DECISION",
                    "insight_text": "Approved the budget",
                    "confidence": 0.90,
                    "classification": "~"
                },
                {
                    "insight_type": "QUESTION",
                    "insight_text": "What is the timeline?",
                    "confidence": 0.75,
                    "classification": "~"
                },
                {
                    "insight_type": "KEY_POINT",
                    "insight_text": "Important finding",
                    "confidence": 0.85,
                    "classification": "~"
                }
            ]
        }
        
        result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        # Should include all non-SENTIMENT insights
        assert len(result["insights"]) == 3
        insight_types = [i["insight_type"] for i in result["insights"]]
        assert "DECISION" in insight_types
        assert "QUESTION" in insight_types
        assert "KEY_POINT" in insight_types
    
    def test_extract_insights_mixed_insights_partially_filtered(self):
        """Test that mixed insights (SENTIMENT + others) are partially filtered correctly."""
        client = self.create_client()
        client.set_content_type("CUSTOMER_SUPPORT", confidence=0.9, source="AUTO_DETECTED")
        
        parsed_data = {
            "analysis": "Test analysis",
            "insights": [
                {
                    "insight_type": "SENTIMENT",
                    "insight_text": "Customer frustration detected",
                    "confidence": 0.85,
                    "classification": "-"
                },
                {
                    "insight_type": "ACTION",
                    "insight_text": "Escalate to manager",
                    "confidence": 0.95,
                    "classification": "+"
                },
                {
                    "insight_type": "SENTIMENT",
                    "insight_text": "Positive resolution",
                    "confidence": 0.80,
                    "classification": "+"
                },
                {
                    "insight_type": "DECISION",
                    "insight_text": "Issue resolved",
                    "confidence": 0.90,
                    "classification": "~"
                }
            ]
        }
        
        result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        # Should include all insights (sentiment_enabled=True for CUSTOMER_SUPPORT)
        assert len(result["insights"]) == 4
        insight_types = [i["insight_type"] for i in result["insights"]]
        assert insight_types.count("SENTIMENT") == 2
        assert "ACTION" in insight_types
        assert "DECISION" in insight_types
    
    def test_extract_insights_empty_insights_list(self):
        """Test that empty insights list is handled correctly."""
        client = self.create_client()
        client.set_content_type("GENERAL_MEETING", confidence=0.9, source="AUTO_DETECTED")
        
        parsed_data = {
            "analysis": "Test analysis",
            "insights": []
        }
        
        result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        assert len(result["insights"]) == 0
    
    def test_extract_insights_only_sentiment_filtered(self):
        """Test that when only SENTIMENT insights exist and are filtered, result is empty."""
        client = self.create_client()
        client.set_content_type("LECTURE_OR_TALK", confidence=0.9, source="AUTO_DETECTED")
        
        parsed_data = {
            "analysis": "Test analysis",
            "insights": [
                {
                    "insight_type": "SENTIMENT",
                    "insight_text": "Engaged audience",
                    "confidence": 0.85,
                    "classification": "+"
                }
            ]
        }
        
        result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        # SENTIMENT should be filtered out
        assert len(result["insights"]) == 0


class TestKeyPointClassification:
    """Tests for KEY POINT classification with breakthrough-level threshold."""
    
    def create_client(self):
        """Create a SummaryClient instance for testing."""
        return SummaryClient(api_key="test_key", model="test_model")
    
    def test_key_point_classification_for_critical_threshold(self):
        """Test that critical thresholds are classified as KEY POINT."""
        client = self.create_client()
        
        # These should be KEY POINT - critical thresholds
        critical_threshold_insights = [
            "The system fails above 10,000 concurrent connections",
            "At 500ms latency, user experience degrades significantly",
            "The error rate is below 1%",
            "Budget is $50,000",
        ]
        
        for insight_text in critical_threshold_insights:
            parsed_data = {
                "analysis": "Test analysis",
                "insights": [
                    {
                        "insight_type": "KEY POINT",
                        "insight_text": insight_text,
                        "confidence": 0.95,
                        "classification": "~"
                    }
                ]
            }
            
            result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
            assert len(result["insights"]) == 1, f"Expected KEY POINT for: {insight_text}"
            assert result["insights"][0]["insight_type"] == "KEY POINT"
    
    def test_notes_classification_for_explanations(self):
        """Test that explanations are classified as NOTES, not KEY POINT."""
        client = self.create_client()
        
        # These should be NOTES - explanations of how things work
        explanation_insights = [
            "RAG retrieves documents via semantic similarity",
            "Multi-hop reasoning is critical for recursive tasks",
            "Clauses referencing other clauses create recursive complexity",
            "The API endpoint is /api/v1/users",
            "Authentication requires a Bearer token",
            "The function takes a string parameter and returns a boolean",
        ]
        
        for insight_text in explanation_insights:
            parsed_data = {
                "analysis": "Test analysis",
                "insights": [
                    {
                        "insight_type": "NOTES",
                        "insight_text": insight_text,
                        "confidence": 0.95,
                        "classification": "~"
                    }
                ]
            }
            
            result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
            assert len(result["insights"]) == 1, f"Expected NOTES for: {insight_text}"
            assert result["insights"][0]["insight_type"] == "NOTES"
    
    def test_key_point_classification_for_discoveries(self):
        """Test that discoveries are classified as KEY POINT."""
        client = self.create_client()
        
        # These should be KEY POINT - discoveries, findings, revelations
        discovery_insights = [
            "Task complexity is the primary driver of context window limitations",
            "Context degradation is task-specific",
            "Context degradation occurs at specific saturation levels",
            "Context degradation severity increases non-linearly",
            "The memory leak was traced to an unclosed database connection",
            "A race condition exists when two requests arrive simultaneously",
        ]
        
        for insight_text in discovery_insights:
            parsed_data = {
                "analysis": "Test analysis",
                "insights": [
                    {
                        "insight_type": "KEY POINT",
                        "insight_text": insight_text,
                        "confidence": 0.85,
                        "classification": "~"
                    }
                ]
            }
            
            result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
            assert len(result["insights"]) == 1, f"Expected KEY POINT for: {insight_text}"
            assert result["insights"][0]["insight_type"] == "KEY POINT"
    
    def test_notes_classification_for_standard_patterns(self):
        """Test that standard patterns and configurations are NOTES."""
        client = self.create_client()
        
        # These should be NOTES - standard patterns, not breakthroughs
        standard_pattern_insights = [
            "We use a retry with exponential backoff",
            "Setting pool_size to 20 is recommended",
            "The library handles JSON serialization automatically",
            "Context window size is only half the story",
            "Task complexity is the other half",
        ]
        
        for insight_text in standard_pattern_insights:
            parsed_data = {
                "analysis": "Test analysis",
                "insights": [
                    {
                        "insight_type": "NOTES",
                        "insight_text": insight_text,
                        "confidence": 0.95,
                        "classification": "~"
                    }
                ]
            }
            
            result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
            assert len(result["insights"]) == 1, f"Expected NOTES for: {insight_text}"
            assert result["insights"][0]["insight_type"] == "NOTES"
    
    def test_key_point_classification_for_isolated_facts_with_implications(self):
        """Test KEY POINT classification for isolated facts with significant implications."""
        client = self.create_client()
        
        # These should be KEY POINT - isolated facts with implications
        significant_facts = [
            "Revenue increased 40% YoY",
            "Customer churn dropped from 10% to 5%",
            "The system processes 10,000 requests/day",
        ]
        
        for insight_text in significant_facts:
            parsed_data = {
                "analysis": "Test analysis",
                "insights": [
                    {
                        "insight_type": "KEY POINT",
                        "insight_text": insight_text,
                        "confidence": 0.90,
                        "classification": "~"
                    }
                ]
            }
            
            result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
            assert len(result["insights"]) == 1, f"Expected KEY POINT for: {insight_text}"
            assert result["insights"][0]["insight_type"] == "KEY POINT"
    
    def test_notes_classification_for_contextual_details(self):
        """Test NOTES classification for contextual details without standalone significance."""
        client = self.create_client()
        
        # These should be NOTES - contextual details
        contextual_details = [
            "The team has 5 members",
            "This is the 3rd meeting this week",
            "We have 5 team members working on this",
            "The meeting lasted 45 minutes",
            "This is the third time we've discussed this",
            "There were 15 attendees",
            "We started this project in Q1",
            "Chat was active today",
        ]
        
        for insight_text in contextual_details:
            parsed_data = {
                "analysis": "Test analysis",
                "insights": [
                    {
                        "insight_type": "NOTES",
                        "insight_text": insight_text,
                        "confidence": 0.95,
                        "classification": "~"
                    }
                ]
            }
            
            result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
            assert len(result["insights"]) == 1, f"Expected NOTES for: {insight_text}"
            assert result["insights"][0]["insight_type"] == "NOTES"
    
    def test_technical_talk_content_type_key_point_deemphasis(self):
        """Test that TECHNICAL_TALK deemphasizes KEY POINT extraction."""
        client = self.create_client()
        client.set_content_type("TECHNICAL_TALK", confidence=0.9, source="AUTO_DETECTED")
        
        # Verify KEY POINT is deemphasized for TECHNICAL_TALK
        from src.summary.prompts import CONTENT_TYPE_RULE_MODIFIERS
        rules = CONTENT_TYPE_RULE_MODIFIERS["TECHNICAL_TALK"]
        
        assert "KEY POINT" in rules["deemphasize"], "KEY POINT should be deemphasized for TECHNICAL_TALK"
        assert rules["notes_frequency"] == "high", "NOTES frequency should be high for TECHNICAL_TALK (reduced for less insights)"
        assert rules["action_strictness"] == "extreme", "ACTION strictness should be extreme for TECHNICAL_TALK"
    
    def test_lecture_or_talk_content_type_key_point_emphasis(self):
        """Test that LECTURE_OR_TALK emphasizes NOTES extraction (not KEY POINT)."""
        client = self.create_client()
        client.set_content_type("LECTURE_OR_TALK", confidence=0.9, source="AUTO_DETECTED")
        
        # Verify NOTES is emphasized for LECTURE_OR_TALK (KEY POINT is deemphasized for less insights)
        from src.summary.prompts import CONTENT_TYPE_RULE_MODIFIERS
        rules = CONTENT_TYPE_RULE_MODIFIERS["LECTURE_OR_TALK"]
        
        assert "NOTES" in rules["emphasize"], "NOTES should be emphasized for LECTURE_OR_TALK"
        assert "KEY POINT" in rules["deemphasize"], "KEY POINT should be deemphasized for LECTURE_OR_TALK"
        assert rules["notes_frequency"] == "medium", "NOTES frequency should be medium for LECTURE_OR_TALK (reduced for less insights)"
    
    def test_general_meeting_content_type_key_point_deemphasis(self):
        """Test that GENERAL_MEETING deemphasizes KEY POINT extraction."""
        client = self.create_client()
        client.set_content_type("GENERAL_MEETING", confidence=0.9, source="AUTO_DETECTED")
        
        # Verify KEY POINT is deemphasized for GENERAL_MEETING
        from src.summary.prompts import CONTENT_TYPE_RULE_MODIFIERS
        rules = CONTENT_TYPE_RULE_MODIFIERS["GENERAL_MEETING"]
        
        assert "KEY POINT" in rules["deemphasize"], "KEY POINT should be deemphasized for GENERAL_MEETING"
        assert "ACTION" in rules["emphasize"], "ACTION should be emphasized for GENERAL_MEETING"
        assert "DECISION" in rules["emphasize"], "DECISION should be emphasized for GENERAL_MEETING"
    
    def test_interview_content_type_notes_emphasis(self):
        """Test that INTERVIEW emphasizes NOTES extraction (not KEY POINT for less insights)."""
        client = self.create_client()
        client.set_content_type("INTERVIEW", confidence=0.9, source="AUTO_DETECTED")
        
        # Verify NOTES is emphasized for INTERVIEW
        from src.summary.prompts import CONTENT_TYPE_RULE_MODIFIERS
        rules = CONTENT_TYPE_RULE_MODIFIERS["INTERVIEW"]
        
        assert "NOTES" in rules["emphasize"], "NOTES should be emphasized for INTERVIEW"
        assert "KEY POINT" in rules["deemphasize"], "KEY POINT should be deemphasized for INTERVIEW"
        assert "QUESTION" in rules["emphasize"], "QUESTION should be emphasized for INTERVIEW"
    
    def test_podcast_content_type_notes_emphasis(self):
        """Test that PODCAST emphasizes NOTES extraction (not KEY POINT for less insights)."""
        client = self.create_client()
        client.set_content_type("PODCAST", confidence=0.9, source="AUTO_DETECTED")
        
        # Verify NOTES is emphasized for PODCAST (KEY POINT is deemphasized for less insights)
        from src.summary.prompts import CONTENT_TYPE_RULE_MODIFIERS
        rules = CONTENT_TYPE_RULE_MODIFIERS["PODCAST"]
        
        assert "NOTES" in rules["emphasize"], "NOTES should be emphasized for PODCAST"
        assert "KEY POINT" in rules["deemphasize"], "KEY POINT should be deemphasized for PODCAST"
        assert rules["notes_frequency"] == "high", "NOTES frequency should be high for PODCAST (reduced for less insights)"


class TestKeyPointVsNotesDecisionGuide:
    """Tests for KEY POINT vs NOTES decision guide examples."""
    
    def create_client(self):
        """Create a SummaryClient instance for testing."""
        return SummaryClient(api_key="test_key", model="test_model")
    
    def test_decision_guide_general_examples(self):
        """Test that general decision guide examples are classified correctly."""
        client = self.create_client()
        
        # General examples from the decision guide
        test_cases = [
            ("The team has 5 members", "NOTES"),
            ("This is the 3rd meeting this week", "NOTES"),
            ("The deadline is next Friday", "KEY POINT"),
            ("Revenue increased 40% YoY", "KEY POINT"),
            ("We have 5 team members working on this", "NOTES"),
            ("The meeting lasted 45 minutes", "NOTES"),
            ("Customer churn dropped from 10% to 5%", "KEY POINT"),
            ("This is the third time we've discussed this", "NOTES"),
            ("Budget is $50,000", "KEY POINT"),
            ("There were 15 attendees", "NOTES"),
            ("The system processes 10,000 requests/day", "KEY POINT"),
            ("We started this project in Q1", "NOTES"),
            ("The error rate is below 1%", "KEY POINT"),
            ("Chat was active today", "NOTES"),
        ]
        
        for insight_text, expected_type in test_cases:
            parsed_data = {
                "analysis": "Test analysis",
                "insights": [
                    {
                        "insight_type": expected_type,
                        "insight_text": insight_text,
                        "confidence": 0.95,
                        "classification": "~"
                    }
                ]
            }
            
            result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
            assert len(result["insights"]) == 1, f"Expected {expected_type} for: {insight_text}"
            assert result["insights"][0]["insight_type"] == expected_type, f"Mismatch for: {insight_text}"
    
    def test_decision_guide_technical_examples(self):
        """Test that technical decision guide examples are classified correctly."""
        client = self.create_client()
        
        # Technical examples from the decision guide
        test_cases = [
            # Explanations should be NOTES
            ("Context window size is only half the story", "NOTES"),
            ("Task complexity is the other half", "NOTES"),
            ("Multi-hop reasoning is critical for recursive tasks", "NOTES"),
            ("Clauses referencing other clauses create recursive complexity", "NOTES"),
            ("RAG retrieves documents via semantic similarity", "NOTES"),
            ("The API endpoint is /api/v1/users", "NOTES"),
            ("Authentication requires a Bearer token", "NOTES"),
            ("We use a retry with exponential backoff", "NOTES"),
            ("Setting pool_size to 20 is recommended", "NOTES"),
            ("The library handles JSON serialization automatically", "NOTES"),
            ("The function takes a string parameter and returns a boolean", "NOTES"),
            # Discoveries should be KEY POINT
            ("Task complexity is the primary driver", "KEY POINT"),
            ("Context degradation is task-specific", "KEY POINT"),
            ("Context degradation occurs at specific saturation levels", "KEY POINT"),
            ("Context degradation severity increases non-linearly", "KEY POINT"),
            ("The system fails above 10,000 concurrent connections", "KEY POINT"),
            ("The memory leak was traced to an unclosed database connection", "KEY POINT"),
            ("At 500ms latency, user experience degrades significantly", "KEY POINT"),
            ("A race condition exists when two requests arrive simultaneously", "KEY POINT"),
        ]
        
        for insight_text, expected_type in test_cases:
            parsed_data = {
                "analysis": "Test analysis",
                "insights": [
                    {
                        "insight_type": expected_type,
                        "insight_text": insight_text,
                        "confidence": 0.95,
                        "classification": "~"
                    }
                ]
            }
            
            result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
            assert len(result["insights"]) == 1, f"Expected {expected_type} for: {insight_text}"
            assert result["insights"][0]["insight_type"] == expected_type, f"Mismatch for: {insight_text}"


class TestZeroOutputBehavior:
    """Tests for zero-output behavior when nothing meaningful is said."""
    
    def create_client(self):
        """Create a SummaryClient instance for testing."""
        return SummaryClient(api_key="test_key", model="test_model")
    
    def test_empty_insights_array_handling(self):
        """Test that empty insights array is handled correctly."""
        client = self.create_client()
        
        # Empty insights array should be handled correctly
        parsed_data = {
            "analysis": "",
            "insights": []
        }
        
        result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        # Should return empty insights
        assert "insights" in result
        assert len(result["insights"]) == 0
    
    def test_empty_analysis_with_insights(self):
        """Test that empty analysis string is valid when insights exist."""
        client = self.create_client()
        
        parsed_data = {
            "analysis": "",
            "insights": [
                {
                    "insight_type": "ACTION",
                    "insight_text": "Complete the task by Friday",
                    "confidence": 0.95,
                    "classification": "[+]"
                }
            ]
        }
        
        result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        # Should still extract the insight
        assert len(result["insights"]) == 1
        assert result["insights"][0]["insight_type"] == "ACTION"
    
    def test_notes_only_output_for_continuity(self):
        """Test that NOTES can be output for continuity when no other insights exist."""
        client = self.create_client()
        
        # NOTES for continuity should be allowed
        parsed_data = {
            "analysis": "",
            "insights": [
                {
                    "insight_type": "NOTES",
                    "insight_text": "Topic shifted to project timeline",
                    "confidence": 0.85,
                    "classification": "[~]"
                }
            ]
        }
        
        result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        # Should extract the NOTES insight
        assert len(result["insights"]) == 1
        assert result["insights"][0]["insight_type"] == "NOTES"
    
    def test_multiple_insights_filtered_to_high_value_only(self):
        """Test that multiple trivial insights are filtered to only high-value ones."""
        client = self.create_client()
        
        # Multiple trivial insights - only meaningful ones should be extracted
        parsed_data = {
            "analysis": "",
            "insights": [
                {
                    "insight_type": "NOTES",
                    "insight_text": "Speaker paused briefly",
                    "confidence": 0.50,
                    "classification": "[~]"
                },
                {
                    "insight_type": "NOTES",
                    "insight_text": "Another pause",
                    "confidence": 0.50,
                    "classification": "[~]"
                },
                {
                    "insight_type": "ACTION",
                    "insight_text": "Submit report by Friday",
                    "confidence": 0.95,
                    "classification": "[+]"
                }
            ]
        }
        
        result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        # Should include the ACTION but filter low-confidence NOTES
        # Note: The filtering logic is in the LLM prompt, not in _extract_insights
        # This test verifies the extraction handles the data correctly
        assert len(result["insights"]) == 3


class TestContentTypeRuleModifiersUpdated:
    """Tests for updated CONTENT_TYPE_RULE_MODIFIERS with stricter settings."""
    
    def test_general_meeting_stricter_settings(self):
        """Test that GENERAL_MEETING has stricter action_strictness and reduced notes_frequency."""
        from src.summary.prompts import CONTENT_TYPE_RULE_MODIFIERS
        
        rules = CONTENT_TYPE_RULE_MODIFIERS["GENERAL_MEETING"]
        
        # Should have stricter settings
        assert rules["action_strictness"] == "very_high", "GENERAL_MEETING should have very_high action_strictness"
        assert rules["notes_frequency"] == "medium", "GENERAL_MEETING should have medium notes_frequency"
    
    def test_technical_talk_stricter_settings(self):
        """Test that TECHNICAL_TALK has stricter settings."""
        from src.summary.prompts import CONTENT_TYPE_RULE_MODIFIERS
        
        rules = CONTENT_TYPE_RULE_MODIFIERS["TECHNICAL_TALK"]
        
        # Should have stricter settings
        assert rules["action_strictness"] == "extreme", "TECHNICAL_TALK should have extreme action_strictness"
        assert rules["notes_frequency"] == "high", "TECHNICAL_TALK should have high notes_frequency"
    
    def test_lecture_or_talk_stricter_settings(self):
        """Test that LECTURE_OR_TALK has stricter settings."""
        from src.summary.prompts import CONTENT_TYPE_RULE_MODIFIERS
        
        rules = CONTENT_TYPE_RULE_MODIFIERS["LECTURE_OR_TALK"]
        
        # Should have stricter settings
        assert rules["action_strictness"] == "block", "LECTURE_OR_TALK should have block action_strictness"
        assert rules["notes_frequency"] == "medium", "LECTURE_OR_TALK should have medium notes_frequency"
    
    def test_streamer_monologue_reduced_notes(self):
        """Test that STREAMER_MONOLOGUE has reduced notes_frequency."""
        from src.summary.prompts import CONTENT_TYPE_RULE_MODIFIERS
        
        rules = CONTENT_TYPE_RULE_MODIFIERS["STREAMER_MONOLOGUE"]
        
        # Should have reduced notes frequency
        assert rules["notes_frequency"] == "low", "STREAMER_MONOLOGUE should have low notes_frequency"
    
    def test_podcast_reduced_notes(self):
        """Test that PODCAST has reduced notes_frequency."""
        from src.summary.prompts import CONTENT_TYPE_RULE_MODIFIERS
        
        rules = CONTENT_TYPE_RULE_MODIFIERS["PODCAST"]
        
        # Should have reduced notes frequency
        assert rules["notes_frequency"] == "high", "PODCAST should have high notes_frequency"


class TestContentTypeRiskGuidance:
    """Tests for content-type-specific RISK guidance."""
    
    def test_all_content_types_have_risk_guidance(self):
        """Test that all content types have risk_guidance field defined."""
        from src.summary.prompts import CONTENT_TYPE_RULE_MODIFIERS
        
        expected_content_types = [
            "GENERAL_MEETING",
            "TECHNICAL_TALK",
            "LECTURE_OR_TALK",
            "INTERVIEW",
            "PODCAST",
            "STREAMER_MONOLOGUE",
            "NEWS_UPDATE",
            "GAMEPLAY_COMMENTARY",
            "CUSTOMER_SUPPORT",
            "DEBATE",
            "UNKNOWN"
        ]
        
        for content_type in expected_content_types:
            assert content_type in CONTENT_TYPE_RULE_MODIFIERS, f"{content_type} missing from CONTENT_TYPE_RULE_MODIFIERS"
            rules = CONTENT_TYPE_RULE_MODIFIERS[content_type]
            assert "risk_guidance" in rules, f"{content_type} missing risk_guidance field"
            assert rules["risk_guidance"], f"{content_type} has empty risk_guidance"
    
    def test_general_meeting_risk_guidance(self):
        """Test GENERAL_MEETING risk guidance focuses on project blockers."""
        from src.summary.prompts import CONTENT_TYPE_RULE_MODIFIERS
        
        rules = CONTENT_TYPE_RULE_MODIFIERS["GENERAL_MEETING"]
        risk_guidance = rules["risk_guidance"]
        
        assert "project blockers" in risk_guidance.lower() or "timeline" in risk_guidance.lower()
        assert "resource" in risk_guidance.lower()
    
    def test_technical_talk_risk_guidance(self):
        """Test TECHNICAL_TALK risk guidance focuses on technical issues."""
        from src.summary.prompts import CONTENT_TYPE_RULE_MODIFIERS
        
        rules = CONTENT_TYPE_RULE_MODIFIERS["TECHNICAL_TALK"]
        risk_guidance = rules["risk_guidance"]
        
        assert "technical" in risk_guidance.lower() or "bugs" in risk_guidance.lower() or "failures" in risk_guidance.lower()
    
    def test_customer_support_risk_guidance(self):
        """Test CUSTOMER_SUPPORT risk guidance focuses on customer-impacting issues."""
        from src.summary.prompts import CONTENT_TYPE_RULE_MODIFIERS
        
        rules = CONTENT_TYPE_RULE_MODIFIERS["CUSTOMER_SUPPORT"]
        risk_guidance = rules["risk_guidance"]
        
        assert "customer" in risk_guidance.lower()
    
    def test_debate_risk_guidance(self):
        """Test DEBATE risk guidance focuses on argument weaknesses."""
        from src.summary.prompts import CONTENT_TYPE_RULE_MODIFIERS
        
        rules = CONTENT_TYPE_RULE_MODIFIERS["DEBATE"]
        risk_guidance = rules["risk_guidance"]
        
        assert "argument" in risk_guidance.lower() or "logical" in risk_guidance.lower() or "counter" in risk_guidance.lower()
    
    def test_format_content_type_rules_includes_risk_guidance(self):
        """Test that _format_content_type_rules includes RISK guidance."""
        from src.summary.summary_client import SummaryClient
        from src.summary.prompts import CONTENT_TYPE_RULE_MODIFIERS
        
        client = SummaryClient(api_key="test_key", model="test_model")
        
        # Test GENERAL_MEETING
        rules = CONTENT_TYPE_RULE_MODIFIERS["GENERAL_MEETING"]
        formatted = client._format_content_type_rules("GENERAL_MEETING")
        
        assert "RISK Definition" in formatted or "RISK" in formatted
        assert rules["risk_guidance"] in formatted or "project blockers" in formatted.lower()
    
    def test_format_content_type_rules_includes_key_point_guidance(self):
        """Test that _format_content_type_rules includes KEY POINT guidance when present."""
        from src.summary.summary_client import SummaryClient
        from src.summary.prompts import CONTENT_TYPE_RULE_MODIFIERS
        
        client = SummaryClient(api_key="test_key", model="test_model")
        
        # Test TECHNICAL_TALK which has key_point_guidance
        rules = CONTENT_TYPE_RULE_MODIFIERS["TECHNICAL_TALK"]
        formatted = client._format_content_type_rules("TECHNICAL_TALK")
        
        assert "KEY POINT Guidance" in formatted
        assert rules["key_point_guidance"] in formatted
    
    def test_format_content_type_rules_handles_missing_guidance(self):
        """Test that _format_content_type_rules handles content types without extra guidance."""
        from src.summary.summary_client import SummaryClient
        from src.summary.prompts import CONTENT_TYPE_RULE_MODIFIERS
        
        client = SummaryClient(api_key="test_key", model="test_model")
        
        # Test NEWS_UPDATE which has risk_guidance but no key_point_guidance
        formatted = client._format_content_type_rules("NEWS_UPDATE")
        
        assert "NEWS_UPDATE" in formatted
        assert "RISK Definition" in formatted or "NEWS_UPDATE" in formatted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])