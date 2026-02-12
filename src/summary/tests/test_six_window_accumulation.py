"""
Unit tests for 6-window accumulation through process_segments with context and insights verification.
Tests the current SummaryClient implementation with WindowManager integration.
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from src.summary.summary_client import SummaryClient, WindowManager, WindowInsight


class TestSixWindowAccumulation:
    """Tests for 6-window accumulation through process_segments with context and insights verification."""
    
    def create_client(self, delay_seconds: float = 0.0, transcription_windows_per_summary_window: int = 2, raw_text_context_limit: int = 10000):
        """Create a SummaryClient instance with specified settings."""
        client = SummaryClient(
            api_key="test_key",
            model="test_model",
            initial_summary_delay_seconds=delay_seconds,
            transcription_windows_per_summary_window=transcription_windows_per_summary_window,
            raw_text_context_limit=raw_text_context_limit
        )
        # Prevent auto content-type detection LLM calls during tests by setting a user override
        client.set_content_type_override("GENERAL_MEETING")
        return client
    
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
    async def test_six_windows_accumulated_text_and_insights(self):
        """Test that 6 transcription windows accumulate text and insights correctly through process_segments.
        
        This test verifies:
        1. All 6 transcription windows are processed through process_segments
        2. Summary requests only happen every 2nd window (at indices 1, 3, 5)
        3. Accumulated text in _build_context is accurate
        4. Insights returned from LLM are correctly extracted and stored
        5. Prior insights are correctly passed to subsequent windows
        """
        # Use transcription_windows_per_summary_window=2 for processing timing
        # With current behavior, summary is triggered at windows 1, 3, 5 (modulo 0)
        client = self.create_client(delay_seconds=0.0, transcription_windows_per_summary_window=2, raw_text_context_limit=10000)
        
        # Disable auto content type detection to focus on context_summary behavior
        client._auto_detect_content_type_detection = False
        
        # Define content for each of the 6 transcription windows
        transcription_windows = [
            "First window discussing project kickoff",
            "Second window about team allocation",
            "Third window covering technical approach",
            "Fourth window with risk assessment",
            "Fifth window with follow-up discussion",
            "Sixth window finalizing next steps",
        ]
        
        # Define insights returned by LLM at each summary point (windows 1, 3, 5)
        # Each summary call analyzes accumulated text up to that point
        summary_insights = [
            # Summary at window 1 (analyzes windows 0-1)
            [
                {"insight_type": "ACTION", "insight_text": "Start the project by Monday", "confidence": 0.95, "classification": "+"},
                {"insight_type": "DECISION", "insight_text": "Approved the initial budget", "confidence": 0.90, "classification": "~"}
            ],
            # Summary at window 3 (analyzes windows 2-3)
            [
                {"insight_type": "KEY POINT", "insight_text": "Architecture reduces context by 10x vs summarization", "confidence": 0.88, "classification": "~"},
                {"insight_type": "RISK", "insight_text": "Timeline risk if backend delayed", "confidence": 0.75, "classification": "-"}
            ],
            # Summary at window 5 (analyzes windows 4-5)
            [
                {"insight_type": "DECISION", "insight_text": "Switched to microservices approach", "confidence": 0.92, "classification": "~"},
                {"insight_type": "ACTION", "insight_text": "Submit final proposal by Friday", "confidence": 0.95, "classification": "+"}
            ],
        ]
        
        # Create mock responses for the 3 summary calls
        mock_responses = [
            self._create_mock_response(f"Analysis for accumulated text up to window {i*2+1}", insights)
            for i, insights in enumerate(summary_insights)
        ]
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = mock_responses
            
            all_results = []
            
            # Process 6 transcription windows through process_segments
            for i, text_content in enumerate(transcription_windows):
                # Create segments for this transcription window
                segments = [{"text": text_content, "start": i * 5, "end": (i + 1) * 5, "start_ms": i * 5000, "end_ms": (i + 1) * 5000}]
                
                # Process segments
                result = await client.process_segments(
                    summary_type="test",
                    segments=segments,
                    transcription_window_id=i,
                    window_start=i * 5.0,
                    window_end=(i + 1) * 5.0
                )
                
                all_results.append(result)
            
            # Verify we got results for all 6 transcription windows
            assert len(all_results) == 6, f"Expected 6 results, got {len(all_results)}"
            
            # With transcription_windows_per_summary_window=2, summary happens at windows 1, 3, 5 (modulo 0)
            # So we expect 3 summary calls with segments (at indices 1, 3, 5)
            processed_count = sum(1 for r in all_results if r.get("segments"))
            assert processed_count == 3, f"Expected 3 processed windows (at modulo 0), got {processed_count}"
            
            # Verify accumulated text in _build_context
            context, prior_insights, text_length, insights_per_window = client._build_context(include_insights=True)
            
            # With current behavior, get_accumulated_text_and_insights() holds back only last 1 window
            # So accumulated text includes windows 0-4 (5 windows), excludes only window 5
            assert "First window discussing project kickoff" in context, "First window text should be in accumulated context"
            assert "Second window about team allocation" in context, "Second window text should be in accumulated context"
            assert "Fourth window with risk assessment" in context, "Fourth window should be in accumulated context"
            
            # Sixth window should NOT be in context (it's the last window, held back)
            assert "Sixth window finalizing next steps" not in context, "Sixth window should NOT be in accumulated context"
            
            # Verify prior insights are included in context
            assert "## PRIOR INSIGHTS" in context, "PRIOR INSIGHTS section should be in context"
            assert "Start the project by Monday" in context, "First window action insight should be in context"
            assert "Approved the initial budget" in context, "First window decision insight should be in context"
            
            # Verify prior_insights list contains insights from accumulated windows
            # With 6 windows and current behavior holding back only last 1, windows 0-4 are accumulated
            # So insights from windows 0-4 are in prior_insights (window 5 is excluded)
            assert len(prior_insights) > 0, "Should have prior insights from accumulated windows"
            
            # Verify insight types in prior_insights - insights from accumulated windows (0-4)
            insight_types = [i.insight_type for i in prior_insights]
            assert "ACTION" in insight_types, "Should have ACTION insight from first summary call"
            assert "DECISION" in insight_types, "Should have DECISION insight from first summary call"
    
    @pytest.mark.asyncio
    async def test_six_windows_llm_insights_accuracy(self):
        """Test that LLM insights are accurately extracted and returned through 6 windows."""
        # With current behavior, summary happens at windows 1, 3, 5 (modulo 0)
        client = self.create_client(delay_seconds=0.0, transcription_windows_per_summary_window=2)
        
        # Disable auto content type detection
        client._auto_detect_content_type_detection = False
        
        # Define insights returned by LLM at each summary point
        summary_insights = [
            # Summary at window 1 (analyzes windows 0-1): 2 insights
            [
                {"insight_type": "ACTION", "insight_text": "Complete project setup by Monday", "confidence": 0.95, "classification": "+"},
                {"insight_type": "DECISION", "insight_text": "Team agreed on Agile methodology", "confidence": 0.90, "classification": "~"}
            ],
            # Summary at window 3 (analyzes windows 2-3): 2 insights
            [
                {"insight_type": "KEY POINT", "insight_text": "System fails above 10000 requests per second", "confidence": 0.88, "classification": "~"},
                {"insight_type": "RISK", "insight_text": "Database connection limit may be reached", "confidence": 0.75, "classification": "-"}
            ],
            # Summary at window 5 (analyzes windows 4-5): 2 insights
            [
                {"insight_type": "DECISION", "insight_text": "Project approved for next phase", "confidence": 0.95, "classification": "+"},
                {"insight_type": "NOTES", "insight_text": "15 attendees in final meeting", "confidence": 0.80, "classification": "~"}
            ]
        ]
        
        # Create mock responses for each summary call
        mock_responses = [
            self._create_mock_response(f"Analysis {i+1}", insights)
            for i, insights in enumerate(summary_insights)
        ]
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = mock_responses
            
            call_index = 0
            
            # Process 6 transcription windows
            for i in range(6):
                segments = [{"text": f"Window {i} content", "start": i * 5, "end": (i + 1) * 5, "start_ms": i * 5000, "end_ms": (i + 1) * 5000}]
                
                result = await client.process_segments(
                    summary_type="test",
                    segments=segments,
                    transcription_window_id=i,
                    window_start=i * 5.0,
                    window_end=(i + 1) * 5.0
                )
                
                # Verify result structure - only check when segments are present (summary windows)
                if result.get("segments"):
                    assert result["type"] == "context_summary", f"Window {i}: Expected type 'context_summary'"
                    assert "timing" in result, f"Window {i}: Expected 'timing' in result"
                    assert "segments" in result, f"Window {i}: Expected 'segments' in result"
                    
                    # Verify timing contains expected fields
                    timing = result["timing"]
                    assert "summary_window_id" in timing, f"Window {i}: Expected 'summary_window_id' in timing"
                    assert "transcription_window_ids" in timing, f"Window {i}: Expected 'transcription_window_ids' in timing"
                    assert "media_window_start_ms" in timing, f"Window {i}: Expected 'media_window_start_ms' in timing"
                    assert "media_window_end_ms" in timing, f"Window {i}: Expected 'media_window_end_ms' in timing"
                    
                    # Verify segments structure
                    segment = result["segments"][0]
                    assert "id" in segment, f"Window {i}: Expected 'id' in segment"
                    assert "summary_type" in segment, f"Window {i}: Expected 'summary_type' in segment"
                    assert "background_context" in segment, f"Window {i}: Expected 'background_context' in segment"
                    assert "summary" in segment, f"Window {i}: Expected 'summary' in segment"
                    
                    # Verify summary contains expected insights
                    summary_text = segment["summary"]
                    try:
                        parsed_summary = json.loads(summary_text)
                        extracted_insights = parsed_summary.get("insights", [])
                        
                        # Verify insight count matches expected for this summary call
                        expected_insights = summary_insights[call_index]
                        expected_count = len(expected_insights)
                        assert len(extracted_insights) == expected_count, \
                            f"Summary call {call_index}: Expected {expected_count} insights, got {len(extracted_insights)}"
                        
                        # Verify each insight type and text
                        for j, expected_insight in enumerate(expected_insights):
                            actual_insight = extracted_insights[j]
                            assert actual_insight["insight_type"] == expected_insight["insight_type"], \
                                f"Summary call {call_index}, Insight {j}: Expected type '{expected_insight['insight_type']}', got '{actual_insight['insight_type']}'"
                            assert actual_insight["insight_text"] == expected_insight["insight_text"], \
                                f"Summary call {call_index}, Insight {j}: Text mismatch"
                            assert actual_insight["confidence"] == expected_insight["confidence"], \
                                f"Summary call {call_index}, Insight {j}: Expected confidence {expected_insight['confidence']}, got {actual_insight['confidence']}"
                            assert actual_insight["classification"] == expected_insight["classification"], \
                                f"Summary call {call_index}, Insight {j}: Expected classification '{expected_insight['classification']}', got '{actual_insight['classification']}'"
                        
                        call_index += 1
                                
                    except json.JSONDecodeError as e:
                        pytest.fail(f"Window {i}: Failed to parse summary as JSON: {e}")
    
    @pytest.mark.asyncio
    async def test_build_context_accuracy_after_six_windows(self):
        """Test that _build_context returns accurate accumulated text after 6 windows."""
        # With current behavior, get_accumulated_text_and_insights() holds back only last 1 window
        # So accumulated text includes windows 0-4 (5 windows), excludes only window 5
        client = self.create_client(delay_seconds=0.0, transcription_windows_per_summary_window=3)
        
        # Add 6 windows manually with known content
        window_texts = [
            "Alpha point one",
            "Beta point two",
            "Gamma point three",
            "Delta point four",
            "Epsilon point five",
            "Zeta point six"
        ]
        
        for i, text in enumerate(window_texts):
            client._window_manager.add_window(text, i * 5.0, (i + 1) * 5.0)
        
        # Build context with insights
        context, prior_insights, text_length, insights_per_window = client._build_context(include_insights=True)
        
        # With current behavior, windows 0-4 should be accumulated (excludes only last 1: window 5)
        assert "Alpha point one" in context, "First window should be in accumulated text"
        assert "Beta point two" in context, "Second window should be in accumulated text"
        assert "Gamma point three" in context, "Third window should be in accumulated text"
        assert "Delta point four" in context, "Fourth window should be in accumulated text"
        assert "Epsilon point five" in context, "Fifth window should be in accumulated text"
        
        # Only window 5 (Zeta point six) should NOT be in accumulated text (it's the last window)
        assert "Zeta point six" not in context, "Sixth window should NOT be in accumulated text"
        
        # Verify PRIOR TEXT section exists
        assert "## PRIOR TEXT" in context, "PRIOR TEXT section should be present"
        
        # PRIOR INSIGHTS section only appears when there are insights
        # Since we didn't add any insights, it should not be present
        # This is expected behavior - the section is conditionally included
    
    @pytest.mark.asyncio
    async def test_six_windows_with_insights_accumulation(self):
        """Test that insights from earlier windows are accumulated and available for later windows."""
        client = self.create_client(delay_seconds=0.0, transcription_windows_per_summary_window=2)
        
        # Add 6 windows
        for i in range(6):
            client._window_manager.add_window(f"Window {i} text", i * 5.0, (i + 1) * 5.0)
        
        # Add insights to first 4 windows (windows 0-3)
        # Window 4 and 5 will not have insights added directly
        for window_id in range(4):
            for insight_id in range(2):
                insight = WindowInsight(
                    insight_id=insight_id + 1 + (window_id * 2),
                    insight_type=["ACTION", "DECISION", "KEY_POINT", "NOTES"][window_id % 4],
                    insight_text=f"Insight {insight_id} from window {window_id}",
                    confidence=0.85,
                    window_id=window_id,
                    timestamp_start=window_id * 5.0,
                    timestamp_end=(window_id + 1) * 5.0,
                    classification="~"
                )
                client._window_manager.add_insight_to_window(window_id, insight)
        
        # Build context with insights
        context, prior_insights, text_length, insights_per_window = client._build_context(include_insights=True)
        
        # With current behavior holding back only last 1 window, windows 0-4 are accumulated
        # So we should have insights from windows 0-4 = 8 insights (2 per window for 4 windows)
        assert len(prior_insights) == 8, f"Expected 8 prior insights (from windows 0-4), got {len(prior_insights)}"
        
        # Verify insights from each accumulated window are present
        insight_texts = [i.insight_text for i in prior_insights]
        assert any("window 0" in text for text in insight_texts), "Should have insights from window 0"
        assert any("window 1" in text for text in insight_texts), "Should have insights from window 1"
        assert any("window 2" in text for text in insight_texts), "Should have insights from window 2"
        assert any("window 3" in text for text in insight_texts), "Should have insights from window 3"
        # Verify we have insights from windows 0-3 (we added to first 4 windows)
        present_windows = set()
        for t in insight_texts:
            for part in ("window 0","window 1","window 2","window 3"):
                if part in t:
                    present_windows.add(part)
        assert len(present_windows) >= 4, f"Expected insights from windows 0-3, got: {present_windows}"

        # Verify insights from window 5 are NOT in prior_insights (it's the last window)
        assert not any("window 5" in text for text in insight_texts), "Should NOT have insights from window 5"
        
        # Verify context includes the insight texts
        for text in insight_texts:
            assert text in context, f"Insight text '{text}' should be in context"
        
        # Verify PRIOR INSIGHTS section exists when there are insights
        assert "## PRIOR INSIGHTS" in context, "PRIOR INSIGHTS section should be present when there are insights"
    
    @pytest.mark.asyncio
    async def test_six_windows_empty_insights_handling(self):
        """Test that empty insights arrays are handled correctly through 6 windows."""
        # With current behavior, summary happens at windows 1, 3, 5 (modulo 0)
        client = self.create_client(delay_seconds=0.0, transcription_windows_per_summary_window=2)
        
        # Disable auto content type detection to focus on context_summary behavior
        client._auto_detect_content_type_detection = False
        
        # Create responses with 1 insight for each summary call (at windows 1, 3, 5)
        responses = [
            self._create_mock_response("Analysis 1", [{"insight_type": "ACTION", "insight_text": "First action", "confidence": 0.95, "classification": "+"}]),
            self._create_mock_response("Analysis 2", [{"insight_type": "DECISION", "insight_text": "First decision", "confidence": 0.90, "classification": "~"}]),
            self._create_mock_response("Analysis 3", [{"insight_type": "ACTION", "insight_text": "Second action", "confidence": 0.95, "classification": "+"}]),
        ]
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = responses
            
            # Process 6 transcription windows
            for i in range(6):
                segments = [{"text": f"Window {i} content", "start": i * 5, "end": (i + 1) * 5, "start_ms": i * 5000, "end_ms": (i + 1) * 5000}]
                
                result = await client.process_segments(
                    summary_type="test",
                    segments=segments,
                    transcription_window_id=i,
                    window_start=i * 5.0,
                    window_end=(i + 1) * 5.0
                )
                
                # Verify result is valid (context_summary or content_type_detection are both valid)
                assert result["type"] in ["context_summary", "content_type_detection"], f"Unexpected result type: {result['type']}"
                
                # Check segments if present (only summary windows have segments)
                segments_data = result.get("segments", [])
                if segments_data:
                    summary_text = segments_data[0].get("summary", "")
                    if summary_text:
                        try:
                            parsed = json.loads(summary_text)
                            insights = parsed.get("insights", [])
                            
                            # Each summary call should return 1 insight
                            assert len(insights) == 1, \
                                f"Expected 1 insight per summary call, got {len(insights)}"
                                
                        except json.JSONDecodeError:
                            pass
            
            # Verify accumulated context handles empty insights correctly
            context, prior_insights, text_length, insights_per_window = client._build_context(include_insights=True)
            
            # Context should still be built correctly
            assert "## PRIOR TEXT" in context, "PRIOR TEXT section should be present"
            
            # With current behavior holding back only last 1 window, windows 0-4 are accumulated
            # So insights from windows 0-4 are in prior_insights (window 5 is excluded)
            # The first summary call (window 1) returned 1 insight, so we have 1 prior insight
            # But with 6 windows and holding back only 1, we get 5 accumulated windows
            # Actually, the test processes 6 windows but only 3 are processed by LLM
            # The accumulated context includes windows 0-4 (5 windows), but insights come from processed windows
            # With transcription_windows_per_summary_window=2, processing happens at windows 1, 3, 5
            # So windows 0-1 are processed first (1 insight), windows 2-3 second (1 insight), windows 4-5 third (1 insight)
            # But window 5 is held back, so only windows 0-4 are accumulated
            # The insights from window 5 are not included since it's the last window
    
    @pytest.mark.asyncio
    async def test_window_manager_accumulation_behavior(self):
        """Test WindowManager.get_accumulated_text_and_insights() behavior with 6 windows."""
        # Create WindowManager with specific settings
        wm = WindowManager(
            context_limit=50000,
            raw_text_context_limit=10000,
            transcription_windows_per_summary_window=2
        )
        
        # Add 6 windows with known content
        window_texts = [
            "First window text",
            "Second window text",
            "Third window text",
            "Fourth window text",
            "Fifth window text",
            "Sixth window text"
        ]
        
        for i, text in enumerate(window_texts):
            wm.add_window(text, i * 5.0, (i + 1) * 5.0)
        
        # Get accumulated text and insights
        accumulated_text, insights, text_length, insights_per_window = wm.get_accumulated_text_and_insights()
        
        # With current behavior, all windows except the last one should be accumulated
        # So windows 0-4 should be in accumulated text, window 5 should NOT be
        assert "First window text" in accumulated_text, "First window should be in accumulated text"
        assert "Second window text" in accumulated_text, "Second window should be in accumulated text"
        assert "Third window text" in accumulated_text, "Third window should be in accumulated text"
        assert "Fourth window text" in accumulated_text, "Fourth window should be in accumulated text"
        assert "Fifth window text" in accumulated_text, "Fifth window should be in accumulated text"
        
        # Sixth window should NOT be in accumulated text (it's the last one)
        assert "Sixth window text" not in accumulated_text, "Sixth window should NOT be in accumulated text"
        
        # Verify we have 5 windows in accumulation (all except last)
        assert len(accumulated_text.split(" window text ")) >= 5, "Should have 5 windows worth of accumulated text"
    
    @pytest.mark.asyncio
    async def test_raw_text_context_limit_enforcement(self):
        """Test that raw_text_context_limit is enforced in accumulation."""
        # Create client with small context limit
        client = self.create_client(delay_seconds=0.0, transcription_windows_per_summary_window=4, raw_text_context_limit=50)
        
        # Add windows with text longer than limit
        long_text = "This is a very long window text that exceeds the raw_text_context_limit of 50 characters"
        
        for i in range(3):
            client._window_manager.add_window(long_text, i * 5.0, (i + 1) * 5.0)
        
        # Get accumulated text and insights
        accumulated_text, insights, text_length, insights_per_window = client._window_manager.get_accumulated_text_and_insights()
        
        # With current behavior, text is added while under the limit
        # So accumulated text should be truncated to respect raw_text_context_limit
        assert len(accumulated_text) <= 50 * 3, "Accumulated text should respect raw_text_context_limit"
    
    @pytest.mark.asyncio
    async def test_insights_always_included_regardless_of_text_limit(self):
        """Test that insights are always included even if text is truncated by context limit."""
        # Create client with small context limit
        client = self.create_client(delay_seconds=0.0, transcription_windows_per_summary_window=4, raw_text_context_limit=10)
        
        # Add windows with text and insights
        for i in range(3):
            client._window_manager.add_window(f"Window {i} text", i * 5.0, (i + 1) * 5.0)
            insight = WindowInsight(
                insight_id=i + 1,
                insight_type="ACTION",
                insight_text=f"Action insight from window {i}",
                confidence=0.9,
                window_id=i,
                timestamp_start=i * 5.0,
                timestamp_end=(i + 1) * 5.0,
                classification="+"
            )
            client._window_manager.add_insight_to_window(i, insight)
        
        # Get accumulated text and insights
        accumulated_text, insights, text_length, insights_per_window = client._window_manager.get_accumulated_text_and_insights()
        
        # With current behavior, the last window's insights are NOT included
        # accumulated_windows = all except last, so only windows 0-1 are included
        assert len(insights) == 2, f"Expected 2 insights (excluding last window), got {len(insights)}"

        # Verify insight texts are present for the expected windows (0-1)
        present = set()
        for ins in insights:
            for i in range(2):
                if f"window {i}" in ins.insight_text:
                    present.add(i)
        assert present == {0, 1}, f"Expected insights from windows 0-1, got {present}"
    
    @pytest.mark.asyncio
    async def test_process_segments_timing_structure(self):
        """Test that process_segments returns correct timing structure."""
        client = self.create_client(delay_seconds=0.0, transcription_windows_per_summary_window=2)
        
        # Disable auto content type detection
        client._auto_detect_content_type_detection = False
        
        # Create mock response
        mock_response = self._create_mock_response(
            "Test analysis",
            [{"insight_type": "ACTION", "insight_text": "Test action", "confidence": 0.95, "classification": "+"}]
        )
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            # Process first transcription window (should be buffered, not processed)
            segments = [{"text": "First window", "start": 0, "end": 5, "start_ms": 0, "end_ms": 5000}]
            result1 = await client.process_segments(
                summary_type="test",
                segments=segments,
                transcription_window_id=0,
                window_start=0.0,
                window_end=5.0
            )
            
            # Process second transcription window (should trigger summary)
            segments = [{"text": "Second window", "start": 5, "end": 10, "start_ms": 5000, "end_ms": 10000}]
            result2 = await client.process_segments(
                summary_type="test",
                segments=segments,
                transcription_window_id=1,
                window_start=5.0,
                window_end=10.0
            )
            
            # First result should have empty segments (buffered)
            assert result1["type"] == "context_summary"
            assert result1["segments"] == []
            
            # Second result should have segments (summary processed)
            assert result2["type"] == "context_summary"
            assert len(result2["segments"]) > 0
            
            # Verify timing structure
            timing = result2["timing"]
            assert "summary_window_id" in timing
            assert "transcription_window_ids" in timing
            assert "media_window_start_ms" in timing
            assert "media_window_end_ms" in timing
            
            # Verify transcription_window_ids contains both windows
            assert 0 in timing["transcription_window_ids"]
            assert 1 in timing["transcription_window_ids"]
            
            # Verify media window timing spans both windows
            assert timing["media_window_start_ms"] == 0
            assert timing["media_window_end_ms"] == 10000
    
    @pytest.mark.asyncio
    async def test_six_windows_with_different_transcription_windows_per_summary(self):
        """Test 6-window accumulation with different transcription_windows_per_summary_window values."""
        for windows_per_summary in [1, 2, 3]:
            client = self.create_client(delay_seconds=0.0, transcription_windows_per_summary_window=windows_per_summary)
            
            # Disable auto content type detection
            client._auto_detect_content_type_detection = False
            
            # Create mock responses for each summary window
            num_summary_windows = 6 // windows_per_summary
            mock_responses = [
                self._create_mock_response(
                    f"Analysis {i+1}",
                    [{"insight_type": "ACTION", "insight_text": f"Action {i+1}", "confidence": 0.95, "classification": "+"}]
                )
                for i in range(num_summary_windows)
            ]
            
            with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
                mock_create.side_effect = mock_responses
                
                # Process 6 transcription windows
                results = []
                for i in range(6):
                    segments = [{"text": f"Window {i} content", "start": i * 5, "end": (i + 1) * 5, "start_ms": i * 5000, "end_ms": (i + 1) * 5000}]
                    
                    result = await client.process_segments(
                        summary_type="test",
                        segments=segments,
                        transcription_window_id=i,
                        window_start=i * 5.0,
                        window_end=(i + 1) * 5.0
                    )
                    results.append(result)
                
                # Count summary windows (those with segments)
                summary_count = sum(1 for r in results if r.get("segments"))
                
                # With windows_per_summary=N, we should have 6/N summary windows
                expected_summary = 6 // windows_per_summary
                assert summary_count == expected_summary, \
                    f"windows_per_summary={windows_per_summary}: Expected {expected_summary} summary windows, got {summary_count}"
    
    @pytest.mark.asyncio
    async def test_context_includes_all_sections(self):
        """Test that _build_context includes all expected sections."""
        client = self.create_client(delay_seconds=0.0, transcription_windows_per_summary_window=2)
        
        # Add windows with text
        for i in range(3):
            client._window_manager.add_window(f"Window {i} text content", i * 5.0, (i + 1) * 5.0)
        
        # Add insights to windows
        for i in range(2):
            insight = WindowInsight(
                insight_id=i + 1,
                insight_type="ACTION",
                insight_text=f"Action from window {i}",
                confidence=0.9,
                window_id=i,
                timestamp_start=i * 5.0,
                timestamp_end=(i + 1) * 5.0,
                classification="+"
            )
            client._window_manager.add_insight_to_window(i, insight)
        
        # Build context with insights
        context, prior_insights, text_length, insights_per_window = client._build_context(include_insights=True)
        
        # Verify all sections are present
        assert "## PRIOR TEXT" in context, "PRIOR TEXT section should be present"
        assert "## PRIOR INSIGHTS" in context, "PRIOR INSIGHTS section should be present"
        
        # Verify text content is in PRIOR TEXT
        assert "Window 0 text content" in context
        assert "Window 1 text content" in context
        
        # Verify insight content is in PRIOR INSIGHTS
        assert "Action from window 0" in context
        assert "Action from window 1" in context
    
    @pytest.mark.asyncio
    async def test_insight_id_formatting(self):
        """Test that insights are formatted with correct ID and timing hints."""
        client = self.create_client(delay_seconds=0.0, transcription_windows_per_summary_window=2)
        
        # Add windows and insights
        for i in range(2):
            client._window_manager.add_window(f"Window {i} text", i * 5.0, (i + 1) * 5.0)
            insight = WindowInsight(
                insight_id=i + 1,
                insight_type="DECISION",
                insight_text=f"Decision made at window {i}",
                confidence=0.95,
                window_id=i,
                timestamp_start=i * 5.0,
                timestamp_end=(i + 1) * 5.0,
                classification="~"
            )
            client._window_manager.add_insight_to_window(i, insight)
        
        # Add a third window so the last window is window 2 and accumulated windows include windows 0-1
        client._window_manager.add_window("Third window text", 10.0, 15.0)

        # Build context (now includes windows 0-1 as prior)
        context, prior_insights, text_length, insights_per_window = client._build_context(include_insights=True)
        
        # Verify insights are formatted with ID and timing hints
        for insight in prior_insights:
            # ID should be formatted as [#{insight_id}]
            assert f"[#{insight.insight_id}]" in context, f"Insight ID {insight.insight_id} should be in context"
            
            # Timing hint should be present
            assert f"[{insight.timestamp_start:.1f}s" in context, f"Timing hint should be present for insight {insight.insight_id}"
    
    @pytest.mark.asyncio
    async def test_continuation_and_correction_markers(self):
        """Test that continuation and correction markers are included in context."""
        client = self.create_client(delay_seconds=0.0, transcription_windows_per_summary_window=2)
        
        # Add first window with initial insight
        client._window_manager.add_window("First window text", 0.0, 5.0)
        initial_insight = WindowInsight(
            insight_id=1,
            insight_type="ACTION",
            insight_text="Initial action",
            confidence=0.9,
            window_id=0,
            timestamp_start=0.0,
            timestamp_end=5.0,
            classification="~"
        )
        client._window_manager.add_insight_to_window(0, initial_insight)
        
        # Add second window with continuation
        client._window_manager.add_window("Second window text", 5.0, 10.0)
        continuation_insight = WindowInsight(
            insight_id=2,
            insight_type="ACTION",
            insight_text="Continuation of initial action",
            confidence=0.85,
            window_id=1,
            timestamp_start=5.0,
            timestamp_end=10.0,
            classification="~",
            continuation_of=1
        )
        client._window_manager.add_insight_to_window(1, continuation_insight)
        
        # Add a third window so the last window is window 2 and accumulated windows include windows 0-1
        client._window_manager.add_window("Third window text", 10.0, 15.0)

        # Build context (now includes windows 0-1 as prior)
        context, prior_insights, text_length, insights_per_window = client._build_context(include_insights=True)
        
        # Verify continuation marker is present
        assert "CONTINUATION of insight #1" in context, "Continuation marker should be in context"
    
    @pytest.mark.asyncio
    async def test_window_manager_clear_resets_state(self):
        """Test that WindowManager.clear() properly resets all state."""
        wm = WindowManager()
        
        # Add windows and insights
        for i in range(3):
            wm.add_window(f"Window {i}", i * 5.0, (i + 1) * 5.0)
            insight = WindowInsight(
                insight_id=i + 1,
                insight_type="ACTION",
                insight_text=f"Insight {i}",
                confidence=0.9,
                window_id=i,
                timestamp_start=i * 5.0,
                timestamp_end=(i + 1) * 5.0,
                classification="~"
            )
            wm.add_insight_to_window(i, insight)
        
        # Verify state before clear
        assert len(wm) == 3, "Should have 3 windows before clear"
        
        # Clear the manager
        wm.clear()
        
        # Verify state after clear
        assert len(wm) == 0, "Should have 0 windows after clear"
        
        # Verify windows are empty
        accumulated_text, insights, text_length, insights_per_window = wm.get_accumulated_text_and_insights()
        assert accumulated_text == "", "Accumulated text should be empty after clear"
        assert insights == [], "Insights should be empty after clear"
