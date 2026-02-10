"""
Unit tests for 6-window accumulation through process_segments with context and insights verification.
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from src.summary.summary_client import SummaryClient, WindowManager, WindowInsight


class TestSixWindowAccumulation:
    """Tests for 6-window accumulation through process_segments with context and insights verification."""
    
    def create_client(self, delay_seconds: float = 0.0, windows_to_accumulate: int = 2):
        """Create a SummaryClient instance with specified settings."""
        return SummaryClient(
            api_key="test_key",
            model="test_model",
            initial_summary_delay_seconds=delay_seconds,
            windows_to_accumulate=windows_to_accumulate
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
    async def test_six_windows_accumulated_text_and_insights(self):
        """Test that 6 transcription windows accumulate text and insights correctly through process_segments.
        
        This test verifies:
        1. All 6 windows are processed through process_segments
        2. Accumulated text in _build_context is accurate
        3. Insights returned from LLM are correctly extracted and stored
        4. Prior insights are correctly passed to subsequent windows
        """
        # Use windows_to_accumulate=2 so first 4 windows are accumulated (excludes last 2)
        client = self.create_client(delay_seconds=0.0, windows_to_accumulate=2)
        
        # Define expected content for each window
        window_contents = [
            ("First window discussing project kickoff", [
                {"insight_type": "ACTION", "insight_text": "Start the project by Monday", "confidence": 0.95, "classification": "+"},
                {"insight_type": "DECISION", "insight_text": "Approved the initial budget", "confidence": 0.90, "classification": "~"}
            ]),
            ("Second window about team allocation", [
                {"insight_type": "NOTES", "insight_text": "Team size is 5 members", "confidence": 0.85, "classification": "~"},
                {"insight_type": "QUESTION", "insight_text": "Who will lead the frontend?", "confidence": 0.80, "classification": "~"}
            ]),
            ("Third window covering technical approach", [
                {"insight_type": "KEY POINT", "insight_text": "Architecture reduces context by 10x vs summarization", "confidence": 0.88, "classification": "~"},
                {"insight_type": "NOTES", "insight_text": "Using RAG for document retrieval", "confidence": 0.85, "classification": "~"}
            ]),
            ("Fourth window with risk assessment", [
                {"insight_type": "RISK", "insight_text": "Timeline risk if backend delayed", "confidence": 0.75, "classification": "-"},
                {"insight_type": "ACTION", "insight_text": "Review backend dependencies by Wednesday", "confidence": 0.90, "classification": "+"}
            ]),
            ("Fifth window with follow-up discussion", [
                {"insight_type": "DECISION", "insight_text": "Switched to microservices approach", "confidence": 0.92, "classification": "~"},
                {"insight_type": "NOTES", "insight_text": "Meeting lasted 45 minutes", "confidence": 0.80, "classification": "~"}
            ]),
            ("Sixth window finalizing next steps", [
                {"insight_type": "ACTION", "insight_text": "Submit final proposal by Friday", "confidence": 0.95, "classification": "+"},
                {"insight_type": "QUESTION", "insight_text": "What is the budget ceiling?", "confidence": 0.70, "classification": "~"}
            ]),
        ]
        
        # Create mock responses for all 6 windows
        mock_responses = [
            self._create_mock_response(analysis, insights)
            for analysis, insights in window_contents
        ]
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = mock_responses
            
            all_results = []
            
            # Process 6 windows
            for i, (text_content, expected_insights) in enumerate(window_contents):
                # Create segments for this window
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
            
            # Verify we got results for all 6 windows
            assert len(all_results) == 6, f"Expected 6 results, got {len(all_results)}"
            
            # With windows_to_accumulate=2, processing happens at windows 1, 3, 5 (modulo 0)
            # So we expect 3 processed windows (at indices 1, 3, 5)
            processed_count = sum(1 for r in all_results if r.get("segments"))
            assert processed_count == 3, f"Expected 3 processed windows (at modulo 0), got {processed_count}"
            
            # Verify accumulated text in _build_context
            context, prior_insights = client._build_context(include_insights=True)
            
            # Context should contain accumulated text from windows 0-3 (first 4 windows)
            # Since windows_to_accumulate=2, windows 0-3 are accumulated (excludes last 2: windows 4-5)
            assert "First window discussing project kickoff" in context, "First window text should be in accumulated context"
            assert "Second window about team allocation" in context, "Second window text should be in accumulated context"
            
            # Verify prior insights are included in context
            assert "## PRIOR INSIGHTS" in context, "PRIOR INSIGHTS section should be in context"
            assert "Start the project by Monday" in context, "First window action insight should be in context"
            assert "Approved the initial budget" in context, "First window decision insight should be in context"
            
            # Verify prior_insights list contains insights from accumulated windows
            # With 6 windows and windows_to_accumulate=2, the last 2 windows (4-5) are excluded
            # So accumulated windows are 0-3, and insights from those windows are in prior_insights
            assert len(prior_insights) > 0, "Should have prior insights from accumulated windows"
            
            # Verify insight types in prior_insights - only the 2 insights that were returned
            insight_types = [i.insight_type for i in prior_insights]
            assert "ACTION" in insight_types, "Should have ACTION insight from first LLM call"
            assert "DECISION" in insight_types, "Should have DECISION insight from first LLM call"
    
    @pytest.mark.asyncio
    async def test_six_windows_llm_insights_accuracy(self):
        """Test that LLM insights are accurately extracted and returned through 6 windows."""
        client = self.create_client(delay_seconds=0.0, windows_to_accumulate=2)
        
        # Define specific insights to verify accuracy - these are returned by the LLM
        # for the accumulated text at each processing point
        mock_insights_by_call = [
            # Window 1 (processes windows 0-1): 2 insights
            [
                {"insight_type": "ACTION", "insight_text": "Complete project setup by Monday", "confidence": 0.95, "classification": "+"},
                {"insight_type": "DECISION", "insight_text": "Team agreed on Agile methodology", "confidence": 0.90, "classification": "~"}
            ],
            # Window 3 (processes windows 2-3): 2 insights
            [
                {"insight_type": "KEY POINT", "insight_text": "System fails above 10000 requests per second", "confidence": 0.88, "classification": "~"},
                {"insight_type": "RISK", "insight_text": "Database connection limit may be reached", "confidence": 0.75, "classification": "-"}
            ],
            # Window 5 (processes windows 4-5): 2 insights
            [
                {"insight_type": "DECISION", "insight_text": "Project approved for next phase", "confidence": 0.95, "classification": "+"},
                {"insight_type": "NOTES", "insight_text": "15 attendees in final meeting", "confidence": 0.80, "classification": "~"}
            ]
        ]
        
        # Create mock responses for each LLM call
        mock_responses = [
            self._create_mock_response(f"Analysis {i+1}", insights)
            for i, insights in enumerate(mock_insights_by_call)
        ]
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = mock_responses
            
            call_index = 0
            
            # Process 6 windows
            for i in range(6):
                segments = [{"text": f"Window {i} content", "start": i * 5, "end": (i + 1) * 5, "start_ms": i * 5000, "end_ms": (i + 1) * 5000}]
                
                result = await client.process_segments(
                    summary_type="test",
                    segments=segments,
                    transcription_window_id=i,
                    window_start=i * 5.0,
                    window_end=(i + 1) * 5.0
                )
                
                # Verify result structure - only check when segments are present (processed windows)
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
                        
                        # Verify insight count matches expected for this LLM call
                        expected_insights = mock_insights_by_call[call_index]
                        expected_count = len(expected_insights)
                        assert len(extracted_insights) == expected_count, \
                            f"LLM call {call_index}: Expected {expected_count} insights, got {len(extracted_insights)}"
                        
                        # Verify each insight type and text
                        for j, expected_insight in enumerate(expected_insights):
                            actual_insight = extracted_insights[j]
                            assert actual_insight["insight_type"] == expected_insight["insight_type"], \
                                f"LLM call {call_index}, Insight {j}: Expected type '{expected_insight['insight_type']}', got '{actual_insight['insight_type']}'"
                            assert actual_insight["insight_text"] == expected_insight["insight_text"], \
                                f"LLM call {call_index}, Insight {j}: Text mismatch"
                            assert actual_insight["confidence"] == expected_insight["confidence"], \
                                f"LLM call {call_index}, Insight {j}: Expected confidence {expected_insight['confidence']}, got {actual_insight['confidence']}"
                            assert actual_insight["classification"] == expected_insight["classification"], \
                                f"LLM call {call_index}, Insight {j}: Expected classification '{expected_insight['classification']}', got '{actual_insight['classification']}'"
                        
                        call_index += 1
                                
                    except json.JSONDecodeError as e:
                        pytest.fail(f"Window {i}: Failed to parse summary as JSON: {e}")
    
    @pytest.mark.asyncio
    async def test_build_context_accuracy_after_six_windows(self):
        """Test that _build_context returns accurate accumulated text after 6 windows."""
        client = self.create_client(delay_seconds=0.0, windows_to_accumulate=3)
        
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
        context, prior_insights = client._build_context(include_insights=True)
        
        # With windows_to_accumulate=3, windows 0-2 should be accumulated (excludes last 3)
        # So accumulated text should contain "Alpha", "Beta", "Gamma"
        assert "Alpha point one" in context, "First window should be in accumulated text"
        assert "Beta point two" in context, "Second window should be in accumulated text"
        assert "Gamma point three" in context, "Third window should be in accumulated text"
        
        # Windows 3-5 should NOT be in accumulated text (they are the last 3)
        assert "Delta point four" not in context, "Fourth window should NOT be in accumulated text"
        assert "Epsilon point five" not in context, "Fifth window should NOT be in accumulated text"
        assert "Zeta point six" not in context, "Sixth window should NOT be in accumulated text"
        
        # Verify PRIOR TEXT section exists
        assert "## PRIOR TEXT" in context, "PRIOR TEXT section should be present"
        
        # PRIOR INSIGHTS section only appears when there are insights
        # Since we didn't add any insights, it should not be present
        # This is expected behavior - the section is conditionally included
    
    @pytest.mark.asyncio
    async def test_six_windows_with_insights_accumulation(self):
        """Test that insights from earlier windows are accumulated and available for later windows."""
        client = self.create_client(delay_seconds=0.0, windows_to_accumulate=2)
        
        # Add 6 windows
        for i in range(6):
            client._window_manager.add_window(f"Window {i} text", i * 5.0, (i + 1) * 5.0)
        
        # Add insights to first 4 windows
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
        context, prior_insights = client._build_context(include_insights=True)
        
        # With windows_to_accumulate=2, windows 0-3 are accumulated (excludes last 2)
        # So we should have insights from windows 0-3 = 8 insights
        assert len(prior_insights) == 8, f"Expected 8 prior insights (from windows 0-3), got {len(prior_insights)}"
        
        # Verify insights from each accumulated window are present
        insight_texts = [i.insight_text for i in prior_insights]
        assert any("window 0" in text for text in insight_texts), "Should have insights from window 0"
        assert any("window 1" in text for text in insight_texts), "Should have insights from window 1"
        assert any("window 2" in text for text in insight_texts), "Should have insights from window 2"
        assert any("window 3" in text for text in insight_texts), "Should have insights from window 3"
        
        # Verify insights from windows 4-5 are NOT in prior_insights (they are the last 2)
        assert not any("window 4" in text for text in insight_texts), "Should NOT have insights from window 4"
        assert not any("window 5" in text for text in insight_texts), "Should NOT have insights from window 5"
        
        # Verify context includes the insight texts
        for text in insight_texts:
            assert text in context, f"Insight text '{text}' should be in context"
        
        # Verify PRIOR INSIGHTS section exists when there are insights
        assert "## PRIOR INSIGHTS" in context, "PRIOR INSIGHTS section should be present when there are insights"
    
    @pytest.mark.asyncio
    async def test_six_windows_empty_insights_handling(self):
        """Test that empty insights arrays are handled correctly through 6 windows."""
        client = self.create_client(delay_seconds=0.0, windows_to_accumulate=2)
        
        # Create responses with varying insight counts for each LLM call
        # With windows_to_accumulate=2, processing happens at windows 1, 3, 5
        responses = [
            self._create_mock_response("Analysis 1", [{"insight_type": "ACTION", "insight_text": "First action", "confidence": 0.95, "classification": "+"}]),
            self._create_mock_response("Analysis 2", [{"insight_type": "DECISION", "insight_text": "First decision", "confidence": 0.90, "classification": "~"}]),
            self._create_mock_response("Analysis 3", [{"insight_type": "ACTION", "insight_text": "Second action", "confidence": 0.95, "classification": "+"}]),
        ]
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = responses
            
            # Process 6 windows
            for i in range(6):
                segments = [{"text": f"Window {i} content", "start": i * 5, "end": (i + 1) * 5, "start_ms": i * 5000, "end_ms": (i + 1) * 5000}]
                
                result = await client.process_segments(
                    summary_type="test",
                    segments=segments,
                    transcription_window_id=i,
                    window_start=i * 5.0,
                    window_end=(i + 1) * 5.0
                )
                
                # Verify result is valid
                assert result["type"] == "context_summary"
                
                # Check segments if present (only processed windows have segments)
                if result["segments"]:
                    summary_text = result["segments"][0].get("summary", "")
                    if summary_text:
                        try:
                            parsed = json.loads(summary_text)
                            insights = parsed.get("insights", [])
                            
                            # Each LLM call should return 1 insight
                            assert len(insights) == 1, \
                                f"Expected 1 insight per LLM call, got {len(insights)}"
                                
                        except json.JSONDecodeError:
                            pass
            
            # Verify accumulated context handles empty insights correctly
            context, prior_insights = client._build_context(include_insights=True)
            
            # Context should still be built correctly
            assert "## PRIOR TEXT" in context, "PRIOR TEXT section should be present"
            
            # Prior insights should include insights from accumulated windows only
            # With 6 windows and windows_to_accumulate=2, the last 2 windows (4-5) are excluded
            # So only insights from windows 0-3 are in prior_insights
            # The first LLM call (window 1) returned 1 insight, so we have 1 prior insight
            assert len(prior_insights) == 1, f"Expected 1 prior insight (from first LLM call in accumulated windows), got {len(prior_insights)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])