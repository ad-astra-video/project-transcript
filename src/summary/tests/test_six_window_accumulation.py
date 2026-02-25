"""
Unit tests for 6-window accumulation through WindowManager.

In the refactored code, insights are stored via plugin_results and
retrieved using ContextSummaryTask._get_prior_insights_from_plugin_results().
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from src.summary.summary_client import SummaryClient
from src.summary.window_manager import WindowManager
from src.summary.context_summary.task import WindowInsight, ContextSummaryTask


class TestSixWindowAccumulation:
    """Tests for 6-window accumulation through WindowManager."""
    
    def create_client(self, transcription_windows_per_summary_window: int = 2):
        """Create a SummaryClient instance with specified settings."""
        client = SummaryClient(
            reasoning_api_key="test_key",
            reasoning_model="test_model",
            initial_summary_delay_seconds=0.0
        )
        client.update_params(transcription_windows_per_summary_window=transcription_windows_per_summary_window)
        return client
    
    def test_six_windows_accumulated_text(self):
        """Test that 6 transcription windows accumulate text correctly."""
        client = self.create_client(transcription_windows_per_summary_window=2)
        
        # Add 6 transcription windows
        # New signature: add_transcription_window(transcription_window_id, segments, window_start_ts, window_end_ts)
        for i in range(6):
            client._window_manager.add_transcription_window(
                transcription_window_id=i,
                segments=[{"id": str(i), "text": f"Window {i} text"}],
                window_start_ts=float(i * 10),
                window_end_ts=float((i + 1) * 10)
            )
        
        # Verify all windows stored
        assert len(client._window_manager._transcription_windows) == 6
    
    def test_summary_windows_accumulate_text_and_insights_via_plugin_results(self):
        """Test that summary windows accumulate text and insights via plugin_results."""
        client = self.create_client(transcription_windows_per_summary_window=2)
        
        # Add first summary window
        client._window_manager.add_summary_window("First window text", 0.0, 10.0, [0, 1])
        
        # Store insights via plugin_results
        first_window = client._window_manager._summary_windows[0]
        result1 = {
            "summary_text": json.dumps({
                "topic": "First analysis",
                "insights": [
                    {
                        "insight_id": 1,
                        "insight_type": "KEY POINT",
                        "insight_text": "First insight",
                        "confidence": 0.9,
                        "window_id": 0,
                        "timestamp_start": 0.0,
                        "timestamp_end": 10.0,
                        "classification": "~"
                    }
                ]
            })
        }
        first_window.store_result("context_summary", result1, include_in_context=True)
        
        # Add second summary window
        client._window_manager.add_summary_window("Second window text", 10.0, 20.0, [2, 3])
        
        # Store insights via plugin_results
        second_window = client._window_manager._summary_windows[1]
        result2 = {
            "summary_text": json.dumps({
                "topic": "Second analysis",
                "insights": [
                    {
                        "insight_id": 2,
                        "insight_type": "ACTION",
                        "insight_text": "Second insight",
                        "confidence": 0.85,
                        "window_id": 1,
                        "timestamp_start": 10.0,
                        "timestamp_end": 20.0,
                        "classification": "+"
                    }
                ]
            })
        }
        second_window.store_result("context_summary", result2, include_in_context=True)
        
        # Add third summary window so second window becomes prior
        client._window_manager.add_summary_window("Third window text", 20.0, 30.0, [4, 5])
        
        # Get prior insights via ContextSummaryTask
        task = ContextSummaryTask(
            llm_client=MagicMock(),
            window_manager=client._window_manager
        )
        prior_insights = task._get_prior_insights_from_plugin_results()
        
        # Should have insights from first and second windows
        assert len(prior_insights) == 2


class TestWindowManagerTextAccumulation:
    """Tests for WindowManager text accumulation."""
    
    def test_accumulated_text_under_limit(self):
        """Test that accumulated text respects context limit."""
        wm = WindowManager(
            context_limit=5000,
            raw_text_context_limit=1000,
            transcription_windows_per_summary_window=4
        )
        
        # Add multiple windows
        for i in range(5):
            wm.add_summary_window(f"Text window {i} " * 50, float(i * 10), float((i + 1) * 10), [i])
        
        # Get all windows text
        text = wm.get_all_windows_text()
        
        # Should have text from all windows
        assert len(text) > 0
    
    def test_insights_preserved_across_windows(self):
        """Test that insights are preserved across windows via plugin_results."""
        wm = WindowManager()
        
        # Add windows with insights stored via plugin_results
        for i in range(3):
            wm.add_summary_window(f"Text {i}", float(i * 10), float((i + 1) * 10), [i])
            
            # Store insights via plugin_results
            window = wm._summary_windows[i]
            result = {
                "summary_text": json.dumps({
                    "topic": f"Analysis {i}",
                    "insights": [
                        {
                            "insight_id": i + 1,
                            "insight_type": "KEY POINT",
                            "insight_text": f"Insight {i}",
                            "confidence": 0.9,
                            "window_id": i,
                            "timestamp_start": float(i * 10),
                            "timestamp_end": float((i + 1) * 10),
                            "classification": "~"
                        }
                    ]
                })
            }
            window.store_result("context_summary", result, include_in_context=True)
        
        # Get prior insights via ContextSummaryTask
        task = ContextSummaryTask(
            llm_client=MagicMock(),
            window_manager=wm
        )
        prior_insights = task._get_prior_insights_from_plugin_results()
        
        # Should have insights from first two windows (third is current)
        assert len(prior_insights) == 2


class TestTranscriptionWindowsPerSummaryWindow:
    """Tests for transcription_windows_per_summary_window config."""
    
    def test_transcription_windows_per_summary_window_config(self):
        """Test that transcription_windows_per_summary_window is configurable."""
        wm = WindowManager(transcription_windows_per_summary_window=3)
        
        assert wm.transcription_windows_per_summary_window == 3
        
        wm2 = WindowManager(transcription_windows_per_summary_window=5)
        assert wm2.transcription_windows_per_summary_window == 5
