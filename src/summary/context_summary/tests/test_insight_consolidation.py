"""
Tests for insight consolidation feature.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

from src.summary.context_summary.task import (
    ContextSummaryTask,
    InsightType,
    ClassificationField,
    WindowInsight
)
from src.summary.context_summary.prompts import InsightConsolidationResponse


class TestFindDuplicateInsightTypes:
    """Tests for _find_duplicate_insight_types method."""
    
    @pytest.fixture
    def task(self):
        """Create a ContextSummaryTask instance for testing."""
        return ContextSummaryTask()
    
    def test_no_duplicates_returns_empty(self, task):
        """When no duplicate types, should return empty dict."""
        insights = [
            WindowInsight(insight_type=InsightType.ACTION, window_id=1, timestamp_start=0.0, timestamp_end=1.0,
                insight_text="Do something",
                confidence=0.9,
                classification=ClassificationField.POSITIVE,
                continuation_of=None,
                correction_of=None
            ),
            WindowInsight(insight_type=InsightType.DECISION, window_id=1, timestamp_start=0.0, timestamp_end=1.0,
                insight_text="Decided X",
                confidence=0.8,
                classification=ClassificationField.NEUTRAL,
                continuation_of=None,
                correction_of=None
            )
        ]
        
        result = task._find_duplicate_insight_types(insights)
        
        assert result == {}
    
    def test_duplicates_found(self, task):
        """When duplicate types exist, should return grouped dict."""
        insights = [
            WindowInsight(insight_type=InsightType.KEY_POINT, window_id=1, timestamp_start=0.0, timestamp_end=1.0,
                insight_text="First key point",
                confidence=0.9,
                classification=ClassificationField.POSITIVE,
                continuation_of=None,
                correction_of=None
            ),
            WindowInsight(insight_type=InsightType.KEY_POINT, window_id=1, timestamp_start=0.0, timestamp_end=1.0,
                insight_text="Second key point",
                confidence=0.8,
                classification=ClassificationField.NEUTRAL,
                continuation_of=None,
                correction_of=None
            ),
            WindowInsight(insight_type=InsightType.ACTION, window_id=1, timestamp_start=0.0, timestamp_end=1.0,
                insight_text="Take action",
                confidence=0.7,
                classification=ClassificationField.POSITIVE,
                continuation_of=None,
                correction_of=None
            )
        ]
        
        result = task._find_duplicate_insight_types(insights)
        
        assert InsightType.KEY_POINT.value in result
        assert len(result[InsightType.KEY_POINT.value]) == 2
        assert InsightType.ACTION.value not in result
    
    def test_continuations_excluded(self, task):
        """Insights with continuation_of should be excluded from duplicate check."""
        insights = [
            WindowInsight(insight_type=InsightType.KEY_POINT, window_id=1, timestamp_start=0.0, timestamp_end=1.0,
                insight_text="Original key point",
                confidence=0.9,
                classification=ClassificationField.POSITIVE,
                continuation_of=None,
                correction_of=None
            ),
            WindowInsight(insight_type=InsightType.KEY_POINT, window_id=1, timestamp_start=0.0, timestamp_end=1.0,
                insight_text="Continuation of key point",
                confidence=0.8,
                classification=ClassificationField.NEUTRAL,
                continuation_of=1,  # This should be excluded
                correction_of=None
            )
        ]
        
        result = task._find_duplicate_insight_types(insights)
        
        # Should be empty because continuation is excluded
        assert result == {}
    
    def test_corrections_excluded(self, task):
        """Insights with correction_of should be excluded from duplicate check."""
        insights = [
            WindowInsight(insight_type=InsightType.KEY_POINT, window_id=1, timestamp_start=0.0, timestamp_end=1.0,
                insight_text="Original key point",
                confidence=0.9,
                classification=ClassificationField.POSITIVE,
                continuation_of=None,
                correction_of=None
            ),
            WindowInsight(insight_type=InsightType.KEY_POINT, window_id=1, timestamp_start=0.0, timestamp_end=1.0,
                insight_text="Correction of key point",
                confidence=0.8,
                classification=ClassificationField.NEUTRAL,
                continuation_of=None,
                correction_of=1  # This should be excluded
            )
        ]
        
        result = task._find_duplicate_insight_types(insights)
        
        # Should be empty because correction is excluded
        assert result == {}
    
    def test_mixed_continuations_and_duplicates(self, task):
        """Should find duplicates when some insights are continuations."""
        insights = [
            WindowInsight(insight_type=InsightType.KEY_POINT, window_id=1, timestamp_start=0.0, timestamp_end=1.0,
                insight_text="Original key point",
                confidence=0.9,
                classification=ClassificationField.POSITIVE,
                continuation_of=None,
                correction_of=None
            ),
            WindowInsight(insight_type=InsightType.KEY_POINT, window_id=1, timestamp_start=0.0, timestamp_end=1.0,
                insight_text="Second key point",
                confidence=0.8,
                classification=ClassificationField.NEUTRAL,
                continuation_of=None,
                correction_of=None
            ),
            WindowInsight(insight_type=InsightType.KEY_POINT, window_id=1, timestamp_start=0.0, timestamp_end=1.0,
                insight_text="Continuation of key point",
                confidence=0.7,
                classification=ClassificationField.POSITIVE,
                continuation_of=1,  # This should be excluded
                correction_of=None
            )
        ]
        
        result = task._find_duplicate_insight_types(insights)
        
        # Should find 2 non-continuation KEY_POINT insights
        assert InsightType.KEY_POINT.value in result
        assert len(result[InsightType.KEY_POINT.value]) == 2


class TestConsolidateSimilarInsights:
    """Tests for _consolidate_similar_insights method."""
    
    @pytest.fixture
    def mock_rapid_llm(self):
        """Create a mock rapid LLM client."""
        mock = AsyncMock()
        return mock
    
    @pytest.fixture
    def task_with_rapid(self, mock_rapid_llm):
        """Create task with rapid LLM client."""
        return ContextSummaryTask(rapid_llm_client=mock_rapid_llm)
    
    @pytest.mark.asyncio
    async def test_no_rapid_client_returns_unchanged(self):
        """When no rapid LLM client, should return insights unchanged."""
        task = ContextSummaryTask(rapid_llm_client=None)
        
        insights = [
            WindowInsight(insight_type=InsightType.KEY_POINT, window_id=1, timestamp_start=0.0, timestamp_end=1.0,
                insight_text="Test",
                confidence=0.9,
                classification=ClassificationField.POSITIVE,
                continuation_of=None,
                correction_of=None
            )
        ]
        
        result = await task._consolidate_similar_insights(insights)
        
        assert len(result) == 1
        assert result[0].insight_text == "Test"
    
    @pytest.mark.asyncio
    async def test_single_insight_returns_unchanged(self, task_with_rapid):
        """When only one insight, should return unchanged."""
        insights = [
            WindowInsight(insight_type=InsightType.KEY_POINT, window_id=1, timestamp_start=0.0, timestamp_end=1.0,
                insight_text="Only one insight",
                confidence=0.9,
                classification=ClassificationField.POSITIVE,
                continuation_of=None,
                correction_of=None
            )
        ]
        
        result = await task_with_rapid._consolidate_similar_insights(insights)
        
        assert len(result) == 1
    
    @pytest.mark.asyncio
    async def test_no_duplicates_returns_unchanged(self, task_with_rapid, mock_rapid_llm):
        """When no duplicate types, should return unchanged without calling LLM."""
        insights = [
            WindowInsight(insight_type=InsightType.ACTION, window_id=1, timestamp_start=0.0, timestamp_end=1.0,
                insight_text="Action 1",
                confidence=0.9,
                classification=ClassificationField.POSITIVE,
                continuation_of=None,
                correction_of=None
            ),
            WindowInsight(insight_type=InsightType.DECISION, window_id=1, timestamp_start=0.0, timestamp_end=1.0,
                insight_text="Decision 1",
                confidence=0.8,
                classification=ClassificationField.NEUTRAL,
                continuation_of=None,
                correction_of=None
            )
        ]
        
        result = await task_with_rapid._consolidate_similar_insights(insights)
        
        # Should not call LLM when no duplicates
        mock_rapid_llm.create_completion.assert_not_called()
        assert len(result) == 2
    
    @pytest.mark.asyncio
    async def test_consolidation_when_similar(self, task_with_rapid, mock_rapid_llm):
        """When LLM says similar, should consolidate into single insight."""
        # Setup mock to return "similar" response
        mock_rapid_llm.create_completion = AsyncMock(return_value=(
            "",  # reasoning
            '{"are_similar": true, "consolidated_text": "Consolidated key point about the topic"}',  # content
            100,  # input_tokens
            50,   # output_tokens
            0     # reasoning_tokens
        ))
        
        insights = [
            WindowInsight(insight_type=InsightType.KEY_POINT, window_id=1, timestamp_start=0.0, timestamp_end=1.0,
                insight_text="The project is behind schedule",
                confidence=0.9,
                classification=ClassificationField.NEGATIVE,
                continuation_of=None,
                correction_of=None
            ),
            WindowInsight(insight_type=InsightType.KEY_POINT, window_id=1, timestamp_start=0.0, timestamp_end=1.0,
                insight_text="The timeline for the project is delayed",
                confidence=0.8,
                classification=ClassificationField.NEGATIVE,
                continuation_of=None,
                correction_of=None
            )
        ]
        
        result = await task_with_rapid._consolidate_similar_insights(insights)
        
        # Should consolidate to single insight
        assert len(result) == 1
        assert "Consolidated key point" in result[0].insight_text
        assert result[0].insight_type == InsightType.KEY_POINT
    
    @pytest.mark.asyncio
    async def test_no_consolidation_when_not_similar(self, task_with_rapid, mock_rapid_llm):
        """When LLM says not similar, should keep all insights."""
        mock_rapid_llm.create_completion = AsyncMock(return_value=(
            "",
            '{"are_similar": false, "consolidated_text": ""}',
            100,
            50,
            0
        ))
        
        insights = [
            WindowInsight(insight_type=InsightType.KEY_POINT, window_id=1, timestamp_start=0.0, timestamp_end=1.0,
                insight_text="Project is behind schedule",
                confidence=0.9,
                classification=ClassificationField.NEGATIVE,
                continuation_of=None,
                correction_of=None
            ),
            WindowInsight(insight_type=InsightType.KEY_POINT, window_id=1, timestamp_start=0.0, timestamp_end=1.0,
                insight_text="Weather was nice today",
                confidence=0.8,
                classification=ClassificationField.POSITIVE,
                continuation_of=None,
                correction_of=None
            )
        ]
        
        result = await task_with_rapid._consolidate_similar_insights(insights)
        
        # Should keep both insights
        assert len(result) == 2
    
    @pytest.mark.asyncio
    async def test_preserves_non_duplicate_insights(self, task_with_rapid, mock_rapid_llm):
        """Should preserve insights that are not part of duplicate types."""
        mock_rapid_llm.create_completion = AsyncMock(return_value=(
            "",
            '{"are_similar": true, "consolidated_text": "Consolidated"}',
            100,
            50,
            0
        ))
        
        insights = [
            WindowInsight(insight_type=InsightType.KEY_POINT, window_id=1, timestamp_start=0.0, timestamp_end=1.0,
                insight_text="Key point 1",
                confidence=0.9,
                classification=ClassificationField.POSITIVE,
                continuation_of=None,
                correction_of=None
            ),
            WindowInsight(insight_type=InsightType.KEY_POINT, window_id=1, timestamp_start=0.0, timestamp_end=1.0,
                insight_text="Key point 2",
                confidence=0.8,
                classification=ClassificationField.POSITIVE,
                continuation_of=None,
                correction_of=None
            ),
            WindowInsight(insight_type=InsightType.ACTION, window_id=1, timestamp_start=0.0, timestamp_end=1.0,
                insight_text="Take action",
                confidence=0.7,
                classification=ClassificationField.POSITIVE,
                continuation_of=None,
                correction_of=None
            )
        ]
        
        result = await task_with_rapid._consolidate_similar_insights(insights)
        
        # Should have consolidated KEY_POINT insights + preserved ACTION
        assert len(result) == 2
        # ACTION should be preserved
        assert any(i.insight_type == InsightType.ACTION for i in result)
    
    @pytest.mark.asyncio
    async def test_uses_highest_confidence(self, task_with_rapid, mock_rapid_llm):
        """When consolidating, should use highest confidence from group."""
        mock_rapid_llm.create_completion = AsyncMock(return_value=(
            "",
            '{"are_similar": true, "consolidated_text": "Consolidated text"}',
            100,
            50,
            0
        ))
        
        insights = [
            WindowInsight(insight_type=InsightType.KEY_POINT, window_id=1, timestamp_start=0.0, timestamp_end=1.0,
                insight_text="Low confidence point",
                confidence=0.5,
                classification=ClassificationField.POSITIVE,
                continuation_of=None,
                correction_of=None
            ),
            WindowInsight(insight_type=InsightType.KEY_POINT, window_id=1, timestamp_start=0.0, timestamp_end=1.0,
                insight_text="High confidence point",
                confidence=0.9,
                classification=ClassificationField.POSITIVE,
                continuation_of=None,
                correction_of=None
            )
        ]
        
        result = await task_with_rapid._consolidate_similar_insights(insights)
        
        # Should use highest confidence (0.9)
        assert len(result) == 1
        assert result[0].confidence == 0.9
    
    @pytest.mark.asyncio
    async def test_handles_llm_error_gracefully(self, task_with_rapid, mock_rapid_llm):
        """When LLM call fails, should return original insights."""
        mock_rapid_llm.create_completion = AsyncMock(side_effect=Exception("LLM error"))
        
        insights = [
            WindowInsight(insight_type=InsightType.KEY_POINT, window_id=1, timestamp_start=0.0, timestamp_end=1.0,
                insight_text="Point 1",
                confidence=0.9,
                classification=ClassificationField.POSITIVE,
                continuation_of=None,
                correction_of=None
            ),
            WindowInsight(insight_type=InsightType.KEY_POINT, window_id=1, timestamp_start=0.0, timestamp_end=1.0,
                insight_text="Point 2",
                confidence=0.8,
                classification=ClassificationField.POSITIVE,
                continuation_of=None,
                correction_of=None
            )
        ]
        
        result = await task_with_rapid._consolidate_similar_insights(insights)
        
        # Should keep both insights on error
        assert len(result) == 2
    
    @pytest.mark.asyncio
    async def test_handles_parse_error_gracefully(self, task_with_rapid, mock_rapid_llm):
        """When LLM response can't be parsed, should return original insights."""
        mock_rapid_llm.create_completion = AsyncMock(return_value=(
            "",
            "invalid json response",
            100,
            50
        ))
        
        insights = [
            WindowInsight(insight_type=InsightType.KEY_POINT, window_id=1, timestamp_start=0.0, timestamp_end=1.0,
                insight_text="Point 1",
                confidence=0.9,
                classification=ClassificationField.POSITIVE,
                continuation_of=None,
                correction_of=None
            ),
            WindowInsight(insight_type=InsightType.KEY_POINT, window_id=1, timestamp_start=0.0, timestamp_end=1.0,
                insight_text="Point 2",
                confidence=0.8,
                classification=ClassificationField.POSITIVE,
                continuation_of=None,
                correction_of=None
            )
        ]
        
        result = await task_with_rapid._consolidate_similar_insights(insights)
        
        # Should keep both insights on parse error
        assert len(result) == 2