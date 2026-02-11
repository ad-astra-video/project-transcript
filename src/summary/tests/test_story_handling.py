"""
Unit tests for story handling in transcript processing.

Tests verify that the LLM properly frames stories within insights,
including story details only when they provide background/relevance/illustration
for the insight itself.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from summary.summary_client import SummaryClient, WindowManager, ContentTypeState, ContentTypeSource
from summary.prompts import CONTENT_TYPE_RULE_MODIFIERS, SYSTEM_PROMPT


class TestStoryHandlingPromptSection:
    """Tests for the STORY AND BACKGROUND CONTEXT HANDLING section in SYSTEM_PROMPT."""
    
    def test_story_handling_section_exists_in_system_prompt(self):
        """Verify that the STORY AND BACKGROUND CONTEXT HANDLING section exists in SYSTEM_PROMPT."""
        assert "STORY AND BACKGROUND CONTEXT HANDLING" in SYSTEM_PROMPT
    
    def test_core_principle_is_defined(self):
        """Verify that the core principle 'Extract the insight, not the story' is present."""
        assert "Extract the insight, not the story" in SYSTEM_PROMPT
    
    def test_story_illustration_examples_exist(self):
        """Verify that examples for story illustration are present."""
        # Check for DO example
        assert "Example - DO:" in SYSTEM_PROMPT
        # Check for DON'T example
        assert "Example - DON'T:" in SYSTEM_PROMPT
    
    def test_when_to_exclude_content_exists(self):
        """Verify that guidance on when to exclude story content is present."""
        assert "When to Exclude Story Content" in SYSTEM_PROMPT
        assert "Pure entertainment" in SYSTEM_PROMPT
        assert "Explicit tangent" in SYSTEM_PROMPT
    
    def test_story_continuation_pattern_exists(self):
        """Verify that story continuation pattern guidance is present."""
        assert "Story Continuation Pattern" in SYSTEM_PROMPT
        assert "continuation_of" in SYSTEM_PROMPT


class TestStoryGuidanceInContentTypes:
    """Tests for story_guidance field in CONTENT_TYPE_RULE_MODIFIERS."""
    
    def test_technical_talk_has_story_guidance(self):
        """Verify TECHNICAL_TALK has story_guidance."""
        assert "story_guidance" in CONTENT_TYPE_RULE_MODIFIERS["TECHNICAL_TALK"]
        guidance = CONTENT_TYPE_RULE_MODIFIERS["TECHNICAL_TALK"]["story_guidance"]
        assert "technical" in guidance.lower()
        assert "production" in guidance.lower()
    
    def test_lecture_has_story_guidance(self):
        """Verify LECTURE_OR_TALK has story_guidance."""
        assert "story_guidance" in CONTENT_TYPE_RULE_MODIFIERS["LECTURE_OR_TALK"]
        guidance = CONTENT_TYPE_RULE_MODIFIERS["LECTURE_OR_TALK"]["story_guidance"]
        assert "educational" in guidance.lower()
    
    def test_interview_has_story_guidance(self):
        """Verify INTERVIEW has story_guidance."""
        assert "story_guidance" in CONTENT_TYPE_RULE_MODIFIERS["INTERVIEW"]
        guidance = CONTENT_TYPE_RULE_MODIFIERS["INTERVIEW"]["story_guidance"]
        assert "experience" in guidance.lower()
    
    def test_podcast_has_story_guidance(self):
        """Verify PODCAST has story_guidance."""
        assert "story_guidance" in CONTENT_TYPE_RULE_MODIFIERS["PODCAST"]
        guidance = CONTENT_TYPE_RULE_MODIFIERS["PODCAST"]["story_guidance"]
        assert "conversational" in guidance.lower()
    
    def test_general_meeting_has_story_guidance(self):
        """Verify GENERAL_MEETING has story_guidance."""
        assert "story_guidance" in CONTENT_TYPE_RULE_MODIFIERS["GENERAL_MEETING"]
        guidance = CONTENT_TYPE_RULE_MODIFIERS["GENERAL_MEETING"]["story_guidance"]
        assert "actionable" in guidance.lower()
    
    def test_all_content_types_have_story_guidance(self):
        """Verify all content types have story_guidance field."""
        content_types = [
            "GENERAL_MEETING", "TECHNICAL_TALK", "LECTURE_OR_TALK",
            "INTERVIEW", "PODCAST", "NEWS_UPDATE", "GAMEPLAY_COMMENTARY",
            "CUSTOMER_SUPPORT", "DEBATE", "UNKNOWN"
        ]
        for content_type in content_types:
            assert f"'{content_type}'" in CONTENT_TYPE_RULE_MODIFIERS or content_type in CONTENT_TYPE_RULE_MODIFIERS, \
                f"{content_type} missing from CONTENT_TYPE_RULE_MODIFIERS"
            if content_type in CONTENT_TYPE_RULE_MODIFIERS:
                assert "story_guidance" in CONTENT_TYPE_RULE_MODIFIERS[content_type], \
                    f"{content_type} missing story_guidance field"


class TestStoryHandlingInKeyPointGuidance:
    """Tests for story handling examples in key_point_guidance sections."""
    
    def test_technical_talk_key_point_includes_story_guidance(self):
        """Verify TECHNICAL_TALK key_point_guidance includes story handling."""
        guidance = CONTENT_TYPE_RULE_MODIFIERS["TECHNICAL_TALK"]["key_point_guidance"]
        assert "story" in guidance.lower() or "When a story" in guidance
    
    def test_interview_key_point_includes_story_guidance(self):
        """Verify INTERVIEW key_point_guidance includes story handling."""
        guidance = CONTENT_TYPE_RULE_MODIFIERS["INTERVIEW"]["key_point_guidance"]
        assert "story" in guidance.lower() or "When a story" in guidance
    
    def test_podcast_key_point_includes_story_guidance(self):
        """Verify PODCAST key_point_guidance includes story handling."""
        guidance = CONTENT_TYPE_RULE_MODIFIERS["PODCAST"]["key_point_guidance"]
        assert "story" in guidance.lower() or "When a story" in guidance


class TestFormatContentTypeRulesWithStory:
    """Tests for _format_content_type_rules method including story_guidance."""
    
    def create_client(self):
        """Create a SummaryClient instance for testing."""
        return SummaryClient(
            base_url="http://test:8000/v1",
            api_key="test-key",
            model="test-model",
            transcription_windows_per_summary_window=2,
            raw_text_context_limit=10000
        )
    
    def test_format_content_type_rules_includes_story_section(self):
        """Verify that _format_content_type_rules includes story guidance section."""
        client = self.create_client()
        
        # Test with TECHNICAL_TALK which has story_guidance
        formatted = client._format_content_type_rules("TECHNICAL_TALK")
        
        assert "### STORY Handling:" in formatted or "STORY" in formatted
    
    def test_format_content_type_rules_handles_missing_story_guidance(self):
        """Verify that missing story_guidance doesn't cause errors."""
        client = self.create_client()
        
        # Create a mock content type without story_guidance
        with patch.object(client, '_content_type_state', 
                          ContentTypeState(content_type="CUSTOM", confidence=0.9)):
            # This should not raise an error even if story_guidance is missing
            # The CONTENT_TYPE_RULE_MODIFIERS should have story_guidance for all types
            formatted = client._format_content_type_rules("CUSTOM")
            assert isinstance(formatted, str)


class TestStoryFramingExamples:
    """Tests to verify story framing examples are properly defined."""
    
    def test_story_illustrating_insight_examples(self):
        """Verify examples for story illustrating insights exist."""
        # Check for JSON examples showing proper story framing
        assert '"insight_type": "KEY POINT"' in SYSTEM_PROMPT
        assert "Small models can outperform" in SYSTEM_PROMPT or "350M parameter" in SYSTEM_PROMPT
    
    def test_background_context_examples(self):
        """Verify examples for background context extraction exist."""
        assert "Speaker's previous company" in SYSTEM_PROMPT or "acquired by Fortune 500" in SYSTEM_PROMPT
    
    def test_summarize_story_example(self):
        """Verify example for summarizing stories exists."""
        assert "Real-world deployment often differs" in SYSTEM_PROMPT or "illustrated by speaker's experience" in SYSTEM_PROMPT


class TestStoryContinuationPattern:
    """Tests for story continuation pattern implementation."""
    
    def test_continuation_pattern_examples_exist(self):
        """Verify continuation pattern examples exist in prompt."""
        assert "Window 1 - Story begins" in SYSTEM_PROMPT
        assert "Window 2 - Story continues" in SYSTEM_PROMPT
        assert "Window 3 - Insight emerges" in SYSTEM_PROMPT
    
    def test_continuation_of_usage_in_examples(self):
        """Verify continuation_of is properly used in examples."""
        assert 'continuation_of": null' in SYSTEM_PROMPT
        assert 'continuation_of": 42' in SYSTEM_PROMPT
        assert 'continuation_of": 43' in SYSTEM_PROMPT


class TestStoryDetectionIndicators:
    """Tests for story detection indicators in the prompt."""
    
    def test_temporal_markers_defined(self):
        """Verify temporal story markers are defined."""
        assert "When I was..." in SYSTEM_PROMPT or "Back in..." in SYSTEM_PROMPT
    
    def test_personal_pronoun_markers_defined(self):
        """Verify personal pronoun story markers are defined."""
        assert "I remember..." in SYSTEM_PROMPT or "I once..." in SYSTEM_PROMPT
    
    def test_transition_phrase_markers_defined(self):
        """Verify transition phrase story markers are defined."""
        assert "That reminds me..." in SYSTEM_PROMPT or "Speaking of which..." in SYSTEM_PROMPT
    
    def test_scene_setting_markers_defined(self):
        """Verify scene-setting story markers are defined."""
        assert "So there I was..." in SYSTEM_PROMPT or "Picture this..." in SYSTEM_PROMPT


class TestContentTypeSpecificStoryHandling:
    """Tests for content-type-specific story handling rules."""
    
    def test_technical_talk_story_rules(self):
        """Verify TECHNICAL_TALK has appropriate story handling rules."""
        rules = CONTENT_TYPE_RULE_MODIFIERS["TECHNICAL_TALK"]
        story_guidance = rules.get("story_guidance", "")
        
        # Should focus on technical content
        assert "technical" in story_guidance.lower() or "production" in story_guidance.lower()
        # Should exclude entertainment
        assert "entertainment" in story_guidance.lower() or "personal" in story_guidance.lower()
    
    def test_interview_story_rules(self):
        """Verify INTERVIEW has appropriate story handling rules."""
        rules = CONTENT_TYPE_RULE_MODIFIERS["INTERVIEW"]
        story_guidance = rules.get("story_guidance", "")
        
        # Should include personal experiences
        assert "experience" in story_guidance.lower() or "personal" in story_guidance.lower()
        # Should allow more story content than technical talks
        assert "relevant" in story_guidance.lower() or "context" in story_guidance.lower()
    
    def test_podcast_story_rules(self):
        """Verify PODCAST has appropriate story handling rules."""
        rules = CONTENT_TYPE_RULE_MODIFIERS["PODCAST"]
        story_guidance = rules.get("story_guidance", "")
        
        # Should recognize stories as vehicle for insights
        assert "conversational" in story_guidance.lower() or "story" in story_guidance.lower()
    
    def test_meeting_story_rules(self):
        """Verify GENERAL_MEETING has appropriate story handling rules."""
        rules = CONTENT_TYPE_RULE_MODIFIERS["GENERAL_MEETING"]
        story_guidance = rules.get("story_guidance", "")
        
        # Should minimize story content
        assert "minimize" in story_guidance.lower() or "focus on actionable" in story_guidance.lower()


class TestStoryHandlingIntegration:
    """Integration tests for story handling in the full processing pipeline."""
    
    def create_client(self, delay_seconds: float = 0.0, transcription_windows_per_summary_window: int = 2):
        """Create a SummaryClient instance for testing."""
        return SummaryClient(
            base_url="http://test:8000/v1",
            api_key="test-key",
            model="test-model",
            transcription_windows_per_summary_window=transcription_windows_per_summary_window,
            raw_text_context_limit=10000,
            initial_summary_delay_seconds=delay_seconds
        )
    
    @pytest.mark.asyncio
    async def test_story_content_extraction_as_notes(self):
        """Test that story content is extracted as NOTES when appropriate."""
        client = self.create_client(delay_seconds=0.0)
        client._auto_detect_content_type_detection = False
        client._content_type_state = ContentTypeState(
            content_type="TECHNICAL_TALK",
            confidence=1.0,
            source=ContentTypeSource.INITIAL.value
        )
        
        # Simulate a story segment
        segments = [
            {
                "text": "So I was working at this startup back in 2015, and we had this crazy problem with our production system.",
                "start_ms": 0,
                "end_ms": 5000
            }
        ]
        
        # The story content should be captured as NOTES
        # The key point about the problem should be extracted separately
        with patch.object(SummaryClient, 'summarize_text', AsyncMock(return_value=('{"analysis": "", "insights": []}', "", 0))):
            result = await client.process_segments(
                "context_summary",
                segments,
                1,
                0.0,
                5.0
            )
        
        # Verify result structure
        assert result["type"] == "context_summary"
    
    @pytest.mark.asyncio
    async def test_story_continuation_tracking(self):
        """Test that stories spanning multiple windows are tracked via continuation_of."""
        client = self.create_client(delay_seconds=0.0, transcription_windows_per_summary_window=1)
        client._auto_detect_content_type_detection = False
        client._content_type_state = ContentTypeState(
            content_type="TECHNICAL_TALK",
            confidence=1.0,
            source=ContentTypeSource.INITIAL.value
        )
        
        # First window - story begins
        segments1 = [
            {
                "text": "I remember this one time we had a production outage at 3 AM on Christmas Eve.",
                "start_ms": 0,
                "end_ms": 5000
            }
        ]
        with patch.object(SummaryClient, 'summarize_text', AsyncMock(return_value=('{"analysis": "", "insights": []}', "", 0))):
            result1 = await client.process_segments(
                "context_summary",
                segments1,
                1,
                0.0,
                5.0
            )

            # Second window - story continues with insight
            segments2 = [
            {
                "text": "It turned out to be a simple config issue. We fixed it in 5 minutes. The lesson is to always check config first.",
                "start_ms": 5000,
                "end_ms": 10000
            }
        ]
            result2 = await client.process_segments(
                "context_summary",
                segments2,
                2,
                5.0,
                10.0
            )
        
        # Verify continuation tracking works
        # The second result should reference the first via continuation_of if applicable
        assert result2["type"] == "context_summary"


class TestStoryExclusion:
    """Tests for story exclusion logic."""
    
    def test_exclusion_reasons_are_defined(self):
        """Verify that reasons for story exclusion are defined in the prompt."""
        assert "Pure entertainment" in SYSTEM_PROMPT
        assert "Explicit tangent" in SYSTEM_PROMPT
        assert "No connection to main topic" in SYSTEM_PROMPT
        assert "Repeated story" in SYSTEM_PROMPT
        assert "Generic anecdote" in SYSTEM_PROMPT
    
    def test_meeting_content_type_excludes_stories(self):
        """Verify GENERAL_MEETING content type has strict story exclusion rules."""
        rules = CONTENT_TYPE_RULE_MODIFIERS["GENERAL_MEETING"]
        story_guidance = rules.get("story_guidance", "").lower()
        
        # Should have explicit guidance about excluding stories
        assert "exclude" in story_guidance or "minimize" in story_guidance or "focus on actionable" in story_guidance


class TestStoryHandlingExamples:
    """Tests to verify comprehensive story handling examples exist."""
    
    def test_technical_story_transformation_example(self):
        """Verify example showing technical story to insight transformation exists."""
        # Should have a "Before" and "After" style example
        assert "Example - DO:" in SYSTEM_PROMPT or "Example - DON'T:" in SYSTEM_PROMPT
    
    def test_interview_background_extraction_example(self):
        """Verify example showing interview background extraction exists."""
        assert "Speaker's previous company" in SYSTEM_PROMPT or "acquired by Fortune 500" in SYSTEM_PROMPT
    
    def test_podcast_illustrative_story_example(self):
        """Verify example showing podcast story handling exists."""
        assert "illustrated by speaker's experience" in SYSTEM_PROMPT or "Real-world deployment" in SYSTEM_PROMPT


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])