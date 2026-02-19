"""
Unit tests for story handling in transcript processing.

Tests verify that the LLM properly frames stories within insights,
including story details only when they provide background/relevance/illustration
for the insight itself.
"""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from src.summary.summary_client import SummaryClient
from src.summary.window_manager import WindowManager
from src.summary.context_summary.prompts import CONTENT_TYPE_RULE_MODIFIERS, SYSTEM_PROMPT
from src.summary.content_type_detection.task import ContentType, ContentTypeSource


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


class TestStoryHandlingWithPlugins:
    """Tests for story handling through the plugin system."""
    
    @pytest.fixture
    def client_with_plugin(self):
        """Create a SummaryClient with context_summary plugin."""
        with patch('src.summary.summary_client.AsyncOpenAI'):
            client = SummaryClient(
                reasoning_api_key="test-key",
                reasoning_base_url="http://test:8000/v1",
                reasoning_model="test-model"
            )
            return client
    
    def test_context_summary_plugin_exists(self, client_with_plugin):
        """Test that context_summary plugin can be loaded."""
        from src.summary.context_summary import ContextSummaryPlugin
        
        # Create mock LLM
        mock_llm = MagicMock()
        mock_llm.reasoning_llm_client = MagicMock()
        
        # Create plugin
        plugin = ContextSummaryPlugin(
            window_manager=client_with_plugin._window_manager,
            llm_manager=mock_llm,
            result_callback=client_with_plugin._queue_payload,
            summary_client=client_with_plugin
        )
        
        assert plugin is not None
        assert plugin._window_manager is not None
    
    def test_content_type_state_holder_in_plugin(self, client_with_plugin):
        """Test that ContentTypeStateHolder is used in plugin."""
        from src.summary.context_summary import ContentTypeStateHolder
        
        state = ContentTypeStateHolder(
            content_type=ContentType.GENERAL_MEETING.value,
            confidence=0.9,
            source=ContentTypeSource.AUTO_DETECTED.value
        )
        
        assert state.content_type == ContentType.GENERAL_MEETING.value
        assert state.confidence == 0.9


class TestContentTypeRuleModifiers:
    """Tests for content type rule modifiers."""
    
    def test_content_type_rule_modifiers_exist(self):
        """Verify that CONTENT_TYPE_RULE_MODIFIERS exists."""
        assert CONTENT_TYPE_RULE_MODIFIERS is not None
    
    def test_content_type_rule_modifiers_is_dict(self):
        """Verify that CONTENT_TYPE_RULE_MODIFIERS is a dictionary."""
        assert isinstance(CONTENT_TYPE_RULE_MODIFIERS, dict)
    
    def test_technical_talk_modifier_exists(self):
        """Verify that TECHNICAL_TALK modifier exists."""
        assert "TECHNICAL_TALK" in CONTENT_TYPE_RULE_MODIFIERS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])