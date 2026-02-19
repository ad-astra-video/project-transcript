"""
Unit tests for content type rule modifiers.
"""

import pytest
from src.summary.summary_client import SummaryClient
from src.summary.context_summary.prompts import CONTENT_TYPE_RULE_MODIFIERS


class TestContentTypeRuleModifiersUpdated:
    """Tests for updated CONTENT_TYPE_RULE_MODIFIERS with stricter settings."""
    
    def test_general_meeting_stricter_settings(self):
        """Test that GENERAL_MEETING has stricter action_strictness and reduced notes_frequency."""
        rules = CONTENT_TYPE_RULE_MODIFIERS["GENERAL_MEETING"]
        
        # Should have stricter settings
        assert rules["action_strictness"] == "very_high", "GENERAL_MEETING should have very_high action_strictness"
        assert rules["notes_frequency"] == "medium", "GENERAL_MEETING should have medium notes_frequency"
    
    def test_technical_talk_stricter_settings(self):
        """Test that TECHNICAL_TALK has stricter settings."""
        rules = CONTENT_TYPE_RULE_MODIFIERS["TECHNICAL_TALK"]
        
        # Should have stricter settings
        assert rules["action_strictness"] == "extreme", "TECHNICAL_TALK should have extreme action_strictness"
        assert rules["notes_frequency"] == "high", "TECHNICAL_TALK should have high notes_frequency"
    
    def test_lecture_or_talk_stricter_settings(self):
        """Test that LECTURE_OR_TALK has stricter settings."""
        rules = CONTENT_TYPE_RULE_MODIFIERS["LECTURE_OR_TALK"]
        
        # Should have stricter settings
        assert rules["action_strictness"] == "block", "LECTURE_OR_TALK should have block action_strictness"
        assert rules["notes_frequency"] == "very_high", "LECTURE_OR_TALK should have very_high notes_frequency"
    
    def test_podcast_reduced_notes(self):
        """Test that PODCAST has reduced notes_frequency."""
        rules = CONTENT_TYPE_RULE_MODIFIERS["PODCAST"]
        
        # Should have reduced notes frequency
        assert rules["notes_frequency"] == "high", "PODCAST should have high notes_frequency"
    
    def test_podcast_reduced_notes(self):
        """Test that PODCAST has reduced notes_frequency."""
        rules = CONTENT_TYPE_RULE_MODIFIERS["PODCAST"]
        
        # Should have reduced notes frequency
        assert rules["notes_frequency"] == "high", "PODCAST should have high notes_frequency"


class TestContentTypeRiskGuidance:
    """Tests for content-type-specific RISK guidance."""
    
    def test_all_content_types_have_risk_guidance(self):
        """Test that all content types have risk_guidance field defined."""
        expected_content_types = [
            "GENERAL_MEETING",
            "TECHNICAL_TALK",
            "LECTURE_OR_TALK",
            "INTERVIEW",
            "PODCAST",
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
        rules = CONTENT_TYPE_RULE_MODIFIERS["GENERAL_MEETING"]
        risk_guidance = rules["risk_guidance"]
        
        assert "project blockers" in risk_guidance.lower() or "timeline" in risk_guidance.lower()
        assert "resource" in risk_guidance.lower()
    
    def test_technical_talk_risk_guidance(self):
        """Test TECHNICAL_TALK risk guidance focuses on technical issues."""
        rules = CONTENT_TYPE_RULE_MODIFIERS["TECHNICAL_TALK"]
        risk_guidance = rules["risk_guidance"]
        
        assert "technical" in risk_guidance.lower() or "bugs" in risk_guidance.lower() or "failures" in risk_guidance.lower()
    
    def test_customer_support_risk_guidance(self):
        """Test CUSTOMER_SUPPORT risk guidance focuses on customer-impacting issues."""
        rules = CONTENT_TYPE_RULE_MODIFIERS["CUSTOMER_SUPPORT"]
        risk_guidance = rules["risk_guidance"]
        
        assert "customer" in risk_guidance.lower()
    
    def test_debate_risk_guidance(self):
        """Test DEBATE risk guidance focuses on argument weaknesses."""
        rules = CONTENT_TYPE_RULE_MODIFIERS["DEBATE"]
        risk_guidance = rules["risk_guidance"]
        
        assert "argument" in risk_guidance.lower() or "logical" in risk_guidance.lower() or "counter" in risk_guidance.lower()
    
    def test_format_content_type_rules_includes_risk_guidance(self):
        """Test that _format_content_type_rules includes RISK guidance."""
        from src.summary.context_summary.task import ContextSummaryTask
        from unittest.mock import MagicMock
        
        # Create a mock LLM client
        mock_llm_client = MagicMock()
        
        # Create task instance
        task = ContextSummaryTask(llm_client=mock_llm_client)
        
        # Test GENERAL_MEETING
        rules = CONTENT_TYPE_RULE_MODIFIERS["GENERAL_MEETING"]
        formatted = task._format_content_type_rules("GENERAL_MEETING")
        
        assert "RISK Definition" in formatted or "RISK" in formatted
        assert rules["risk_guidance"] in formatted or "project blockers" in formatted.lower()
    
    def test_format_content_type_rules_includes_key_point_guidance(self):
        """Test that _format_content_type_rules includes KEY POINT guidance when present."""
        from src.summary.context_summary.task import ContextSummaryTask
        from unittest.mock import MagicMock
        
        # Create a mock LLM client
        mock_llm_client = MagicMock()
        
        # Create task instance
        task = ContextSummaryTask(llm_client=mock_llm_client)
        
        # Test TECHNICAL_TALK which has key_point_guidance
        rules = CONTENT_TYPE_RULE_MODIFIERS["TECHNICAL_TALK"]
        formatted = task._format_content_type_rules("TECHNICAL_TALK")
        
        assert "KEY POINT Guidance" in formatted
        assert rules["key_point_guidance"] in formatted
    
    def test_format_content_type_rules_handles_missing_guidance(self):
        """Test that _format_content_type_rules handles content types without extra guidance."""
        from src.summary.context_summary.task import ContextSummaryTask
        from unittest.mock import MagicMock
        
        # Create a mock LLM client
        mock_llm_client = MagicMock()
        
        # Create task instance
        task = ContextSummaryTask(llm_client=mock_llm_client)
        
        # Test NEWS_UPDATE which has risk_guidance but no key_point_guidance
        formatted = task._format_content_type_rules("NEWS_UPDATE")
        
        assert "NEWS_UPDATE" in formatted
        assert "RISK Definition" in formatted or "NEWS_UPDATE" in formatted


class TestContentTypeParticipantsEnabled:
    """Tests for participants_enabled field in CONTENT_TYPE_RULE_MODIFIERS."""
    
    def test_all_content_types_have_participants_enabled(self):
        """Test that all content types have participants_enabled field defined."""
        expected_content_types = [
            "GENERAL_MEETING",
            "TECHNICAL_TALK",
            "LECTURE_OR_TALK",
            "INTERVIEW",
            "PODCAST",
            "NEWS_UPDATE",
            "GAMEPLAY_COMMENTARY",
            "CUSTOMER_SUPPORT",
            "DEBATE",
            "UNKNOWN"
        ]
        
        for content_type in expected_content_types:
            assert content_type in CONTENT_TYPE_RULE_MODIFIERS, f"{content_type} missing from CONTENT_TYPE_RULE_MODIFIERS"
            rules = CONTENT_TYPE_RULE_MODIFIERS[content_type]
            assert "participants_enabled" in rules, f"{content_type} missing participants_enabled field"
            assert isinstance(rules["participants_enabled"], bool), f"{content_type} participants_enabled must be a boolean"
    
    def test_multi_speaker_content_types_have_participants_enabled_true(self):
        """Test that content types with multiple speakers have participants_enabled=True."""
        enabled_types = [
            "GENERAL_MEETING",
            "TECHNICAL_TALK",
            "LECTURE_OR_TALK",
            "INTERVIEW",
            "PODCAST",
            "CUSTOMER_SUPPORT",
            "DEBATE",
            "UNKNOWN"
        ]
        
        for content_type in enabled_types:
            rules = CONTENT_TYPE_RULE_MODIFIERS[content_type]
            assert rules["participants_enabled"] is True, f"{content_type} should have participants_enabled=True"
    
    def test_single_speaker_content_types_have_participants_enabled_false(self):
        """Test that single-speaker content types have participants_enabled=False."""
        disabled_types = [
            "NEWS_UPDATE",
            "GAMEPLAY_COMMENTARY"
        ]
        
        for content_type in disabled_types:
            rules = CONTENT_TYPE_RULE_MODIFIERS[content_type]
            assert rules["participants_enabled"] is False, f"{content_type} should have participants_enabled=False"
    
    def test_format_content_type_rules_includes_participants_tracking(self):
        """Test that _format_content_type_rules includes participant tracking status."""
        from src.summary.context_summary.task import ContextSummaryTask
        from unittest.mock import MagicMock
        
        # Create a mock LLM client
        mock_llm_client = MagicMock()
        
        # Create task instance
        task = ContextSummaryTask(llm_client=mock_llm_client)
        
        # Test GENERAL_MEETING (enabled)
        formatted = task._format_content_type_rules("GENERAL_MEETING")
        assert "Participant Tracking: ENABLED" in formatted
        
        # Test NEWS_UPDATE (disabled)
        formatted = task._format_content_type_rules("NEWS_UPDATE")
        assert "Participant Tracking: DISABLED" in formatted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])