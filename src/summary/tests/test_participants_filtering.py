"""
Unit tests for participants tracking functionality.
"""

import pytest
from src.summary.summary_client import SummaryClient


class TestParticipantsEnabledFiltering:
    """Tests for participants_enabled filtering in _extract_insights."""
    
    def create_client(self):
        """Create a SummaryClient instance for testing."""
        return SummaryClient(api_key="test_key", model="test_model")
    
    def test_is_participants_enabled_returns_true_for_general_meeting(self):
        """Test that is_participants_enabled returns True for GENERAL_MEETING."""
        client = self.create_client()
        client.set_content_type("GENERAL_MEETING", confidence=0.9, source="AUTO_DETECTED")
        assert client.is_participants_enabled() is True
    
    def test_is_participants_enabled_returns_true_for_technical_talk(self):
        """Test that is_participants_enabled returns True for TECHNICAL_TALK."""
        client = self.create_client()
        client.set_content_type("TECHNICAL_TALK", confidence=0.9, source="AUTO_DETECTED")
        assert client.is_participants_enabled() is True
    
    def test_is_participants_enabled_returns_true_for_lecture_or_talk(self):
        """Test that is_participants_enabled returns True for LECTURE_OR_TALK."""
        client = self.create_client()
        client.set_content_type("LECTURE_OR_TALK", confidence=0.9, source="AUTO_DETECTED")
        assert client.is_participants_enabled() is True
    
    def test_is_participants_enabled_returns_true_for_interview(self):
        """Test that is_participants_enabled returns True for INTERVIEW."""
        client = self.create_client()
        client.set_content_type("INTERVIEW", confidence=0.9, source="AUTO_DETECTED")
        assert client.is_participants_enabled() is True
    
    def test_is_participants_enabled_returns_true_for_podcast(self):
        """Test that is_participants_enabled returns True for PODCAST."""
        client = self.create_client()
        client.set_content_type("PODCAST", confidence=0.9, source="AUTO_DETECTED")
        assert client.is_participants_enabled() is True
    
    def test_is_participants_enabled_returns_false_for_news_update(self):
        """Test that is_participants_enabled returns False for NEWS_UPDATE."""
        client = self.create_client()
        client.set_content_type("NEWS_UPDATE", confidence=0.9, source="AUTO_DETECTED")
        assert client.is_participants_enabled() is False
    
    def test_is_participants_enabled_returns_false_for_gameplay_commentary(self):
        """Test that is_participants_enabled returns False for GAMEPLAY_COMMENTARY."""
        client = self.create_client()
        client.set_content_type("GAMEPLAY_COMMENTARY", confidence=0.9, source="AUTO_DETECTED")
        assert client.is_participants_enabled() is False
    
    def test_is_participants_enabled_returns_true_for_customer_support(self):
        """Test that is_participants_enabled returns True for CUSTOMER_SUPPORT."""
        client = self.create_client()
        client.set_content_type("CUSTOMER_SUPPORT", confidence=0.9, source="AUTO_DETECTED")
        assert client.is_participants_enabled() is True
    
    def test_is_participants_enabled_returns_true_for_debate(self):
        """Test that is_participants_enabled returns True for DEBATE."""
        client = self.create_client()
        client.set_content_type("DEBATE", confidence=0.9, source="AUTO_DETECTED")
        assert client.is_participants_enabled() is True
    
    def test_is_participants_enabled_returns_true_for_unknown(self):
        """Test that is_participants_enabled returns True for UNKNOWN."""
        client = self.create_client()
        client.set_content_type("UNKNOWN", confidence=0.9, source="AUTO_DETECTED")
        assert client.is_participants_enabled() is True
    
    def test_extract_insights_includes_participants_when_enabled(self):
        """Test that PARTICIPANTS insights are included when participants_enabled=True."""
        client = self.create_client()
        client.set_content_type("GENERAL_MEETING", confidence=0.9, source="AUTO_DETECTED")
        
        parsed_data = {
            "analysis": "Test analysis",
            "insights": [
                {
                    "insight_type": "PARTICIPANTS",
                    "insight_text": "John Smith (CEO) - main speaker throughout the meeting",
                    "confidence": 0.85,
                    "classification": "~"
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
        
        # Should include both insights (PARTICIPANTS + ACTION)
        assert len(result["insights"]) == 2
        insight_types = [i["insight_type"] for i in result["insights"]]
        assert "PARTICIPANTS" in insight_types
        assert "ACTION" in insight_types
    
    def test_extract_insights_filters_participants_when_disabled(self):
        """Test that PARTICIPANTS insights are filtered when participants_enabled=False."""
        client = self.create_client()
        client.set_content_type("STREAMER_MONOLOGUE", confidence=0.9, source="AUTO_DETECTED")
        
        parsed_data = {
            "analysis": "Test analysis",
            "insights": [
                {
                    "insight_type": "PARTICIPANTS",
                    "insight_text": "Streamer introduced themselves",
                    "confidence": 0.85,
                    "classification": "~"
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
        
        # Should only include ACTION insight (PARTICIPANTS filtered out)
        assert len(result["insights"]) == 1
        assert result["insights"][0]["insight_type"] == "ACTION"
    
    def test_extract_insights_non_participants_always_included(self):
        """Test that non-PARTICIPANTS insights are always included regardless of participants_enabled."""
        client = self.create_client()
        client.set_content_type("STREAMER_MONOLOGUE", confidence=0.9, source="AUTO_DETECTED")
        
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
        
        # Should include all non-PARTICIPANTS insights
        assert len(result["insights"]) == 3
        insight_types = [i["insight_type"] for i in result["insights"]]
        assert "DECISION" in insight_types
        assert "QUESTION" in insight_types
        assert "KEY_POINT" in insight_types
    
    def test_extract_insights_mixed_insights_partially_filtered(self):
        """Test that mixed insights (PARTICIPANTS + others) are partially filtered correctly."""
        client = self.create_client()
        client.set_content_type("INTERVIEW", confidence=0.9, source="AUTO_DETECTED")
        
        parsed_data = {
            "analysis": "Test analysis",
            "insights": [
                {
                    "insight_type": "PARTICIPANTS",
                    "insight_text": "Jane Doe (CTO) - being interviewed about AI",
                    "confidence": 0.85,
                    "classification": "~"
                },
                {
                    "insight_type": "ACTION",
                    "insight_text": "Follow up on AI implementation",
                    "confidence": 0.95,
                    "classification": "+"
                },
                {
                    "insight_type": "PARTICIPANTS",
                    "insight_text": "Interviewer: Bob from TechNews",
                    "confidence": 0.80,
                    "classification": "~"
                },
                {
                    "insight_type": "DECISION",
                    "insight_text": "Schedule part 2",
                    "confidence": 0.90,
                    "classification": "~"
                }
            ]
        }
        
        result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        # Should include all insights (participants_enabled=True for INTERVIEW)
        assert len(result["insights"]) == 4
        insight_types = [i["insight_type"] for i in result["insights"]]
        assert insight_types.count("PARTICIPANTS") == 2
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
    
    def test_extract_insights_only_participants_filtered(self):
        """Test that when only PARTICIPANTS insights exist and are filtered, result is empty."""
        client = self.create_client()
        client.set_content_type("STREAMER_MONOLOGUE", confidence=0.9, source="AUTO_DETECTED")
        
        parsed_data = {
            "analysis": "Test analysis",
            "insights": [
                {
                    "insight_type": "PARTICIPANTS",
                    "insight_text": "Streamer introduced themselves",
                    "confidence": 0.85,
                    "classification": "~"
                }
            ]
        }
        
        result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        # PARTICIPANTS should be filtered out
        assert len(result["insights"]) == 0


class TestSelfIntroductionCapture:
    """Tests for self-introduction capture behavior in prompts."""
    
    def create_client(self):
        """Create a SummaryClient instance for testing."""
        return SummaryClient(api_key="test_key", model="test_model")
    
    def test_self_introduction_prompt_includes_examples(self):
        """Test that the system prompt includes self-introduction examples."""
        from src.summary.prompts import SYSTEM_PROMPT
        
        # Check that self-introduction examples are in the prompt
        assert "I'm Tim from Cello" in SYSTEM_PROMPT or "Self-Introduction" in SYSTEM_PROMPT
    
    def test_self_introduction_splitting_guidance(self):
        """Test that prompt includes guidance for splitting self-introduction content."""
        from src.summary.prompts import SYSTEM_PROMPT
        
        # Check that splitting guidance is in the prompt
        assert "Splitting Self-Introduction" in SYSTEM_PROMPT or "speaker's identifying information" in SYSTEM_PROMPT
    
    def test_correction_handling_guidance(self):
        """Test that prompt includes guidance for handling corrections."""
        from src.summary.prompts import SYSTEM_PROMPT
        
        # Check that correction handling is in the prompt
        assert "Handling Corrections" in SYSTEM_PROMPT or "correction" in SYSTEM_PROMPT.lower()
    
    def test_participants_definition_includes_self_introductions(self):
        """Test that PARTICIPANTS definition explicitly mentions self-introductions."""
        from src.summary.prompts import SYSTEM_PROMPT
        
        # Check that self-introductions are mentioned in PARTICIPANTS section
        assert "self-introduction" in SYSTEM_PROMPT.lower() or "introduces themselves" in SYSTEM_PROMPT.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])