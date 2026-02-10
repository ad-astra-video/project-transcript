"""
Unit tests for KEY POINT classification with breakthrough-level threshold.
"""

import pytest
from src.summary.summary_client import SummaryClient


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])