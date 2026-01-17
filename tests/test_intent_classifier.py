"""
Unit tests for intent classifier module.
Tests intent detection, confidence scoring, and response generation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from inference.intent_classifier import IntentClassifier
from inference.utils import InvalidInputError, ModelNotFoundError


class TestIntentClassifierInitialization:
    """Test intent classifier initialization."""
    
    def test_initialization_with_defaults(self):
        """Test classifier initializes with default values."""
        classifier = IntentClassifier()
        assert classifier.model_name == "deepscaler-chat"
        assert classifier.confidence_threshold == 0.7
    
    def test_initialization_with_custom_values(self):
        """Test classifier initializes with custom values."""
        classifier = IntentClassifier(
            model_name="custom-model",
            confidence_threshold=0.8
        )
        assert classifier.model_name == "custom-model"
        assert classifier.confidence_threshold == 0.8


class TestIntentDetection:
    """Test intent detection logic."""
    
    def test_greeting_intent_detection(self, intent_classifier):
        """Test greeting intent is detected."""
        intent, confidence = intent_classifier._extract_intent("Hello there!")
        assert intent == "greeting"
        assert confidence > 0.5
    
    def test_farewell_intent_detection(self, intent_classifier):
        """Test farewell intent is detected."""
        intent, confidence = intent_classifier._extract_intent("Goodbye, see you soon")
        assert intent == "farewell"
        assert confidence > 0.5
    
    def test_help_intent_detection(self, intent_classifier):
        """Test help intent is detected."""
        intent, confidence = intent_classifier._extract_intent("Can you help me?")
        assert intent == "help"
        assert confidence > 0.5
    
    def test_information_intent_detection(self, intent_classifier):
        """Test information intent is detected."""
        intent, confidence = intent_classifier._extract_intent("Explain machine learning")
        assert intent == "information"
        assert confidence > 0.5
    
    def test_default_intent_for_unknown(self, intent_classifier):
        """Test unknown input defaults to general_query."""
        intent, confidence = intent_classifier._extract_intent("xyz abc 123")
        assert intent == "general_query"
        assert confidence == 0.5
    
    def test_confidence_increases_with_keyword_frequency(self, intent_classifier):
        """Test confidence increases with multiple keywords."""
        intent1, conf1 = intent_classifier._extract_intent("hello")
        intent2, conf2 = intent_classifier._extract_intent("hello hello hello")
        
        assert intent1 == intent2 == "greeting"
        assert conf2 > conf1  # Multiple keywords boost confidence


class TestInputValidation:
    """Test input validation in classification."""
    
    def test_empty_input_raises_error(self, intent_classifier):
        """Test empty input raises InvalidInputError."""
        with pytest.raises(InvalidInputError):
            intent_classifier.classify("")
    
    def test_whitespace_only_input_raises_error(self, intent_classifier):
        """Test whitespace-only input raises InvalidInputError."""
        with pytest.raises(InvalidInputError):
            intent_classifier.classify("   ")
    
    def test_valid_input_accepted(self, intent_classifier):
        """Test valid input is accepted."""
        with patch.object(intent_classifier, '_get_model_response', return_value="test response"):
            result = intent_classifier.classify("valid input")
            assert result is not None
            assert "intent" in result


class TestBatchProcessing:
    """Test batch classification."""
    
    def test_batch_classification(self, intent_classifier):
        """Test batch processing multiple inputs."""
        inputs = ["hello", "what is this?", "goodbye"]
        
        with patch.object(intent_classifier, '_get_model_response', return_value="response"):
            results = intent_classifier.batch_classify(inputs)
            
            assert len(results) == 3
            assert all("intent" in r or "error" in r for r in results)
    
    def test_batch_handles_failures_gracefully(self, intent_classifier):
        """Test batch processing continues after errors."""
        inputs = ["valid", "", "another"]
        
        with patch.object(intent_classifier, '_get_model_response', return_value="response"):
            results = intent_classifier.batch_classify(inputs)
            
            # Should have 3 results, even with one error
            assert len(results) == 3
            # Second result should have error
            assert "error" in results[1]


class TestClassificationResponse:
    """Test classification response structure."""
    
    def test_response_structure(self, intent_classifier):
        """Test response has required fields."""
        with patch.object(intent_classifier, '_get_model_response', return_value="test response"):
            result = intent_classifier.classify("test input")
            
            assert "intent" in result
            assert "confidence" in result
            assert "response" in result
            assert "input_length" in result
            assert "model" in result
            assert "metadata" in result
    
    def test_response_confidence_is_float(self, intent_classifier):
        """Test confidence is returned as float."""
        with patch.object(intent_classifier, '_get_model_response', return_value="test"):
            result = intent_classifier.classify("test")
            
            assert isinstance(result["confidence"], float)
            assert 0 <= result["confidence"] <= 1
