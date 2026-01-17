"""
Pytest configuration and shared fixtures for testing.
"""

import pytest
from inference.config import Settings
from inference.intent_classifier import IntentClassifier


@pytest.fixture
def test_settings() -> Settings:
    """Provide test configuration settings."""
    return Settings(
        model_name="deepscaler-chat",
        max_input_length=512,
        inference_timeout_seconds=5,
        intent_confidence_threshold=0.7,
        environment="test"
    )


@pytest.fixture
def intent_classifier(test_settings) -> IntentClassifier:
    """Provide a test intent classifier instance."""
    return IntentClassifier(
        model_name=test_settings.model_name,
        confidence_threshold=test_settings.intent_confidence_threshold
    )
