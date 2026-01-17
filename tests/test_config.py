"""
Unit tests for configuration module.
Tests validation, environment variable loading, and settings.
"""

import pytest
from inference.config import Settings


class TestSettingsValidation:
    """Test configuration validation."""
    
    def test_default_settings_valid(self):
        """Test that default settings are valid."""
        settings = Settings()
        assert settings.validate() is True
    
    def test_memory_validation_positive(self):
        """Test memory settings must be positive."""
        with pytest.raises(ValueError, match="max_memory_mb must be positive"):
            settings = Settings(max_memory_mb=-1)
            settings.validate()
    
    def test_memory_threshold_validation(self):
        """Test memory threshold cannot exceed max memory."""
        with pytest.raises(ValueError, match="memory_warning_threshold_mb cannot exceed max_memory_mb"):
            settings = Settings(
                max_memory_mb=1000,
                memory_warning_threshold_mb=2000
            )
            settings.validate()
    
    def test_intent_confidence_range(self):
        """Test confidence threshold must be 0-1."""
        with pytest.raises(ValueError, match="intent_confidence_threshold must be between 0 and 1"):
            settings = Settings(intent_confidence_threshold=1.5)
            settings.validate()
    
    def test_port_validation(self):
        """Test port must be in valid range."""
        with pytest.raises(ValueError, match="api_port must be between 1 and 65535"):
            settings = Settings(api_port=99999)
            settings.validate()
    
    def test_log_level_validation(self):
        """Test log level must be valid."""
        with pytest.raises(ValueError, match="log_level must be one of"):
            settings = Settings(log_level="INVALID")
            settings.validate()
    
    def test_environment_validation(self):
        """Test environment must be valid."""
        with pytest.raises(ValueError, match="environment must be one of"):
            settings = Settings(environment="invalid")
            settings.validate()


class TestSettingsSummary:
    """Test configuration summary generation."""
    
    def test_get_summary_contains_all_keys(self):
        """Test summary includes all important settings."""
        settings = Settings()
        summary = settings.get_summary()
        
        assert "model_name" in summary
        assert "environment" in summary
        assert "api_host" in summary
        assert "api_port" in summary
        assert "max_memory_mb" in summary
        assert "inference_timeout_seconds" in summary
    
    def test_get_summary_values_correct(self):
        """Test summary has correct values."""
        settings = Settings(
            model_name="test-model",
            api_port=9000,
            max_memory_mb=4096
        )
        summary = settings.get_summary()
        
        assert summary["model_name"] == "test-model"
        assert summary["api_port"] == 9000
        assert summary["max_memory_mb"] == 4096


class TestSettingsEnvironment:
    """Test environment-based configuration."""
    
    def test_settings_initialization(self):
        """Test settings can be initialized with custom values."""
        settings = Settings(
            model_name="custom-model",
            api_port=8080,
            environment="production"
        )
        
        assert settings.model_name == "custom-model"
        assert settings.api_port == 8080
        assert settings.environment == "production"
