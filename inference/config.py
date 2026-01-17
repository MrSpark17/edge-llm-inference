"""
Configuration management for edge-optimized LLM inference.
Supports environment variables for production deployments.
"""

import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings

import logging

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Application configuration with environment variable support.
    Use environment variables to override defaults.
    
    Example:
        export MODEL_NAME="deepscaler-chat"
        export INFERENCE_TIMEOUT_SECONDS=10
    """
    
    # Model Configuration
    model_name: str = Field(
        default="deepscaler-chat",
        description="Ollama model name to use for inference"
    )
    
    # API Configuration
    api_host: str = Field(
        default="0.0.0.0",
        description="API server host"
    )
    api_port: int = Field(
        default=8000,
        description="API server port"
    )
    
    # Inference Configuration
    max_input_length: int = Field(
        default=512,
        description="Maximum input message length in characters"
    )
    inference_timeout_seconds: int = Field(
        default=5,
        description="Maximum inference timeout in seconds"
    )
    
    # Memory Configuration
    max_memory_mb: int = Field(
        default=2048,
        description="Maximum allowed memory usage in MB"
    )
    memory_warning_threshold_mb: int = Field(
        default=1800,
        description="Memory usage warning threshold in MB"
    )
    
    # Intent Classification Configuration
    intent_confidence_threshold: float = Field(
        default=0.7,
        description="Minimum confidence score for intent classification (0-1)"
    )
    
    # Logging Configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    log_file: str = Field(
        default="inference.log",
        description="Path to log file"
    )
    
    # Performance Configuration
    enable_performance_monitoring: bool = Field(
        default=True,
        description="Enable detailed performance monitoring"
    )
    batch_processing_enabled: bool = Field(
        default=True,
        description="Enable batch inference processing"
    )
    max_batch_size: int = Field(
        default=32,
        description="Maximum batch size for batch inference"
    )
    
    # Environment Configuration
    environment: str = Field(
        default="development",
        description="Deployment environment (development, production, edge)"
    )
    
    class Config:
        """Pydantic config for environment variable loading."""
        env_file = ".env"
        case_sensitive = False
        # Allow environment variable override
        extra = "allow"
    
    def validate(self) -> bool:
        """
        Validate configuration values.
        
        Returns:
            bool: True if valid, raises ValueError otherwise
        """
        errors = []
        
        # Validate memory settings
        if self.max_memory_mb <= 0:
            errors.append("max_memory_mb must be positive")
        if self.memory_warning_threshold_mb > self.max_memory_mb:
            errors.append("memory_warning_threshold_mb cannot exceed max_memory_mb")
        
        # Validate intent threshold
        if not (0 <= self.intent_confidence_threshold <= 1):
            errors.append("intent_confidence_threshold must be between 0 and 1")
        
        # Validate input length
        if self.max_input_length <= 0:
            errors.append("max_input_length must be positive")
        
        # Validate timeout
        if self.inference_timeout_seconds <= 0:
            errors.append("inference_timeout_seconds must be positive")
        
        # Validate port
        if not (1 <= self.api_port <= 65535):
            errors.append("api_port must be between 1 and 65535")
        
        # Validate batch size
        if self.max_batch_size <= 0:
            errors.append("max_batch_size must be positive")
        
        # Validate log level
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_levels:
            errors.append(f"log_level must be one of {valid_levels}")
        
        # Validate environment
        valid_envs = ["development", "production", "edge"]
        if self.environment.lower() not in valid_envs:
            errors.append(f"environment must be one of {valid_envs}")
        
        if errors:
            error_msg = "\n".join(errors)
            logger.error(f"Configuration validation failed:\n{error_msg}")
            raise ValueError(f"Configuration errors:\n{error_msg}")
        
        logger.info("Configuration validation passed")
        return True
    
    def get_summary(self) -> dict:
        """
        Get a summary of active configuration.
        
        Returns:
            dict: Configuration summary (safe to log/display)
        """
        return {
            "model_name": self.model_name,
            "environment": self.environment,
            "api_host": self.api_host,
            "api_port": self.api_port,
            "max_input_length": self.max_input_length,
            "max_memory_mb": self.max_memory_mb,
            "inference_timeout_seconds": self.inference_timeout_seconds,
            "intent_confidence_threshold": self.intent_confidence_threshold,
            "log_level": self.log_level,
            "batch_processing_enabled": self.batch_processing_enabled,
            "performance_monitoring_enabled": self.enable_performance_monitoring
        }


# Global settings instance (singleton pattern)
try:
    settings = Settings()
    settings.validate()
    logger.info(f"Configuration loaded: {settings.get_summary()}")
except ValueError as e:
    logger.error(f"Configuration error: {str(e)}")
    raise
except Exception as e:
    logger.error(f"Unexpected configuration error: {str(e)}")
    raise


def get_settings() -> Settings:
    """
    Get global settings instance.
    
    Returns:
        Settings: Application configuration
    """
    return settings