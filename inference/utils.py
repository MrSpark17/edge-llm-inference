"""
Utility module for performance monitoring, logging, and error handling.
Essential for production-grade inference systems.
"""

import logging
import time
from functools import wraps
from typing import Any, Callable
import psutil
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor system performance metrics during inference."""
    
    @staticmethod
    def get_memory_usage() -> dict:
        """
        Get current memory usage in MB.
        
        Returns:
            dict: Contains 'rss_mb' (Resident Set Size) and 'vms_mb' (Virtual Memory Size)
        """
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
        }
    
    @staticmethod
    def log_inference_metrics(latency_ms: float, memory_mb: float, model: str) -> None:
        """
        Log inference performance metrics.
        
        Args:
            latency_ms (float): Inference latency in milliseconds
            memory_mb (float): Memory usage in MB
            model (str): Model name
        """
        logger.info(
            f"Inference Complete | Model: {model} | "
            f"Latency: {latency_ms:.2f}ms | Memory: {memory_mb:.2f}MB"
        )


def measure_performance(func: Callable) -> Callable:
    """
    Decorator to measure function execution time and memory usage.
    Logs performance metrics and catches exceptions.
    
    Args:
        func (Callable): Function to decorate
        
    Returns:
        Callable: Wrapped function with performance tracking
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_memory = PerformanceMonitor.get_memory_usage()
        start_time = time.time()
        
        try:
            logger.info(f"Starting: {func.__name__}")
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = PerformanceMonitor.get_memory_usage()
            
            latency_ms = (end_time - start_time) * 1000
            memory_delta = end_memory["rss_mb"] - start_memory["rss_mb"]
            
            logger.info(
                f"Completed: {func.__name__} | "
                f"Latency: {latency_ms:.2f}ms | Memory Delta: {memory_delta:.2f}MB"
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            raise
    
    return wrapper


# Custom Exception Classes
class ChatbotError(Exception):
    """Base exception for chatbot errors."""
    pass


class ModelNotFoundError(ChatbotError):
    """Raised when the model is not available."""
    pass


class InferenceTimeoutError(ChatbotError):
    """Raised when inference exceeds timeout."""
    pass


class InvalidInputError(ChatbotError):
    """Raised when input validation fails."""
    pass