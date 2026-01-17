"""
Production-grade inference runtime for edge-optimized LLM.
Handles model loading, inference, error handling, and performance monitoring.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import ollama
import logging
from typing import Optional, Dict, Any
from .utils import measure_performance, PerformanceMonitor, ModelNotFoundError, InvalidInputError

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Edge-Optimized LLM Inference",
    description="Sub-second inference on resource-constrained edge devices",
    version="1.0.0"
)

from .config import get_settings

# Load configuration
config = get_settings()



# Pydantic Models for Request/Response validation
class ChatRequest(BaseModel):
    """Validated chat request."""
    message: str = Field(..., min_length=1, max_length=config.max_input_length)
    model: Optional[str] = Field(default=config.model_name, description="Model name")
    
    @validator('message')
    def message_not_empty(cls, v: str) -> str:
        """Ensure message is not just whitespace."""
        if not v.strip():
            raise ValueError("Message cannot be empty or whitespace")
        return v.strip()


class ChatResponse(BaseModel):
    """Validated chat response."""
    response: str
    model: str
    latency_ms: float
    memory_mb: float
    status: str = "success"


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_available: bool
    memory_usage_mb: float


# Error response handler
@app.exception_handler(InvalidInputError)
async def invalid_input_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": f"Invalid input: {str(exc)}"}
    )


@app.exception_handler(ModelNotFoundError)
async def model_not_found_handler(request, exc):
    return JSONResponse(
        status_code=503,
        content={"detail": f"Model unavailable: {str(exc)}"}
    )


# Routes
@app.get("/", tags=["Health"])
def home() -> Dict[str, str]:
    """
    Home endpoint - returns welcome message.
    """
    logger.info("Home endpoint accessed")
    return {
        "message": "Welcome to Edge-Optimized LLM Inference",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", tags=["Health"], response_model=HealthResponse)
@measure_performance
def health_check() -> HealthResponse:
    """
    Health check endpoint - verifies system status and model availability.
    """
    logger.info("Health check initiated")
    
    try:
        memory_usage = PerformanceMonitor.get_memory_usage()
        
        # Verify model availability with a lightweight test
        try:
            ollama.chat(
                model=config.model_name,
                messages=[{"role": "user", "content": "ping"}],
                stream=False
            )
            model_available = True
        except Exception as e:
            logger.warning(f"Model health check failed: {str(e)}")
            model_available = False
        
        return HealthResponse(
            status="operational" if model_available else "degraded",
            model_available=model_available,
            memory_usage_mb=memory_usage["rss_mb"]
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Health check failed")


@app.post("/chat", tags=["Inference"], response_model=ChatResponse)
@measure_performance
def chat(request: ChatRequest) -> ChatResponse:
    """
    Chat endpoint - processes user message and returns model response.
    
    Args:
        request (ChatRequest): Validated chat request with message and optional model name
        
    Returns:
        ChatResponse: Model response with performance metrics
        
    Raises:
        HTTPException: If model is unavailable or inference fails
    """
    logger.info(f"Chat request received | Message length: {len(request.message)}")
    
    try:
        # Record initial state
        memory_before = PerformanceMonitor.get_memory_usage()
        import time
        start_time = time.time()
        
        # Call model with error handling
        try:
            response = ollama.chat(
                model=request.model,
                messages=[{"role": "user", "content": request.message}],
                stream=False
            )
        except Exception as e:
            logger.error(f"Model inference failed: {str(e)}")
            raise ModelNotFoundError(f"Failed to get response from model: {str(e)}")
        
        # Calculate metrics
        latency_ms = (time.time() - start_time) * 1000
        memory_after = PerformanceMonitor.get_memory_usage()
        memory_used = memory_after["rss_mb"] - memory_before["rss_mb"]
        
        # Validate response
        if not response or "message" not in response:
            logger.error("Invalid model response structure")
            raise InvalidInputError("Model returned invalid response")
        
        result = ChatResponse(
            response=response["message"]["content"],
            model=request.model,
            latency_ms=latency_ms,
            memory_mb=memory_used,
            status="success"
        )
        
        # Log metrics
        PerformanceMonitor.log_inference_metrics(latency_ms, memory_after["rss_mb"], request.model)
        logger.info(f"Chat response successful | Latency: {latency_ms:.2f}ms")
        
        return result
    
    except (ModelNotFoundError, InvalidInputError):
        raise
    
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/metrics", tags=["Monitoring"])
@measure_performance
def get_metrics() -> Dict[str, Any]:
    """
    Metrics endpoint - returns current system metrics.
    Useful for monitoring edge device constraints.
    """
    logger.info("Metrics endpoint accessed")
    
    memory = PerformanceMonitor.get_memory_usage()
    
    return {
        "memory_rss_mb": round(memory["rss_mb"], 2),
        "memory_vms_mb": round(memory["vms_mb"], 2),
        "max_allowed_mb": 2048,
        "status": "within_limits" if memory["rss_mb"] < 2048 else "exceeded"
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Log startup and verify model availability."""
    logger.info("=" * 50)
    logger.info("Edge-Optimized LLM Inference Server Starting")
    logger.info(f"Default Model: {config.model_name}")
    logger.info(f"Max Input Length: {config.max_input_length}")
    logger.info(f"Inference Timeout: {config.inference_timeout_seconds}s")
    logger.info("=" * 50)


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Log shutdown."""
    logger.info("Edge-Optimized LLM Inference Server Shutting Down")


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server")
    uvicorn.run(app, host="0.0.0.0", port=8000)
