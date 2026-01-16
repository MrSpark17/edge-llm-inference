"""
Intent Classification module for edge-optimized LLM.
Separates inference logic from UI, enabling testing and reusability.
"""

import logging
from typing import Dict, List, Tuple
import ollama
from .utils import measure_performance, PerformanceMonitor, ModelNotFoundError, InvalidInputError

logger = logging.getLogger(__name__)


class IntentClassifier:
    """
    Production-grade intent classifier using quantized LLM.
    Handles inference, intent extraction, and confidence scoring.
    """
    
    def __init__(self, model_name: str = "deepscaler-chat", confidence_threshold: float = 0.7):
        """
        Initialize the intent classifier.
        
        Args:
            model_name (str): Name of the Ollama model to use
            confidence_threshold (float): Minimum confidence score for intent classification (0-1)
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.intent_templates = {
            "greeting": ["hello", "hi", "hey", "greetings"],
            "farewell": ["goodbye", "bye", "see you", "farewell"],
            "help": ["help", "assist", "support", "how do i"],
            "information": ["what", "how", "explain", "tell me"],
            "action": ["do", "perform", "execute", "run"]
        }
        logger.info(f"IntentClassifier initialized with model: {model_name}")
    
    @measure_performance
    def classify(self, user_input: str) -> Dict[str, any]:
        """
        Classify user intent and generate response.
        
        Args:
            user_input (str): User message to classify
            
        Returns:
            Dict: Contains 'intent', 'confidence', 'response', and 'metadata'
            
        Raises:
            InvalidInputError: If input is empty or invalid
            ModelNotFoundError: If model is unavailable
        """
        if not user_input or not user_input.strip():
            raise InvalidInputError("Input message cannot be empty")
        
        user_input = user_input.strip()
        logger.info(f"Classifying intent for: {user_input[:50]}...")
        
        try:
            # Extract intent from input
            detected_intent, confidence = self._extract_intent(user_input)
            logger.info(f"Detected intent: {detected_intent} (confidence: {confidence:.2f})")
            
            # Get model response
            model_response = self._get_model_response(user_input)
            
            result = {
                "intent": detected_intent,
                "confidence": round(confidence, 3),
                "response": model_response,
                "input_length": len(user_input),
                "model": self.model_name,
                "metadata": {
                    "threshold": self.confidence_threshold,
                    "meets_threshold": confidence >= self.confidence_threshold
                }
            }
            
            logger.info(f"Classification complete | Intent: {detected_intent} | Response length: {len(model_response)}")
            return result
        
        except ModelNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Intent classification failed: {str(e)}", exc_info=True)
            raise ModelNotFoundError(f"Classification error: {str(e)}")
    
    def _extract_intent(self, user_input: str) -> Tuple[str, float]:
        """
        Extract intent from user input using keyword matching.
        
        Args:
            user_input (str): User message
            
        Returns:
            Tuple[str, float]: (intent_name, confidence_score)
        """
        user_input_lower = user_input.lower()
        
        # Simple keyword-based intent detection
        for intent, keywords in self.intent_templates.items():
            for keyword in keywords:
                if keyword in user_input_lower:
                    # Confidence based on keyword position and frequency
                    keyword_count = user_input_lower.count(keyword)
                    confidence = min(0.9, 0.5 + (keyword_count * 0.1))
                    logger.debug(f"Intent '{intent}' detected via keyword '{keyword}'")
                    return intent, confidence
        
        # Default intent if no keyword match
        logger.debug("No specific intent detected, defaulting to 'information'")
        return "general_query", 0.5
    
    @measure_performance
    def _get_model_response(self, user_input: str) -> str:
        """
        Get response from the quantized LLM model.
        
        Args:
            user_input (str): User message
            
        Returns:
            str: Model response
            
        Raises:
            ModelNotFoundError: If model inference fails
        """
        try:
            logger.info(f"Calling model: {self.model_name}")
            
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": user_input
                    }
                ],
                stream=False
            )
            
            if not response or "message" not in response:
                raise ModelNotFoundError("Invalid response structure from model")
            
            model_text = response["message"]["content"]
            logger.info(f"Model response received | Length: {len(model_text)} chars")
            
            return model_text
        
        except ollama.ResponseError as e:
            logger.error(f"Ollama API error: {str(e)}")
            raise ModelNotFoundError(f"Model error: {str(e)}")
        except Exception as e:
            logger.error(f"Model inference error: {str(e)}", exc_info=True)
            raise ModelNotFoundError(f"Inference failed: {str(e)}")
    
    def batch_classify(self, inputs: List[str]) -> List[Dict]:
        """
        Classify multiple inputs in batch.
        
        Args:
            inputs (List[str]): List of user messages
            
        Returns:
            List[Dict]: List of classification results
        """
        logger.info(f"Batch classification initiated | Batch size: {len(inputs)}")
        
        results = []
        for idx, user_input in enumerate(inputs, 1):
            try:
                result = self.classify(user_input)
                results.append(result)
                logger.info(f"Batch item {idx}/{len(inputs)} completed")
            except Exception as e:
                logger.error(f"Batch item {idx} failed: {str(e)}")
                results.append({
                    "error": str(e),
                    "input": user_input,
                    "status": "failed"
                })
        
        logger.info(f"Batch classification complete | Success: {len([r for r in results if 'error' not in r])}/{len(inputs)}")
        return results


# Gradio UI (Optional - can be run separately)
def launch_gradio_ui():
    """
    Launch Gradio interface for the intent classifier.
    Run this separately: python -c "from inference.intent_classifier import launch_gradio_ui; launch_gradio_ui()"
    """
    try:
        import gradio as gr
        
        logger.info("Launching Gradio UI")
        
        # Initialize classifier
        classifier = IntentClassifier()
        
        def chat_interface(user_message: str) -> str:
            """Gradio interface function."""
            try:
                result = classifier.classify(user_message)
                return f"**Intent:** {result['intent']}\n\n**Response:** {result['response']}"
            except Exception as e:
                return f"Error: {str(e)}"
        
        # Create interface
        interface = gr.Interface(
            fn=chat_interface,
            inputs=gr.Textbox(
                label="Chat Message",
                placeholder="Enter your message here...",
                lines=3
            ),
            outputs=gr.Textbox(
                label="Bot Response",
                lines=5
            ),
            title="Edge-Optimized LLM Intent Classifier",
            description="Real-time intent classification and response generation on edge hardware",
            examples=[
                ["Hello, how are you?"],
                ["What is machine learning?"],
                ["Help me with this issue"],
                ["Goodbye!"]
            ]
        )
        
        logger.info("Gradio interface created, launching...")
        interface.launch()
    
    except ImportError:
        logger.error("Gradio not installed. Install with: pip install gradio")
    except Exception as e:
        logger.error(f"Failed to launch Gradio UI: {str(e)}", exc_info=True)


if __name__ == "__main__":
    # For standalone execution
    launch_gradio_ui()
