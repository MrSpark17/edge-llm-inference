"""Intent classification module for edge LLM inference."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class IntentClassifier:
    """Classify user intents from text input."""

    def __init__(
        self,
        model_name: str = "deepscaler-chat",
        confidence_threshold: float = 0.7,
    ):
        """Initialize intent classifier.

        Args:
            model_name: Name of the model to use for classification
            confidence_threshold: Minimum confidence threshold for intent detection
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold

        # Intent keyword templates with priority ordering
        # More specific intents first
        self.intent_templates = {
            "information": ["explain", "what is", "how does", "tell me about"],
            "help": ["help", "assist", "support", "how do i"],
            "greeting": ["hello", "hi", "hey", "greetings"],
            "farewell": ["goodbye", "bye", "see you", "farewell"],
            "action": ["do", "perform", "execute", "run"],
        }

        logger.info(f"IntentClassifier initialized with model: {model_name}")

    def _get_model_response(self, text: str) -> str:
        """Get response from the model.
        
        Args:
            text: Input text
            
        Returns:
            Model response
        """
        # Placeholder for model integration
        return f"Response to: {text}"

    def _extract_intent(self, text: str) -> tuple[str, float]:
        """Extract intent from text using keyword matching.

        Args:
            text: Input text to classify

        Returns:
            Tuple of (intent, confidence_score)
        """
        text_lower = text.lower()
        intent_scores = {}
        
        # Check all intents and count keyword occurrences
        for intent, keywords in self.intent_templates.items():
            # Count total occurrences of all keywords in this intent
            total_matches = sum(text_lower.count(keyword) for keyword in keywords)
            
            if total_matches > 0:
                # Calculate confidence: 0.6 + (matches * 0.1)
                # 1 occurrence = 0.7, 2 occurrences = 0.8, 3 occurrences = 0.9, etc.
                confidence = min(0.95, 0.6 + (total_matches * 0.1))
                intent_scores[intent] = (intent, confidence)
        
        # Return the intent with highest confidence
        if intent_scores:
            best_intent = max(intent_scores.values(), key=lambda x: x[1])
            return best_intent
        
        # Default to general_query if no keywords matched
        return "general_query", 0.5

    def classify(self, text: str) -> dict:
        """Classify intent from user input text.

        Args:
            text: User input text to classify

        Returns:
            Dict with intent, confidence, response, and metadata

        Raises:
            InvalidInputError: If text is empty or whitespace only
        """
        from inference.utils import InvalidInputError
        
        if not text or not text.strip():
            raise InvalidInputError("Input text cannot be empty or whitespace only")

        intent, confidence = self._extract_intent(text)

        # Get model response
        response = self._get_model_response(text)

        return {
            "intent": intent,
            "confidence": confidence,
            "response": response,
            "input_length": len(text),
            "model": self.model_name,
            "metadata": {
                "threshold": self.confidence_threshold,
                "matched": confidence >= self.confidence_threshold,
            }
        }

    def batch_classify(self, texts: list[str]) -> list[dict]:
        """Classify multiple texts for intent.

        Args:
            texts: List of input texts to classify

        Returns:
            List of result dicts (with "error" key if failed)
        """
        from inference.utils import InvalidInputError
        
        results = []
        for text in texts:
            try:
                result = self.classify(text)
                results.append(result)
            except InvalidInputError as e:
                logger.warning(f"Failed to classify text: {e}")
                # Add error result instead of skipping
                results.append({
                    "error": str(e),
                    "input": text,
                    "intent": None,
                    "confidence": 0.0
                })

        return results