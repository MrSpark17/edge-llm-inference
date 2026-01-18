"""Inference module for edge LLM."""

from .intent_classifier import IntentClassifier
from .config import Settings

__all__ = [
    "IntentClassifier",
    "Settings",
]

__version__ = "0.1.0"
