"""
classifiers/final_classifier_engine.py — P72: Final Classifier Engine.

See classifiers/__init__.py for the implementation.
This module re-exports for convenience.
"""

from classifiers import (
    CLASSIFIER_ML_READY,
    CLASSIFIER_RULE_FALLBACK,
    CLASSIFIER_BLOCKED,
    run_final_classifier,
)

__all__ = [
    "CLASSIFIER_ML_READY",
    "CLASSIFIER_RULE_FALLBACK",
    "CLASSIFIER_BLOCKED",
    "run_final_classifier",
]
