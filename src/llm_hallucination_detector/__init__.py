"""
llm-hallucination-detector
==========================

Detect hallucinations in LLM outputs using NLI + self-consistency + grounding.

Quickstart
----------
>>> from llm_hallucination_detector import detect
>>> r = detect(question="Q?", answer="A", context="...")
>>> r.score, r.verdict
"""

from .core import detect, detect_self_consistency, Detector, Config
from .guard import guard
from .batch import BatchDetector
from .types import DetectionResult, Verdict

__version__ = "0.1.0"
__all__ = [
    "detect",
    "detect_self_consistency",
    "Detector",
    "Config",
    "guard",
    "BatchDetector",
    "DetectionResult",
    "Verdict",
]
