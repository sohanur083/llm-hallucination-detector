"""Decorator-based guard for LLM functions."""
from __future__ import annotations
import functools
from typing import Callable, Tuple, Any
from .core import Detector, Config
from .types import DetectionResult


def guard(
    threshold: float = 0.7,
    context_arg: str = "context",
) -> Callable:
    """Decorate an LLM function so every call returns (answer, DetectionResult).

    The wrapped function must return a string. It must accept `question` as
    its first positional arg and optionally `context` as a kwarg.
    """
    detector = Detector(Config(threshold=threshold))

    def deco(fn: Callable[..., str]) -> Callable[..., Tuple[str, DetectionResult]]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Tuple[str, DetectionResult]:
            answer = fn(*args, **kwargs)
            question = args[0] if args else kwargs.get("question", "")
            context = kwargs.get(context_arg, "")
            result = detector.check(question=question, answer=answer, context=context)
            return answer, result
        return wrapper
    return deco
