"""Core detection logic."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional
import re
from collections import Counter

from .types import DetectionResult, Verdict


@dataclass
class Config:
    """Detector configuration."""
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    consistency_samples: int = 5
    grounding_top_k: int = 3
    threshold: float = 0.7
    weights: dict = field(default_factory=lambda: {
        "nli": 0.45, "consistency": 0.25, "grounding": 0.30
    })
    device: str = "cpu"


# ---- lightweight heuristic backends (no heavy deps by default) ----

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _cosine(a: Counter, b: Counter) -> float:
    keys = set(a) | set(b)
    num = sum(a[k] * b[k] for k in keys)
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    return num / (na * nb) if na and nb else 0.0


def _nli_heuristic(premise: str, hypothesis: str) -> float:
    """Proxy entailment via weighted token overlap + negation penalty."""
    p, h = Counter(_tokenize(premise)), Counter(_tokenize(hypothesis))
    overlap = _cosine(p, h)
    negations_h = len(re.findall(r"\b(not|never|no|cannot)\b", hypothesis.lower()))
    negations_p = len(re.findall(r"\b(not|never|no|cannot)\b", premise.lower()))
    if abs(negations_h - negations_p) >= 1:
        overlap *= 0.5
    return max(0.0, min(1.0, overlap))


def _grounding_score(answer: str, context: str, top_k: int = 3) -> float:
    """Fraction of answer sentences grounded in context."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    if not sents:
        return 1.0
    grounded = 0
    for s in sents:
        if _nli_heuristic(context, s) > 0.4:
            grounded += 1
    return grounded / len(sents)


def _semantic_consistency(samples: List[str]) -> float:
    """Average pairwise similarity across sampled generations."""
    if len(samples) < 2:
        return 1.0
    counters = [Counter(_tokenize(s)) for s in samples]
    sims = []
    for i in range(len(counters)):
        for j in range(i + 1, len(counters)):
            sims.append(_cosine(counters[i], counters[j]))
    return sum(sims) / len(sims)


# ---- main API ----

class Detector:
    """Pluggable hallucination detector."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

    def check(
        self,
        question: str,
        answer: str,
        context: str = "",
        samples: Optional[List[str]] = None,
    ) -> DetectionResult:
        signals = {}
        if context:
            signals["nli"] = _nli_heuristic(context, answer)
            signals["grounding"] = _grounding_score(answer, context, self.config.grounding_top_k)
        else:
            signals["nli"] = 0.6  # neutral prior when no context
            signals["grounding"] = 0.6

        if samples and len(samples) >= 2:
            signals["consistency"] = _semantic_consistency(samples)
        else:
            signals["consistency"] = 0.8

        w = self.config.weights
        score = sum(signals[k] * w.get(k, 0) for k in signals) / sum(w.get(k, 0) for k in signals)

        if score >= self.config.threshold:
            verdict = Verdict.FAITHFUL
        elif score >= self.config.threshold - 0.2:
            verdict = Verdict.SUSPECT
        else:
            verdict = Verdict.HALLUCINATION

        suspect = self._find_suspect_spans(answer, context) if verdict != Verdict.FAITHFUL else []

        return DetectionResult(
            score=score,
            verdict=verdict,
            signals=signals,
            suspect_spans=suspect,
            explanation=self._explain(signals, verdict),
        )

    def _find_suspect_spans(self, answer: str, context: str):
        spans = []
        for m in re.finditer(r"[^.!?]+[.!?]", answer):
            sent = m.group().strip()
            if context and _nli_heuristic(context, sent) < 0.3:
                spans.append((m.start(), m.end(), sent))
        return spans

    def _explain(self, signals, verdict):
        weak = [k for k, v in signals.items() if v < 0.5]
        if verdict == Verdict.FAITHFUL:
            return "All signals strong — answer is well-grounded."
        return f"Weak signal(s): {', '.join(weak) or 'composite below threshold'}."


# Convenience top-level API
_default = Detector()


def detect(question: str, answer: str, context: str = "") -> DetectionResult:
    """Single check with optional grounding context."""
    return _default.check(question, answer, context)


def detect_self_consistency(
    question: str,
    llm_fn: Callable[[str], str],
    n_samples: int = 5,
    context: str = "",
) -> DetectionResult:
    """Sample N generations, check semantic consistency."""
    samples = [llm_fn(question) for _ in range(n_samples)]
    primary = samples[0]
    return _default.check(question, primary, context, samples=samples)
