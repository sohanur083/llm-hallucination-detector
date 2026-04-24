"""Type definitions."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple


class Verdict(str, Enum):
    FAITHFUL = "FAITHFUL"
    SUSPECT = "SUSPECT"
    HALLUCINATION = "HALLUCINATION"


@dataclass
class DetectionResult:
    """Result of a hallucination detection check."""
    score: float
    verdict: Verdict
    signals: Dict[str, float] = field(default_factory=dict)
    suspect_spans: List[Tuple[int, int, str]] = field(default_factory=list)
    explanation: str = ""

    def __repr__(self) -> str:
        return (
            f"DetectionResult(score={self.score:.3f}, "
            f"verdict={self.verdict.value}, "
            f"signals={ {k: round(v, 3) for k, v in self.signals.items()} })"
        )
