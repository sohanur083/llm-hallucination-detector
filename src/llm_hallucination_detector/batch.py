"""Batch evaluation over CSV / JSONL datasets."""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Iterable, List, Dict
from .core import Detector, Config


class BatchDetector:
    """Evaluate a dataset of (question, answer, context) rows."""

    def __init__(self, config: Config | None = None):
        self.detector = Detector(config)

    def evaluate_rows(self, rows: Iterable[Dict]) -> List[Dict]:
        out = []
        for row in rows:
            r = self.detector.check(
                question=row.get("question", ""),
                answer=row.get("answer", ""),
                context=row.get("context", ""),
            )
            out.append({
                **row,
                "score": r.score,
                "verdict": r.verdict.value,
                **{f"signal_{k}": v for k, v in r.signals.items()},
            })
        return out

    def evaluate_dataset(self, path: str | Path) -> List[Dict]:
        path = Path(path)
        if path.suffix == ".csv":
            with path.open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
        return self.evaluate_rows(rows)
