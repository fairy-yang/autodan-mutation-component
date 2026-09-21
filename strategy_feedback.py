#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Strategy feedback utilities for the lightweight mutation component.

The paper's full method updates a strategy library from multi-round attack logs.
This lightweight module keeps the update point explicit and auditable: callers
can record structured evaluation feedback, inspect aggregate statistics, and
append human-reviewed strategy notes. It does not call a target model or infer
new adversarial strategies automatically.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class StrategyFeedbackStore:
    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.records: List[Dict[str, Any]] = []
        if path and os.path.exists(path):
            self.records = self._load_jsonl(path)

    @staticmethod
    def _load_jsonl(path: str) -> List[Dict[str, Any]]:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def record(
        self,
        strategy: str,
        score: Optional[float] = None,
        outcome: Optional[str] = None,
        notes: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy": strategy,
            "score": score,
            "outcome": outcome,
            "notes": notes,
            "metadata": metadata or {},
        }
        self.records.append(record)

        if self.path:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return record

    def report(self) -> Dict[str, Any]:
        by_strategy: Dict[str, Dict[str, Any]] = {}

        for record in self.records:
            strategy = record.get("strategy") or "UNKNOWN"
            bucket = by_strategy.setdefault(
                strategy,
                {
                    "count": 0,
                    "scored_count": 0,
                    "score_sum": 0.0,
                    "outcomes": {},
                },
            )
            bucket["count"] += 1

            score = record.get("score")
            if isinstance(score, (int, float)):
                bucket["scored_count"] += 1
                bucket["score_sum"] += float(score)

            outcome = record.get("outcome") or "unspecified"
            bucket["outcomes"][outcome] = bucket["outcomes"].get(outcome, 0) + 1

        for bucket in by_strategy.values():
            if bucket["scored_count"]:
                bucket["average_score"] = bucket["score_sum"] / bucket["scored_count"]
            else:
                bucket["average_score"] = None
            del bucket["score_sum"]

        return {
            "total_records": len(self.records),
            "strategies": by_strategy,
        }


def append_reviewed_strategy(
    strategy_library_path: str,
    strategy_name: str,
    definition: str,
    example: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Append a human-reviewed strategy note to the JSON strategy library.

    The library format in this repository is a dict of strategy objects. The new
    key is deterministic enough for review and avoids overwriting existing keys.
    """
    if not os.path.exists(strategy_library_path):
        raise FileNotFoundError(f"Strategy library not found: {strategy_library_path}")

    with open(strategy_library_path, "r", encoding="utf-8") as f:
        library = json.load(f)

    if not isinstance(library, dict):
        raise ValueError("Strategy library must be a JSON object.")

    base_key = strategy_name.strip().replace(" ", "_") or "reviewed_strategy"
    key = base_key
    counter = 1
    while key in library:
        counter += 1
        key = f"{base_key}_{counter}"

    library[key] = {
        "Strategy": strategy_name,
        "Definition": definition,
        "Example": example,
        "metadata": metadata or {},
        "source": "human_reviewed_feedback",
    }

    with open(strategy_library_path, "w", encoding="utf-8") as f:
        json.dump(library, f, ensure_ascii=False, indent=2)

    return key
