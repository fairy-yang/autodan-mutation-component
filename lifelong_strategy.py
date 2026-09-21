#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lightweight lifelong strategy learning utilities.

This module implements the engineering shape of AutoDAN-Turbo's strategy loop:
load strategy records, retrieve strategies for the current item, record feedback,
update strategy statistics, and export an auditable updated library.

It deliberately uses structured feedback supplied by the caller. It does not call
a target model or automatically optimize adversarial success.
"""

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


class LifelongStrategyLibrary:
    def __init__(
        self,
        base_library_path: str,
        learned_library_path: Optional[str] = None,
        feedback_log_path: Optional[str] = None,
    ):
        self.base_library_path = base_library_path
        self.learned_library_path = learned_library_path
        self.feedback_log_path = feedback_log_path
        self.strategies: Dict[str, Dict[str, Any]] = {}
        self.feedback: List[Dict[str, Any]] = []
        self.load()

    def load(self) -> None:
        self.strategies = {}
        self._load_strategy_file(self.base_library_path, source="base")

        if self.learned_library_path and os.path.exists(self.learned_library_path):
            self._load_strategy_file(self.learned_library_path, source="learned")

        if self.feedback_log_path and os.path.exists(self.feedback_log_path):
            self.feedback = self._load_jsonl(self.feedback_log_path)
            for record in self.feedback:
                self._apply_feedback(record, persist=False)

    def _load_strategy_file(self, path: str, source: str) -> None:
        if not os.path.exists(path):
            if source == "base":
                raise FileNotFoundError(f"Strategy library not found: {path}")
            return

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, dict):
            raise ValueError(f"Strategy library must be a JSON object: {path}")

        for key, value in raw.items():
            if not isinstance(value, dict) or not value.get("Strategy"):
                continue

            name = str(value["Strategy"])
            strategy_id = value.get("id") or key
            self.strategies[strategy_id] = {
                **value,
                "id": strategy_id,
                "Strategy": name,
                "source": value.get("source", source),
                "usage_count": int(value.get("usage_count", 0) or 0),
                "score_sum": _safe_float(value.get("score_sum")),
                "average_score": value.get("average_score"),
                "success_count": int(value.get("success_count", 0) or 0),
                "failure_count": int(value.get("failure_count", 0) or 0),
                "tags": value.get("tags", []),
                "updated_at": value.get("updated_at"),
            }

    @staticmethod
    def _load_jsonl(path: str) -> List[Dict[str, Any]]:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def list_strategies(self) -> List[Dict[str, Any]]:
        return list(self.strategies.values())

    def retrieve(
        self,
        text: str,
        risk_category: Optional[str] = None,
        top_k: int = 1,
    ) -> List[Dict[str, Any]]:
        query = f"{text} {risk_category or ''}".lower()
        scored = []

        for strategy in self.strategies.values():
            name = str(strategy.get("Strategy", ""))
            definition = str(strategy.get("Definition", ""))
            tags = " ".join(str(tag) for tag in strategy.get("tags", []))
            haystack = f"{name} {definition} {tags}".lower()

            lexical_score = sum(1 for token in query.split() if token and token in haystack)
            average_score = strategy.get("average_score")
            if not isinstance(average_score, (int, float)):
                average_score = 0.0

            source_bonus = 0.1 if strategy.get("source") == "learned" else 0.0
            usage_penalty = min(strategy.get("usage_count", 0), 20) * 0.005
            score = lexical_score + float(average_score) + source_bonus - usage_penalty
            scored.append((score, strategy))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [strategy for _, strategy in scored[:max(top_k, 1)]]

    def record_feedback(
        self,
        strategy_id: str,
        item_id: Any = None,
        round_index: int = 0,
        attempt_index: int = 0,
        score: Optional[float] = None,
        outcome: Optional[str] = None,
        notes: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        record = {
            "timestamp": _now(),
            "strategy_id": strategy_id,
            "item_id": item_id,
            "round": round_index,
            "attempt": attempt_index,
            "score": score,
            "outcome": outcome,
            "notes": notes,
            "metadata": metadata or {},
        }
        self.feedback.append(record)
        self._apply_feedback(record, persist=True)
        return record

    def _apply_feedback(self, record: Dict[str, Any], persist: bool) -> None:
        strategy_id = str(record.get("strategy_id"))
        strategy = self.strategies.get(strategy_id)
        if not strategy:
            return

        strategy["usage_count"] = int(strategy.get("usage_count", 0) or 0) + 1

        score = record.get("score")
        if isinstance(score, (int, float)):
            strategy["score_sum"] = _safe_float(strategy.get("score_sum")) + float(score)
            strategy["average_score"] = strategy["score_sum"] / strategy["usage_count"]

        outcome = str(record.get("outcome") or "").lower()
        if outcome in {"success", "valid", "valid_mutation", "passed"}:
            strategy["success_count"] = int(strategy.get("success_count", 0) or 0) + 1
        elif outcome:
            strategy["failure_count"] = int(strategy.get("failure_count", 0) or 0) + 1

        strategy["updated_at"] = _now()

        if persist and self.feedback_log_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.feedback_log_path)), exist_ok=True)
            with open(self.feedback_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def add_learned_strategy(
        self,
        strategy_name: str,
        definition: str,
        example: str = "",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        base_id = strategy_name.strip().replace(" ", "_") or "learned_strategy"
        strategy_id = base_id
        counter = 1
        while strategy_id in self.strategies:
            counter += 1
            strategy_id = f"{base_id}_{counter}"

        self.strategies[strategy_id] = {
            "id": strategy_id,
            "Strategy": strategy_name,
            "Definition": definition,
            "Example": example,
            "tags": tags or [],
            "metadata": metadata or {},
            "source": "learned",
            "usage_count": 0,
            "score_sum": 0.0,
            "average_score": None,
            "success_count": 0,
            "failure_count": 0,
            "created_at": _now(),
            "updated_at": _now(),
        }
        return strategy_id

    def summarize_feedback(
        self,
        min_average_score: float = 0.7,
        min_usage_count: int = 2,
    ) -> List[Dict[str, Any]]:
        candidates = []
        by_outcome: Dict[str, int] = defaultdict(int)

        for strategy in self.strategies.values():
            usage_count = int(strategy.get("usage_count", 0) or 0)
            average_score = strategy.get("average_score")
            if (
                usage_count >= min_usage_count
                and isinstance(average_score, (int, float))
                and average_score >= min_average_score
            ):
                candidates.append({
                    "strategy_id": strategy.get("id"),
                    "Strategy": strategy.get("Strategy"),
                    "average_score": average_score,
                    "usage_count": usage_count,
                    "source": strategy.get("source"),
                })

        for record in self.feedback:
            by_outcome[str(record.get("outcome") or "unspecified")] += 1

        return [{
            "generated_at": _now(),
            "total_feedback": len(self.feedback),
            "outcomes": dict(by_outcome),
            "high_score_strategies": candidates,
        }]

    def save_learned_library(self) -> Optional[str]:
        if not self.learned_library_path:
            return None

        learned = {
            strategy_id: strategy
            for strategy_id, strategy in self.strategies.items()
            if strategy.get("source") == "learned"
        }

        os.makedirs(os.path.dirname(os.path.abspath(self.learned_library_path)), exist_ok=True)
        with open(self.learned_library_path, "w", encoding="utf-8") as f:
            json.dump(learned, f, ensure_ascii=False, indent=2)

        return self.learned_library_path

    def export_updated_library(self, output_path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.strategies, f, ensure_ascii=False, indent=2)
        return output_path
