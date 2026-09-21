#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch mutation runner with a lightweight lifelong strategy loop.

The runner performs multiple rounds and attempts per item. It retrieves a
strategy, calls the mutation service, records structured metadata, and optionally
accepts externally supplied feedback for strategy statistics.
"""

import json
from typing import Any, Dict, Iterable, List, Optional, Union

from autodan_mutation_service import get_service
from lifelong_strategy import LifelongStrategyLibrary


def _as_items(items: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if isinstance(items, dict):
        return [items]
    if isinstance(items, list):
        return items
    raise TypeError("items must be a dict or a list of dicts")


def run_lifelong_mutation(
    items: Union[Dict[str, Any], List[Dict[str, Any]]],
    model_config: Optional[Dict[str, Any]] = None,
    rounds: int = 3,
    attempts_per_item: int = 5,
    top_k: int = 1,
    feedback_by_attempt: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    model_config = model_config or {}
    service = get_service(model_config)

    library = LifelongStrategyLibrary(
        base_library_path=service.strategy_library_path,
        learned_library_path=model_config.get("learned_strategy_library_path"),
        feedback_log_path=model_config.get("feedback_log_path"),
    )

    outputs: List[Dict[str, Any]] = []
    feedback_by_attempt = feedback_by_attempt or {}

    for item in _as_items(items):
        original_text = str(item.get("text", ""))
        for round_index in range(1, rounds + 1):
            for attempt_index in range(1, attempts_per_item + 1):
                retrieved = library.retrieve(
                    text=original_text,
                    risk_category=item.get("risk_category"),
                    top_k=top_k,
                )
                strategy = retrieved[0]
                service.strategies = [strategy]

                result = service.mutate_one(item)
                result.update({
                    "round": round_index,
                    "attempt": attempt_index,
                    "strategy_id": strategy.get("id"),
                    "strategy_source": strategy.get("source", "base"),
                })

                attempt_key = f"{item.get('id')}:{round_index}:{attempt_index}"
                feedback = feedback_by_attempt.get(attempt_key)
                if feedback:
                    library.record_feedback(
                        strategy_id=strategy.get("id"),
                        item_id=item.get("id"),
                        round_index=round_index,
                        attempt_index=attempt_index,
                        score=feedback.get("score"),
                        outcome=feedback.get("outcome"),
                        notes=feedback.get("notes"),
                        metadata=feedback.get("metadata"),
                    )

                outputs.append(result)

    if model_config.get("updated_strategy_library_path"):
        library.export_updated_library(model_config["updated_strategy_library_path"])

    if model_config.get("learned_strategy_library_path"):
        library.save_learned_library()

    return outputs


def run_lifelong_mutation_jsonl(
    input_path: str,
    output_path: str,
    model_config: Optional[Dict[str, Any]] = None,
    rounds: int = 3,
    attempts_per_item: int = 5,
) -> str:
    items: List[Dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    outputs = run_lifelong_mutation(
        items=items,
        model_config=model_config,
        rounds=rounds,
        attempts_per_item=attempts_per_item,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        for item in outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return output_path
