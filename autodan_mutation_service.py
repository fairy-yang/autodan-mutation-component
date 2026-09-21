#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Platform-facing lightweight AutoDAN-Turbo mutation service.

This module keeps the platform API small while making long input handling explicit:
- Load the existing strategy library once.
- Reuse a single Qwen2.5-1.5B-Instruct model instance.
- Accept one item or a list of items.
- Report prompt-budget truncation metadata instead of hiding length failures.

NOTE:
This lightweight component loads an existing Strategy Library for text mutation.
It does not run the paper's warm-up, target-model judging, or lifelong strategy
learning loop.
"""

import json
import os
from typing import Any, Dict, List, Union


from llm.huggingface_models import HuggingFaceModel
from framework.attacker import Attacker
from strategy_feedback import StrategyFeedbackStore, append_reviewed_strategy


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
CONFIG_DIR = os.path.join(BASE_DIR, "llm", "chat_templates")
CONFIG_NAME = "qwen2-instruct"
STRATEGY_LIBRARY = os.path.join(
    BASE_DIR,
    "strategies",
    "strategy_library.json",
)


class AutoDANMutationService:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        config_dir: str = CONFIG_DIR,
        config_name: str = CONFIG_NAME,
        strategy_library_path: str = STRATEGY_LIBRARY,
        temperature: float = 0.8,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
        max_context_tokens: int = None,
        allow_input_truncation: bool = True,
        truncation_strategy: str = "middle",
        feedback_store_path: str = None,
    ):
        self.model_name = model_name
        self.config_dir = config_dir
        self.config_name = config_name
        self.strategy_library_path = strategy_library_path
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.max_context_tokens = max_context_tokens
        self.allow_input_truncation = allow_input_truncation
        self.truncation_strategy = truncation_strategy
        self.feedback_store_path = feedback_store_path
        self.feedback_store = StrategyFeedbackStore(feedback_store_path)

        self.strategies = self._load_strategies(strategy_library_path)

        self.model = HuggingFaceModel(
            repo_name=model_name,
            config_dir=config_dir,
            config_name=config_name,
            token=None,
        )
        self.attacker = Attacker(self.model)

        self._strategy_cursor = 0

    @staticmethod
    def _load_strategies(path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Strategy library not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        strategies = [
            v for v in raw.values()
            if isinstance(v, dict) and v.get("Strategy")
        ]

        if not strategies:
            raise ValueError("No valid strategies found in strategy library.")

        return strategies

    def _next_strategy(self) -> Dict[str, Any]:
        strategy = self.strategies[self._strategy_cursor % len(self.strategies)]
        self._strategy_cursor += 1
        return strategy

    def mutate_one(self, item: Dict[str, Any]) -> Dict[str, Any]:
        if "text" not in item:
            raise ValueError("Input item must contain field: text")

        text = str(item["text"])
        strategy = self._next_strategy()

        mutated_text, _ = self.attacker.use_strategy(
            request=text,
            strategy_list=[strategy],
            max_new_tokens=self.max_new_tokens,
            max_context_tokens=self.max_context_tokens,
            allow_input_truncation=self.allow_input_truncation,
            truncation_strategy=self.truncation_strategy,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        generation_stats = getattr(self.model, "last_generation_stats", {}) or {}

        return {
            "id": item.get("id"),
            "original_text": text,
            "mutated_text": mutated_text,
            "strategy": strategy.get("Strategy"),
            "risk_category": item.get("risk_category"),
            "source": item.get("source", "generated"),
            "parent_id": item.get("parent_id"),
            "version": item.get("version", "v0.1"),
            "method": "AutoDAN-Turbo/Attacker.use_strategy",
            "model": self.model_name,
            "generation": generation_stats,
        }

    def record_feedback(
        self,
        strategy: str,
        score: float = None,
        outcome: str = None,
        notes: str = None,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        return self.feedback_store.record(
            strategy=strategy,
            score=score,
            outcome=outcome,
            notes=notes,
            metadata=metadata,
        )

    def feedback_report(self) -> Dict[str, Any]:
        return self.feedback_store.report()

    def add_reviewed_strategy(
        self,
        strategy_name: str,
        definition: str,
        example: str = "",
        metadata: Dict[str, Any] = None,
    ) -> str:
        key = append_reviewed_strategy(
            strategy_library_path=self.strategy_library_path,
            strategy_name=strategy_name,
            definition=definition,
            example=example,
            metadata=metadata,
        )
        self.strategies = self._load_strategies(self.strategy_library_path)
        return key

    def mutate(
        self,
        items: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        if isinstance(items, dict):
            return self.mutate_one(items)

        if isinstance(items, list):
            return [self.mutate_one(item) for item in items]

        raise TypeError("items must be a dict or a list of dicts")


_services = {}


def get_service(
    model_config: Dict[str, Any] = None
) -> AutoDANMutationService:
    """
    Get or create a reusable mutation service.

    model_config example:
    {
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "config_dir": "llm/chat_templates",
        "config_name": "qwen2-instruct",
        "strategy_library_path": "strategies/strategy_library.json",
        "max_new_tokens": 512,
        "max_context_tokens": 8192,
        "allow_input_truncation": true,
        "truncation_strategy": "middle",
        "feedback_store_path": "runs/strategy_feedback.jsonl"
    }
    """
    model_config = model_config or {}

    model_name = model_config.get("model_name", MODEL_NAME)
    config_dir = model_config.get("config_dir", CONFIG_DIR)
    config_name = model_config.get("config_name", CONFIG_NAME)
    strategy_library_path = model_config.get(
        "strategy_library_path",
        STRATEGY_LIBRARY
    )
    temperature = model_config.get("temperature", 0.8)
    top_p = model_config.get("top_p", 0.9)
    max_new_tokens = model_config.get("max_new_tokens", 512)
    max_context_tokens = model_config.get("max_context_tokens")
    allow_input_truncation = model_config.get("allow_input_truncation", True)
    truncation_strategy = model_config.get("truncation_strategy", "middle")
    feedback_store_path = model_config.get("feedback_store_path")

    service_key = (
        model_name,
        config_dir,
        config_name,
        strategy_library_path,
        temperature,
        top_p,
        max_new_tokens,
        max_context_tokens,
        allow_input_truncation,
        truncation_strategy,
        feedback_store_path,
    )

    if service_key not in _services:
        _services[service_key] = AutoDANMutationService(
            model_name=model_name,
            config_dir=config_dir,
            config_name=config_name,
            strategy_library_path=strategy_library_path,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            max_context_tokens=max_context_tokens,
            allow_input_truncation=allow_input_truncation,
            truncation_strategy=truncation_strategy,
            feedback_store_path=feedback_store_path,
        )

    return _services[service_key]


def run_mutation(
    items: Union[Dict[str, Any], List[Dict[str, Any]]],
    model_config: Dict[str, Any] = None,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    service = get_service(model_config)
    return service.mutate(items)


def record_strategy_feedback(
    strategy: str,
    score: float = None,
    outcome: str = None,
    notes: str = None,
    metadata: Dict[str, Any] = None,
    model_config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    service = get_service(model_config)
    return service.record_feedback(
        strategy=strategy,
        score=score,
        outcome=outcome,
        notes=notes,
        metadata=metadata,
    )


def get_strategy_feedback_report(
    model_config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    service = get_service(model_config)
    return service.feedback_report()


def add_reviewed_strategy(
    strategy_name: str,
    definition: str,
    example: str = "",
    metadata: Dict[str, Any] = None,
    model_config: Dict[str, Any] = None,
) -> str:
    service = get_service(model_config)
    return service.add_reviewed_strategy(
        strategy_name=strategy_name,
        definition=definition,
        example=example,
        metadata=metadata,
    )

try:
    from lifelong_runner import run_lifelong_mutation, run_lifelong_mutation_jsonl
except ImportError:
    run_lifelong_mutation = None
    run_lifelong_mutation_jsonl = None
