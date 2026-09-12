#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Platform-facing lightweight AutoDAN-Turbo mutation service.

Design:
- Reuse original repository HuggingFaceModel
- Reuse original framework.attacker.Attacker.use_strategy()
- Load existing lifelong strategy library
- Load Qwen2.5-1.5B once and reuse it
- Accept one item or a list of items
- Return structured Python dict/list for platform integration

NOTE:
This is the lightweight mutation component extracted from AutoDAN-Turbo.
It does not rerun warm-up/lifelong strategy training.
"""

import json
import os
from typing import Dict, List, Union, Any

from llm.huggingface_models import HuggingFaceModel
from framework.attacker import Attacker


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
    ):
        self.model_name = model_name
        self.config_dir = config_dir
        self.config_name = config_name
        self.strategy_library_path = strategy_library_path
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

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
            max_length=10000,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
        )

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
        }

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
        "strategy_library_path": "strategies/strategy_library.json"
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

    service_key = (
        model_name,
        config_dir,
        config_name,
        strategy_library_path,
        temperature,
        top_p,
        max_new_tokens,
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
        )

    return _services[service_key]


def run_mutation(
    items: Union[Dict[str, Any], List[Dict[str, Any]]],
    model_config: Dict[str, Any] = None,
):
    """
    Platform entrypoint.

    Supports:
    - default local configuration
    - platform-provided model configuration
    - single item or batch input
    """
    return get_service(model_config=model_config).mutate(items)
