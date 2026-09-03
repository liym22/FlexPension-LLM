import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "03-baseline-model-selection"
    / "screening"
    / "evaluate_screening.py"
)
SPEC = importlib.util.spec_from_file_location("evaluate_screening", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

RUNNER_PATH = MODULE_PATH.with_name("run_screening.py")
RUNNER_SPEC = importlib.util.spec_from_file_location("run_screening", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = RUNNER
RUNNER_SPEC.loader.exec_module(RUNNER)


def test_screening_summary_uses_seed_average_and_majority_vote():
    ground_truth = {
        "a": {"decision": "参保", "type": "城镇职工养老保险"},
        "b": {"decision": "参保", "type": "城乡居民养老保险"},
        "c": {"decision": "不参保", "type": "不参保"},
    }
    runs = {
        42: {
            "a": ("参保", "城镇职工养老保险"),
            "b": ("不参保", "不参保"),
            "c": ("不参保", "不参保"),
        },
        123: {
            "a": ("参保", "城镇职工养老保险"),
            "b": ("参保", "城乡居民养老保险"),
            "c": ("参保", "城乡居民养老保险"),
        },
        456: {
            "a": ("参保", "城镇职工养老保险"),
            "b": ("参保", "城乡居民养老保险"),
            "c": ("不参保", "不参保"),
        },
    }

    summary = MODULE.summarize_model_runs(ground_truth, runs)

    expected_average = sum(MODULE.compute_metrics(ground_truth, run)["composite_f1"] for run in runs.values()) / 3
    assert summary["seed_count"] == 3
    assert summary["average_composite_f1"] == expected_average
    assert summary["vote_composite_f1"] == 1.0


def test_three_way_type_tie_uses_seed_42_prediction():
    runs = {
        42: {"a": ("参保", "城乡居民养老保险")},
        123: {"a": ("参保", "城镇职工养老保险")},
        456: {"a": ("参保", "其他养老保险")},
    }

    voted = MODULE.vote_predictions(runs)

    assert voted["a"] == ("参保", "城乡居民养老保险")


def test_screening_payload_preserves_historical_seed_and_temperature():
    payload = RUNNER.build_payload(
        model_id="example/model",
        prompt="case prompt",
        seed=123,
        temperature=0.5,
        max_tokens=5000,
    )

    assert payload["model"] == "example/model"
    assert payload["seed"] == 123
    assert payload["temperature"] == 0.5
    assert payload["max_tokens"] == 5000
    assert payload["response_format"] == {"type": "json_object"}


def test_validate_model_runs_rejects_missing_seed():
    ground_truth = {"a": {"decision": "不参保", "type": "不参保"}}
    runs = {42: {"a": ("不参保", "不参保")}}

    with pytest.raises(FileNotFoundError, match="123, 456"):
        MODULE.validate_model_runs(ground_truth, runs, [42, 123, 456])


def test_validate_model_runs_rejects_incomplete_predictions():
    ground_truth = {
        "a": {"decision": "不参保", "type": "不参保"},
        "b": {"decision": "参保", "type": "城乡居民养老保险"},
    }
    runs = {
        42: {"a": ("不参保", "不参保")},
        123: {
            "a": ("不参保", "不参保"),
            "b": ("参保", "城乡居民养老保险"),
        },
        456: {
            "a": ("不参保", "不参保"),
            "b": ("参保", "城乡居民养老保险"),
        },
    }

    with pytest.raises(ValueError, match="seed 42.*missing.*b"):
        MODULE.validate_model_runs(ground_truth, runs, [42, 123, 456])
