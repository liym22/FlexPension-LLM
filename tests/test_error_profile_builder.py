import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "11-figure-generation"
    / "build_error_profile.py"
)
SPEC = importlib.util.spec_from_file_location("build_error_profile", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_error_classification_matches_reported_categories():
    assert MODULE.classify_error("参保", "城乡居民养老保险", "不参保", "不参保") == "A_false_negative"
    assert MODULE.classify_error("不参保", "不参保", "参保", "城乡居民养老保险") == "C_false_positive"
    assert MODULE.classify_error("参保", "城乡居民养老保险", "参保", "城镇职工养老保险") == "other_error"
    assert MODULE.classify_error("参保", "城乡居民养老保险", "参保", "城乡居民养老保险") == "correct"


def test_profile_counts_sum_to_all_samples():
    truth = {
        "a": ("参保", "城乡居民养老保险"),
        "b": ("不参保", "不参保"),
        "c": ("参保", "城镇职工养老保险"),
    }
    predictions = {
        "a": ("不参保", "不参保"),
        "b": ("参保", "城乡居民养老保险"),
        "c": ("参保", "城乡居民养老保险"),
    }

    profile = MODULE.build_profile("Example", truth, predictions)

    assert profile["total"] == 3
    assert profile["total_error"] == 3
    assert profile["A_false_negative"] == 1
    assert profile["C_false_positive"] == 1
    assert profile["other_error"] == 1


def test_profile_rejects_missing_predictions():
    with pytest.raises(ValueError, match="missing predictions"):
        MODULE.build_profile(
            "Example",
            {"a": ("参保", "城乡居民养老保险"), "b": ("不参保", "不参保")},
            {"a": ("参保", "城乡居民养老保险")},
        )


def test_restrict_ground_truth_uses_blind_ids_only():
    restricted = MODULE.restrict_ground_truth(
        {"a": ("参保", "城乡居民养老保险"), "b": ("不参保", "不参保")},
        ["b"],
    )

    assert restricted == {"b": ("不参保", "不参保")}
