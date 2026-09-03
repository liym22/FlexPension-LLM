from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "code"
    / "08-validation-experiments"
    / "build_external_student_dataset.py"
)


def load_module():
    assert MODULE_PATH.exists(), "external student dataset builder is missing"
    spec = importlib.util.spec_from_file_location("external_student_dataset", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_external_rows_matches_prompts_and_ground_truth():
    module = load_module()
    prompts = [
        {
            "id": "h1-p1",
            "household_id": "h1",
            "individual_id": "p1",
            "prompt": "家庭ID: h1\n个人ID: p1\ncase prompt",
        }
    ]
    truths = [
        {
            "id": "h1-p1",
            "decision": "参保",
            "type": "城乡居民养老保险",
        }
    ]

    rows = module.build_external_rows(prompts, truths, expected_size=1)

    assert rows[0]["messages"][0] == {"role": "user", "content": prompts[0]["prompt"]}
    assistant = json.loads(rows[0]["messages"][1]["content"])
    assert assistant == {
        "household_id": "h1",
        "individual_id": "p1",
        "insurance_decision": {
            "action": "参保",
            "insurance_type": "城乡居民养老保险",
        },
    }


def test_build_external_rows_rejects_unmatched_ids():
    module = load_module()
    prompts = [
        {
            "id": "h1-p1",
            "household_id": "h1",
            "individual_id": "p1",
            "prompt": "case prompt",
        }
    ]
    truths = [{"id": "h2-p2", "decision": "不参保", "type": "不参保"}]

    with pytest.raises(ValueError, match="prompt and ground-truth IDs differ"):
        module.build_external_rows(prompts, truths, expected_size=1)
