from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "10-expert-evaluation"
    / "analyze_expert_evaluation.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_expert_evaluation", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_summarize_balanced_anonymous_responses() -> None:
    rows = []
    scores = {
        "R01": {
            "qwen_zs": [(1, 2, 0), (3, 4, 0)],
            "claude_sonnet_4_5_teacher": [(2, 3, 0), (4, 5, 1)],
            "flexpension_llm": [(5, 4, 1), (5, 5, 0)],
        },
        "R02": {
            "qwen_zs": [(2, 2, 0), (2, 2, 0)],
            "claude_sonnet_4_5_teacher": [(3, 3, 1), (3, 3, 0)],
            "flexpension_llm": [(4, 4, 0), (4, 4, 1)],
        },
    }
    for respondent_id, by_model in scores.items():
        for model, case_values in by_model.items():
            for case_index, (soundness, completeness, trusted) in enumerate(
                case_values, start=1
            ):
                rows.append(
                    {
                        "respondent_id": respondent_id,
                        "case_id": f"C{case_index:02d}",
                        "model": model,
                        "soundness": soundness,
                        "completeness": completeness,
                        "trusted": trusted,
                    }
                )

    summary = MODULE.summarize(rows)

    assert summary["design"] == {
        "respondents": 2,
        "cases": 2,
        "models": 3,
        "ratings_per_model": 4,
        "trust_choices": 4,
    }
    flex = summary["models"]["flexpension_llm"]
    assert flex["soundness_mean"] == pytest.approx(4.5)
    assert flex["completeness_mean"] == pytest.approx(4.25)
    assert flex["mean_rating"] == pytest.approx(4.375)
    assert flex["trust_count"] == 2
    assert flex["trust_share"] == pytest.approx(0.5)
    assert summary["respondent_comparisons"]["flexpension_llm_over_qwen_zs"] == {
        "higher_mean_rating": 2,
        "higher_trust_share": 2,
        "respondents": 2,
    }


def test_rejects_more_than_one_trusted_model_per_case() -> None:
    rows = [
        {
            "respondent_id": "R01",
            "case_id": "C01",
            "model": model,
            "soundness": 3,
            "completeness": 3,
            "trusted": int(model != "qwen_zs"),
        }
        for model in MODULE.MODELS
    ]

    with pytest.raises(ValueError, match="exactly one trusted model"):
        MODULE.summarize(rows)
