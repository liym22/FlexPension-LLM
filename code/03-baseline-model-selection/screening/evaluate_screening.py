#!/usr/bin/env python
"""Reproduce the 30-case screening metrics reported in the supplement."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


PARTICIPATE = "参保"
NON_PARTICIPATE = "不参保"
EMPLOYEE_PENSION = "城镇职工养老保险"


def _binary_f1(y_true: list[bool], y_pred: list[bool]) -> float:
    true_positive = sum(truth and prediction for truth, prediction in zip(y_true, y_pred))
    false_positive = sum(not truth and prediction for truth, prediction in zip(y_true, y_pred))
    false_negative = sum(truth and not prediction for truth, prediction in zip(y_true, y_pred))
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


def compute_metrics(
    ground_truth: dict[str, dict[str, str]],
    predictions: dict[str, tuple[str, str]],
) -> dict[str, float]:
    true_actions: list[bool] = []
    predicted_actions: list[bool] = []
    true_types: list[bool] = []
    predicted_types: list[bool] = []

    for sample_id, truth in ground_truth.items():
        predicted_action, predicted_type = predictions.get(sample_id, ("", ""))
        true_actions.append(truth["decision"] == PARTICIPATE)
        predicted_actions.append(predicted_action == PARTICIPATE)

        if truth["decision"] == PARTICIPATE and predicted_action == PARTICIPATE:
            true_types.append(truth["type"] == EMPLOYEE_PENSION)
            predicted_types.append(predicted_type == EMPLOYEE_PENSION)

    action_f1 = _binary_f1(true_actions, predicted_actions)
    type_f1 = _binary_f1(true_types, predicted_types) if true_types else 0.0
    return {
        "action_f1": action_f1,
        "type_f1": type_f1,
        "composite_f1": 0.6 * action_f1 + 0.4 * type_f1,
    }


def _majority(values: list[str]) -> str:
    return Counter(values).most_common(1)[0][0]


def vote_predictions(runs: dict[int, dict[str, tuple[str, str]]]) -> dict[str, tuple[str, str]]:
    if not runs:
        return {}
    sample_ids = sorted({sample_id for run in runs.values() for sample_id in run})
    seed_42 = runs.get(42, runs[min(runs)])
    voted: dict[str, tuple[str, str]] = {}

    for sample_id in sample_ids:
        action_votes = [run.get(sample_id, ("", ""))[0] for run in runs.values()]
        action = _majority(action_votes)
        if action != PARTICIPATE:
            voted[sample_id] = (NON_PARTICIPATE, NON_PARTICIPATE)
            continue

        type_votes = [
            run.get(sample_id, ("", ""))[1]
            for run in runs.values()
            if run.get(sample_id, ("", ""))[1] not in {"", NON_PARTICIPATE}
        ]
        if not type_votes:
            insurance_type = NON_PARTICIPATE
        else:
            counts = Counter(type_votes).most_common()
            insurance_type = counts[0][0] if counts[0][1] >= 2 else seed_42.get(sample_id, ("", ""))[1]
        voted[sample_id] = (action, insurance_type)
    return voted


def summarize_model_runs(
    ground_truth: dict[str, dict[str, str]],
    runs: dict[int, dict[str, tuple[str, str]]],
) -> dict[str, float | int]:
    seed_metrics = [compute_metrics(ground_truth, predictions) for predictions in runs.values()]
    vote_metrics = compute_metrics(ground_truth, vote_predictions(runs))
    return {
        "seed_count": len(seed_metrics),
        "average_composite_f1": sum(row["composite_f1"] for row in seed_metrics) / len(seed_metrics),
        "vote_composite_f1": vote_metrics["composite_f1"],
    }


def validate_model_runs(
    ground_truth: dict[str, dict[str, str]],
    runs: dict[int, dict[str, tuple[str, str]]],
    expected_seeds: list[int],
) -> None:
    missing_seeds = [seed for seed in expected_seeds if seed not in runs]
    if missing_seeds:
        joined = ", ".join(str(seed) for seed in missing_seeds)
        raise FileNotFoundError(f"Missing screening result seeds: {joined}")

    expected_ids = set(ground_truth)
    for seed in expected_seeds:
        actual_ids = set(runs[seed])
        missing_ids = sorted(expected_ids - actual_ids)
        extra_ids = sorted(actual_ids - expected_ids)
        if missing_ids or extra_ids:
            parts = []
            if missing_ids:
                parts.append("missing sample IDs: " + ", ".join(missing_ids))
            if extra_ids:
                parts.append("unexpected sample IDs: " + ", ".join(extra_ids))
            raise ValueError(f"seed {seed}: " + "; ".join(parts))


def load_ground_truth(path: Path) -> dict[str, dict[str, str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["id"]): {"decision": row["decision"], "type": row["type"]} for row in rows}


def load_result(path: Path) -> dict[str, tuple[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(row["sample_id"]): (
            row.get("predicted_action", ""),
            row.get("predicted_insurance_type", ""),
        )
        for row in payload.get("tests", [])
        if row.get("success")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--model", action="append", required=True, metavar="SHORT_NAME")
    parser.add_argument("--seeds", nargs="+", default=[42, 123, 456], type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    ground_truth = load_ground_truth(args.ground_truth)
    summaries = []
    for model_name in args.model:
        runs = {}
        for seed in args.seeds:
            path = args.results_dir / f"{model_name}_seed{seed}_temp05_results.json"
            if path.exists():
                runs[seed] = load_result(path)
        validate_model_runs(ground_truth, runs, args.seeds)
        summaries.append({"model": model_name, **summarize_model_runs(ground_truth, runs)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    main()
