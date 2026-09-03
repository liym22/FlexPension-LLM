#!/usr/bin/env python
"""Build non-identifying error-profile aggregates from private predictions."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


def classify_error(true_action: str, true_type: str, predicted_action: str, predicted_type: str) -> str:
    if predicted_action == true_action:
        if true_action == "不参保" or predicted_type == true_type:
            return "correct"
        return "other_error"
    if true_action == "参保" and predicted_action == "不参保":
        return "A_false_negative"
    if true_action == "不参保" and predicted_action == "参保":
        return "C_false_positive"
    return "other_error"


def build_profile(
    model: str,
    ground_truth: dict[str, tuple[str, str]],
    predictions: dict[str, tuple[str, str]],
) -> dict[str, str | int | float]:
    missing = sorted(set(ground_truth) - set(predictions))
    if missing:
        raise ValueError(f"{model}: missing predictions for {len(missing)} samples")
    counts = Counter(
        classify_error(*truth, *predictions[sample_id])
        for sample_id, truth in ground_truth.items()
    )
    total = len(ground_truth)
    total_error = total - counts["correct"]
    return {
        "model": model,
        "total": total,
        "total_error": total_error,
        "accuracy": counts["correct"] / total,
        "A_false_negative": counts["A_false_negative"],
        "C_false_positive": counts["C_false_positive"],
        "other_error": counts["other_error"],
        "A_share": counts["A_false_negative"] / total_error if total_error else 0.0,
        "C_share": counts["C_false_positive"] / total_error if total_error else 0.0,
        "other_share": counts["other_error"] / total_error if total_error else 0.0,
    }


def load_ground_truth(path: Path) -> dict[str, tuple[str, str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["id"]): (row["decision"], row["type"]) for row in rows}


def restrict_ground_truth(
    ground_truth: dict[str, tuple[str, str]], sample_ids: list[str]
) -> dict[str, tuple[str, str]]:
    missing = [sample_id for sample_id in sample_ids if sample_id not in ground_truth]
    if missing:
        raise ValueError(f"Ground truth is missing {len(missing)} requested sample IDs")
    return {sample_id: ground_truth[sample_id] for sample_id in sample_ids}


def _json_from_text(text: str) -> dict:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("No JSON object found in model response")
    return json.JSONDecoder().raw_decode(cleaned[start:])[0]


def load_predictions(path: Path) -> dict[str, tuple[str, str]]:
    if path.suffix == ".jsonl":
        predictions = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                labels = _json_from_text(row["labels"])
                response = _json_from_text(row["response"])
                sample_id = f"{labels['household_id']}-{labels['individual_id']}"
                decision = response.get("insurance_decision") or {}
                predictions[sample_id] = (decision.get("action", ""), decision.get("insurance_type", ""))
        return predictions

    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(row["sample_id"]): (
            row.get("predicted_action", ""),
            row.get("predicted_insurance_type", ""),
        )
        for row in payload.get("tests", [])
        if row.get("success", True)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--sample-ids", required=True, type=Path)
    parser.add_argument("--prediction", action="append", required=True, metavar="MODEL=PATH")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    sample_ids = [str(value) for value in json.loads(args.sample_ids.read_text(encoding="utf-8"))]
    ground_truth = restrict_ground_truth(load_ground_truth(args.ground_truth), sample_ids)
    profiles = []
    for item in args.prediction:
        model, raw_path = item.split("=", 1)
        profiles.append(build_profile(model, ground_truth, load_predictions(Path(raw_path))))

    for profile in profiles:
        for field in ("accuracy", "A_share", "C_share", "other_share"):
            profile[field] = f"{float(profile[field]):.6f}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(profiles[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(profiles)


if __name__ == "__main__":
    main()
