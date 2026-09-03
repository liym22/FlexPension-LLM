from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


MODELS = (
    "qwen_zs",
    "claude_sonnet_4_5_teacher",
    "flexpension_llm",
)
REQUIRED_COLUMNS = {
    "respondent_id",
    "case_id",
    "model",
    "soundness",
    "completeness",
    "trusted",
}


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
        return list(reader)


def _normalize(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        item = {
            "respondent_id": str(row["respondent_id"]).strip(),
            "case_id": str(row["case_id"]).strip(),
            "model": str(row["model"]).strip(),
            "soundness": int(row["soundness"]),
            "completeness": int(row["completeness"]),
            "trusted": int(row["trusted"]),
        }
        if not item["respondent_id"] or not item["case_id"]:
            raise ValueError("respondent_id and case_id must be non-empty")
        if item["model"] not in MODELS:
            raise ValueError(f"unknown model: {item['model']}")
        if item["soundness"] not in range(1, 6):
            raise ValueError("soundness must be an integer from 1 to 5")
        if item["completeness"] not in range(1, 6):
            raise ValueError("completeness must be an integer from 1 to 5")
        if item["trusted"] not in (0, 1):
            raise ValueError("trusted must be 0 or 1")
        key = (item["respondent_id"], item["case_id"], item["model"])
        if key in seen:
            raise ValueError(f"duplicate response row: {key}")
        seen.add(key)
        normalized.append(item)
    if not normalized:
        raise ValueError("response file is empty")
    return normalized


def _validate_balanced(rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    respondents = sorted({row["respondent_id"] for row in rows})
    cases = sorted({row["case_id"] for row in rows})
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["respondent_id"], row["case_id"])].append(row)

    expected_groups = {(respondent, case) for respondent in respondents for case in cases}
    if set(grouped) != expected_groups:
        raise ValueError("every respondent must rate every case")
    for key, group in grouped.items():
        if {row["model"] for row in group} != set(MODELS):
            raise ValueError(f"each respondent-case pair must rate all models: {key}")
        if sum(row["trusted"] for row in group) != 1:
            raise ValueError(f"each respondent-case pair must have exactly one trusted model: {key}")
    return respondents, cases


def _sample_sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    normalized = _normalize(rows)
    respondents, cases = _validate_balanced(normalized)
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_respondent_model: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        by_model[row["model"]].append(row)
        by_respondent_model[(row["respondent_id"], row["model"])].append(row)

    trust_choices = len(respondents) * len(cases)
    model_summary: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        model_rows = by_model[model]
        soundness = [float(row["soundness"]) for row in model_rows]
        completeness = [float(row["completeness"]) for row in model_rows]
        trust_count = sum(row["trusted"] for row in model_rows)
        model_summary[model] = {
            "soundness_mean": statistics.fmean(soundness),
            "soundness_sd": _sample_sd(soundness),
            "completeness_mean": statistics.fmean(completeness),
            "completeness_sd": _sample_sd(completeness),
            "mean_rating": statistics.fmean(soundness + completeness),
            "trust_count": trust_count,
            "trust_share": trust_count / trust_choices,
        }

    comparisons: dict[str, dict[str, int]] = {}
    target = "flexpension_llm"
    for comparator in ("qwen_zs", "claude_sonnet_4_5_teacher"):
        higher_rating = 0
        higher_trust = 0
        for respondent in respondents:
            target_rows = by_respondent_model[(respondent, target)]
            comparator_rows = by_respondent_model[(respondent, comparator)]
            target_rating = statistics.fmean(
                [
                    value
                    for row in target_rows
                    for value in (row["soundness"], row["completeness"])
                ]
            )
            comparator_rating = statistics.fmean(
                [
                    value
                    for row in comparator_rows
                    for value in (row["soundness"], row["completeness"])
                ]
            )
            higher_rating += target_rating > comparator_rating
            higher_trust += sum(row["trusted"] for row in target_rows) > sum(
                row["trusted"] for row in comparator_rows
            )
        comparisons[f"{target}_over_{comparator}"] = {
            "higher_mean_rating": higher_rating,
            "higher_trust_share": higher_trust,
            "respondents": len(respondents),
        }

    return {
        "design": {
            "respondents": len(respondents),
            "cases": len(cases),
            "models": len(MODELS),
            "ratings_per_model": len(respondents) * len(cases),
            "trust_choices": trust_choices,
        },
        "models": model_summary,
        "respondent_comparisons": comparisons,
    }


def write_model_csv(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "soundness_mean",
        "soundness_sd",
        "completeness_mean",
        "completeness_sd",
        "mean_rating",
        "trust_count",
        "trust_share",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model in MODELS:
            writer.writerow({"model": model, **summary["models"][model]})


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze anonymized expert ratings.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()

    summary = summarize(read_rows(args.input))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.output_csv:
        write_model_csv(summary, args.output_csv)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
