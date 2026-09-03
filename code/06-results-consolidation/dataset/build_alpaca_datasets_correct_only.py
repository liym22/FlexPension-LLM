from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_CODE_DIR))

from config.paths import (
    CHFS2019_DKI_GROUND_TRUTH_FILE,
    CHFS2019_DKI_PROMPTS_FILE,
    CHFS2019_TEACHER_RESULTS_FILE,
    DISTILLATION_ANALYSIS_DIR,
    DISTILLATION_DATASETS_CORRECT_ONLY_DIR,
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_output_from_parsed(parsed_json: dict):
    return json.dumps(parsed_json, ensure_ascii=False, indent=2)


def build_test_output(sample: dict):
    return json.dumps(
        {
            "household_id": sample.get("household_id", ""),
            "individual_id": sample.get("individual_id", ""),
            "insurance_decision": {
                "action": sample.get("decision", ""),
                "insurance_type": sample.get("type", ""),
            },
        },
        ensure_ascii=False,
        indent=2,
    )


def main():
    dataset_dir = DISTILLATION_ANALYSIS_DIR
    correct_only_dir = dataset_dir / "correct_only"
    output_dir = DISTILLATION_DATASETS_CORRECT_ONLY_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts_path = CHFS2019_DKI_PROMPTS_FILE
    gt_path = CHFS2019_DKI_GROUND_TRUTH_FILE
    claude_results_path = CHFS2019_TEACHER_RESULTS_FILE

    # Use the split IDs generated for the correct-only ablation.
    train_ids_path = correct_only_dir / "train_ids.json"
    val_ids_path = correct_only_dir / "val_ids.json"
    test_ids_path = correct_only_dir / "test_ids.json"

    if not train_ids_path.exists() or not val_ids_path.exists() or not test_ids_path.exists():
        raise FileNotFoundError("Missing train/val/test ID files under correct_only")

    prompts = load_json(prompts_path)
    ground_truth = load_json(gt_path)

    prompt_map = {item["id"]: item["prompt"] for item in prompts}
    gt_map = {item["id"]: item for item in ground_truth}

    train_ids = set(load_json(train_ids_path))
    val_ids = set(load_json(val_ids_path))
    test_ids = set(load_json(test_ids_path))

    claude_results = load_json(claude_results_path)
    claude_tests = claude_results.get("tests", [])
    claude_map = {
        item.get("sample_id"): item
        for item in claude_tests
        if item.get("sample_id")
    }

    train_rows = []
    val_rows = []
    test_rows = []
    missing_outputs = []
    length_stats = {
        "train_chars": [],
        "val_chars": [],
        "test_chars": [],
    }
    output_length_stats = {
        "train_output_chars": [],
        "val_output_chars": [],
        "test_output_chars": [],
    }

    # Build the unchanged test split.
    for sample_id in sorted(test_ids):
        prompt = prompt_map.get(sample_id)
        gt = gt_map.get(sample_id)
        if not prompt or not gt:
            missing_outputs.append(sample_id)
            continue
        output_text = build_test_output(gt)
        length_stats["test_chars"].append(len(prompt) + len(output_text))
        output_length_stats["test_output_chars"].append(len(output_text))
        test_rows.append(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": output_text},
                ]
            }
        )

    # Build train and validation splits from teacher-correct cases only.
    for sample_id in sorted(train_ids | val_ids):
        prompt = prompt_map.get(sample_id)
        gt = gt_map.get(sample_id)
        if not prompt or not gt:
            missing_outputs.append(sample_id)
            continue

        # These cases were prefiltered, so their teacher JSON is label-correct.
        claude_item = claude_map.get(sample_id)
        if not claude_item or not claude_item.get("parse_success") or not claude_item.get("success"):
            missing_outputs.append(sample_id)
            continue

        parsed_json = claude_item.get("parsed_json")
        if not parsed_json:
            missing_outputs.append(sample_id)
            continue

        output_text = build_output_from_parsed(parsed_json)
        split_key = "train" if sample_id in train_ids else "val"
        length_stats[f"{split_key}_chars"].append(len(prompt) + len(output_text))
        output_length_stats[f"{split_key}_output_chars"].append(len(output_text))
        row = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": output_text},
            ]
        }

        if sample_id in train_ids:
            train_rows.append(row)
        else:
            val_rows.append(row)

    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "val.jsonl", val_rows)
    write_jsonl(output_dir / "test.jsonl", test_rows)

    def recommend_max_length(lengths):
        if not lengths:
            return {
                "max_chars": 0,
                "p95_chars": 0,
                "p99_chars": 0,
                "recommended_max_tokens": 0,
            }
        sorted_lengths = sorted(lengths)
        p95_index = max(0, int(math.ceil(len(sorted_lengths) * 0.95)) - 1)
        p99_index = max(0, int(math.ceil(len(sorted_lengths) * 0.99)) - 1)
        p95_chars = sorted_lengths[p95_index]
        p99_chars = sorted_lengths[p99_index]
        max_chars = sorted_lengths[-1]
        # Approximate one token per Chinese character, with a small margin.
        recommended = int(math.ceil(max(p99_chars, max_chars) * 1.05))
        return {
            "max_chars": max_chars,
            "p95_chars": p95_chars,
            "p99_chars": p99_chars,
            "recommended_max_tokens": recommended,
        }

    summary = {
        "train_total": len(train_ids),
        "val_total": len(val_ids),
        "test_total": len(test_ids),
        "train_written": len(train_rows),
        "val_written": len(val_rows),
        "test_written": len(test_rows),
        "missing_outputs": len(missing_outputs),
        "missing_sample_ids": missing_outputs,
        "max_length_recommendation": {
            "train": recommend_max_length(length_stats["train_chars"]),
            "val": recommend_max_length(length_stats["val_chars"]),
            "test": recommend_max_length(length_stats["test_chars"]),
            "combined": recommend_max_length(
                length_stats["train_chars"]
                + length_stats["val_chars"]
                + length_stats["test_chars"]
            ),
        },
        "max_new_tokens_recommendation": {
            "train": recommend_max_length(output_length_stats["train_output_chars"]),
            "val": recommend_max_length(output_length_stats["val_output_chars"]),
            "test": recommend_max_length(output_length_stats["test_output_chars"]),
            "combined": recommend_max_length(
                output_length_stats["train_output_chars"]
                + output_length_stats["val_output_chars"]
                + output_length_stats["test_output_chars"]
            ),
        },
    }

    summary_path = output_dir / "alpaca_dataset_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("✓ Alpaca dataset files generated for correct_only split!")
    print(f"  Output directory: {output_dir}")
    print("\nDataset statistics:")
    print(f"  Train: {summary['train_written']}/{summary['train_total']}")
    print(f"  Validation: {summary['val_written']}/{summary['val_total']}")
    print(f"  Test: {summary['test_written']}/{summary['test_total']}")
    print(f"  Missing samples: {summary['missing_outputs']}")
    print(f"\nRecommended max_length: {summary['max_length_recommendation']['combined']['recommended_max_tokens']}")
    print(f"Recommended max_new_tokens: {summary['max_new_tokens_recommendation']['combined']['recommended_max_tokens']}")
    print(f"\nDetailed statistics saved to: {summary_path}")


if __name__ == "__main__":
    main()
