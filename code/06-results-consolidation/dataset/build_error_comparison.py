from __future__ import annotations

import json
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_CODE_DIR))

from config.paths import (
    CHFS2019_DKI_GROUND_TRUTH_FILE,
    CHFS2019_TEACHER_RESULTS_FILE,
    DISTILLATION_ANALYSIS_DIR,
    DISTILLATION_REGEN_RESULTS_FILE,
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_prediction_correct(pred_action: str, pred_type: str, gt_action: str, gt_type: str) -> bool:
    """Return whether both hierarchical decision fields match."""
    return pred_action == gt_action and pred_type == gt_type


def main():
    dataset_dir = DISTILLATION_ANALYSIS_DIR
    output_path = dataset_dir / "error_comparison.jsonl"

    # Load dataset split IDs.
    train_ids = set(load_json(dataset_dir / "train_ids.json"))
    val_ids = set(load_json(dataset_dir / "val_ids.json"))

    # Load ground truth labels.
    gt_path = CHFS2019_DKI_GROUND_TRUTH_FILE
    ground_truth = load_json(gt_path)
    gt_map = {item["id"]: item for item in ground_truth}

    # Load original Claude teacher predictions.
    claude_results_path = CHFS2019_TEACHER_RESULTS_FILE
    claude_results = load_json(claude_results_path)
    claude_tests = claude_results.get("tests", [])
    claude_map = {
        item.get("sample_id"): item
        for item in claude_tests
        if item.get("sample_id")
    }

    # Load regenerated teacher-error outputs.
    renew_results_path = DISTILLATION_REGEN_RESULTS_FILE
    renew_map = {}
    if renew_results_path.exists():
        renew_results = load_json(renew_results_path)
        renew_tests = renew_results.get("tests", []) if isinstance(renew_results, dict) else []
        renew_map = {
            item.get("sample_id"): item
            for item in renew_tests
            if item.get("sample_id")
        }

    comparison_rows = []
    stats = {
        "total_errors": 0,
        "train_errors": 0,
        "val_errors": 0,
        "corrected_by_rerun": 0,
        "still_wrong_after_rerun": 0,
        "no_rerun_result": 0,
    }

    # Compare cases in the train and validation splits.
    for sample_id in sorted(train_ids | val_ids):
        gt = gt_map.get(sample_id)
        claude_item = claude_map.get(sample_id)

        if not gt or not claude_item:
            continue

        # Read the ground truth decision.
        gt_action = gt.get("decision", "")
        gt_type = gt.get("type", "")

        # Read the original teacher prediction.
        claude_parsed = claude_item.get("parsed_json", {})
        claude_insurance = claude_parsed.get("insurance_decision", {})
        claude_action = claude_insurance.get("action", "")
        claude_type = claude_insurance.get("insurance_type", "")
        claude_decision_process = claude_parsed.get("decision_process", {})

        # Check whether the original teacher prediction is correct.
        is_original_correct = is_prediction_correct(claude_action, claude_type, gt_action, gt_type)

        # This analysis covers teacher-error cases only.
        if is_original_correct:
            continue

        stats["total_errors"] += 1
        if sample_id in train_ids:
            stats["train_errors"] += 1
            split = "train"
        else:
            stats["val_errors"] += 1
            split = "val"

        # Build the before/after comparison record.
        comparison = {
            "sample_id": sample_id,
            "split": split,
            "ground_truth": {
                "action": gt_action,
                "insurance_type": gt_type,
            },
            "claude_original": {
                "action": claude_action,
                "insurance_type": claude_type,
                "decision_process": claude_decision_process,
            },
        }

        # Attach the regenerated result when available.
        renew_item = renew_map.get(sample_id)
        if renew_item and renew_item.get("success") and renew_item.get("parse_success"):
            renew_parsed = renew_item.get("parsed_json", {})
            renew_insurance = renew_parsed.get("insurance_decision", {})
            renew_action = renew_insurance.get("action", "")
            renew_type = renew_insurance.get("insurance_type", "")
            renew_decision_process = renew_parsed.get("decision_process", {})

            comparison["claude_rerun"] = {
                "action": renew_action,
                "insurance_type": renew_type,
                "decision_process": renew_decision_process,
            }

            # Check whether regeneration recovers the observed label.
            is_rerun_correct = is_prediction_correct(renew_action, renew_type, gt_action, gt_type)
            comparison["is_corrected"] = is_rerun_correct

            if is_rerun_correct:
                stats["corrected_by_rerun"] += 1
            else:
                stats["still_wrong_after_rerun"] += 1
        else:
            comparison["claude_rerun"] = None
            comparison["is_corrected"] = False
            stats["no_rerun_result"] += 1

        comparison_rows.append(comparison)

    # Write case-level comparisons to JSONL.
    write_jsonl(output_path, comparison_rows)

    # Save aggregate comparison statistics.
    summary = {
        "total_errors": stats["total_errors"],
        "by_split": {
            "train": stats["train_errors"],
            "val": stats["val_errors"],
        },
        "rerun_results": {
            "corrected": stats["corrected_by_rerun"],
            "still_wrong": stats["still_wrong_after_rerun"],
            "no_result": stats["no_rerun_result"],
        },
        "correction_rate": f"{stats['corrected_by_rerun'] / stats['total_errors'] * 100:.2f}%" if stats["total_errors"] > 0 else "N/A",
    }

    summary_path = dataset_dir / "error_comparison_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Teacher-error comparison dataset generated.")
    print(f"  Output file: {output_path}")
    print(f"  Summary: {summary_path}")
    print("\nTeacher-error statistics:")
    print(f"  Total errors: {summary['total_errors']}")
    print(f"  Train errors: {summary['by_split']['train']}")
    print(f"  Validation errors: {summary['by_split']['val']}")
    print("\nRegeneration results:")
    print(f"  Corrected: {summary['rerun_results']['corrected']}")
    print(f"  Still wrong: {summary['rerun_results']['still_wrong']}")
    print(f"  Missing result: {summary['rerun_results']['no_result']}")
    print(f"  Correction rate: {summary['correction_rate']}")


if __name__ == "__main__":
    main()
