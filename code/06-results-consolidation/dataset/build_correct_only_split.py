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
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def is_prediction_correct(pred: dict, gt: dict) -> bool:
    """Return whether both hierarchical decision fields match."""
    pred_action = pred.get("decision", "")
    pred_type = pred.get("type", "")
    gt_action = gt.get("decision", "")
    gt_type = gt.get("type", "")

    return pred_action == gt_action and pred_type == gt_type


def main():
    dataset_dir = DISTILLATION_ANALYSIS_DIR

    # Write the ablation split under a dedicated correct-only directory.
    output_dir = dataset_dir / "correct_only"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the original split IDs.
    train_ids = load_json(dataset_dir / "train_ids.json")
    val_ids = load_json(dataset_dir / "val_ids.json")
    test_ids = load_json(dataset_dir / "test_ids.json")

    # Load ground truth labels.
    gt_path = CHFS2019_DKI_GROUND_TRUTH_FILE
    ground_truth = load_json(gt_path)
    gt_map = {item["id"]: item for item in ground_truth}

    # Load Claude teacher predictions.
    claude_results_path = CHFS2019_TEACHER_RESULTS_FILE
    claude_results = load_json(claude_results_path)
    claude_tests = claude_results.get("tests", [])

    # Index teacher predictions by sample ID.
    pred_map = {}
    for test in claude_tests:
        sample_id = test.get("sample_id")  # Teacher output uses sample_id rather than id.
        parsed = test.get("parsed_json", {})
        if sample_id and parsed:
            # Read the hierarchical decision from parsed_json.
            insurance_decision = parsed.get("insurance_decision", {})
            pred_map[sample_id] = {
                "decision": insurance_decision.get("action", ""),
                "type": insurance_decision.get("insurance_type", "")
            }

    # Keep only teacher-correct samples.
    correct_ids = set()
    for sample_id in set(train_ids) | set(val_ids):
        if sample_id in pred_map and sample_id in gt_map:
            if is_prediction_correct(pred_map[sample_id], gt_map[sample_id]):
                correct_ids.add(sample_id)

    # Filter train and validation IDs.
    train_ids_correct = [sid for sid in train_ids if sid in correct_ids]
    val_ids_correct = [sid for sid in val_ids if sid in correct_ids]

    # Keep the test split unchanged.
    test_ids_correct = test_ids

    # Save the ablation splits.
    save_json(output_dir / "train_ids.json", train_ids_correct)
    save_json(output_dir / "val_ids.json", val_ids_correct)
    save_json(output_dir / "test_ids.json", test_ids_correct)

    # Save split statistics.
    summary = {
        "original": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
            "total": len(train_ids) + len(val_ids) + len(test_ids)
        },
        "correct_only": {
            "train": len(train_ids_correct),
            "val": len(val_ids_correct),
            "test": len(test_ids_correct),
            "total": len(train_ids_correct) + len(val_ids_correct) + len(test_ids_correct)
        },
        "removed": {
            "train": len(train_ids) - len(train_ids_correct),
            "val": len(val_ids) - len(val_ids_correct),
            "test": 0
        },
        "retention_rate": {
            "train": f"{len(train_ids_correct)/len(train_ids)*100:.2f}%",
            "val": f"{len(val_ids_correct)/len(val_ids)*100:.2f}%"
        }
    }

    save_json(output_dir / "split_summary.json", summary)

    print("Correct-only split generated.")
    print(f"  Output directory: {output_dir}")
    print("\nOriginal dataset:")
    print(f"  Train: {summary['original']['train']}")
    print(f"  Validation: {summary['original']['val']}")
    print(f"  Test: {summary['original']['test']}")
    print("\nAfter retaining teacher-correct samples:")
    print(f"  Train: {summary['correct_only']['train']} (retention: {summary['retention_rate']['train']})")
    print(f"  Validation: {summary['correct_only']['val']} (retention: {summary['retention_rate']['val']})")
    print(f"  Test: {summary['correct_only']['test']} (unchanged)")
    print("\nRemoved samples:")
    print(f"  Train: {summary['removed']['train']}")
    print(f"  Validation: {summary['removed']['val']}")


if __name__ == "__main__":
    main()
