from __future__ import annotations

import json
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_CODE_DIR))

from config.paths import (
    CHFS2019_DKI_GROUND_TRUTH_FILE,
    CHFS2019_DKI_PROMPTS_FILE,
    CHFS2019_TEACHER_RESULTS_FILE,
    DISTILLATION_ANALYSIS_DIR,
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
    output_dir = dataset_dir / "test_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test split IDs.
    test_ids = set(load_json(dataset_dir / "test_ids.json"))

    # Load prompts.
    prompts_path = CHFS2019_DKI_PROMPTS_FILE
    prompts = load_json(prompts_path)
    prompt_map = {item["id"]: item["prompt"] for item in prompts}

    # Load ground truth labels.
    gt_path = CHFS2019_DKI_GROUND_TRUTH_FILE
    ground_truth = load_json(gt_path)
    gt_map = {item["id"]: item for item in ground_truth}

    # Load Claude teacher predictions.
    claude_results_path = CHFS2019_TEACHER_RESULTS_FILE
    claude_results = load_json(claude_results_path)
    claude_tests = claude_results.get("tests", [])
    claude_map = {
        item.get("sample_id"): item
        for item in claude_tests
        if item.get("sample_id")
    }

    correct_samples = []
    incorrect_samples = []

    for sample_id in sorted(test_ids):
        gt = gt_map.get(sample_id)
        claude_item = claude_map.get(sample_id)
        prompt = prompt_map.get(sample_id)

        if not gt or not claude_item or not prompt:
            continue

        # Read the ground truth decision.
        gt_action = gt.get("decision", "")
        gt_type = gt.get("type", "")

        # Read the teacher prediction.
        claude_parsed = claude_item.get("parsed_json", {})
        claude_insurance = claude_parsed.get("insurance_decision", {})
        claude_action = claude_insurance.get("action", "")
        claude_type = claude_insurance.get("insurance_type", "")
        claude_decision_process = claude_parsed.get("decision_process", {})

        # Check hierarchical prediction correctness.
        is_correct = is_prediction_correct(claude_action, claude_type, gt_action, gt_type)

        sample_record = {
            "sample_id": sample_id,
            "prompt": prompt,
            "ground_truth": {
                "action": gt_action,
                "insurance_type": gt_type,
            },
            "claude_prediction": {
                "action": claude_action,
                "insurance_type": claude_type,
                "decision_process": claude_decision_process,
            },
        }

        if is_correct:
            correct_samples.append(sample_record)
        else:
            incorrect_samples.append(sample_record)

    # Write case-level analysis to JSONL.
    write_jsonl(output_dir / "test_correct.jsonl", correct_samples)
    write_jsonl(output_dir / "test_incorrect.jsonl", incorrect_samples)

    # Save aggregate statistics.
    total = len(correct_samples) + len(incorrect_samples)
    summary = {
        "total_test_samples": total,
        "correct_predictions": len(correct_samples),
        "incorrect_predictions": len(incorrect_samples),
        "accuracy": f"{len(correct_samples) / total * 100:.2f}%" if total > 0 else "N/A",
        "files": {
            "correct_samples": str(output_dir / "test_correct.jsonl"),
            "incorrect_samples": str(output_dir / "test_incorrect.jsonl"),
        },
    }

    summary_path = output_dir / "test_analysis_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Test-set prediction analysis complete.")
    print(f"  Output directory: {output_dir}")
    print("\nTest-set statistics:")
    print(f"  Total samples: {summary['total_test_samples']}")
    print(f"  Correct predictions: {summary['correct_predictions']}")
    print(f"  Incorrect predictions: {summary['incorrect_predictions']}")
    print(f"  Accuracy: {summary['accuracy']}")
    print("\nOutput files:")
    print("  Correct samples: test_correct.jsonl")
    print("  Incorrect samples: test_incorrect.jsonl")
    print("  Summary: test_analysis_summary.json")


if __name__ == "__main__":
    main()
