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


def main():
    dataset_dir = DISTILLATION_ANALYSIS_DIR

    train_ids_path = dataset_dir / "train_ids.json"
    val_ids_path = dataset_dir / "val_ids.json"

    gt_path = CHFS2019_DKI_GROUND_TRUTH_FILE
    claude_results_path = CHFS2019_TEACHER_RESULTS_FILE
    renew_results_path = DISTILLATION_REGEN_RESULTS_FILE

    if not train_ids_path.exists() or not val_ids_path.exists():
        raise FileNotFoundError("Missing train_ids.json or val_ids.json")

    train_ids = set(load_json(train_ids_path))
    val_ids = set(load_json(val_ids_path))
    trainval_ids = train_ids | val_ids

    ground_truth = load_json(gt_path)
    gt_map = {item["id"]: item for item in ground_truth}

    claude_results = load_json(claude_results_path)
    claude_tests = claude_results.get("tests", [])
    claude_map = {item.get("sample_id"): item for item in claude_tests if item.get("sample_id")}

    renew_map = {}
    if renew_results_path.exists():
        renew_results = load_json(renew_results_path)
        renew_tests = renew_results.get("tests", []) if isinstance(renew_results, dict) else []
        renew_map = {item.get("sample_id"): item for item in renew_tests if item.get("sample_id")}

    missing = []
    wrong = []
    matched = []
    regenerated = []

    for sample_id in sorted(trainval_ids):
        gt = gt_map.get(sample_id)
        if not gt:
            missing.append({"sample_id": sample_id, "reason": "missing_ground_truth"})
            continue

        decision = gt.get("decision")
        gt_type = gt.get("type")
        claude_item = claude_map.get(sample_id)
        if claude_item:
            if claude_item.get("parse_success") and claude_item.get("predicted_action") == decision:
                if decision == "不参保" or claude_item.get("predicted_insurance_type") == gt_type:
                    matched.append(sample_id)
                    continue

        renew_item = renew_map.get(sample_id)
        if renew_item and renew_item.get("success") and renew_item.get("parse_success"):
            regenerated.append(sample_id)
            continue

        if claude_item and not claude_item.get("success"):
            reason = claude_item.get("error", "claude_error")
        elif renew_item and not renew_item.get("success"):
            reason = renew_item.get("error", "renew_error")
        elif renew_item and renew_item.get("success") and not renew_item.get("parse_success"):
            reason = renew_item.get("parse_error", "renew_parse_failed")
        else:
            reason = "missing_or_incorrect_demo"

        wrong.append({"sample_id": sample_id, "reason": reason})

    summary = {
        "train_total": len(train_ids),
        "val_total": len(val_ids),
        "trainval_total": len(trainval_ids),
        "matched_correct": len(matched),
        "regenerated_correct": len(regenerated),
        "missing_ground_truth": len([m for m in missing if m["reason"] == "missing_ground_truth"]),
        "missing_or_incorrect_demo": len(wrong),
        "has_renew_results": renew_results_path.exists(),
    }

    payload = {
        "summary": summary,
        "matched_sample_ids": matched,
        "regenerated_sample_ids": regenerated,
        "missing_samples": missing,
        "incorrect_or_missing_samples": wrong,
    }

    output_path = dataset_dir / "trainval_demo_check.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("Train/Val demo check completed.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
