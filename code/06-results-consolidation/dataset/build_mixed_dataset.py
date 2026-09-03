from __future__ import annotations

import json
import random
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


RANDOM_SEED = 42
TRAINVAL_RATIO = 0.85
TRAIN_RATIO = 0.8


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def stratified_split(ids_by_label: dict[str, list[str]], split_ratio: float, seed: int):
    rng = random.Random(seed)
    left_ids: list[str] = []
    right_ids: list[str] = []
    for label, ids in ids_by_label.items():
        ids_copy = list(ids)
        rng.shuffle(ids_copy)
        split_idx = int(round(len(ids_copy) * split_ratio))
        left_ids.extend(ids_copy[:split_idx])
        right_ids.extend(ids_copy[split_idx:])
    rng.shuffle(left_ids)
    rng.shuffle(right_ids)
    return left_ids, right_ids


def main():
    gt_path = CHFS2019_DKI_GROUND_TRUTH_FILE
    prompts_path = CHFS2019_DKI_PROMPTS_FILE
    claude_results_path = CHFS2019_TEACHER_RESULTS_FILE
    output_dir = DISTILLATION_ANALYSIS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth = load_json(gt_path)
    prompts = load_json(prompts_path)
    claude_results = load_json(claude_results_path)

    ids_by_label = {
        "不参保": [],
        "城乡居民养老保险": [],
        "城镇职工养老保险": [],
    }
    for item in ground_truth:
        decision = item.get("decision")
        insurance_type = item.get("type")
        sample_id = item.get("id")
        if not sample_id:
            continue
        if decision == "不参保":
            ids_by_label["不参保"].append(sample_id)
        elif decision == "参保" and insurance_type in ids_by_label:
            ids_by_label[insurance_type].append(sample_id)

    # Step 1: stratified split into train+val and test
    trainval_ids, test_ids = stratified_split(
        ids_by_label,
        TRAINVAL_RATIO,
        RANDOM_SEED,
    )

    # Step 2: stratified split train+val into train and val
    id_to_decision = {item["id"]: item.get("decision") for item in ground_truth}
    trainval_by_label = {
        "不参保": [],
        "城乡居民养老保险": [],
        "城镇职工养老保险": [],
    }
    id_to_type = {item["id"]: item.get("type") for item in ground_truth}
    for sample_id in trainval_ids:
        decision = id_to_decision.get(sample_id)
        insurance_type = id_to_type.get(sample_id)
        if decision == "不参保":
            trainval_by_label["不参保"].append(sample_id)
        elif decision == "参保" and insurance_type in trainval_by_label:
            trainval_by_label[insurance_type].append(sample_id)

    train_ids, val_ids = stratified_split(
        trainval_by_label,
        TRAIN_RATIO,
        RANDOM_SEED,
    )

    write_json(output_dir / "train_ids.json", train_ids)
    write_json(output_dir / "val_ids.json", val_ids)
    write_json(output_dir / "test_ids.json", test_ids)

    # Step 3: build the train/validation error list from teacher results.
    prompt_map = {item["id"]: item["prompt"] for item in prompts}
    gt_map = {item["id"]: item for item in ground_truth}
    trainval_set = set(trainval_ids)

    errors = []
    matched = []
    for test in claude_results.get("tests", []):
        sample_id = test.get("sample_id")
        if sample_id not in trainval_set:
            continue
        if sample_id not in prompt_map or sample_id not in gt_map:
            continue
        gt_item = gt_map[sample_id]
        gt_decision = gt_item.get("decision")
        gt_type = gt_item.get("type")
        parse_ok = test.get("parse_success", False)
        claude_action = test.get("predicted_action")
        claude_type = test.get("predicted_insurance_type")
        payload = {
            "sample_id": sample_id,
            "prompt": prompt_map[sample_id],
            "gt": gt_item,
            "claude": test,
        }

        action_match = parse_ok and claude_action == gt_decision
        type_match = (
            parse_ok
            and gt_decision == "参保"
            and claude_action == "参保"
            and claude_type == gt_type
        )

        if action_match and (gt_decision == "不参保" or type_match):
            matched.append(payload)
        else:
            errors.append(payload)

    write_json(output_dir / "trainval_need_regen.json", errors)
    write_json(output_dir / "trainval_matched_claude.json", matched)

    summary = {
        "total_samples": len(ground_truth),
        "strata_counts": {k: len(v) for k, v in ids_by_label.items()},
        "train_total": len(train_ids),
        "val_total": len(val_ids),
        "test_total": len(test_ids),
        "trainval_ratio": TRAINVAL_RATIO,
        "train_ratio": TRAIN_RATIO,
        "seed": RANDOM_SEED,
        "trainval_need_regen": len(errors),
        "trainval_matched": len(matched),
    }
    write_json(output_dir / "split_summary.json", summary)

    print("Dataset split completed.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
