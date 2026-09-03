from __future__ import annotations

import json
from pathlib import Path
from sklearn.metrics import f1_score
import pandas as pd
import sys

# Add the project root for local imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.paths import (
    CHFS2019_DKI_GROUND_TRUTH_FILE,
    DISTILLATION_ANALYSIS_DIR,
    DISTILLATION_RESULTS_DIR,
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path):
    """Load a JSONL file."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def calc_f1_metrics(y_true_action, y_pred_action, y_true_type, y_pred_type):
    """Compute Action F1, Type F1, and their weighted composite.

    Type F1 is evaluated where both labels indicate participation.
    Weighted F1 = 0.6 * Action F1 + 0.4 * Type F1.
    """
    if len(y_true_action) == 0:
        return None, None, None

    # Action F1: participation (1) versus non-participation (0).
    action_f1 = f1_score(y_true_action, y_pred_action, zero_division=0)

    # Type F1 is evaluated where both labels indicate participation.
    if len(y_true_type) == 0:
        return action_f1, None, action_f1

    type_f1 = f1_score(y_true_type, y_pred_type, zero_division=0)
    weighted_f1 = 0.6 * action_f1 + 0.4 * type_f1

    return action_f1, type_f1, weighted_f1


def extract_prediction(result: dict):
    """Extract a hierarchical prediction from model output."""
    # First parse the response field.
    response = result.get("response", "")
    if response:
        import re
        # Remove the optional <think> block before parsing JSON.
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                import json as json_lib
                parsed = json_lib.loads(json_match.group())
                insurance = parsed.get("insurance_decision", {})
                action = insurance.get("action", "")
                ins_type = insurance.get("insurance_type", "")
                return action, ins_type
            except:
                pass

    # Fall back to the parsed_json field.
    parsed = result.get("parsed_json", {})
    insurance = parsed.get("insurance_decision", {})
    action = insurance.get("action", "")
    ins_type = insurance.get("insurance_type", "")
    return action, ins_type


def extract_sample_id(result: dict):
    """Extract sample_id from model output."""
    # Fall back to parsing the raw response.
    response = result.get("response", "")
    if response:
        import re
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                import json as json_lib
                parsed = json_lib.loads(json_match.group())
                household_id = parsed.get("household_id", "")
                individual_id = parsed.get("individual_id", "")
                if household_id and individual_id:
                    return f"{household_id}-{individual_id}"
            except:
                pass
    return None


def main():
    dataset_dir = DISTILLATION_ANALYSIS_DIR
    test_analysis_dir = dataset_dir / "test_analysis"
    output_dir = dataset_dir / "model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load teacher-correct and teacher-error sample groups.
    claude_correct_ids = set()
    claude_incorrect_ids = set()

    claude_correct = load_jsonl(test_analysis_dir / "test_correct.jsonl")
    claude_incorrect = load_jsonl(test_analysis_dir / "test_incorrect.jsonl")

    for item in claude_correct:
        claude_correct_ids.add(item["sample_id"])
    for item in claude_incorrect:
        claude_incorrect_ids.add(item["sample_id"])

    print(f"✓ Correct Claude samples: {len(claude_correct_ids)}")
    print(f"✓ Incorrect Claude samples: {len(claude_incorrect_ids)}")

    # Load ground truth labels.
    gt_path = CHFS2019_DKI_GROUND_TRUTH_FILE
    ground_truth = load_json(gt_path)
    gt_map = {item["id"]: item for item in ground_truth}

    # Load predictions from both student variants.
    rationalization_path = DISTILLATION_RESULTS_DIR / "infer_results" / "data_scaling_data100_checkpoint-1998.jsonl"
    correct_only_path = DISTILLATION_RESULTS_DIR / "infer_results" / "correct_only_results.jsonl"

    rationalization_results = load_jsonl(rationalization_path)
    correct_only_results = load_jsonl(correct_only_path)

    # Index predictions by sample ID.
    rationalization_map = {}
    for r in rationalization_results:
        sample_id = extract_sample_id(r)
        if sample_id:
            rationalization_map[sample_id] = r

    correct_only_map = {}
    for r in correct_only_results:
        sample_id = extract_sample_id(r)
        if sample_id:
            correct_only_map[sample_id] = r

    print(f"✓ Rationalization model results: {len(rationalization_map)}")
    print(f"✓ Correct-only model results: {len(correct_only_map)}")

    # Evaluate each teacher-correct/error subgroup.
    groups = {
        "claude_correct": claude_correct_ids,
        "claude_incorrect": claude_incorrect_ids,
    }

    results = {}

    for group_name, sample_ids in groups.items():
        print(f"\n{'='*60}")
        print(f"Evaluation group: {group_name} ({len(sample_ids)} samples)")
        print(f"{'='*60}")

        for model_name, pred_map in [
            ("rationalization", rationalization_map),
            ("correct_only", correct_only_map),
        ]:
            y_true_action = []
            y_pred_action = []
            y_true_type = []
            y_pred_type = []

            for sample_id in sample_ids:
                gt = gt_map.get(sample_id)
                pred = pred_map.get(sample_id)

                if not gt or not pred:
                    continue

                gt_action = gt.get("decision", "")
                gt_type = gt.get("type", "")

                pred_action, pred_type = extract_prediction(pred)

                # Action: participation (1) versus non-participation (0).
                y_true_action.append(1 if gt_action == "参保" else 0)
                y_pred_action.append(1 if pred_action == "参保" else 0)

                # Type is evaluated only when both labels indicate participation.
                if gt_action == "参保" and pred_action == "参保":
                    # Employee scheme (1) versus resident scheme (0).
                    y_true_type.append(1 if gt_type == "城镇职工养老保险" else 0)
                    y_pred_type.append(1 if pred_type == "城镇职工养老保险" else 0)

            action_f1, type_f1, weighted_f1 = calc_f1_metrics(
                y_true_action, y_pred_action, y_true_type, y_pred_type
            )

            results[f"{group_name}_{model_name}"] = {
                "group": group_name,
                "model": model_name,
                "n_samples": len(sample_ids),
                "n_evaluated": len(y_true_action),
                "n_type_evaluated": len(y_true_type),
                "action_f1": action_f1,
                "type_f1": type_f1,
                "weighted_f1": weighted_f1,
            }

            print(f"\n{model_name}:")
            print(f"  Evaluated samples: {len(y_true_action)}/{len(sample_ids)}")
            print(f"  Action F1: {action_f1:.4f}" if action_f1 is not None else "  Action F1: N/A")
            print(f"  Type F1: {type_f1:.4f} (n={len(y_true_type)})" if type_f1 is not None else "  Type F1: N/A")
            print(f"  Weighted F1: {weighted_f1:.4f}" if weighted_f1 is not None else "  Weighted F1: N/A")

    # Save detailed metrics.
    summary_path = output_dir / "cross_analysis_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Print the comparison table.
    df_rows = []
    for key, val in results.items():
        df_rows.append({
            "分组": "Claude正确" if val["group"] == "claude_correct" else "Claude错误",
            "模型": "Rationalization" if val["model"] == "rationalization" else "Correct-only",
            "样本数": val["n_samples"],
            "评估数": val["n_evaluated"],
            "Action F1": f"{val['action_f1']:.4f}" if val['action_f1'] is not None else "N/A",
            "Type F1": f"{val['type_f1']:.4f}" if val['type_f1'] is not None else "N/A",
            "Weighted F1": f"{val['weighted_f1']:.4f}" if val['weighted_f1'] is not None else "N/A",
        })

    df = pd.DataFrame(df_rows)
    csv_path = output_dir / "cross_analysis_table.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"\n{'='*60}")
    print("Comparison table:")
    print(f"{'='*60}")
    print(df.to_string(index=False))

    print(f"\n✓ Results saved:")
    print(f"  JSON: {summary_path}")
    print(f"  CSV: {csv_path}")


if __name__ == "__main__":
    main()
