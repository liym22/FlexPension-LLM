from __future__ import annotations

from typing import Any, Dict, Iterable

from datasets import EvalSample

ACTION_POSITIVE = "参保"
TYPE_POSITIVE = "城镇职工养老保险"


def _binary_f1(y_true: list[str], y_pred: list[str], positive_label: str) -> float:
    tp = sum(t == positive_label and p == positive_label for t, p in zip(y_true, y_pred))
    fp = sum(t != positive_label and p == positive_label for t, p in zip(y_true, y_pred))
    fn = sum(t == positive_label and p != positive_label for t, p in zip(y_true, y_pred))
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 0.0


def evaluate_predictions(
    samples: Iterable[EvalSample],
    predictions: Dict[str, Dict[str, Any]],
    require_complete: bool = True,
) -> dict[str, Any]:
    y_true_action: list[str] = []
    y_pred_action: list[str] = []
    y_true_type: list[str] = []
    y_pred_type: list[str] = []
    success_count = 0
    parse_count = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost_rmb = 0.0

    missing_count = 0

    for sample in samples:
        if not require_complete and sample.sample_id not in predictions:
            missing_count += 1
            continue
        pred = predictions.get(sample.sample_id, {})
        success = bool(pred.get("success", False))
        parse_ok = bool(pred.get("parse_ok", pred.get("parse_success", False)))
        if success:
            success_count += 1
        if parse_ok:
            parse_count += 1

        usage = pred.get("usage") or {}
        total_input_tokens += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        total_output_tokens += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        total_cost_rmb += float(pred.get("cost_rmb") or usage.get("cost_rmb") or usage.get("cost") or 0.0)

        pred_action = (pred.get("action") or pred.get("predicted_action")) if parse_ok else "不参保"
        pred_type = (pred.get("insurance_type") or pred.get("predicted_insurance_type")) if parse_ok else "不参保"

        y_true_action.append(sample.ground_truth_action)
        y_pred_action.append(pred_action or "不参保")

        if parse_ok and sample.ground_truth_action == "参保" and pred_action == "参保":
            y_true_type.append(sample.ground_truth_type)
            y_pred_type.append(pred_type or "__INVALID__")

    action_f1 = _binary_f1(y_true_action, y_pred_action, ACTION_POSITIVE)
    type_f1 = _binary_f1(y_true_type, y_pred_type, TYPE_POSITIVE) if y_true_type else 0.0
    composite_f1 = 0.6 * action_f1 + 0.4 * type_f1
    n_total = len(y_true_action)

    return {
        "n_total": n_total,
        "n_missing_predictions": missing_count,
        "n_success": success_count,
        "n_parse_success": parse_count,
        "api_success_rate": success_count / n_total if n_total else 0.0,
        "parse_success_rate": parse_count / n_total if n_total else 0.0,
        "n_action_correct": sum(t == p for t, p in zip(y_true_action, y_pred_action)),
        "n_type_eval": len(y_true_type),
        "action_f1": action_f1,
        "type_f1": type_f1,
        "composite_f1": composite_f1,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cost_rmb": total_cost_rmb,
    }
