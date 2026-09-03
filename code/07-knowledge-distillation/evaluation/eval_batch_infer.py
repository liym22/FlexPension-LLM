#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch inference evaluation script.

Metrics:
1) action F1
2) insurance_type F1, only for samples where action is correct (true=参保 & pred=参保 & parse_ok)
3) weighted F1 = 0.6 * action_f1 + 0.4 * type_f1
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

# Add the repository root to the Python import path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.paths import SERVER_BASE, SERVER_OUTPUT_DIR, SERVER_DATASET_DIR

try:
    from sklearn.metrics import f1_score
except Exception:  # pragma: no cover
    f1_score = None

DEFAULT_RESULT_PATH = f"{SERVER_OUTPUT_DIR}/batch_infer_results.jsonl"
DEFAULT_GROUND_TRUTH_PATH = f"{SERVER_DATASET_DIR}/test.jsonl"
DEFAULT_OUTPUT_DIR = f"{SERVER_OUTPUT_DIR}/evaluation_results"

ACTION_POSITIVE = "参保"
TYPE_LABELS = ["城乡居民养老保险", "城镇职工养老保险"]


def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        return None


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    code_block = re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    for block in reversed(code_block):
        parsed = _safe_json_loads(block)
        if parsed:
            return parsed

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    return None


def parse_prediction(response_text: str) -> Tuple[Optional[str], Optional[str], bool]:
    payload = extract_json_from_text(response_text)
    if not payload:
        return None, None, False
    decision = payload.get("insurance_decision") or payload.get("insuranceDecision") or {}
    action = decision.get("action")
    ins_type = decision.get("insurance_type") or decision.get("insuranceType")
    return action, ins_type, True


def load_ground_truth(ground_truth_path: str) -> Dict[Tuple[str, str], Tuple[Optional[str], Optional[str]]]:
    mapping: Dict[Tuple[str, str], Tuple[Optional[str], Optional[str]]] = {}
    if not os.path.exists(ground_truth_path):
        raise FileNotFoundError(f"Missing ground truth file: {ground_truth_path}")
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = _safe_json_loads(line)
            if not row:
                continue
            messages = row.get("messages", [])
            if not messages:
                continue
            assistant_msg = messages[-1]
            if assistant_msg.get("role") != "assistant":
                continue
            payload = _safe_json_loads(assistant_msg.get("content", ""))
            if not payload:
                continue
            household_id = str(payload.get("household_id") or "")
            individual_id = str(payload.get("individual_id") or "")
            if not household_id or not individual_id:
                continue
            decision = payload.get("insurance_decision") or {}
            mapping[(household_id, individual_id)] = (
                decision.get("action"),
                decision.get("insurance_type"),
            )
    if not mapping:
        raise ValueError("Ground truth mapping is empty. Check test.jsonl format.")
    return mapping


def compute_f1(y_true: List[str], y_pred: List[str], labels: Optional[List[str]] = None) -> float:
    if not y_true:
        return 0.0
    if f1_score is None:
        correct = sum(t == p for t, p in zip(y_true, y_pred))
        return correct / len(y_true)
    if labels:
        return f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    return f1_score(y_true, y_pred, average="binary", pos_label=ACTION_POSITIVE, zero_division=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-path", default=DEFAULT_RESULT_PATH, help="Inference-result file")
    parser.add_argument("--ground-truth", default=DEFAULT_GROUND_TRUTH_PATH, help="Test-set file")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Evaluation output directory")
    args = parser.parse_args()

    if not os.path.exists(args.result_path):
        raise FileNotFoundError(f"Missing results file: {args.result_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    gt_map = load_ground_truth(args.ground_truth)

    y_true_action: List[str] = []
    y_pred_action: List[str] = []
    y_true_type: List[str] = []
    y_pred_type: List[str] = []

    with open(args.result_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = _safe_json_loads(line)
            if not row:
                continue

            pred_action, pred_type, parse_ok = parse_prediction(row.get("response", ""))
            messages = row.get("messages", [])
            if not messages:
                continue
            user_content = messages[0].get("content", "") if messages else ""
            household_id = None
            individual_id = None
            for match in re.finditer(r"家庭ID:\s*(\S+)", user_content):
                household_id = match.group(1)
                break
            for match in re.finditer(r"个人ID:\s*(\S+)", user_content):
                individual_id = match.group(1)
                break
            if not household_id or not individual_id:
                continue

            true_action, true_type = gt_map.get((household_id, individual_id), (None, None))
            if true_action is None:
                continue

            # Action evaluation treats parse failures as non-participation.
            y_true_action.append(true_action)
            y_pred_action.append(pred_action if parse_ok and pred_action else "不参保")

            # Type evaluation requires true and predicted participation plus a valid parse.
            if parse_ok and true_action == "参保" and pred_action == "参保":
                y_true_type.append(true_type or "__INVALID__")
                y_pred_type.append(pred_type or "__INVALID__")

    action_f1 = compute_f1(y_true_action, y_pred_action)
    type_f1 = compute_f1(y_true_type, y_pred_type, labels=TYPE_LABELS)
    weighted_f1 = 0.6 * action_f1 + 0.4 * type_f1

    result = {
        "result_path": args.result_path,
        "ground_truth_path": args.ground_truth,
        "n_total": len(y_true_action),
        "n_action_correct": sum(t == p for t, p in zip(y_true_action, y_pred_action)),
        "n_type_eval": len(y_true_type),
        "action_f1": action_f1,
        "type_f1": type_f1,
        "weighted_f1": weighted_f1,
        "timestamp": datetime.now().isoformat(),
    }

    out_path = os.path.join(args.output_dir, f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
