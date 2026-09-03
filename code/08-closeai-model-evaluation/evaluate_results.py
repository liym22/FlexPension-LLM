from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_distill_jsonl, load_external_json
from metrics import evaluate_predictions


def load_predictions(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    metadata: dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not metadata:
                metadata = {
                    "run_id": row.get("run_id"),
                    "run_type": row.get("run_type"),
                    "model_id": row.get("model_id"),
                    "model_short_name": row.get("model_short_name") or path.name.removesuffix("_results.jsonl"),
                }
            sample_id = str(row["sample_id"])
            predictions[sample_id] = {
                "success": row.get("success", False),
                "parse_ok": row.get("parse_success", row.get("parse_ok", False)),
                "action": row.get("predicted_action") or row.get("action"),
                "insurance_type": row.get("predicted_insurance_type") or row.get("insurance_type"),
                "usage": row.get("usage", {}),
                "cost_rmb": row.get("cost_rmb"),
            }
    return predictions, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-jsonl", required=True, type=Path)
    parser.add_argument("--dataset-kind", required=True, choices=["distill_jsonl", "external_json"])
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--prompts-path", type=Path)
    parser.add_argument("--ground-truth-path", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    if args.dataset_kind == "distill_jsonl":
        if not args.dataset_path:
            raise SystemExit("--dataset-path is required for distill_jsonl")
        samples = load_distill_jsonl(args.dataset_path)
        dataset_label = str(args.dataset_path)
    else:
        if not args.prompts_path or not args.ground_truth_path:
            raise SystemExit("--prompts-path and --ground-truth-path are required for external_json")
        samples = load_external_json(args.prompts_path, args.ground_truth_path)
        dataset_label = str(args.prompts_path)

    predictions, metadata = load_predictions(args.result_jsonl)
    result = evaluate_predictions(samples, predictions, require_complete=args.require_complete)
    result.update(metadata)
    result.update(
        {
            "result_jsonl": str(args.result_jsonl),
            "dataset": dataset_label,
        }
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
