#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def iter_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        rows = payload.get("tests") or payload.get("results")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    raise ValueError("Unsupported teacher result format: expected dict with tests/results or list")


def normalize_row(row: dict[str, Any], fallback_run_id: str, fallback_model: str) -> dict[str, Any]:
    sample_id = row.get("sample_id")
    household_id = row.get("household_id")
    individual_id = row.get("individual_id")
    if sample_id is None and household_id is not None and individual_id is not None:
        sample_id = f"{household_id}-{individual_id}"
    if sample_id is None:
        raise ValueError("Row is missing sample_id and household_id/individual_id")

    return {
        "run_id": row.get("run_id") or fallback_run_id,
        "run_type": row.get("run_type") or "converted_teacher",
        "model_id": row.get("model_id") or row.get("model_used") or fallback_model,
        "model_short_name": row.get("model_short_name") or "claude_sonnet_4_5",
        "sample_id": str(sample_id),
        "household_id": str(household_id) if household_id is not None else str(sample_id).split("-", 1)[0],
        "individual_id": str(individual_id) if individual_id is not None else str(sample_id).split("-", 1)[1],
        "success": bool(row.get("success", False)),
        "parse_success": bool(row.get("parse_success", row.get("parse_ok", False))),
        "response": row.get("response") or row.get("raw_response"),
        "parsed_json": row.get("parsed_json"),
        "predicted_action": row.get("predicted_action") or row.get("action"),
        "predicted_insurance_type": row.get("predicted_insurance_type") or row.get("insurance_type"),
        "usage": row.get("usage"),
        "cost_rmb": row.get("cost_rmb"),
        "error": row.get("error"),
        "created_at": row.get("created_at") or row.get("test_time"),
    }


def normalize_rows(
    rows: list[dict[str, Any]],
    fallback_run_id: str,
    fallback_model: str,
    *,
    latest_per_id: bool = False,
) -> list[dict[str, Any]]:
    normalized_rows = [normalize_row(row, fallback_run_id, fallback_model) for row in rows]
    if not latest_per_id:
        return normalized_rows

    latest: dict[str, dict[str, Any]] = {}
    latest_position: dict[str, int] = {}
    for idx, row in enumerate(normalized_rows):
        sample_id = row["sample_id"]
        latest[sample_id] = row
        latest_position[sample_id] = idx
    return [latest[sample_id] for sample_id in sorted(latest_position, key=latest_position.get)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert old teacher JSON result files to closeai-style JSONL.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-id", default="anthropic/claude-sonnet-4.5")
    parser.add_argument(
        "--latest-per-id",
        action="store_true",
        help="Keep only the last attempt for each sample_id. Use for unique-ID datasets, not CLDS row-level data.",
    )
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = iter_rows(payload)

    normalized_rows = normalize_rows(rows, args.run_id, args.model_id, latest_per_id=args.latest_per_id)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in normalized_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    unique_ids = len({row["sample_id"] for row in normalized_rows})
    print(f"Converted {len(rows)} rows to {len(normalized_rows)} row-level predictions ({unique_ids} unique ids): {args.output}")


if __name__ == "__main__":
    main()
