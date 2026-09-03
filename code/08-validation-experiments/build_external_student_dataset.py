#!/usr/bin/env python
"""Convert paired external prompt/label JSON files to ms-swift evaluation JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _index_by_id(rows: list[dict], label: str) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for row in rows:
        sample_id = str(row["id"])
        if sample_id in indexed:
            raise ValueError(f"duplicate {label} ID: {sample_id}")
        indexed[sample_id] = row
    return indexed


def build_external_rows(
    prompts: list[dict],
    ground_truth: list[dict],
    expected_size: int | None = None,
) -> list[dict]:
    prompts_by_id = _index_by_id(prompts, "prompt")
    truth_by_id = _index_by_id(ground_truth, "ground-truth")
    if set(prompts_by_id) != set(truth_by_id):
        raise ValueError("prompt and ground-truth IDs differ")
    if expected_size is not None and len(prompts) != expected_size:
        raise ValueError(f"loaded {len(prompts)} samples but expected {expected_size}")

    output = []
    for prompt_row in prompts:
        sample_id = str(prompt_row["id"])
        truth = truth_by_id[sample_id]
        household_id = str(prompt_row.get("household_id") or truth.get("household_id") or "")
        individual_id = str(prompt_row.get("individual_id") or truth.get("individual_id") or "")
        if not household_id or not individual_id:
            raise ValueError(f"missing household_id or individual_id for {sample_id}")
        assistant = {
            "household_id": household_id,
            "individual_id": individual_id,
            "insurance_decision": {
                "action": truth["decision"],
                "insurance_type": truth["type"],
            },
        }
        output.append(
            {
                "messages": [
                    {"role": "user", "content": prompt_row["prompt"]},
                    {
                        "role": "assistant",
                        "content": json.dumps(assistant, ensure_ascii=False, separators=(",", ":")),
                    },
                ]
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-size", type=int, default=500)
    args = parser.parse_args()

    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    ground_truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    rows = build_external_rows(prompts, ground_truth, args.expected_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} samples to {args.output}")


if __name__ == "__main__":
    main()
