#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validate response JSON format in batch_infer_results.jsonl.

Focuses only on the key fields required for evaluation:
- insurance_decision.action
- insurance_decision.insurance_type

Outputs a summary and a JSON report with per-sample errors.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add the repository root to the Python import path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.paths import SERVER_OUTPUT_DIR

DEFAULT_INPUT = f"{SERVER_OUTPUT_DIR}/batch_infer_results.jsonl"
DEFAULT_OUTPUT_DIR = f"{SERVER_OUTPUT_DIR}/format_validation"

ACTION_VALUES = {"参保", "不参保"}
TYPE_VALUES = {"不参保", "城乡居民养老保险", "城镇职工养老保险"}


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


def extract_ids_from_prompt(text: str) -> Tuple[Optional[str], Optional[str]]:
    household_id = None
    individual_id = None
    if text:
        match = re.search(r"家庭ID:\s*(\S+)", text)
        if match:
            household_id = match.group(1)
        match = re.search(r"个人ID:\s*(\S+)", text)
        if match:
            individual_id = match.group(1)
    return household_id, individual_id


def validate_payload(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    if not isinstance(payload, dict):
        return ["payload is not a JSON object"]

    insurance_decision = payload.get("insurance_decision")
    if isinstance(insurance_decision, dict):
        for key in ["action", "insurance_type"]:
            if key not in insurance_decision:
                errors.append(f"insurance_decision is missing field: {key}")

        action = insurance_decision.get("action")
        if action not in ACTION_VALUES:
            errors.append(f"invalid action: {action}")

        ins_type = insurance_decision.get("insurance_type")
        if ins_type not in TYPE_VALUES:
            errors.append(f"invalid insurance_type: {ins_type}")

        if action == "不参保" and ins_type not in {"不参保", None}:
            errors.append("action is non-enrollment but insurance_type is inconsistent")
    elif insurance_decision is not None:
        errors.append("insurance_decision is not an object")
    else:
        errors.append("missing field: insurance_decision")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Result file")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Report output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    total = 0
    valid = 0
    invalid = 0
    records: List[Dict[str, Any]] = []

    with open(args.input, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            row = _safe_json_loads(line)
            if not row:
                records.append({"index": idx, "errors": ["line is not valid JSON"], "valid": False})
                invalid += 1
                continue

            response_text = row.get("response", "")
            payload = extract_json_from_text(response_text)
            if not payload:
                household_id, individual_id = extract_ids_from_prompt(
                    row.get("messages", [{}])[0].get("content", "")
                )
                records.append({
                    "index": idx,
                    "household_id": household_id,
                    "individual_id": individual_id,
                    "errors": ["could not parse JSON from response"],
                    "valid": False,
                })
                invalid += 1
                continue

            errors = validate_payload(payload)
            is_valid = len(errors) == 0
            if is_valid:
                valid += 1
            else:
                invalid += 1

            prompt_household_id, prompt_individual_id = extract_ids_from_prompt(
                row.get("messages", [{}])[0].get("content", "")
            )

            records.append({
                "index": idx,
                "household_id": payload.get("household_id") or prompt_household_id,
                "individual_id": payload.get("individual_id") or prompt_individual_id,
                "valid": is_valid,
                "errors": errors,
            })

    summary = {
        "input": args.input,
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "timestamp": datetime.now().isoformat(),
    }

    out_path = os.path.join(args.output_dir, f"format_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
