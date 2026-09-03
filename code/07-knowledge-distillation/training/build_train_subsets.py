#!/usr/bin/env python3
import argparse
import json
import os
import random
from collections import defaultdict
from typing import Dict, List


def _extract_assistant_json(content: str) -> dict:
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not parse JSON from the assistant content")
    return json.loads(content[start : end + 1])


def _get_category(record: dict) -> str:
    messages = record.get("messages", [])
    if not messages:
        raise ValueError("Record is missing the messages field")

    assistant_msg = None
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            assistant_msg = msg
            break
    if not assistant_msg:
        raise ValueError("Record is missing an assistant message")

    assistant_json = _extract_assistant_json(assistant_msg.get("content", ""))
    insurance_type = (
        assistant_json.get("insurance_decision", {}).get("insurance_type") or ""
    )

    if "不参保" in insurance_type:
        return "不参保"
    if "城镇职工" in insurance_type:
        return "城镇保"
    if "居民" in insurance_type:
        return "居民保"

    raise ValueError(f"Unrecognized insurance type: {insurance_type}")


def _build_ordered_buckets(
    buckets: Dict[str, List[str]], seed: int
) -> Dict[str, List[str]]:
    rng = random.Random(seed)
    ordered: Dict[str, List[str]] = {}
    for category, items in buckets.items():
        copied = list(items)
        rng.shuffle(copied)
        ordered[category] = copied
    return ordered


def _sample_records(
    ordered_buckets: Dict[str, List[str]], ratio: float
) -> List[str]:
    sampled: List[str] = []

    for category, items in ordered_buckets.items():
        if not items:
            continue
        target = int(round(len(items) * ratio))
        target = max(1, target) if ratio > 0 else 0
        target = min(len(items), target)
        sampled.extend(items[:target])

    random.shuffle(sampled)
    return sampled


def main() -> None:
    random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to train.jsonl")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]

    buckets: Dict[str, List[str]] = defaultdict(list)
    for line in lines:
        record = json.loads(line)
        category = _get_category(record)
        buckets[category].append(line)

    ratios = {
        "10": 0.10,
        "25": 0.25,
        "50": 0.50,
        "75": 0.75,
    }

    os.makedirs(args.output_dir, exist_ok=True)

    ordered_buckets = _build_ordered_buckets(buckets, args.seed)

    for label, ratio in ratios.items():
        sampled = _sample_records(ordered_buckets, ratio)
        output_path = os.path.join(args.output_dir, f"train_{label}.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(sampled)

    print("Total samples:", len(lines))
    for category, items in buckets.items():
        print(f"{category}: {len(items)}")

    for label, ratio in ratios.items():
        sampled = _sample_records(ordered_buckets, ratio)
        print(f"train_{label}.jsonl: {len(sampled)}")


if __name__ == "__main__":
    main()
