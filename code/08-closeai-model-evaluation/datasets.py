from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class EvalSample:
    sample_id: str
    household_id: str
    individual_id: str
    prompt: str
    ground_truth_action: str
    ground_truth_type: str


def _load_json_text(text: str) -> dict:
    return json.loads(text)


def _decision_from_payload(payload: dict) -> tuple[str, str]:
    decision = payload.get("insurance_decision") or {}
    return decision.get("action", ""), decision.get("insurance_type", "")


def terminal_label(action: str, insurance_type: str) -> str:
    if action != "参保":
        return "不参保"
    return insurance_type or "__UNKNOWN__"


def load_distill_jsonl(path: Path) -> List[EvalSample]:
    samples: List[EvalSample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row["messages"]
            user_msg = next(m for m in messages if m.get("role") == "user")
            assistant_msg = next(m for m in messages if m.get("role") == "assistant")
            truth_payload = _load_json_text(assistant_msg["content"])
            household_id = str(truth_payload["household_id"])
            individual_id = str(truth_payload["individual_id"])
            action, ins_type = _decision_from_payload(truth_payload)
            samples.append(
                EvalSample(
                    sample_id=f"{household_id}-{individual_id}",
                    household_id=household_id,
                    individual_id=individual_id,
                    prompt=user_msg["content"],
                    ground_truth_action=action,
                    ground_truth_type=ins_type,
                )
            )
    return samples


def load_external_json(prompts_path: Path, ground_truth_path: Path) -> List[EvalSample]:
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    truths = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    truth_by_id = {str(item["id"]): item for item in truths}
    samples: List[EvalSample] = []
    for prompt_row in prompts:
        sample_id = str(prompt_row["id"])
        truth = truth_by_id[sample_id]
        samples.append(
            EvalSample(
                sample_id=sample_id,
                household_id=str(prompt_row.get("household_id", "")),
                individual_id=str(prompt_row.get("individual_id", "")),
                prompt=prompt_row["prompt"],
                ground_truth_action=truth.get("decision", ""),
                ground_truth_type=truth.get("type", ""),
            )
        )
    return samples


def stratified_sample(samples: list[EvalSample], sample_size: int, seed: int) -> list[EvalSample]:
    groups: dict[str, list[EvalSample]] = {}
    for sample in samples:
        label = terminal_label(sample.ground_truth_action, sample.ground_truth_type)
        groups.setdefault(label, []).append(sample)

    rng = random.Random(seed)
    for rows in groups.values():
        rng.shuffle(rows)

    total = len(samples)
    selected: list[EvalSample] = []
    labels = sorted(groups)
    for label in labels:
        quota = round(sample_size * len(groups[label]) / total)
        selected.extend(groups[label][:quota])

    if len(selected) < sample_size:
        selected_ids = {row.sample_id for row in selected}
        leftovers = [row for label in labels for row in groups[label] if row.sample_id not in selected_ids]
        selected.extend(leftovers[: sample_size - len(selected)])
    elif len(selected) > sample_size:
        selected = selected[:sample_size]

    return selected


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--distill", type=Path)
    parser.add_argument("--external", nargs=2, type=Path, metavar=("PROMPTS", "GROUND_TRUTH"))
    args = parser.parse_args()

    if args.check and args.distill:
        print(f"distill_samples={len(load_distill_jsonl(args.distill))}")
    if args.check and args.external:
        print(f"external_samples={len(load_external_json(args.external[0], args.external[1]))}")


if __name__ == "__main__":
    _main()

