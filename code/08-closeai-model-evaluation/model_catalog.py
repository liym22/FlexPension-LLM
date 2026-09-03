from __future__ import annotations

import argparse
import asyncio
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from closeai_config import get_closeai_admin_key, get_closeai_models_url


def fetch_models() -> dict[str, Any]:
    request = Request(
        get_closeai_models_url(),
        headers={"Authorization": f"Bearer {get_closeai_admin_key()}"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"CloseAI model catalog failed: HTTP {exc.code}: {body[:500]}") from exc


def flatten_models(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("models") or payload.get("data") or []
    flat: list[dict[str, Any]] = []
    for item in rows:
        pricing = item.get("pricing") or {}
        flat.append(
            {
                "model": item.get("model") or item.get("id") or "",
                "vendor": item.get("vendor", ""),
                "category": item.get("category", ""),
                "context_window": item.get("context_window", ""),
                "max_output_tokens": item.get("max_output_tokens", ""),
                "text_input_price_rmb_per_1m": pricing.get("text_input_price", ""),
                "text_output_price_rmb_per_1m": pricing.get("text_output_price", ""),
                "cache_read_rate": pricing.get("cache_read_rate", ""),
            }
        )
    return sorted(flat, key=lambda row: str(row["model"]))


def candidate_rows(flat: list[dict[str, Any]]) -> list[dict[str, Any]]:
    family_keywords = {
        "claude": ["claude"],
        "gemini": ["gemini"],
        "openai": ["gpt", "o3", "o4", "o5"],
        "qwen": ["qwen"],
        "deepseek": ["deepseek"],
        "llama": ["llama"],
        "minimax": ["minimax"],
    }
    candidates: list[dict[str, Any]] = []
    for row in flat:
        model = str(row["model"]).lower()
        for family, keywords in family_keywords.items():
            if any(keyword in model for keyword in keywords):
                enriched = dict(row)
                enriched["family"] = family
                enriched["required_priority"] = model == "claude-sonnet-5"
                candidates.append(enriched)
                break
    return candidates


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else ["model"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "family",
        "model",
        "vendor",
        "context_window",
        "max_output_tokens",
        "text_input_price_rmb_per_1m",
        "text_output_price_rmb_per_1m",
        "required_priority",
    ]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="code/08-closeai-model-evaluation/outputs/model_catalog")
    args = parser.parse_args()

    payload = fetch_models()
    flat = flatten_models(payload)
    candidates = candidate_rows(flat)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    write_rows(flat, output_dir / f"{stamp}_closeai_models_flat.json")
    write_rows(flat, output_dir / f"{stamp}_closeai_models_flat.csv")
    write_rows(candidates, output_dir / f"{stamp}_candidate_models.csv")
    write_markdown(candidates, output_dir / f"{stamp}_candidate_models.md")

    claude5 = [row["model"] for row in candidates if row.get("required_priority")]
    print(f"models={len(flat)}")
    print(f"candidates={len(candidates)}")
    print("claude5_sonnet_candidates=" + json.dumps(claude5, ensure_ascii=False))
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
