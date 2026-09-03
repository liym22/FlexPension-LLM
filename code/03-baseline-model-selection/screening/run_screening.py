#!/usr/bin/env python
"""Run the historical 30-case, three-seed model screening through OpenRouter."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml


def build_payload(
    *,
    model_id: str,
    prompt: str,
    seed: int,
    temperature: float,
    max_tokens: int,
) -> dict:
    return {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "response_format": {"type": "json_object"},
    }


def _parse_prediction(content: str) -> tuple[dict, str, str]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
    parsed = json.loads(cleaned)
    decision = parsed.get("insurance_decision") or {}
    return parsed, decision.get("action", ""), decision.get("insurance_type", "")


def _request(url: str, api_key: str, payload: dict, timeout: int) -> dict:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _run_one(url: str, api_key: str, payload: dict, retries: int, timeout: int) -> dict:
    for attempt in range(retries + 1):
        try:
            return _request(url, api_key, payload, timeout)
        except (HTTPError, URLError, TimeoutError):
            if attempt == retries:
                raise
            time.sleep(min(30, 2 ** (attempt + 1)))
    raise RuntimeError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required")
    api_url = os.environ.get("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    temperature = float(config.get("temperature", 0.5))
    max_tokens = int(config.get("max_tokens", 5000))
    retries = int(config.get("retries", 10))
    timeout = int(config.get("timeout_seconds", 120))

    for model in config["models"]:
        for seed in model.get("seeds", config.get("seeds", [42, 123, 456])):
            tests = []
            for row in prompts:
                payload = build_payload(
                    model_id=model["model_id"],
                    prompt=row["prompt"],
                    seed=int(seed),
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                response = _run_one(api_url, api_key, payload, retries, timeout)
                content = response["choices"][0]["message"]["content"]
                parsed, action, insurance_type = _parse_prediction(content)
                tests.append(
                    {
                        "sample_id": str(row["id"]),
                        "success": True,
                        "parse_success": True,
                        "response": content,
                        "parsed_json": parsed,
                        "predicted_action": action,
                        "predicted_insurance_type": insurance_type,
                        "usage": response.get("usage") or {},
                    }
                )
            output = {
                "model_id": model["model_id"],
                "model_short_name": model["short_name"],
                "seed": int(seed),
                "temperature": temperature,
                "tests": tests,
            }
            path = args.output_dir / f"{model['short_name']}_seed{seed}_temp05_results.json"
            path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
