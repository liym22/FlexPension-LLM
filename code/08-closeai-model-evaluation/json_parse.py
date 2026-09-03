from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ParsedPrediction:
    action: str
    insurance_type: str
    parse_ok: bool
    raw_json: dict[str, Any] | None = None


def _loads(text: str) -> Optional[dict[str, Any]]:
    try:
        obj = json.loads(text)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _extract_json_object(text: str) -> Optional[dict[str, Any]]:
    fenced = re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    for block in reversed(fenced):
        parsed = _loads(block)
        if parsed is not None:
            return parsed

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def extract_prediction(text: str) -> ParsedPrediction:
    payload = _extract_json_object(text or "")
    if payload is None:
        return ParsedPrediction(action="不参保", insurance_type="不参保", parse_ok=False, raw_json=None)
    decision = payload.get("insurance_decision") or payload.get("insuranceDecision") or {}
    action = decision.get("action") or "不参保"
    insurance_type = decision.get("insurance_type") or decision.get("insuranceType") or "不参保"
    return ParsedPrediction(action=action, insurance_type=insurance_type, parse_ok=True, raw_json=payload)

