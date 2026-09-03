#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regenerate label-consistent rationales for teacher-error cases.

Input: ``trainval_need_regen.json`` from ``build_mixed_dataset.py``.
Output: ``renew_claude_errors_results.json`` for SFT dataset construction.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from json_repair import repair_json as _repair_json

    _HAS_JSON_REPAIR = True
except ImportError:
    _HAS_JSON_REPAIR = False

# Paths and API configuration
_CODE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_CODE_DIR))

from config.paths import DISTILLATION_REGEN_INPUT_FILE, DISTILLATION_REGEN_RESULTS_FILE

ERROR_SAMPLES_PATH = DISTILLATION_REGEN_INPUT_FILE
RESULTS_PATH = DISTILLATION_REGEN_RESULTS_FILE

API_URL = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions")
API_KEY = os.getenv("OPENROUTER_API_KEY")

MODEL_ID = "anthropic/claude-sonnet-4.5"
MODEL_SHORT_NAME = "claude45sonnet"

SEED = 42
TEMPERATURE = 0.5
MAX_CONCURRENT = 15
MAX_API_RETRY = 10
MAX_JSON_RETRY = 3


# Adaptive rate limiter
class AdaptiveRateLimiter:
    def __init__(self, initial_rate=6.0):
        self.base_rate = initial_rate
        self.current_rate = initial_rate
        self.tokens = initial_rate
        self.last_update = time.time()
        self.lock = asyncio.Lock()
        self.last_429_time = 0
        self.consecutive_429 = 0

    async def acquire(self):
        async with self.lock:
            if self.consecutive_429 >= 3:
                elapsed = time.time() - self.last_429_time
                if elapsed < 5.0:
                    await asyncio.sleep(5.0 - elapsed)

            now = time.time()
            elapsed = now - self.last_update
            self.last_update = now
            self.tokens = min(
                self.current_rate, self.tokens + elapsed * self.current_rate
            )

            if self.tokens < 1:
                wait = (1 - self.tokens) / self.current_rate
                await asyncio.sleep(wait)
                self.tokens = 0
            else:
                self.tokens -= 1

    def on_429(self):
        self.consecutive_429 += 1
        self.last_429_time = time.time()
        new_rate = max(2.0, self.current_rate * 0.5)
        if new_rate < self.current_rate:
            old_rate = self.current_rate
            self.current_rate = new_rate
            print(
                f"⚠️  429 circuit breaker: rate {old_rate:.1f} → {new_rate:.1f} QPS ({self.consecutive_429} consecutive responses)"
            )

    def on_success(self):
        self.consecutive_429 = 0
        if hasattr(self, "success_count"):
            self.success_count += 1
        else:
            self.success_count = 1
        if self.success_count % 10 == 0 and self.current_rate < self.base_rate:
            new_rate = min(self.base_rate, self.current_rate * 1.05)
            if new_rate > self.current_rate:
                old_rate = self.current_rate
                self.current_rate = new_rate
                print(f"✅ Rate recovery: {old_rate:.1f} → {new_rate:.1f} QPS")


rate_limiter = AdaptiveRateLimiter(initial_rate=8.0)


# Robust JSON parsing
def fix_common_json_errors(json_str: str):
    json_str = json_str.replace("'", '"')
    json_str = re.sub(r",\s*([\}\]])", r"\1", json_str)
    json_str = re.sub(r"\bNaN\b", "null", json_str)
    json_str = re.sub(r"\bInfinity\b", "null", json_str)

    def replace_newlines_in_strings(match):
        s = match.group(0)
        return s.replace("\n", "\\n").replace("\r", "\\r")

    json_str = re.sub(r'"(?:[^"\\]|\\.)*"', replace_newlines_in_strings, json_str)
    return json_str


def robust_json_parse(content: str, _depth=0):
    if _depth > 2:
        return None, "Nested JSON exceeds the maximum recursion depth (2)"
    try:
        parsed = json.loads(content.strip())
    except json.JSONDecodeError:
        try:
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                parts = content.split("```")
                json_str = parts[1].strip() if len(parts) >= 3 else content
            else:
                json_str = content

            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError:
                if _HAS_JSON_REPAIR:
                    try:
                        repaired = _repair_json(json_str, return_objects=True)
                        if isinstance(repaired, dict):
                            parsed = repaired
                        else:
                            raise ValueError(f"JSON repair returned a non-dictionary value: {type(repaired)}")
                    except Exception:
                        json_str = fix_common_json_errors(json_str)
                        parsed = json.loads(json_str)
                else:
                    json_str = fix_common_json_errors(json_str)
                    parsed = json.loads(json_str)
        except Exception as e:
            return None, f"JSON parsing failed: {str(e)[:100]}"

    if isinstance(parsed, list):
        if len(parsed) == 0:
            return None, "Parsed result is an empty array"
        for item in parsed:
            if isinstance(item, dict):
                parsed = item
                break
        else:
            return None, f"Array contains no dictionary element: {str(parsed)[:100]}"

    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
            parsed, err = robust_json_parse(parsed, _depth + 1)
            if parsed is None:
                return None, f"Nested JSON parsing failed (depth={_depth + 1}): {err}"
        except json.JSONDecodeError as e:
            return None, f"Nested JSON parsing failed: {str(parsed)[:100]} | {str(e)[:60]}"

    if not isinstance(parsed, dict):
        return None, f"Expected a dictionary, received {type(parsed).__name__}: {str(parsed)[:100]}"

    for key in ["decision", "result", "output", "data", "response"]:
        if key in parsed and isinstance(parsed[key], dict):
            nested_dict = parsed.pop(key)
            for k, v in nested_dict.items():
                if k not in parsed:
                    parsed[k] = v
            break

    return parsed, None


def load_previous_results(path: Path):
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError:
        return {}

    tests = payload.get("tests", []) if isinstance(payload, dict) else []
    results = {}
    for entry in tests:
        sample_id = entry.get("sample_id")
        if sample_id:
            results[sample_id] = entry
    return results


def merge_results(existing: dict[str, dict], updated: list[dict]):
    merged = dict(existing)
    for item in updated:
        sample_id = item.get("sample_id")
        if sample_id:
            merged[sample_id] = item
    return merged


# Ground-truth-conditioned prompt construction
NEW_TASK_BACKGROUND = (
    "# 任务背景\n\n"
    "请你根据提供的灵活就业人员个人信息和养老保险参保决策，"
    "补全在给定的政策情景下该样本的思考链，其中养老保险参保决策包括是否参保和参保种类（城乡居民基本养老保险/城镇职工基本养老保险）。\n\n"
)


def update_task_background(prompt: str):
    start_anchor = "# 任务背景"
    end_anchor = "# 个人与家庭数据"
    if start_anchor in prompt and end_anchor in prompt:
        before, rest = prompt.split(start_anchor, 1)
        _, after = rest.split(end_anchor, 1)
        return f"{before}{NEW_TASK_BACKGROUND}{end_anchor}{after}"
    return prompt


def inject_ground_truth_block(prompt: str, action: str, insurance_type: str):
    gt_block = (
        "## 真实参保决策\n\n"
        "- \"action\": \"不参保/参保\"\n"
        "- \"insurance_type\": \"不参保/城乡居民养老保险/城镇职工养老保险\"\n\n"
        f"- 真实参保决策: {action}\n"
        f"- 真实参保类型: {insurance_type}\n\n"
    )

    anchor = "# 政策情景（基于户籍地）"
    if anchor in prompt:
        return prompt.replace(anchor, gt_block + anchor)
    return prompt + "\n\n" + gt_block


# Teacher API calls
async def async_call_api(session, prompt_text: str, sample_id: str, max_retry=MAX_API_RETRY):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": TEMPERATURE,
        "max_tokens": 8000,
        "seed": SEED,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(max_retry):
        try:
            await rate_limiter.acquire()

            async with session.post(API_URL, headers=headers, json=payload, timeout=120) as response:
                if response.status == 200:
                    rate_limiter.on_success()
                    return {"success": True, "data": await response.json(), "sample_id": sample_id}

                if response.status == 429:
                    rate_limiter.on_429()
                    backoff = min(30.0, (2**attempt) * 2.0)
                    print(f"  ⚠️  Sample {sample_id}: HTTP 429 (attempt {attempt + 1}); retrying in {backoff:.1f}s...")
                    await asyncio.sleep(backoff)
                    continue

                if response.status in [502, 503, 504]:
                    backoff = min(20.0, (2**attempt) * 1.5)
                    print(
                        f"  ⚠️  Sample {sample_id}: HTTP {response.status} (attempt {attempt + 1}); retrying in {backoff:.1f}s..."
                    )
                    await asyncio.sleep(backoff)
                    continue

                error_text = (await response.text())[:200]
                if response.status == 403 and "not available in your region" in error_text:
                    raise RuntimeError(
                        f"HTTP 403: model unavailable in this region; execution stopped. Details: {error_text}"
                    )
                return {"success": False, "error": f"HTTP {response.status}: {error_text}", "sample_id": sample_id}

        except asyncio.TimeoutError:
            if attempt < max_retry - 1:
                await asyncio.sleep(min(10.0, 2**attempt))
                continue
            return {"success": False, "error": "Request timeout", "sample_id": sample_id}

        except Exception as e:
            if attempt < max_retry - 1:
                await asyncio.sleep(min(5.0, 2**attempt))
                continue
            return {"success": False, "error": str(e)[:200], "sample_id": sample_id}

    return {"success": False, "error": f"Max retries ({max_retry}) exceeded", "sample_id": sample_id}


async def test_errors(samples, previous_results):
    import aiohttp

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    results = []

    async with aiohttp.ClientSession() as session:

        async def process_sample(sample):
            sample_id = sample["sample_id"]
            gt = sample["gt"]
            base_prompt = update_task_background(sample["prompt"])
            prompt_text = inject_ground_truth_block(
                base_prompt,
                gt.get("decision", ""),
                gt.get("type", ""),
            )

            async with semaphore:
                parse_retry = 0
                while parse_retry <= MAX_JSON_RETRY:
                    api_res = await async_call_api(session, prompt_text, sample_id)
                    result = {
                        "sample_id": sample_id,
                        "household_id": gt.get("household_id"),
                        "individual_id": gt.get("individual_id"),
                        "seed": SEED,
                        "success": api_res["success"],
                        "parse_retry_count": parse_retry,
                    }

                    if not api_res["success"]:
                        result["error"] = api_res.get("error", "Unknown")
                        return result

                    content = api_res["data"]["choices"][0]["message"]["content"]
                    result.update(
                        {
                            "response": content,
                            "usage": api_res["data"].get("usage", {}),
                            "model_used": api_res["data"].get("model", MODEL_ID),
                        }
                    )

                    parsed, parse_err = robust_json_parse(content)
                    if not parsed:
                        parse_retry += 1
                        result["parse_success"] = False
                        result["parse_error"] = f"JSON parsing failed: {parse_err}"
                        if parse_retry <= MAX_JSON_RETRY:
                            await asyncio.sleep(1.5)
                            continue
                        return result

                    ins_dec = parsed.get("insurance_decision")
                    validation_errors = []
                    if not isinstance(ins_dec, dict):
                        validation_errors.append(
                            f"insurance_decision is missing or not a dictionary (type: {type(ins_dec).__name__ if ins_dec is not None else 'None'})"
                        )
                    else:
                        if "action" not in ins_dec:
                            validation_errors.append("Missing action field")
                        if "insurance_type" not in ins_dec:
                            validation_errors.append("Missing insurance_type field")
                        if "action" in ins_dec and ins_dec["action"] not in ["参保", "不参保"]:
                            validation_errors.append(f"Invalid action value: '{ins_dec.get('action')}'")

                    if validation_errors:
                        parse_retry += 1
                        error_msg = " | ".join(validation_errors)
                        result["parse_success"] = False
                        result["parse_error"] = f"Schema validation failed: {error_msg}"
                        if parse_retry <= MAX_JSON_RETRY:
                            await asyncio.sleep(1.5)
                            continue
                        return result

                    result["parsed_json"] = parsed
                    result["parse_success"] = True
                    result.update(
                        {
                            "predicted_action": ins_dec["action"],
                            "predicted_insurance_type": ins_dec["insurance_type"],
                            "predicted_annual_payment": ins_dec.get("annual_payment", 0),
                            "predicted_main_reason": ins_dec.get("main_reason", ""),
                        }
                    )
                    return result

        tasks = [process_sample(sample) for sample in samples]
        start_time = time.time()
        for idx, coro in enumerate(asyncio.as_completed(tasks), 1):
            res = await coro
            pct = idx / len(tasks) * 100 if samples else 100
            status = "✓✓" if res.get("success") and res.get("parse_success") else ("✓✗" if res.get("success") else "✗✗")
            action = res.get("predicted_action", "N/A")[:4]
            ptype = res.get("predicted_insurance_type", "N/A")[:8]
            elapsed = time.time() - start_time
            eta = elapsed / idx * (len(tasks) - idx) if idx else 0

            print(
                f"  [{idx:4d}/{len(tasks)}] {pct:5.1f}% {res['sample_id']:20s} {status} "
                f"{action:4s}/{ptype:8s} | QPS: {rate_limiter.current_rate:.1f} | ETA: {eta:.0f}s",
                end="\r",
            )
            results.append(res)

        print()

    total_time = time.time() - start_time
    merged_results = merge_results(previous_results, results)
    merged_list = list(merged_results.values())

    summary = {
        "total": len(merged_list),
        "success": sum(1 for r in merged_list if r.get("success")),
        "parse_success": sum(1 for r in merged_list if r.get("parse_success")),
        "failed": len(merged_list) - sum(1 for r in merged_list if r.get("success")),
        "parse_failed": sum(1 for r in merged_list if r.get("success") and not r.get("parse_success")),
        "total_tokens": sum(r.get("usage", {}).get("total_tokens", 0) for r in merged_list),
        "total_cost": sum(r.get("usage", {}).get("cost", 0) for r in merged_list),
        "total_time": total_time,
        "ran_samples": len(results),
        "skipped_samples": len(previous_results) - len(results),
    }

    payload = {
        "model_id": MODEL_ID,
        "model_short_name": MODEL_SHORT_NAME,
        "seed": SEED,
        "test_time": datetime.now().isoformat(),
        "temperature": TEMPERATURE,
        "api_params": {
            "temperature": TEMPERATURE,
            "max_tokens": 8000,
            "seed": SEED,
            "response_format": {"type": "json_object"},
        },
        "tests": merged_list,
        "summary": summary,
    }

    tmp_path = str(RESULTS_PATH) + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, RESULTS_PATH)

    print(
        f"  ✓ Complete: {summary['success']}/{summary['total']} API✓, "
        f"{summary['parse_success']}/{summary['success']} JSON✓ | "
        f"Elapsed: {summary['total_time']:.1f}s"
    )


async def main_async():
    if not API_KEY:
        raise RuntimeError(
            "Set OPENROUTER_API_KEY before regenerating teacher-error rationales"
        )

    if not ERROR_SAMPLES_PATH.exists():
        raise FileNotFoundError(f"Sample file not found: {ERROR_SAMPLES_PATH}")

    with ERROR_SAMPLES_PATH.open("r", encoding="utf-8") as f:
        samples = json.load(f)

    previous_results = load_previous_results(RESULTS_PATH)
    failed_sample_ids = {
        sample_id
        for sample_id, result in previous_results.items()
        if result.get("error")
        or result.get("parse_success") is False
        or result.get("success") is False
    }
    existing_sample_ids = set(previous_results.keys())

    if previous_results:
        samples = [
            sample
            for sample in samples
            if sample.get("sample_id") not in existing_sample_ids
            or sample.get("sample_id") in failed_sample_ids
        ]

    print("=" * 80)
    print("🔄 Regenerate Results for Claude Error Samples")
    print(f"Model: {MODEL_ID}")
    print(
        f"Samples: {len(samples)} | Concurrency: {MAX_CONCURRENT} | Initial QPS: {rate_limiter.base_rate:.1f}"
    )
    if previous_results:
        print(
            f"Existing results: {len(previous_results)} | Error samples to rerun: {len(samples)}"
        )
    print("=" * 80)

    await test_errors(samples, previous_results)
    print(f"\n✓ Results file: {RESULTS_PATH}")


if __name__ == "__main__":
    try:
        import aiohttp  # noqa: F401
    except ImportError:
        print("❌ Install required package: pip install aiohttp")
        raise SystemExit(1)

    asyncio.run(main_async())
