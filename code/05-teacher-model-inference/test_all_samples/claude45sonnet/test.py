#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run reliable concurrent inference for all claude-sonnet-4.5 samples.

Model: anthropic/claude-sonnet-4.5
Seed: 42
Retries: API (10), JSON (3), and automatic resume
Rate limiting: request backoff, global 429 circuit breaker, token recalibration
Resume behavior: skip only samples with successful API and JSON results
Initial QPS: 8; halve the rate after HTTP 429 responses
"""

import asyncio
import aiohttp
import json
import time
import os
import re
import sys
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")
try:
    from json_repair import repair_json as _repair_json

    _HAS_JSON_REPAIR = True
except ImportError:
    _HAS_JSON_REPAIR = False

# ==================== Path configuration ====================
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_CODE_DIR))

from config.paths import (
    CHFS2019_DKI_PROMPTS_FILE,
    TEACHER_INFERENCE_CHFS2019_CLAUDE_RESULTS_DIR,
)

PROMPT_PATH = str(CHFS2019_DKI_PROMPTS_FILE)
JSON_OUTPUT_DIR = str(TEACHER_INFERENCE_CHFS2019_CLAUDE_RESULTS_DIR)
os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)

API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = [("anthropic/claude-sonnet-4.5", "claude45sonnet")]

SEEDS = [42]
TEMPERATURE = 0.5
MAX_CONCURRENT_PER_MODEL = 15
MAX_API_RETRY = 10
MAX_JSON_RETRY = 3


# ==================== Adaptive rate limiter with recovery ====================
class AdaptiveRateLimiter:
    """Apply adaptive rate limiting with 429 protection and gradual recovery."""

    def __init__(self, initial_rate=8.0):
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


# Shared rate limiter
rate_limiter = AdaptiveRateLimiter(initial_rate=8.0)


# ==================== API call with three retry layers ====================
async def async_call_api(
    session, model_id, prompt_text, sample_id, seed, max_retry=MAX_API_RETRY
):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": TEMPERATURE,
        "max_tokens": 8000,
        "seed": seed,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(max_retry):
        try:
            await rate_limiter.acquire()

            async with session.post(
                API_URL, headers=headers, json=payload, timeout=120
            ) as response:
                if response.status == 200:
                    rate_limiter.on_success()
                    return {
                        "success": True,
                        "data": await response.json(),
                        "sample_id": sample_id,
                    }

                elif response.status == 429:
                    rate_limiter.on_429()
                    backoff = min(30.0, (2**attempt) * 2.0)
                    print(
                        f"  ⚠️  Sample {sample_id}: HTTP 429 (attempt {attempt + 1}); retrying in {backoff:.1f}s..."
                    )
                    await asyncio.sleep(backoff)
                    continue

                elif response.status in [502, 503, 504]:
                    backoff = min(20.0, (2**attempt) * 1.5)
                    print(
                        f"  ⚠️  Sample {sample_id}: HTTP {response.status} (attempt {attempt + 1}); retrying in {backoff:.1f}s..."
                    )
                    await asyncio.sleep(backoff)
                    continue

                elif response.status == 400:
                    error_text = await response.text()
                    if (
                        "model" in error_text.lower()
                        and "not found" in error_text.lower()
                    ):
                        return {
                            "success": False,
                            "error": f"Invalid model ID: {model_id}",
                            "sample_id": sample_id,
                            "model_invalid": True,
                        }
                    return {
                        "success": False,
                        "error": f"HTTP 400: {error_text[:150]}",
                        "sample_id": sample_id,
                    }

                else:
                    error_text = (await response.text())[:200]
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}: {error_text}",
                        "sample_id": sample_id,
                    }

        except asyncio.TimeoutError:
            if attempt < max_retry - 1:
                await asyncio.sleep(min(10.0, 2**attempt))
                continue
            return {
                "success": False,
                "error": "Request timeout",
                "sample_id": sample_id,
            }

        except Exception as e:
            if attempt < max_retry - 1:
                await asyncio.sleep(min(5.0, 2**attempt))
                continue
            return {"success": False, "error": str(e)[:200], "sample_id": sample_id}

    return {
        "success": False,
        "error": f"Max retries ({max_retry}) exceeded",
        "sample_id": sample_id,
    }


# ==================== JSON parsing ====================
def fix_common_json_errors(json_str):
    json_str = json_str.replace("'", '"')
    json_str = re.sub(r",\s*([\}\]])", r"\1", json_str)
    json_str = re.sub(r"\bNaN\b", "null", json_str)
    json_str = re.sub(r"\bInfinity\b", "null", json_str)

    def replace_newlines_in_strings(match):
        s = match.group(0)
        return s.replace("\n", "\\n").replace("\r", "\\r")

    json_str = re.sub(r'"(?:[^"\\]|\\.)*"', replace_newlines_in_strings, json_str)
    return json_str


def robust_json_parse(content, _depth=0):
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

            # Parse directly first, then repair malformed JSON if needed.
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


# ==================== Single-model, single-seed concurrent run ====================
async def test_single_model_with_seed(model_id, model_short_name, seed, test_prompts):
    result_filename = f"{model_short_name}_seed{seed}_temp{str(TEMPERATURE).replace('.', '')}_results.json"
    result_filepath = os.path.join(JSON_OUTPUT_DIR, result_filename)

    # Resume by skipping only samples with successful API and JSON results.
    existing_results = {}
    if os.path.exists(result_filepath):
        try:
            with open(result_filepath, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                existing_results = {
                    r["sample_id"]: r
                    for r in existing_data.get("tests", [])
                    if r.get("success") and r.get("parse_success")
                }
            print(f"  ✓ Skipped {len(existing_results)} fully successful samples")
        except Exception as e:
            print(f"  ⚠ Failed to load results: {str(e)[:60]}")

    pending = [
        (p, f"{p['household_id']}-{p['individual_id']}")
        for p in test_prompts
        if f"{p['household_id']}-{p['individual_id']}" not in existing_results
    ]

    if not pending:
        print(f"  ✓ All samples succeeded; skipping test")
        with open(result_filepath, "r", encoding="utf-8") as f:
            return json.load(f)["tests"], False, 0.1

    print(
        f"  📊 Pending: {len(pending)}/{len(test_prompts)} | Concurrency: {MAX_CONCURRENT_PER_MODEL} | Initial QPS: {rate_limiter.base_rate:.1f}"
    )

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_PER_MODEL)
    invalid_model = False
    results = []

    async with aiohttp.ClientSession() as session:

        async def process_sample(prompt_info, sample_id, hhid, pid):
            nonlocal invalid_model
            async with semaphore:
                parse_retry = 0
                while parse_retry <= MAX_JSON_RETRY:
                    api_res = await async_call_api(
                        session,
                        model_id,
                        prompt_info["prompt"],
                        sample_id,
                        seed,
                        max_retry=MAX_API_RETRY,
                    )

                    if api_res.get("model_invalid") and not invalid_model:
                        invalid_model = True
                        print(f"\n  ⚠ Invalid model ID: {model_id}")

                    result = {
                        "household_id": hhid,
                        "individual_id": pid,
                        "sample_id": sample_id,
                        "seed": seed,
                        "success": api_res["success"],
                        "parse_retry_count": parse_retry,
                    }

                    if not api_res["success"]:
                        result["error"] = api_res.get("error", "Unknown")
                        if api_res.get("model_invalid"):
                            result["model_invalid"] = True
                        return result

                    content = api_res["data"]["choices"][0]["message"]["content"]
                    result.update(
                        {
                            "response": content,
                            "usage": api_res["data"].get("usage", {}),
                            "model_used": api_res["data"].get("model", model_id),
                        }
                    )

                    parsed, parse_err = robust_json_parse(content)

                    if not parsed:
                        parse_retry += 1
                        result["parse_success"] = False
                        result["parse_error"] = f"JSON parsing failed: {parse_err}"
                        print(f"\n⚠️  Sample {sample_id}: initial JSON parsing failed")
                        print(f"   • Error: {parse_err}")
                        print(f"   • First 300 response characters: {content[:300]}...")
                        if parse_retry <= MAX_JSON_RETRY:
                            await asyncio.sleep(1.5)
                            continue
                        else:
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
                        if "action" in ins_dec and ins_dec["action"] not in [
                            "参保",
                            "不参保",
                        ]:
                            validation_errors.append(
                                f"Invalid action value: '{ins_dec.get('action')}'"
                            )

                    if validation_errors:
                        parse_retry += 1
                        error_msg = " | ".join(validation_errors)
                        result["parse_success"] = False
                        result["parse_error"] = f"Schema validation failed: {error_msg}"
                        print(f"\n⚠️  Sample {sample_id}: schema validation failed: {error_msg}")
                        if parse_retry <= MAX_JSON_RETRY:
                            await asyncio.sleep(1.5)
                            continue
                        else:
                            return result

                    result["parsed_json"] = parsed
                    result["parse_success"] = True
                    result.update(
                        {
                            "predicted_action": ins_dec["action"],
                            "predicted_insurance_type": ins_dec["insurance_type"],
                            "predicted_annual_payment": ins_dec.get(
                                "annual_payment", 0
                            ),
                            "predicted_main_reason": ins_dec.get("main_reason", ""),
                        }
                    )
                    return result

        tasks = [
            process_sample(p, sid, p["household_id"], p["individual_id"])
            for p, sid in pending
        ]

        start_time = time.time()
        for idx, coro in enumerate(asyncio.as_completed(tasks), 1):
            res = await coro
            pct = idx / len(tasks) * 100
            status = (
                "✓✓"
                if res.get("success") and res.get("parse_success")
                else ("✓✗" if res.get("success") else "✗✗")
            )
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

    # Merge prior successful results with the current run.
    all_results = []
    existing_dict = {r["sample_id"]: r for r in existing_results.values()}
    for p in test_prompts:
        sid = f"{p['household_id']}-{p['individual_id']}"
        if sid in existing_dict:
            all_results.append(existing_dict[sid])
        else:
            matches = [r for r in results if r["sample_id"] == sid]
            if matches:
                all_results.append(matches[-1])

    summary = {
        "total": len(test_prompts),
        "success": sum(1 for r in all_results if r.get("success")),
        "parse_success": sum(1 for r in all_results if r.get("parse_success")),
        "failed": len(test_prompts) - sum(1 for r in all_results if r.get("success")),
        "parse_failed": sum(
            1 for r in all_results if r.get("success") and not r.get("parse_success")
        ),
        "total_tokens": sum(
            r.get("usage", {}).get("total_tokens", 0) for r in all_results
        ),
        "total_cost": sum(r.get("usage", {}).get("cost", 0) for r in all_results),
        "model_invalid": invalid_model,
        "total_time": time.time() - start_time,
    }

    # Write atomically to avoid losing results if the process is interrupted.
    tmp_filepath = result_filepath + ".tmp"
    with open(tmp_filepath, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_id": model_id,
                "model_short_name": model_short_name,
                "seed": seed,
                "test_time": datetime.now().isoformat(),
                "temperature": TEMPERATURE,
                "api_params": {
                    "temperature": TEMPERATURE,
                    "max_tokens": 8000,
                    "seed": seed,
                    "response_format": {"type": "json_object"},
                },
                "tests": all_results,
                "summary": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    os.replace(tmp_filepath, result_filepath)  # Atomic on both Windows and Linux.

    parse_rate = (
        f"{summary['parse_success']}/{summary['success']}"
        if summary["success"]
        else "N/A"
    )
    avg_qps = len(pending) / summary["total_time"] if summary["total_time"] > 0 else 0
    print(
        f"  ✓ Complete: {summary['success']}/{summary['total']} API✓, "
        f"{parse_rate} JSON✓ | "
        f"Elapsed: {summary['total_time']:.1f}s | Average QPS: {avg_qps:.1f}"
    )

    return all_results, invalid_model, summary["total_time"]


# ==================== Main workflow ====================
async def main_async():
    if not API_KEY:
        raise RuntimeError("Set OPENROUTER_API_KEY before running teacher inference")

    print("=" * 80)
    print("🔥 Reliable Full-Sample Concurrent Test - claude-sonnet-4.5")
    print(f"Models: {len(MODELS)} | Seeds: {SEEDS} | Concurrency: {MAX_CONCURRENT_PER_MODEL}")
    print(f"Initial QPS: {rate_limiter.base_rate:.1f} | 429 circuit breaker: rate × 0.5")
    print("=" * 80)

    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        test_prompts = json.load(f)

    print(f"\n📥 Samples: {len(test_prompts)}")

    all_summaries = []
    for m_idx, (model_id, short_name) in enumerate(MODELS, 1):
        print(f"\n{'=' * 80}")
        print(f"[Model {m_idx}/{len(MODELS)}] {short_name}")
        print(f"{'=' * 80}")

        for s_idx, seed in enumerate(SEEDS, 1):
            print(f"\n▶ Seed {s_idx}/{len(SEEDS)}: {seed}")
            results, invalid, duration = await test_single_model_with_seed(
                model_id, short_name, seed, test_prompts
            )
            all_summaries.append(
                {
                    "model": short_name,
                    "seed": seed,
                    "success": sum(1 for r in results if r.get("success")),
                    "parse_success": sum(1 for r in results if r.get("parse_success")),
                    "total": len(test_prompts),
                    "time": duration,
                    "invalid": invalid,
                }
            )
            if s_idx < len(SEEDS):
                await asyncio.sleep(2)

        if m_idx < len(MODELS):
            await asyncio.sleep(5)

    total_time = sum(s["time"] for s in all_summaries)
    print(f"\n{'=' * 80}")
    print(f"✅ All complete | Total elapsed: {total_time:.1f}s")
    print(f"{'=' * 80}")
    print(f"{'Model':<15} {'Seed':<6} {'API✓':<8} {'JSON✓':<8} {'Time (s)':<10}")
    for s in all_summaries:
        print(
            f"{s['model']:<15} {s['seed']:<6} {s['success']:<8} {s['parse_success']:<8} {s['time']:<10.1f}"
        )

    total_api = sum(s["success"] for s in all_summaries)
    total_json = sum(s["parse_success"] for s in all_summaries)
    total_samples = len(test_prompts) * len(MODELS) * len(SEEDS)
    if total_api > 0:
        print(
            f"\n📊 Overall: {total_json}/{total_api} valid JSON ({total_json / total_api * 100:.1f}%) | Total samples: {total_samples}"
        )
    print(f"\n✓ Results directory: {JSON_OUTPUT_DIR}")
    print(f"💡 Resume behavior: skip only API✓ + JSON✓ samples; retry failed samples")


if __name__ == "__main__":
    try:
        import aiohttp
    except ImportError:
        print("❌ Install required package: pip install aiohttp")
        exit(1)

    asyncio.run(main_async())
