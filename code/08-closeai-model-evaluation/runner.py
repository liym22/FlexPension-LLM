from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from closeai_config import get_closeai_api_key, get_closeai_chat_url
from config_loader import load_config
from datasets import EvalSample, load_distill_jsonl, load_external_json, stratified_sample
from json_parse import extract_prediction


class BudgetStop(RuntimeError):
    """Raised when configured spend limits require stopping the run."""


class AdaptiveConcurrencyLimiter:
    def __init__(
        self,
        *,
        enabled: bool,
        min_concurrency: int,
        max_concurrency: int,
        increase_every_successes: int,
        cooldown_seconds: float,
        name: str,
        now_func: Callable[[], float] = time.monotonic,
    ) -> None:
        self.enabled = enabled
        self.min_concurrency = max(1, int(min_concurrency))
        self.max_concurrency = max(self.min_concurrency, int(max_concurrency))
        self.increase_every_successes = max(1, int(increase_every_successes))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.name = name
        self._now = now_func
        self._current_limit = self.min_concurrency
        self._active = 0
        self._successes_since_increase = 0
        self._cooldown_until = 0.0
        self._condition = asyncio.Condition()

    @classmethod
    def from_api_config(cls, api_cfg: dict[str, Any], name: str) -> "AdaptiveConcurrencyLimiter":
        fixed = max(1, int(api_cfg.get("concurrency_per_model", 1)))
        adaptive = api_cfg.get("adaptive_concurrency") or {}
        enabled = bool(adaptive.get("enabled", False))
        if not enabled:
            return cls(
                enabled=False,
                min_concurrency=fixed,
                max_concurrency=fixed,
                increase_every_successes=1,
                cooldown_seconds=0,
                name=name,
            )
        return cls(
            enabled=True,
            min_concurrency=int(adaptive.get("min", fixed)),
            max_concurrency=int(adaptive.get("max", fixed)),
            increase_every_successes=int(adaptive.get("increase_every_successes", 25)),
            cooldown_seconds=float(adaptive.get("cooldown_seconds", 60)),
            name=name,
        )

    @property
    def current_limit(self) -> int:
        return self._current_limit

    @property
    def max_workers(self) -> int:
        return self.max_concurrency

    @property
    def in_cooldown(self) -> bool:
        return self._now() < self._cooldown_until

    async def acquire(self) -> None:
        while True:
            async with self._condition:
                now = self._now()
                if now >= self._cooldown_until and self._active < self._current_limit:
                    self._active += 1
                    return
                wait_seconds = max(0.0, self._cooldown_until - now)
                if wait_seconds == 0:
                    await self._condition.wait()
                    continue
            await asyncio.sleep(min(wait_seconds, 1.0))

    async def release(self) -> None:
        async with self._condition:
            self._active = max(0, self._active - 1)
            self._condition.notify_all()

    async def on_success(self) -> None:
        if not self.enabled:
            return
        async with self._condition:
            if self.in_cooldown:
                return
            self._successes_since_increase += 1
            if self._successes_since_increase < self.increase_every_successes:
                return
            self._successes_since_increase = 0
            if self._current_limit < self.max_concurrency:
                self._current_limit += 1
                print(
                    f"[CONCURRENCY] {self.name}: increased to {self._current_limit}/{self.max_concurrency}",
                    flush=True,
                )
                self._condition.notify_all()

    async def on_rate_limited(self) -> None:
        if not self.enabled:
            return
        async with self._condition:
            self._current_limit = self.min_concurrency
            self._successes_since_increase = 0
            self._cooldown_until = max(self._cooldown_until, self._now() + self.cooldown_seconds)
            print(
                f"[CONCURRENCY] {self.name}: HTTP 429, reduced to {self.min_concurrency}; "
                f"cooldown={self.cooldown_seconds:.1f}s",
                flush=True,
            )
            self._condition.notify_all()


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    short_name: str
    family: str = ""
    priority: str = ""
    input_price_rmb_per_1m: float = 0.0
    output_price_rmb_per_1m: float = 0.0
    omit_temperature: bool = False
    max_tokens_param: str = "max_tokens"
    max_tokens: int | None = None
    timeout_seconds: int | None = None
    hard_stop_rmb: float | None = None
    projected_stop_rmb: float | None = None


def load_models(path: Path) -> list[ModelSpec]:
    data = load_config(path)
    models = []
    for row in data.get("models", []):
        models.append(
            ModelSpec(
                model_id=row["model_id"],
                short_name=row["short_name"],
                family=row.get("family", ""),
                priority=row.get("priority", ""),
                input_price_rmb_per_1m=float(row.get("input_price_rmb_per_1m", 0.0) or 0.0),
                output_price_rmb_per_1m=float(row.get("output_price_rmb_per_1m", 0.0) or 0.0),
                omit_temperature=bool(row.get("omit_temperature", False)),
                max_tokens_param=row.get("max_tokens_param", "max_tokens"),
                max_tokens=int(row["max_tokens"]) if row.get("max_tokens") is not None else None,
                timeout_seconds=int(row["timeout_seconds"]) if row.get("timeout_seconds") is not None else None,
                hard_stop_rmb=float(row["hard_stop_rmb"]) if row.get("hard_stop_rmb") is not None else None,
                projected_stop_rmb=(
                    float(row["projected_stop_rmb"]) if row.get("projected_stop_rmb") is not None else None
                ),
            )
        )
    if not models:
        raise ValueError(f"No models found in {path}")
    return models


def load_samples(project_root: Path, config: dict[str, Any], dataset_name: str | None = None) -> list[EvalSample]:
    if config["run_type"] == "external_new":
        if not dataset_name:
            raise ValueError("--dataset-name is required for external_new")
        dataset_cfg = config["datasets"][dataset_name]
        return load_external_json(project_root / dataset_cfg["prompts"], project_root / dataset_cfg["ground_truth"])

    dataset_cfg = config["dataset"]
    kind = dataset_cfg["kind"]
    if kind == "distill_jsonl":
        samples = load_distill_jsonl(project_root / dataset_cfg["path"])
        sample_size = dataset_cfg.get("sample_size")
        if sample_size:
            return samples[: int(sample_size)]
        return samples

    if kind == "stratified_distill_jsonl":
        samples: list[EvalSample] = []
        for rel_path in dataset_cfg["paths"]:
            samples.extend(load_distill_jsonl(project_root / rel_path))
        return stratified_sample(samples, int(dataset_cfg["sample_size"]), int(dataset_cfg.get("seed", 42)))

    raise ValueError(f"Unsupported dataset kind: {kind}")


def validate_samples_for_config(
    samples: list[EvalSample],
    config: dict[str, Any],
    dataset_name: str | None = None,
) -> None:
    run_type = config.get("run_type", "")
    if run_type == "external_new":
        if not dataset_name:
            raise ValueError("--dataset-name is required for external_new")
        dataset_cfg = config["datasets"][dataset_name]
    else:
        dataset_cfg = config.get("dataset") or {}

    if run_type == "blind" and dataset_cfg.get("sample_size") is not None:
        raise ValueError("Blind runs must use the fixed full dataset; remove dataset.sample_size")

    expected_size = dataset_cfg.get("expected_size")
    if expected_size is not None and len(samples) != int(expected_size):
        raise ValueError(f"Loaded {len(samples)} samples but expected_size={expected_size}")


def request_chat(payload: dict[str, Any], api_key: str, timeout: int = 180) -> tuple[int, dict[str, Any] | None, str]:
    request = Request(
        get_closeai_chat_url(),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8")
            return response.status, json.loads(text), text
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return exc.code, None, body
    except URLError as exc:
        return 0, None, str(exc)
    except Exception as exc:
        return 0, None, str(exc)


def completed_keys(result_path: Path) -> set[str]:
    keys: set[str] = set()
    if not result_path.exists():
        return keys
    with result_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("success") and row.get("parse_success"):
                keys.add(f"{row.get('model_id')}::{row.get('sample_id')}")
    return keys


def existing_result_cost_rmb(result_path: Path) -> float:
    if not result_path.exists():
        return 0.0
    total = 0.0
    with result_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            usage = row.get("usage") or {}
            total += float(row.get("cost_rmb") or usage.get("cost_rmb") or usage.get("cost") or 0.0)
    return total


def result_file_stats(result_path: Path) -> tuple[int, float]:
    if not result_path.exists():
        return 0, 0.0
    rows = 0
    cost = 0.0
    with result_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            rows += 1
            usage = row.get("usage") or {}
            cost += float(row.get("cost_rmb") or usage.get("cost_rmb") or usage.get("cost") or 0.0)
    return rows, cost


def run_cost_rmb(output_dir: Path) -> float:
    if not output_dir.exists():
        return 0.0
    return sum(existing_result_cost_rmb(path) for path in output_dir.glob("*_results.jsonl"))


def check_budget_guard(
    model: ModelSpec,
    sample_count: int,
    budget_cfg: dict[str, Any],
    output_dir: Path,
    result_path: Path,
) -> dict[str, float | int | None]:
    run_cost = run_cost_rmb(output_dir)
    hard_stop = float(budget_cfg.get("hard_stop_rmb", 10000))
    if run_cost >= hard_stop:
        raise BudgetStop(f"run hard_stop_rmb reached: run_cost={run_cost:.4f}, limit={hard_stop:.4f}")

    model_rows, model_cost = result_file_stats(result_path)
    model_hard_stop = model.hard_stop_rmb
    if model_hard_stop is None and budget_cfg.get("model_hard_stop_rmb") is not None:
        model_hard_stop = float(budget_cfg["model_hard_stop_rmb"])
    if model_hard_stop is not None and model_cost >= model_hard_stop:
        raise BudgetStop(
            f"{model.short_name} hard_stop_rmb reached: cost={model_cost:.4f}, limit={model_hard_stop:.4f}"
        )

    projected_cost = None
    min_projection_samples = max(1, int(budget_cfg.get("min_projection_samples", 50)))
    projected_stop = model.projected_stop_rmb
    if projected_stop is None and budget_cfg.get("model_projected_stop_rmb") is not None:
        projected_stop = float(budget_cfg["model_projected_stop_rmb"])
    if sample_count > 0 and model_rows >= min_projection_samples:
        projected_cost = model_cost / model_rows * sample_count
        if projected_stop is not None and projected_cost >= projected_stop:
            raise BudgetStop(
                f"{model.short_name} projected_stop_rmb reached: "
                f"projected={projected_cost:.4f}, limit={projected_stop:.4f}, "
                f"rows={model_rows}/{sample_count}, cost={model_cost:.4f}"
            )

    return {
        "run_cost_rmb": run_cost,
        "model_rows": model_rows,
        "model_cost_rmb": model_cost,
        "projected_model_cost_rmb": projected_cost,
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def response_content(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, list):
        return "".join(str(part.get("text", part)) for part in content)
    return str(content)


def cost_from_usage(usage: dict[str, Any]) -> float:
    for key in ("cost_rmb", "cost"):
        value = usage.get(key)
        if value is not None:
            try:
                return float(value)
            except Exception:
                return 0.0
    return 0.0


def estimate_cost_rmb(usage: dict[str, Any], model: ModelSpec) -> float:
    explicit_cost = cost_from_usage(usage)
    if explicit_cost:
        return explicit_cost
    input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    return (
        input_tokens * model.input_price_rmb_per_1m / 1_000_000
        + output_tokens * model.output_price_rmb_per_1m / 1_000_000
    )


def build_payload(model: ModelSpec, prompt: str, api_cfg: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "model": model.model_id,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    payload[model.max_tokens_param] = model.max_tokens or int(api_cfg.get("max_tokens", 2048))
    if not model.omit_temperature:
        payload["temperature"] = float(api_cfg.get("temperature", 0.5))
    return payload


def should_retry_status(status: int) -> bool:
    return status == 0 or status == 408 or status == 409 or status == 429 or status >= 500


async def request_chat_with_retries(
    payload: dict[str, Any],
    api_key: str,
    api_cfg: dict[str, Any],
    rate_limit_callback: Callable[[], Any] | None = None,
) -> tuple[int, dict[str, Any] | None, str]:
    retry_count = int(api_cfg.get("retry_count", 2))
    base_delay = float(api_cfg.get("retry_base_delay", 1.0))
    status = 0
    data: dict[str, Any] | None = None
    raw_text = ""
    for attempt in range(retry_count + 1):
        timeout = int(api_cfg.get("timeout_seconds", 180))
        status, data, raw_text = await asyncio.to_thread(request_chat, payload, api_key, timeout)
        if status == 429 and rate_limit_callback is not None:
            await rate_limit_callback()
        if not should_retry_status(status) or attempt == retry_count:
            return status, data, raw_text
        delay = base_delay * (2 ** attempt)
        print(f"[RETRY] HTTP {status}, attempt={attempt + 1}/{retry_count}, sleep={delay:.1f}s", flush=True)
        await asyncio.sleep(delay)
    return status, data, raw_text


async def run_model(
    model: ModelSpec,
    samples: list[EvalSample],
    config: dict[str, Any],
    run_id: str,
    output_dir: Path,
) -> Path:
    api_key = get_closeai_api_key()
    api_cfg = config.get("api", {})
    model_api_cfg = dict(api_cfg)
    if model.timeout_seconds is not None:
        model_api_cfg["timeout_seconds"] = model.timeout_seconds
    budget_cfg = config.get("budget", {})
    result_path = output_dir / f"{model.short_name}_results.jsonl"
    done = completed_keys(result_path)
    total_cost = existing_result_cost_rmb(result_path)
    write_lock = asyncio.Lock()
    cost_lock = asyncio.Lock()
    budget_check_lock = asyncio.Lock()
    stop_model = asyncio.Event()
    concurrency_limiter = AdaptiveConcurrencyLimiter.from_api_config(api_cfg, model.short_name)
    queue: asyncio.Queue[tuple[int, EvalSample]] = asyncio.Queue()
    budget_check_interval = max(1, int(budget_cfg.get("check_interval_samples", 25)))
    rows_since_budget_check = 0

    check_budget_guard(model, len(samples), budget_cfg, output_dir, result_path)

    for idx, sample in enumerate(samples, 1):
        key = f"{model.model_id}::{sample.sample_id}"
        if key in done:
            continue
        queue.put_nowait((idx, sample))

    async def write_row(row: dict[str, Any]) -> None:
        async with write_lock:
            append_jsonl(result_path, row)

    async def worker() -> None:
        nonlocal rows_since_budget_check, total_cost
        while not stop_model.is_set():
            try:
                idx, sample = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                payload = build_payload(model, sample.prompt, model_api_cfg)

                await concurrency_limiter.acquire()
                try:
                    status, data, raw_text = await request_chat_with_retries(
                        payload,
                        api_key,
                        model_api_cfg,
                        rate_limit_callback=concurrency_limiter.on_rate_limited,
                    )
                    if status == 400 and "response_format" in raw_text:
                        payload.pop("response_format", None)
                        status, data, raw_text = await request_chat_with_retries(
                            payload,
                            api_key,
                            model_api_cfg,
                            rate_limit_callback=concurrency_limiter.on_rate_limited,
                        )
                    if status == 400 and "temperature" in raw_text and (
                        "deprecated" in raw_text or "Only the default" in raw_text or "not supported" in raw_text
                    ):
                        payload.pop("temperature", None)
                        status, data, raw_text = await request_chat_with_retries(
                            payload,
                            api_key,
                            model_api_cfg,
                            rate_limit_callback=concurrency_limiter.on_rate_limited,
                        )
                    if status == 400 and "max_tokens" in raw_text and "max_completion_tokens" in raw_text:
                        max_tokens = payload.pop("max_tokens", None)
                        if max_tokens is not None:
                            payload["max_completion_tokens"] = max_tokens
                            status, data, raw_text = await request_chat_with_retries(
                                payload,
                                api_key,
                                model_api_cfg,
                                rate_limit_callback=concurrency_limiter.on_rate_limited,
                            )
                finally:
                    await concurrency_limiter.release()

                if status == 400:
                    row = {
                        "run_id": run_id,
                        "run_type": config["run_type"],
                        "model_id": model.model_id,
                        "model_short_name": model.short_name,
                        "sample_id": sample.sample_id,
                        "household_id": sample.household_id,
                        "individual_id": sample.individual_id,
                        "success": False,
                        "parse_success": False,
                        "response": "",
                        "usage": {},
                        "error": f"HTTP 400: {raw_text[:500]}",
                        "created_at": datetime.now().isoformat(),
                    }
                    await write_row(row)
                    print(f"[INVALID] {model.short_name}: HTTP 400, stopped model", flush=True)
                    stop_model.set()
                    return

                if status != 200 or data is None:
                    row = {
                        "run_id": run_id,
                        "run_type": config["run_type"],
                        "model_id": model.model_id,
                        "model_short_name": model.short_name,
                        "sample_id": sample.sample_id,
                        "household_id": sample.household_id,
                        "individual_id": sample.individual_id,
                        "success": False,
                        "parse_success": False,
                        "response": "",
                        "usage": {},
                        "error": f"HTTP {status}: {raw_text[:500]}",
                        "created_at": datetime.now().isoformat(),
                    }
                    await write_row(row)
                    await maybe_check_budget()
                    print(f"[FAIL] {model.short_name} {idx}/{len(samples)} {sample.sample_id}: HTTP {status}", flush=True)
                    continue

                await concurrency_limiter.on_success()
                content = response_content(data)
                parsed = extract_prediction(content)
                usage = data.get("usage") or {}
                cost = estimate_cost_rmb(usage, model)
                async with cost_lock:
                    total_cost += cost
                row = {
                    "run_id": run_id,
                    "run_type": config["run_type"],
                    "model_id": model.model_id,
                    "model_short_name": model.short_name,
                    "sample_id": sample.sample_id,
                    "household_id": sample.household_id,
                    "individual_id": sample.individual_id,
                    "success": True,
                    "parse_success": parsed.parse_ok,
                    "response": content,
                    "parsed_json": parsed.raw_json,
                    "predicted_action": parsed.action,
                    "predicted_insurance_type": parsed.insurance_type,
                    "usage": usage,
                    "cost_rmb": cost,
                    "error": None,
                    "created_at": datetime.now().isoformat(),
                }
                await write_row(row)
                await maybe_check_budget()
                print(
                    f"[OK] {model.short_name} {idx}/{len(samples)} {sample.sample_id} "
                    f"parse={parsed.parse_ok} cost={cost:.6f}",
                    flush=True,
                )
                await asyncio.sleep(float(api_cfg.get("request_interval", 0.2)))
            finally:
                queue.task_done()

    async def maybe_check_budget() -> None:
        nonlocal rows_since_budget_check
        async with budget_check_lock:
            rows_since_budget_check += 1
            if rows_since_budget_check < budget_check_interval:
                return
            rows_since_budget_check = 0
            status = check_budget_guard(model, len(samples), budget_cfg, output_dir, result_path)
            warning_rmb = float(budget_cfg.get("warning_rmb", 0))
            run_cost = float(status["run_cost_rmb"] or 0.0)
            if warning_rmb and run_cost >= warning_rmb:
                print(f"[WARN] run cost reached warning_rmb: cost={run_cost:.4f}, limit={warning_rmb:.4f}", flush=True)
            projected = status["projected_model_cost_rmb"]
            projected_text = "n/a" if projected is None else f"{float(projected):.4f}"
            print(
                f"[BUDGET] {model.short_name}: run={run_cost:.4f} RMB, "
                f"model={float(status['model_cost_rmb'] or 0.0):.4f} RMB, "
                f"rows={int(status['model_rows'] or 0)}/{len(samples)}, "
                f"projected={projected_text} RMB",
                flush=True,
            )

    print(
        f"[CONCURRENCY] {model.short_name}: start limit={concurrency_limiter.current_limit}, "
        f"max_workers={concurrency_limiter.max_workers}, adaptive={concurrency_limiter.enabled}",
        flush=True,
    )
    workers = [asyncio.create_task(worker()) for _ in range(concurrency_limiter.max_workers)]
    await asyncio.gather(*workers)
    return result_path


async def main_async() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[2], type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--dataset-name")
    parser.add_argument("--output-root", default=Path("code/08-closeai-model-evaluation/outputs"), type=Path)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    config = load_config(args.config)
    samples = load_samples(args.project_root, config, args.dataset_name)
    validate_samples_for_config(samples, config, args.dataset_name)
    models = load_models(args.models)
    run_output = args.project_root / args.output_root / config["run_type"] / args.run_id
    run_output.mkdir(parents=True, exist_ok=True)
    print(f"run_id={args.run_id}")
    print(f"run_type={config['run_type']}")
    print(f"samples={len(samples)}")
    print(f"models={len(models)}")
    print(f"output={run_output}")

    try:
        for model in models:
            await run_model(model, samples, config, args.run_id, run_output)
    except BudgetStop as exc:
        print(f"[BUDGET_STOP] {exc}", flush=True)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
