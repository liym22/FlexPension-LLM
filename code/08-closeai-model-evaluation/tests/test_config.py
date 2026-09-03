import asyncio

import pytest

import closeai_config
from datasets import EvalSample
from closeai_config import (
    CLOSEAI_CHAT_URL,
    CLOSEAI_MODELS_URL,
    get_closeai_admin_key,
    get_closeai_api_key,
)
from runner import (
    AdaptiveConcurrencyLimiter,
    BudgetStop,
    ModelSpec,
    build_payload,
    check_budget_guard,
    completed_keys,
    estimate_cost_rmb,
    existing_result_cost_rmb,
    request_chat,
    run_cost_rmb,
    run_model,
    should_retry_status,
    validate_samples_for_config,
)


def test_closeai_urls_are_expected():
    assert CLOSEAI_CHAT_URL == "https://api.openai-proxy.org/v1/chat/completions"
    assert CLOSEAI_MODELS_URL == "https://api.openai-proxy.org/api/v1/management/models"


def test_closeai_urls_can_be_overridden(monkeypatch):
    monkeypatch.setenv("CLOSEAI_CHAT_URL", "https://example.test/v1/chat/completions")
    monkeypatch.setenv("CLOSEAI_MODELS_URL", "https://example.test/api/models")

    assert closeai_config.get_closeai_chat_url() == "https://example.test/v1/chat/completions"
    assert closeai_config.get_closeai_models_url() == "https://example.test/api/models"


def test_api_key_loaded_from_environment(monkeypatch):
    monkeypatch.setenv("CLOSEAI_API_KEY", "test-api-key")
    assert get_closeai_api_key() == "test-api-key"


def test_admin_key_loaded_from_environment(monkeypatch):
    monkeypatch.setenv("CLOSEAI_ADMIN_KEY", "test-admin-key")
    assert get_closeai_admin_key() == "test-admin-key"


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("CLOSEAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="CLOSEAI_API_KEY"):
        get_closeai_api_key()


def test_estimate_cost_from_closeai_model_prices():
    model = ModelSpec(
        model_id="claude-sonnet-5",
        short_name="claude_sonnet_5",
        input_price_rmb_per_1m=31.5,
        output_price_rmb_per_1m=157.5,
    )
    usage = {"prompt_tokens": 2000, "completion_tokens": 400}
    assert estimate_cost_rmb(usage, model) == pytest.approx(0.126)


def test_build_payload_can_omit_temperature_for_unsupported_models():
    model = ModelSpec(
        model_id="claude-sonnet-5",
        short_name="claude_sonnet_5",
        omit_temperature=True,
    )
    payload = build_payload(model, "prompt text", {"temperature": 0.5, "max_tokens": 2048})
    assert payload["model"] == "claude-sonnet-5"
    assert "temperature" not in payload
    assert payload["max_tokens"] == 2048


def test_build_payload_can_use_max_completion_tokens_for_openai_models():
    model = ModelSpec(
        model_id="gpt-5.5",
        short_name="gpt_5_5",
        max_tokens_param="max_completion_tokens",
    )
    payload = build_payload(model, "prompt text", {"temperature": 0.5, "max_tokens": 2048})
    assert "max_tokens" not in payload
    assert payload["max_completion_tokens"] == 2048


def test_build_payload_can_override_max_tokens_per_model():
    model = ModelSpec(
        model_id="gemini-3.1-pro-preview",
        short_name="gemini_3_1_pro_preview",
        max_tokens=8192,
    )
    payload = build_payload(model, "prompt text", {"temperature": 0.5, "max_tokens": 2048})
    assert payload["max_tokens"] == 8192


def test_load_models_can_set_timeout_seconds(tmp_path):
    models_path = tmp_path / "models.yaml"
    models_path.write_text(
        "models:\n"
        "  - model_id: gpt-5.4-pro\n"
        "    short_name: gpt_5_4_pro\n"
        "    timeout_seconds: 600\n",
        encoding="utf-8",
    )
    from runner import load_models

    models = load_models(models_path)

    assert models[0].timeout_seconds == 600


def test_retry_policy_retries_transient_failures_only():
    assert should_retry_status(0)
    assert should_retry_status(429)
    assert should_retry_status(500)
    assert should_retry_status(503)
    assert not should_retry_status(400)
    assert not should_retry_status(401)


def test_adaptive_concurrency_increases_after_stable_successes():
    limiter = AdaptiveConcurrencyLimiter(
        enabled=True,
        min_concurrency=2,
        max_concurrency=4,
        increase_every_successes=2,
        cooldown_seconds=60,
        name="test_model",
    )

    async def scenario():
        assert limiter.current_limit == 2
        await limiter.on_success()
        assert limiter.current_limit == 2
        await limiter.on_success()
        assert limiter.current_limit == 3
        await limiter.on_success()
        await limiter.on_success()
        assert limiter.current_limit == 4
        await limiter.on_success()
        await limiter.on_success()
        assert limiter.current_limit == 4

    asyncio.run(scenario())


def test_adaptive_concurrency_rate_limit_resets_and_cools_down():
    now = [100.0]
    limiter = AdaptiveConcurrencyLimiter(
        enabled=True,
        min_concurrency=2,
        max_concurrency=5,
        increase_every_successes=1,
        cooldown_seconds=30,
        name="test_model",
        now_func=lambda: now[0],
    )

    async def scenario():
        await limiter.on_success()
        await limiter.on_success()
        assert limiter.current_limit == 4

        await limiter.on_rate_limited()
        assert limiter.current_limit == 2
        assert limiter.in_cooldown

        await limiter.on_success()
        assert limiter.current_limit == 2

        now[0] = 131.0
        assert not limiter.in_cooldown
        await limiter.on_success()
        assert limiter.current_limit == 3

    asyncio.run(scenario())


def test_adaptive_concurrency_from_api_config_disabled_uses_fixed_limit():
    limiter = AdaptiveConcurrencyLimiter.from_api_config(
        {"concurrency_per_model": 5},
        name="test_model",
    )

    assert not limiter.enabled
    assert limiter.current_limit == 5
    assert limiter.max_workers == 5


def test_request_chat_converts_unexpected_transport_exception_to_status_zero(monkeypatch):
    def fake_urlopen(request, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr("runner.urlopen", fake_urlopen)
    status, data, raw_text = request_chat({"model": "m", "messages": []}, "test-api-key")

    assert status == 0
    assert data is None
    assert "timed out" in raw_text


def test_existing_result_cost_sums_previous_rows(tmp_path):
    result_path = tmp_path / "results.jsonl"
    result_path.write_text(
        '{"success": true, "cost_rmb": 1.25}\n'
        '{"success": false, "cost_rmb": 0.50}\n'
        '{"success": true, "usage": {"cost": 2.0}}\n',
        encoding="utf-8",
    )
    assert existing_result_cost_rmb(result_path) == pytest.approx(3.75)


def test_run_cost_sums_all_result_files(tmp_path):
    (tmp_path / "model_a_results.jsonl").write_text(
        '{"success": true, "cost_rmb": 1.25}\n',
        encoding="utf-8",
    )
    (tmp_path / "model_b_results.jsonl").write_text(
        '{"success": true, "usage": {"cost": 2.0}}\n',
        encoding="utf-8",
    )
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")

    assert run_cost_rmb(tmp_path) == pytest.approx(3.25)


def test_budget_guard_raises_on_projected_model_cost(tmp_path):
    result_path = tmp_path / "model_a_results.jsonl"
    result_path.write_text(
        "".join('{"success": true, "cost_rmb": 1.0}\n' for _ in range(50)),
        encoding="utf-8",
    )

    with pytest.raises(BudgetStop, match="projected"):
        check_budget_guard(
            ModelSpec(
                model_id="model-a",
                short_name="model_a",
                projected_stop_rmb=100,
            ),
            2350,
            {"min_projection_samples": 50, "hard_stop_rmb": 10000},
            tmp_path,
            result_path,
        )


def test_completed_keys_only_includes_successful_rows(tmp_path):
    result_path = tmp_path / "results.jsonl"
    result_path.write_text(
        '{"success": true, "parse_success": true, "model_id": "m1", "sample_id": "s1"}\n'
        '{"success": false, "model_id": "m1", "sample_id": "s2"}\n',
        encoding="utf-8",
    )
    assert completed_keys(result_path) == {"m1::s1"}


def test_completed_keys_excludes_parse_failures(tmp_path):
    result_path = tmp_path / "results.jsonl"
    result_path.write_text(
        '{"success": true, "parse_success": false, "model_id": "m1", "sample_id": "s1"}\n',
        encoding="utf-8",
    )
    assert completed_keys(result_path) == set()


def test_validate_samples_for_config_rejects_expected_size_mismatch():
    samples = [
        EvalSample(
            sample_id="s1",
            household_id="h1",
            individual_id="p1",
            prompt="prompt",
            ground_truth_action="参保",
            ground_truth_type="城乡居民养老保险",
        )
    ]
    config = {
        "run_type": "blind",
        "dataset": {
            "kind": "distill_jsonl",
            "path": "data/distillation/datasets/full/test.jsonl",
            "expected_size": 2350,
        },
    }
    with pytest.raises(ValueError, match="expected_size=2350"):
        validate_samples_for_config(samples, config)


def test_validate_external_samples_uses_selected_dataset_expected_size():
    samples = [
        EvalSample(
            sample_id="s1",
            household_id="h1",
            individual_id="p1",
            prompt="prompt",
            ground_truth_action="参保",
            ground_truth_type="城乡居民养老保险",
        )
    ]
    config = {
        "run_type": "external_new",
        "datasets": {
            "chfs2017_new": {
                "prompts": "prompts.json",
                "ground_truth": "ground_truth.json",
                "expected_size": 500,
            }
        },
    }

    with pytest.raises(ValueError, match="expected_size=500"):
        validate_samples_for_config(samples, config, dataset_name="chfs2017_new")


def test_validate_samples_for_config_rejects_blind_sample_size():
    samples = [
        EvalSample(
            sample_id="s1",
            household_id="h1",
            individual_id="p1",
            prompt="prompt",
            ground_truth_action="参保",
            ground_truth_type="城乡居民养老保险",
        )
    ]
    config = {
        "run_type": "blind",
        "dataset": {
            "kind": "distill_jsonl",
            "path": "data/distillation/datasets/full/test.jsonl",
            "sample_size": 500,
        },
    }
    with pytest.raises(ValueError, match="sample_size"):
        validate_samples_for_config(samples, config)


def test_run_model_continues_after_transient_failure(monkeypatch, tmp_path):
    samples = [
        EvalSample(
            sample_id="s1",
            household_id="h1",
            individual_id="p1",
            prompt="prompt 1",
            ground_truth_action="参保",
            ground_truth_type="城乡居民养老保险",
        ),
        EvalSample(
            sample_id="s2",
            household_id="h2",
            individual_id="p2",
            prompt="prompt 2",
            ground_truth_action="不参保",
            ground_truth_type="不参保",
        ),
    ]
    responses = iter(
        [
            (503, None, "temporary upstream failure"),
            (
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"insurance_decision": {"action": "不参保", '
                                    '"insurance_type": "不参保"}}'
                                )
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
                "",
            ),
        ]
    )

    async def fake_request_chat_with_retries(payload, api_key, api_cfg, rate_limit_callback=None):
        return next(responses)

    monkeypatch.setattr("runner.get_closeai_api_key", lambda: "test-api-key")
    monkeypatch.setattr("runner.request_chat_with_retries", fake_request_chat_with_retries)

    result_path = asyncio.run(
        run_model(
            ModelSpec(model_id="model-a", short_name="model_a"),
            samples,
            {"run_type": "blind", "api": {"concurrency_per_model": 1}, "budget": {"hard_stop_rmb": 10000}},
            "run1",
            tmp_path,
        )
    )

    rows = [line for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 2
    assert '"success": false' in rows[0]
    assert '"sample_id": "s1"' in rows[0]
    assert '"success": true' in rows[1]
    assert '"sample_id": "s2"' in rows[1]


def test_run_model_stops_on_projected_model_budget(monkeypatch, tmp_path):
    samples = [
        EvalSample(
            sample_id=f"s{i}",
            household_id=f"h{i}",
            individual_id=f"p{i}",
            prompt=f"prompt {i}",
            ground_truth_action="参保",
            ground_truth_type="城乡居民养老保险",
        )
        for i in range(4)
    ]

    async def fake_request_chat_with_retries(payload, api_key, api_cfg, rate_limit_callback=None):
        return (
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"insurance_decision": {"action": "参保", '
                                '"insurance_type": "城乡居民养老保险"}}'
                            )
                        }
                    }
                ],
                "usage": {"cost": 10.0},
            },
            "",
        )

    monkeypatch.setattr("runner.get_closeai_api_key", lambda: "test-api-key")
    monkeypatch.setattr("runner.request_chat_with_retries", fake_request_chat_with_retries)

    with pytest.raises(BudgetStop, match="projected"):
        asyncio.run(
            run_model(
                ModelSpec(model_id="model-a", short_name="model_a", projected_stop_rmb=30),
                samples,
                {
                    "run_type": "blind",
                    "api": {"concurrency_per_model": 1},
                    "budget": {
                        "hard_stop_rmb": 10000,
                        "check_interval_samples": 2,
                        "min_projection_samples": 2,
                    },
                },
                "run1",
                tmp_path,
            )
        )

    rows = [line for line in (tmp_path / "model_a_results.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 2
