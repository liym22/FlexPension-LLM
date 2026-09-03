from pathlib import Path

from datasets import load_distill_jsonl, load_external_json

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_distill_jsonl_uses_only_user_prompt_for_request():
    rows = load_distill_jsonl(FIXTURES / "mini_distill.jsonl")
    assert len(rows) == 2
    assert rows[0].sample_id == "1-1"
    assert rows[0].prompt == "家庭ID: 1\n个人ID: 1\nprompt A"
    assert rows[0].ground_truth_action == "参保"
    assert rows[0].ground_truth_type == "城乡居民养老保险"
    assert "assistant" not in rows[0].prompt


def test_load_external_json_maps_prompt_and_ground_truth():
    rows = load_external_json(
        FIXTURES / "mini_external_prompts.json",
        FIXTURES / "mini_external_ground_truth.json",
    )
    assert len(rows) == 1
    assert rows[0].sample_id == "e-1"
    assert rows[0].prompt.endswith("external prompt")
    assert rows[0].ground_truth_action == "参保"
    assert rows[0].ground_truth_type == "城镇职工养老保险"

