from __future__ import annotations

import importlib.util
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
PATHS_FILE = ROOT / "code" / "config" / "paths.py"


def load_paths_module():
    spec = importlib.util.spec_from_file_location("release_paths", PATHS_FILE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_workflow_paths_share_one_release_layout() -> None:
    paths = load_paths_module()

    assert paths.CHFS2019_DKI_PROMPTS_FILE == (
        ROOT / "data/prompts/chfs2019/dki/all_prompts_newera.json"
    )
    assert paths.CHFS2019_DKI_GROUND_TRUTH_FILE == (
        ROOT / "data/prompts/chfs2019/dki/ground_truth_newera.json"
    )
    assert paths.CHFS2019_TEACHER_RESULTS_FILE == (
        ROOT
        / "data/teacher_inference/chfs2019/claude45sonnet/results/"
        "claude45sonnet_seed42_temp05_results.json"
    )
    assert paths.DISTILLATION_REGEN_INPUT_FILE == (
        ROOT / "data/distillation/analysis/trainval_need_regen.json"
    )
    assert paths.DISTILLATION_REGEN_RESULTS_FILE == (
        ROOT / "data/distillation/analysis/renew_claude_errors_results.json"
    )
    assert paths.BURDEN_DATA_FILE == (
        ROOT / "data/prompts/chfs2019/dki/burden_data.csv"
    )


def test_dki_scripts_share_the_configured_burden_file() -> None:
    scripts = [
        ROOT / "code/04-prompt-engineering/calculate/burden_analysis.py",
        ROOT / "code/04-prompt-engineering/generate_prompts_newera.py",
    ]

    for script in scripts:
        source = script.read_text(encoding="utf-8")
        assert "BURDEN_DATA_FILE" in source


def test_main_workflow_scripts_do_not_use_legacy_private_layouts() -> None:
    scripts = [
        ROOT
        / "code/05-teacher-model-inference/test_all_samples/claude45sonnet/test.py",
        ROOT
        / "code/05-teacher-model-inference/test_all_samples/claude45sonnet/eva.py",
        *sorted((ROOT / "code/06-results-consolidation/dataset").glob("*.py")),
    ]
    forbidden = (
        '"test_all_samples"',
        '"train" / "dataset"',
        "prompts_newera",
        "claude45sonnet_v2",
    )

    for script in scripts:
        source = script.read_text(encoding="utf-8")
        for fragment in forbidden:
            assert fragment not in source, f"{script}: legacy path fragment {fragment}"


def test_external_student_wrappers_use_only_four_final_datasets() -> None:
    wrappers = [
        ROOT / "code/07-knowledge-distillation/evaluation/run_external_eval.sh",
        ROOT
        / "code/07-knowledge-distillation/evaluation/"
        "run_external_eval_correct_only.sh",
    ]
    expected = {
        "test_2017_new",
        "test_cfps_new",
        "test_chip_new",
        "test_clds_new",
    }

    for wrapper in wrappers:
        source = wrapper.read_text(encoding="utf-8")
        names = {
            line.strip()
            for line in source.splitlines()
            if line.strip().startswith("test_")
        }
        assert names == expected
        assert "fanhua" not in source.lower()
        assert "_old" not in source


def test_screening_commands_use_generator_output_paths() -> None:
    reproduction = (ROOT / "REPRODUCTION.md").read_text(encoding="utf-8")

    assert "data/prompts/chfs2019/baseline_v3/all_prompts_v3.json" in reproduction
    assert "data/prompts/chfs2019/baseline_v3/ground_truth_v3.json" in reproduction
    assert "data/prompts/screening/" not in reproduction


def test_external_prompt_builders_write_to_runner_paths() -> None:
    expected = {
        "chfs2017/task4_prompts/04_build_prompts_v2.py": (
            "chfs2017",
            "prompts_2017_v2.json",
            "ground_truth_2017_v2.json",
        ),
        "cfps/task4_prompts/04_build_prompts_new.py": (
            "cfps2018",
            "prompts_cfps_new.json",
            "ground_truth_cfps_new.json",
        ),
        "chip/task4_prompts/05_build_prompts_new.py": (
            "chip2018",
            "prompts_chip_new.json",
            "ground_truth_chip_new.json",
        ),
        "clds/task4_prompts/04_build_prompts_new.py": (
            "clds2018",
            "prompts_clds_new.json",
            "ground_truth_clds_new.json",
        ),
    }
    base = ROOT / "code/08-validation-experiments/cross_dataset"

    for relative_path, (dataset_dir, prompts_name, truth_name) in expected.items():
        source = (base / relative_path).read_text(encoding="utf-8")
        compact = re.sub(r"\s+", " ", source)
        assert (
            f'OUTPUT_DIR = PROJECT_ROOT / "data" / "prompts" / "generalization" / "{dataset_dir}"'
            in compact
        )
        assert f'OUT_PROMPTS = OUTPUT_DIR / "{prompts_name}"' in compact
        assert f'OUT_GT = OUTPUT_DIR / "{truth_name}"' in compact


def test_full_data_external_wrapper_searches_nested_checkpoints() -> None:
    source = (
        ROOT / "code/07-knowledge-distillation/evaluation/run_external_eval.sh"
    ).read_text(encoding="utf-8")

    assert "-maxdepth 1" not in source
    assert 'find "${ADAPTERS_ROOT}" -type d -name \'checkpoint-*\'' in source
