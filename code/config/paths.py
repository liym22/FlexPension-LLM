# -*- coding: utf-8 -*-
"""Central path definitions for local and server-side reproduction.

Example:
    from config.paths import DATA_DIR, PROCESSED_DIR, DISTILLATION_DIR
    df = pd.read_csv(PROCESSED_DIR / "all_samples_with_policy.csv")
    prompts = json.load(open(TEACHER_INFERENCE_DIR / "prompts/all_prompts.json"))
"""

from pathlib import Path
import os

# ============================================================================
# Project roots (detected from this file)
# ============================================================================

# Current file: code/config/paths.py
# Project root: code/config/paths.py -> ../../
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CODE_DIR = PROJECT_ROOT / "code"
DATA_DIR = PROJECT_ROOT / "data"

# ============================================================================
# Local data paths
# ============================================================================

# Raw survey data
RAW_DIR = DATA_DIR / "raw"
RAW_CHFS2019_DIR = RAW_DIR / "chfs2019"
RAW_CFPS2018_DIR = RAW_DIR / "cfps2018"
RAW_CHIP2018_DIR = RAW_DIR / "chip2018"
RAW_CLDS2018_DIR = RAW_DIR / "clds2018"
RAW_CHFS2017_DIR = RAW_DIR / "chfs2017"
RAW_POLICY_DIR = RAW_DIR / "policy"

# Processed data
PROCESSED_DIR = DATA_DIR / "processed"

# Intermediate data
INTERMEDIATE_DIR = DATA_DIR / "intermediate"

# Stage-specific intermediate outputs
BASELINE_SELECTION_DIR = INTERMEDIATE_DIR / "03-baseline-selection"
PROMPT_ENGINEERING_DIR = INTERMEDIATE_DIR / "04-prompt-engineering"
VALIDATION_DIR = INTERMEDIATE_DIR / "08-validation"

# Baseline experiment data
BASELINE_DIR = DATA_DIR / "baseline_selection"
BASELINE_PROMPTS_DIR = BASELINE_DIR / "prompts"
BASELINE_API_RESULTS_DIR = BASELINE_DIR / "api_results"
BASELINE_EVALUATION_DIR = BASELINE_DIR / "evaluation"

# Prompts
PROMPTS_DIR = DATA_DIR / "prompts"
PROMPTS_CHFS2019_DIR = PROMPTS_DIR / "chfs2019"
PROMPTS_CHFS2019_BASELINE_DIR = PROMPTS_CHFS2019_DIR / "baseline"
PROMPTS_CHFS2019_DKI_DIR = PROMPTS_CHFS2019_DIR / "dki"
PROMPTS_GENERALIZATION_DIR = PROMPTS_DIR / "generalization"

# Teacher inference
TEACHER_INFERENCE_DIR = DATA_DIR / "teacher_inference"
TEACHER_INFERENCE_CHFS2019_DIR = TEACHER_INFERENCE_DIR / "chfs2019"
TEACHER_INFERENCE_GENERALIZATION_DIR = TEACHER_INFERENCE_DIR / "generalization"

TEACHER_INFERENCE_CHFS2019_CLAUDE_DIR = (
    TEACHER_INFERENCE_CHFS2019_DIR / "claude45sonnet"
)
TEACHER_INFERENCE_CHFS2019_CLAUDE_RESULTS_DIR = (
    TEACHER_INFERENCE_CHFS2019_CLAUDE_DIR / "results"
)
TEACHER_INFERENCE_CHFS2019_CLAUDE_EVALUATION_DIR = (
    TEACHER_INFERENCE_CHFS2019_CLAUDE_DIR / "evaluation"
)

# Result directories retained for script compatibility
RESULTS_DIR = DATA_DIR / "results"
BASELINE_COMPARISON_DIR = RESULTS_DIR / "baseline_comparison"
PROMPT_COMPARISON_DIR = RESULTS_DIR / "prompt_comparison"
VALIDATION_RESULTS_DIR = RESULTS_DIR / "validation_results"

# Probit regression
PROBIT_DIR = DATA_DIR / "probit"
PROBIT_V1_DIR = PROBIT_DIR / "v1_results"
PROBIT_V2_DIR = PROBIT_DIR / "v2_results"
PROBIT_STATS_DIR = PROBIT_DIR / "descriptive_stats"

# Knowledge distillation
DISTILLATION_DIR = DATA_DIR / "distillation"
DISTILLATION_DATASETS_DIR = DISTILLATION_DIR / "datasets"
DISTILLATION_DATASETS_FULL_DIR = DISTILLATION_DATASETS_DIR / "full"
DISTILLATION_DATASETS_CORRECT_ONLY_DIR = DISTILLATION_DATASETS_DIR / "correct_only"
DISTILLATION_DATASETS_DATA10_DIR = DISTILLATION_DATASETS_DIR / "data10"
DISTILLATION_DATASETS_DATA25_DIR = DISTILLATION_DATASETS_DIR / "data25"
DISTILLATION_DATASETS_DATA50_DIR = DISTILLATION_DATASETS_DIR / "data50"
DISTILLATION_DATASETS_DATA75_DIR = DISTILLATION_DATASETS_DIR / "data75"
DISTILLATION_ANALYSIS_DIR = DISTILLATION_DIR / "analysis"
DISTILLATION_RESULTS_DIR = DISTILLATION_DIR / "student_results"
DISTILLATION_CHECKPOINTS_DIR = DISTILLATION_RESULTS_DIR / "checkpoints"
DISTILLATION_TEMPERATURE_DIR = DISTILLATION_RESULTS_DIR / "temperature"
DISTILLATION_ABLATION_DIR = DISTILLATION_RESULTS_DIR / "ablation"
DISTILLATION_GENERALIZATION_DIR = DISTILLATION_RESULTS_DIR / "generalization"

# Paper outputs
OUTPUTS_DIR = DATA_DIR / "outputs"
OUTPUTS_FIGURES_DIR = OUTPUTS_DIR / "figures"
OUTPUTS_TABLES_DIR = OUTPUTS_DIR / "tables"
OUTPUTS_ANALYSIS_DIR = OUTPUTS_DIR / "analysis"

# ============================================================================
# Server paths for training and inference
# ============================================================================

# Read the server root from the environment; no private default is included.
# Configure these variables in a private .env file based on .env.example.
SERVER_BASE = os.getenv("SERVER_BASE")

if SERVER_BASE:
    # Server data paths
    SERVER_DATASET_DIR = os.getenv("SERVER_DATASET_DIR", os.path.join(SERVER_BASE, "datasets"))
    SERVER_OUTPUT_DIR = os.getenv("SERVER_OUTPUT_DIR", os.path.join(SERVER_BASE, "outputs"))
    SERVER_MODEL_DIR = os.getenv("SERVER_MODEL_DIR", os.path.join(SERVER_BASE, "models"))
    SERVER_LOGS_DIR = os.getenv("SERVER_LOGS_DIR", os.path.join(SERVER_OUTPUT_DIR, "logs"))

    # Server environment
    SERVER_CONDA_PATH = os.getenv(
        "SERVER_CONDA_PATH", os.path.join(SERVER_BASE, "miniconda3")
    )
    SERVER_CONDA_ENV = os.getenv("SERVER_CONDA_ENV", "my_qlora_env")
else:
    # Explicit placeholders when server paths are not configured
    SERVER_DATASET_DIR = None
    SERVER_OUTPUT_DIR = None
    SERVER_MODEL_DIR = None
    SERVER_LOGS_DIR = None
    SERVER_CONDA_PATH = None
    SERVER_CONDA_ENV = None

# ============================================================================
# Common file paths
# ============================================================================

# Raw data files
CHFS2019_IND_FILE = RAW_CHFS2019_DIR / "chfs2019_ind_202112.dta"
CHFS2019_HH_FILE = RAW_CHFS2019_DIR / "chfs2019_hh_202112.dta"
CHFS2019_MASTER_FILE = RAW_CHFS2019_DIR / "chfs2019_master_202112.dta"

# Processed data files
REGRESSION_DATA_FILE = PROCESSED_DIR / "regression_data.csv"
SAMPLE_30_FILE = PROCESSED_DIR / "sample_30.csv"
SAMPLED_IDS_FILE = PROCESSED_DIR / "sampled_ids.csv"
SAMPLED_IDS_FILTERED_FILE = PROCESSED_DIR / "sampled_ids_filtered.csv"
ALL_SAMPLES_WITH_POLICY_FILE = PROCESSED_DIR / "all_samples_with_policy.csv"

# Main CHFS 2019 prompt and teacher-inference files
CHFS2019_DKI_PROMPTS_FILE = PROMPTS_CHFS2019_DKI_DIR / "all_prompts_newera.json"
CHFS2019_DKI_GROUND_TRUTH_FILE = (
    PROMPTS_CHFS2019_DKI_DIR / "ground_truth_newera.json"
)
CHFS2019_TEACHER_RESULTS_FILE = (
    TEACHER_INFERENCE_CHFS2019_CLAUDE_RESULTS_DIR
    / "claude45sonnet_seed42_temp05_results.json"
)

# Distillation split, regeneration, and diagnostic files
DISTILLATION_TRAIN_IDS_FILE = DISTILLATION_ANALYSIS_DIR / "train_ids.json"
DISTILLATION_VAL_IDS_FILE = DISTILLATION_ANALYSIS_DIR / "val_ids.json"
DISTILLATION_TEST_IDS_FILE = DISTILLATION_ANALYSIS_DIR / "test_ids.json"
DISTILLATION_REGEN_INPUT_FILE = (
    DISTILLATION_ANALYSIS_DIR / "trainval_need_regen.json"
)
DISTILLATION_REGEN_RESULTS_FILE = (
    DISTILLATION_ANALYSIS_DIR / "renew_claude_errors_results.json"
)

# Probit model files
PROBIT_V1_MODEL1_FILE = PROBIT_V1_DIR / "model1.pkl"
PROBIT_V1_MODEL2_FILE = PROBIT_V1_DIR / "model2.pkl"
PROBIT_V2_MODEL1_FILE = PROBIT_V2_DIR / "model1.pkl"
PROBIT_V2_MODEL2_FILE = PROBIT_V2_DIR / "model2.pkl"

# Training dataset files
TRAIN_FULL_FILE = DISTILLATION_DATASETS_FULL_DIR / "train.jsonl"
VAL_FULL_FILE = DISTILLATION_DATASETS_FULL_DIR / "val.jsonl"
TEST_FULL_FILE = DISTILLATION_DATASETS_FULL_DIR / "test.jsonl"

# Burden-analysis file
BURDEN_DATA_FILE = PROMPTS_CHFS2019_DKI_DIR / "burden_data.csv"

# ============================================================================
# Path validation utilities
# ============================================================================

def validate_data_paths(verbose=True):
    """Return whether all critical project directories exist."""
    critical_paths = [
        ("project root", PROJECT_ROOT),
        ("code directory", CODE_DIR),
        ("data directory", DATA_DIR),
        ("raw data", RAW_DIR),
        ("processed data", PROCESSED_DIR),
        ("Probit regression", PROBIT_DIR),
        ("knowledge distillation", DISTILLATION_DIR),
        ("paper outputs", OUTPUTS_DIR),
    ]

    missing = []
    for name, path in critical_paths:
        if not path.exists():
            missing.append((name, str(path)))
            if verbose:
                print(f"[Missing] {name}: {path}")
        else:
            if verbose:
                print(f"✓ {name}: {path}")

    if missing:
        if verbose:
            print(f"\nFound {len(missing)} missing paths")
        return False

    if verbose:
        print("\nAll critical paths validated")
    return True


def validate_file_exists(file_path, file_description="file"):
    """Return whether a file exists, optionally printing a descriptive error."""
    path = Path(file_path)
    if not path.exists():
        print(f"[Missing] {file_description}: {path}")
        return False
    return True


def get_relative_path(absolute_path):
    """Return a path relative to the project root when possible."""
    path = Path(absolute_path)
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        # Preserve paths outside the project root.
        return path


def ensure_dir_exists(dir_path):
    """Create a directory if needed and return its path."""
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_path(*parts):
    """Return a path under the data directory for legacy callers.

    Example:
        >>> get_data_path("intermediate", "05-teacher-inference", "prompts", "all_prompts.json")
        PosixPath('/path/to/data/intermediate/05-teacher-inference/prompts/all_prompts.json')
    """
    return DATA_DIR.joinpath(*parts)


def get_code_path(*parts):
    """Return a path under the code directory for legacy callers.

    Example:
        >>> get_code_path("03-baseline-model-selection", "test.py")
        PosixPath('/path/to/code/03-baseline-model-selection/test.py')
    """
    return CODE_DIR.joinpath(*parts)


# ============================================================================
# Command-line validation
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Path configuration validation")
    print("=" * 80)
    print()

    print("Project paths:")
    print(f"  PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"  CODE_DIR: {CODE_DIR}")
    print(f"  DATA_DIR: {DATA_DIR}")
    print()

    print("Server paths:")
    print(f"  SERVER_BASE: {SERVER_BASE}")
    print(f"  SERVER_DATASET_DIR: {SERVER_DATASET_DIR}")
    print(f"  SERVER_OUTPUT_DIR: {SERVER_OUTPUT_DIR}")
    print()

    print("Critical-path validation:")
    print("-" * 80)
    validate_data_paths(verbose=True)
