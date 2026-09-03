# -*- coding: utf-8 -*-
"""Shared project configuration and path definitions."""

from .paths import (
    # Project roots
    PROJECT_ROOT,
    CODE_DIR,
    DATA_DIR,

    # Raw data
    RAW_DIR,
    RAW_CHFS2019_DIR,
    RAW_CFPS2018_DIR,
    RAW_CHIP2018_DIR,
    RAW_CLDS2018_DIR,
    RAW_CHFS2017_DIR,

    # Processed data
    PROCESSED_DIR,

    # Probit regression
    PROBIT_DIR,
    PROBIT_V1_DIR,
    PROBIT_V2_DIR,
    PROBIT_STATS_DIR,

    # Knowledge distillation
    DISTILLATION_DIR,
    DISTILLATION_DATASETS_DIR,
    DISTILLATION_ANALYSIS_DIR,
    DISTILLATION_RESULTS_DIR,
    DISTILLATION_CHECKPOINTS_DIR,
    DISTILLATION_TEMPERATURE_DIR,
    DISTILLATION_ABLATION_DIR,
    DISTILLATION_GENERALIZATION_DIR,

    # Paper outputs
    OUTPUTS_DIR,
    OUTPUTS_FIGURES_DIR,
    OUTPUTS_TABLES_DIR,
    OUTPUTS_ANALYSIS_DIR,

    # Prompts
    PROMPTS_DIR,
    PROMPTS_CHFS2019_DKI_DIR,

    # Server paths
    SERVER_BASE,
    SERVER_DATASET_DIR,
    SERVER_OUTPUT_DIR,
    SERVER_MODEL_DIR,

    # Common files
    CHFS2019_IND_FILE,
    CHFS2019_HH_FILE,
    CHFS2019_MASTER_FILE,
    REGRESSION_DATA_FILE,
    SAMPLE_30_FILE,
    BURDEN_DATA_FILE,

    # Utilities
    validate_data_paths,
    validate_file_exists,
    ensure_dir_exists,
)

__all__ = [
    'PROJECT_ROOT',
    'CODE_DIR',
    'DATA_DIR',
    'RAW_DIR',
    'PROCESSED_DIR',
    'PROBIT_DIR',
    'DISTILLATION_DIR',
    'OUTPUTS_DIR',
    'PROMPTS_DIR',
    'SERVER_BASE',
    'validate_data_paths',
]
