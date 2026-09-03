# Reproduction Guide

## 1. Environment

The final LoRA/SFT run used ms-swift 4.0.2 and PEFT 0.18.1. The Qwen 3.5 model loader requires Transformers 5.2 or later. Install the general analysis dependencies from `requirements.txt`, then configure the CUDA, PyTorch, and DeepSpeed versions appropriate for the available GPU system.

Copy `.env.example` to `.env` only in a private working directory. Never place real credentials in the release archive. Teacher inference reads `OPENROUTER_API_KEY`; the unified benchmark runner reads `CLOSEAI_API_KEY` and, when needed, `CLOSEAI_ADMIN_KEY`.

## 2. Controlled-Access Data Layout

Place approved survey files under `data/raw`:

```text
data/raw/
  chfs2019/
    chfs2019_ind_202112.dta
    chfs2019_hh_202112.dta
    chfs2019_master_202112.dta
  chfs2017/
    chfs2017_ind_202206.dta
    chfs2017_hh_202206.dta
    chfs2017_master_202206.dta
  chfs2015/
    chfs2015_hh_20191120_version14.dta
  cfps2018/
    cfps2018person_202512.dta
    cfps2018famecon_202512.dta
    cfps2018famconf_202512.dta
  chip2018/
    chip2018_urban_person.dta
    chip2018_rural_person.dta
    chip2018_urban_household.dta
    chip2018_rural_household.dta
    chip2018_income_consumption.dta
  clds2018/
    18个人.dta
    18家庭.dta
  policy/
    province_policy_2016.xlsx
    province_policy_2018.xlsx
```

The original provider releases may use nested directories. The public scripts use the normalized layout above so that no machine-specific path is required.

## 3. Main Workflow

Run commands from the package root unless noted otherwise.

1. Construct the full CHFS 2019 analysis cohort:

   ```bash
   python code/01-data-preparation/chfs2019/sampling_analysis.py
   python code/01-data-preparation/chfs2019/filter_by_income_core_vars.py
   ```

   The reported run produced 16,822 initial eligible respondents and 15,672 respondents after the final completeness filter. The output IDs are `data/processed/sampled_ids.csv` and `data/processed/sampled_ids_filtered.csv`.

2. Separately construct the 30-case model-screening subset, which is not the main blind split:

   ```bash
   python code/01-data-preparation/chfs2019/stratified_sampling.py
   python code/01-data-preparation/chfs2019/reconstruct_variables.py
   python code/03-baseline-model-selection/prompt_v3/generate_prompts_v3.py
   ```

   The Probit scripts exclude these 30 screening cases. The 85/15 main split is created later by `build_mixed_dataset.py`.

   Run the historical three-seed screening and aggregate A-Comp./V-Comp. with:

   ```bash
   python code/03-baseline-model-selection/screening/run_screening.py \
     --prompts data/prompts/chfs2019/baseline_v3/all_prompts_v3.json \
     --config code/03-baseline-model-selection/screening/screening_models.yaml \
     --output-dir outputs/model_screening/raw
   python code/03-baseline-model-selection/screening/evaluate_screening.py \
     --ground-truth data/prompts/chfs2019/baseline_v3/ground_truth_v3.json \
     --results-dir outputs/model_screening/raw \
     --model claude45sonnet --model gemini3flash --model minimaxm21 \
     --model gpt5mini --model llama3370b --model deepseekv32 --model qwen3max \
     --output outputs/model_screening/screening_summary.csv
   ```

   The prompt and ground-truth files are sample-level controlled-data derivatives and are not redistributed. The included aggregate summary preserves the exact historical model IDs.

3. Reconstruct all 15,672 records and fit the two Probit specifications:

   ```bash
   python code/02-probit-regression/readable_variables.py
   python code/02-probit-regression/transform_variables.py
   python code/02-probit-regression/v1/run_regression.py
   python code/02-probit-regression/v2/run_regression.py
   ```

   Run the marginal-effect and test scripts in the same `v1` and `v2` directories as needed. The full readable table is `data/processed/all_samples_with_policy.csv`.

4. Generate the policy-linked burden indicators and DKI prompt set:

   ```bash
   python code/04-prompt-engineering/calculate/burden_analysis.py
   python code/04-prompt-engineering/generate_prompts_newera.py
   ```

   These scripts write `burden_data.csv`, `all_prompts_newera.json`, and `ground_truth_newera.json` under `data/prompts/chfs2019/dki`.

5. Run Claude Sonnet 4.5 teacher inference and evaluate its output:

   ```bash
   python code/05-teacher-model-inference/test_all_samples/claude45sonnet/test.py
   python code/05-teacher-model-inference/test_all_samples/claude45sonnet/eva.py
   ```

   Teacher predictions are written under `data/teacher_inference/chfs2019/claude45sonnet`.

6. Create the main split, regenerate label-constrained rationales for teacher-error cases, and build the full and correct-only SFT datasets:

   ```bash
   python code/06-results-consolidation/dataset/build_mixed_dataset.py
   python code/06-results-consolidation/dataset/renew_claude_errors.py
   python code/06-results-consolidation/dataset/check_trainval_demo.py
   python code/06-results-consolidation/dataset/build_alpaca_datasets.py
   python code/06-results-consolidation/dataset/build_correct_only_split.py
   python code/06-results-consolidation/dataset/build_alpaca_datasets_correct_only.py
   ```

   The reported split contains 10,657 training, 2,665 validation, and 2,350 blind-test cases with seed 42. Split IDs and diagnostics are written to `data/distillation/analysis`; final JSONL datasets are written to `data/distillation/datasets/full` and `data/distillation/datasets/correct_only`.

7. Configure `.env`, then run `code/07-knowledge-distillation/training/train_100.sh` for the reported full-data model. The 10, 25, 50, and 75 percent scripts support the training-data robustness analysis. Run the corresponding inference and evaluation scripts under `code/07-knowledge-distillation`.

`code/07-knowledge-distillation/training/final_training_config.json` records the sanitized configuration recovered from the final full-data checkpoint. It is the source of truth for training hyperparameters.

## 4. External Evaluation

For each external dataset, run task 1 through task 4 under `code/08-validation-experiments/cross_dataset/<dataset>` in task-number order. These scripts produce the normalized 500-case stratified sample, reconstructed variables, DKI prompts, and ground truth used by the evaluation runners. Task 4 writes the paired JSON files under `data/prompts/generalization/<dataset>`, matching `external_core.yaml`. Intermediate CSV, prompt, and ground-truth files are sample-level artifacts and must remain in the researcher's private working directory.

The package excludes the earlier dataset-specific API scripts. `code/08-closeai-model-evaluation/runner.py` is the unified runner used for the current benchmark. The locked model lists are in its `configs` directory.

Run the full blind benchmark with:

```bash
python code/08-closeai-model-evaluation/runner.py \
  --config code/08-closeai-model-evaluation/configs/blind.yaml \
  --models code/08-closeai-model-evaluation/configs/locked_blind_full_models.yaml
```

For each external dataset name in `external_core.yaml`, run:

```bash
python code/08-closeai-model-evaluation/runner.py \
  --config code/08-closeai-model-evaluation/configs/external_core.yaml \
  --models code/08-closeai-model-evaluation/configs/locked_external_core_models.yaml \
  --dataset-name chfs2017_new
```

For the open student model, convert each paired prompt/label file to the private ms-swift JSONL input expected by the final wrappers:

```bash
python code/08-validation-experiments/build_external_student_dataset.py \
  --prompts data/prompts/generalization/chfs2017/prompts_2017_v2.json \
  --ground-truth data/prompts/generalization/chfs2017/ground_truth_2017_v2.json \
  --output /private/path/external_datasets/test_2017_new.jsonl
python code/08-validation-experiments/build_external_student_dataset.py \
  --prompts data/prompts/generalization/cfps2018/prompts_cfps_new.json \
  --ground-truth data/prompts/generalization/cfps2018/ground_truth_cfps_new.json \
  --output /private/path/external_datasets/test_cfps_new.jsonl
python code/08-validation-experiments/build_external_student_dataset.py \
  --prompts data/prompts/generalization/chip2018/prompts_chip_new.json \
  --ground-truth data/prompts/generalization/chip2018/ground_truth_chip_new.json \
  --output /private/path/external_datasets/test_chip_new.jsonl
python code/08-validation-experiments/build_external_student_dataset.py \
  --prompts data/prompts/generalization/clds2018/prompts_clds_new.json \
  --ground-truth data/prompts/generalization/clds2018/ground_truth_clds_new.json \
  --output /private/path/external_datasets/test_clds_new.jsonl
```

The final external wrappers run only these four reported `*_new` datasets:

```bash
bash code/07-knowledge-distillation/evaluation/run_external_eval.sh
bash code/07-knowledge-distillation/evaluation/run_external_eval_correct_only.sh
```

Set `SERVER_EXTERNAL_DATASET_DIR` to the directory containing the four private JSONL datasets before running these commands.

## 5. Bootstrap Analysis

The scripts under `code/09-bootstrap-evaluation` implement the binary Type-F1 protocol, Composite F1, paired stratified bootstrap intervals with 10,000 resamples, and seed 42.

Sample-level predictions are intentionally excluded. To rerun the four-dataset average, provide a local JSON manifest to `bootstrap_external_average.py`:

```json
{
  "cfps2018_new": {"prompts": "/local/path/prompts.json", "ground_truth": "/local/path/ground_truth.json"},
  "chfs2017_new": {"prompts": "/local/path/prompts.json", "ground_truth": "/local/path/ground_truth.json"},
  "chip2018_new": {"prompts": "/local/path/prompts.json", "ground_truth": "/local/path/ground_truth.json"},
  "clds2018_new": {"prompts": "/local/path/prompts.json", "ground_truth": "/local/path/ground_truth.json"}
}
```

Use `--dataset-manifest`, `--bootstrap-root`, `--output-dir`, and `--type-f1-mode binary`. Aggregate outputs included under `outputs` support direct inspection of the reported intervals without exposing respondent-level records.

## 6. Expert Evaluation

The package contains anonymized numeric ratings for 27 experts, 12 cases, and three systems. It excludes names, IP addresses, timestamps, background fields, and free-text source material. Reproduce the reported summary with:

```bash
python code/10-expert-evaluation/analyze_expert_evaluation.py \
  --input data/expert_evaluation/anonymized_responses.csv \
  --output-json outputs/expert_evaluation/summary.json \
  --output-csv outputs/expert_evaluation/model_summary.csv
```

## 7. Supplementary Figures

The package includes only non-identifying aggregate figure inputs. Recreate the main-paper component-gain figure and supplementary error-profile and loss curves with:

```bash
python code/11-figure-generation/plot_component_gain.py \
  --pdf outputs/figures/component_gain_plot.pdf \
  --svg outputs/figures/component_gain_plot.svg \
  --png outputs/figures/component_gain_plot.png
python code/11-figure-generation/build_error_profile.py \
  --ground-truth /private/path/ground_truth_newera.json \
  --sample-ids /private/path/test_ids.json \
  --prediction 'Baseline Claude=/private/path/baseline_claude.json' \
  --prediction 'Baseline Qwen=/private/path/baseline_qwen.json' \
  --prediction 'DKI-Claude=/private/path/dki_claude.json' \
  --prediction 'Qwen-LoRA=/private/path/student_blind.jsonl' \
  --output outputs/figure_inputs/error_profile_summary.tsv
python code/11-figure-generation/extract_loss_points.py \
  --input /private/path/full_run/logging.jsonl \
  --output outputs/figure_inputs/loss_curve_points.json
python code/11-figure-generation/plot_supplementary_figures.py error-profile \
  --input outputs/figure_inputs/error_profile_summary.tsv \
  --output outputs/figures/error_profile.pdf
python code/11-figure-generation/plot_supplementary_figures.py loss-curves \
  --input outputs/figure_inputs/loss_curve_points.json \
  --output outputs/figures/loss_curves.pdf
```
