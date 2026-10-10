# LCS2 Round 2 — Responsible Medical LLMs for Mental Healthcare

A small, reproducible benchmark that evaluates **Llama-3.2-3B-Instruct** and **Qwen2.5-3B-Instruct** on one fixed
mental-health multiple-choice task (MedMisBench-MentalHealth) across four dimensions — Hallucinations &
Uncertainty, Interpretability & Explainability, Bias & Fairness, Privacy & Security — plus a cross-dimension
analysis (Part C) and one small intervention (Part D). Everything was run on a **free Kaggle T4 GPU**.

The final report (`Sourava_LCS2_Round2_Report.pdf`) is submitted separately. AI use: [AI_USAGE.md](AI_USAGE.md).

```
.
├── README.md, AI_USAGE.md
├── config.yaml             all settings: models + revisions, dataset + revision, subsets, prompts, seed, paths
├── requirements.txt        exact package versions (local runs; Kaggle needs no install)
├── .env.example            Hugging Face token template (local runs only)
├── notebooks/              one notebook per experiment = experiment cells + analysis cells
├── scripts/report_numbers.py   report numbers that are not in an analysis CSV (CIs, evidence echo, flip-rate CIs)
├── kaggle_run_logs/        Kaggle job log of every experiment run (one <part>.log per notebook)
└── results/
    ├── experiments/<part>/     machine-readable clean and perturbed outputs (official Kaggle T4 run)
    └── analysis/<folder>/      every table (*.csv) and figure (*.png) of the analysis
```

---

## 1. README.md with end-to-end reproduction instructions from a fresh Kaggle notebook

### 1.1 One-time setup
1. **Hugging Face:** accept the Llama 3.2 licence at <https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct> and
   create a read token at <https://huggingface.co/settings/tokens>.
2. **Kaggle:** an account with phone verification (needed for GPU and internet).

### 1.2 Run a notebook on Kaggle
1. **Create → New Notebook**, then **File → Import Notebook** and paste the GitHub link of a notebook from
   `notebooks/` (or upload the `.ipynb` file). Start with `4_part_a_baseline.ipynb`.
2. **Settings** (right panel): *Accelerator* → **GPU T4**; *Internet* → **On**.
3. **Add-ons → Secrets:** add a secret named **`HF_TOKEN`** with your token and switch it on for this notebook.
4. Insert one code cell **at the very top** and run it:
   ```python
   !git clone --depth 1 https://github.com/your-username/LCS2-Round2.git
   # to regenerate this part from scratch, also delete its saved records, e.g.
   # !rm -rf LCS2-Round2/results/experiments/4_part_a_baseline
   ```
   Every notebook finds the repository root as the folder that contains `config.yaml`, so no path needs editing.
5. **Run All.** Results are written to `/kaggle/working/LCS2-Round2/results/`; download them from the *Output* panel.

### 1.3 Order, dependencies and run time

| Order | Notebook | Needs | GPU time (T4) |
|---|---|---|---|
| 1 | `4_part_a_baseline` | — | ≈ 30 min |
| 2 | `5_1_hallucination_uncertainty` | Part A | ≈ 12 min |
| 2 | `5_2_interpretability` | Part A | ≈ 3.5 h |
| 2 | `5_3_bias_fairness` | Part A | ≈ 50 min |
| 2 | `5_4_privacy_security` | Part A, 5.1 | ≈ 30 min |
| 3 | `8_part_d_prompt_hardening` | Part A, 5.1, 5.4 | ≈ 30 min |
| 4 | `6_cross_dimension`, `7_overview_all_experiments` | all of the above | CPU, seconds |

Total ≈ 6.2 GPU hours (Kaggle's free weekly quota is 30 h). One model is loaded at a time.

### 1.4 Saved records are reused
The generation cells are **resumable**: they skip rows already present in `results/experiments/<part>/`. Because
the repository ships the official results, a run on a fresh clone re-uses them and only re-runs the analysis. To
regenerate a part, delete its `results/experiments/<part>/` folder first (step 4 above).

Each Stress-set notebook also regenerates three clean items and requires them to match the saved Part A records
token for token. If Kaggle's software image has changed since our run, this check can fail; then delete
`results/experiments/4_part_a_baseline/` too and rerun Part A first in the same environment.

### 1.5 Running locally instead (optional)
Python 3.11+ and an NVIDIA GPU with ≥ 16 GB for the experiment parts (analysis parts run on CPU).
```bash
git clone https://github.com/your-username/LCS2-Round2.git && cd LCS2-Round2
python -m venv .venv && source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
cp .env.example .env                    # paste your Hugging Face token into .env
jupyter nbconvert --to notebook --execute --inplace notebooks/4_part_a_baseline.ipynb
```
Pick the GPU with `cuda_device` in `config.yaml` (default `"0"`).

---

## 2. Code/notebooks, prompts, configs, seed, exact model revisions, dataset revision, and environment/package versions

| Item | Where | Value |
|---|---|---|
| **Code / notebooks** | `notebooks/*.ipynb` | 8 notebooks; each holds the experiment cells and then the analysis cells (table below) |
| **Prompts** | `config.yaml` → `prompts:` | clean system prompt; misleading header; injection note; canary sentence + audit request; bias metadata lines; hardening block. Every output record also stores the exact `prompt` messages |
| **Configs** | `config.yaml` | models, dataset, subsets, inference, generation, prompts, paths |
| **Seed** | `config.yaml` → `seed` | **42** (all preprocessing / sampling code; decoding is greedy) |
| **Model revisions** | `config.yaml` → `models:` | `meta-llama/Llama-3.2-3B-Instruct` @ `0cb88a4f764b7a12671c53f0838cd831a0843b95`<br>`Qwen/Qwen2.5-3B-Instruct` @ `aa8e72537993ba99e69dfaafa59ed015b17504d1` |
| **Dataset revision** | `config.yaml` → `dataset:` | `Johndfm/MedMisBench-MentalHealth` @ `9757095598b6555c3e935679fdb7384aed4eb651`, file `medmisbench_mentalhealth.jsonl`, split `train`; SHA-256 `0c7db2295ca5bf220933bc95d8f613c11e3f8c0d3d5f0235dbf56595d5434c21` |
| **Subsets** | `config.yaml` → `subsets:` | zero-based `row_index` in stored order, no shuffling: Core 0–199, Stress 0–59, Interpretability 0–39 |
| **Inference** | `config.yaml` → `inference:`, `generation:` | float16, no quantization, batch size 1, greedy, `max_new_tokens=200`, `repetition_penalty=1.0`, Llama chat date fixed to `01 Sep 2026` |
| **Environment (official run)** | every `results/experiments/<part>/run_meta.json` | Kaggle, Tesla T4, Python 3.13.15, PyTorch 2.11.0+cu128, transformers 5.16.1, CUDA 12.8 (Kaggle's preinstalled image) |
| **Package versions (local)** | `requirements.txt` | exact versions the notebooks were tested with (torch 2.6.0, transformers 5.18.0, …) |

### Notebooks

| Notebook | Assignment part | Subset / condition |
|---|---|---|
| `4_part_a_baseline` | 4 · Part A baseline | Core (200), clean prompt |
| `5_1_hallucination_uncertainty` | 5.1 · Hallucinations & Uncertainty (+ hypotheses H1, H2) | Stress (60): clean vs `ADDITIONAL UNVERIFIED CONTEXT` with `injections[type1_injection_key]` |
| `5_2_interpretability` | 5.2 · Interpretability & Explainability | Interpretability (40), clean; Integrated Gradients, deletion test with a random control |
| `5_3_bias_fairness` | 5.3 · Bias & Fairness | Stress (60): 6 one-line `PATIENT METADATA` variants |
| `5_4_privacy_security` | 5.4 · Privacy & Security | Stress (60): targeted prompt injection; canary attack + canary control |
| `6_cross_dimension` | 6 · Part C | Stress (60), results of all parts |
| `8_part_d_prompt_hardening` | 7 · Part D | Stress (60): misleading / clean / injection, before vs after prompt hardening |
| `7_overview_all_experiments` | overview | every condition with the same metrics |

---

## 3. Machine-readable clean and perturbed outputs plus scripts used to generate every reported figure/table

### 3.1 Clean and perturbed outputs (`results/experiments/`)

| Folder | Files (`llama_*` and `qwen_*`) | Condition |
|---|---|---|
| `4_part_a_baseline/` | `*_core.jsonl` | clean (Core, 200 items) |
| `5_1_hallucination_uncertainty/` | `*_stress.jsonl` | misleading context |
| `5_2_interpretability/` | `*_interp.jsonl` | clean (attributions of the Part A answers) |
| `5_3_bias_fairness/` | `*_race_black/white`, `*_insurance_uninsured/private`, `*_residence_rural/urban` | 6 metadata variants |
| `5_4_1_prompt_injection/` | `*_injection.jsonl` | prompt injection |
| `5_4_2_canary_leakage/` | `*_canary.jsonl`, `*_canary_control.jsonl` | canary attack, canary control |
| `8_part_d_prompt_hardening/` | `*_hardened_{misleading,clean,injection}.jsonl` | hardened system prompt |

The clean condition of every Stress-set experiment is the Part A record of the same `row_index`. Each folder also
has `run_meta.json` (dataset revision + SHA-256, model revisions, prompts, settings, seed, environment, timing) and
`kaggle_log.log` (the Kaggle job log). **Failed and unparseable generations are kept** (`answer = null`,
`parse_status = invalid`); no result file was edited by hand.

**Fields of every record:** `row_index`, `id` (dataset id), `gold`, `raw_output`, `answer` (parsed),
`confidence_token` (probability of the chosen key at the answer step, renormalised over the item's keys),
`option_probs`, `entropy`, `option_mass`, `evidence`, `correct`, `parse_status`, `truncated`, `model`,
`model_revision`, `generation` (decoding settings, dtype, seed, batch size, chat date), `dataset_revision`,
`prompt` (exact messages), `condition`, `prompt_name`, token counts and timing. Extra fields: **5.1**
`injected_key`, `injected_text`, `injection_content`, `injection_provenance`, `chose_injected` · **5.3**
`variant`, `attribute`, `value`, `metadata_line`, `metadata_conflict` · **5.4.1** `injected_key`, `note_text`,
`chose_injected` · **5.4.2** `canary`, `system_has_canary`, `audit_request` · **Part D** `base_condition`,
`intervention` · **5.2** per-token IG and gradient×input scores, deletion test, evidence check, IG checks.

### 3.2 Kaggle run logs (`kaggle_run_logs/`)

The job log of each Kaggle run (`<part>.log`, JSON lines of everything the run printed). Each log shows the
environment line `torch 2.11.0+cu128 | transformers 5.16.1 | Tesla T4`, the determinism and clean-reuse checks
passing, and the progress and timing of every model run. The same log is also stored next to the results as
`results/experiments/<part>/kaggle_log.log`.

### 3.3 Scripts that generate every reported figure and table

* **Analysis cells** of each notebook (the part titled *"Analysis"*) read only `results/experiments/` and write
  every CSV and PNG in `results/analysis/<folder>/`. They run on CPU; on the shipped results the regenerated CSVs
  are identical to the committed ones.
* **`scripts/report_numbers.py`** computes the few reported numbers that are not in an analysis CSV:
  `python scripts/report_numbers.py part_a 5_1 5_3`.

**Figures in the report** (each is a byte-identical copy of the file listed):

| Report figure | File | Generated by |
|---|---|---|
| Part A reliability diagram | `results/analysis/4_part_a_baseline/reliability_diagram.png` | `4_part_a_baseline` (analysis) |
| Part A confidence of right vs wrong answers | `results/analysis/4_part_a_baseline/conf_right_vs_wrong.png` | `4_part_a_baseline` (analysis) |
| 5.1 clean vs misleading confidence per item | `results/analysis/5_1_hallucination_uncertainty/fig_confidence_pairs.png` | `5_1_hallucination_uncertainty` (analysis) |
| 5.2 deletion curves | `results/analysis/5_2_interpretability/fig_deletion_curves.png` | `5_2_interpretability` (analysis) |
| 5.2 misleading case heat map | `results/analysis/5_2_interpretability/fig_case_misleading.png` | `5_2_interpretability` (analysis) |
| 5.3 flip rate vs control | `results/analysis/5_3_bias_fairness/fig_afr_vs_control.png` | `5_3_bias_fairness` (analysis) |
| 5.4 injection compliance | `results/analysis/5_4_privacy_security/fig1_compliance.png` | `5_4_privacy_security` (analysis) |
| Part C AUROC per failure | `results/analysis/6_cross_dimension/fig_auroc.png` | `6_cross_dimension` |
| Part D before vs after | `results/analysis/8_part_d_prompt_hardening/fig_partd.png` | `8_part_d_prompt_hardening` (analysis) |

**Tables and quoted numbers in the report:**

| Report section / table | Source file(s) | Generated by |
|---|---|---|
| Part A results | `4_part_a_baseline/metrics.csv`, `ece_bins.csv` | `4_part_a_baseline` (analysis); CIs, AUROC, chance accuracy: `report_numbers.py part_a` |
| 5.1 misleading-context results | `5_1_…/metrics.csv`, `clean_vs_misleading.csv`, `ece_clean_vs_misleading.csv` | `5_1_hallucination_uncertainty` (analysis); evidence echo: `report_numbers.py 5_1` |
| 5.1 hypothesis tests H1, H2 | `5_1_…/hypothesis_tests.csv` | `5_1_hallucination_uncertainty` (analysis) |
| 5.1 wrong-but-confident cases | `5_1_…/wrong_confident_cases.csv` | `5_1_hallucination_uncertainty` (analysis) |
| 5.2 faithfulness + extra results | `5_2_…/faithfulness_tests.csv`, `ig_vs_gxi.csv`, `robustness.csv`, `attribution_profile.csv`, `evidence_vs_attribution.csv`, `run_checks.csv` | `5_2_interpretability` (analysis) |
| 5.2 three cases (both models) | `5_2_…/cases.csv`, `deletion_items.csv` | `5_2_interpretability` (analysis) |
| 5.3 fairness results | `5_3_…/fairness_pairs.csv`, `descriptive.csv`, `attribute_mentions.csv`, `usr.csv` | `5_3_bias_fairness` (analysis); flip-rate CIs: `report_numbers.py 5_3` |
| 5.3 manual USR labels | `5_3_…/usr_labels.csv`, `human_check.md` (all 16 answer changes reviewed by the author) | manual review, read by `5_3_bias_fairness` (analysis) |
| 5.4 injection and canary results | `5_4_…/table_injection.csv`, `table_canary.csv`, `injection_metrics.csv`, `canary_metrics.csv`, `leak_cases.csv` | `5_4_privacy_security` (analysis) |
| Part C cross-dimension | `6_cross_dimension/summary.csv`, `primary_auroc.csv`, `secondary_tests.csv` | `6_cross_dimension` |
| Part D before vs after | `8_part_d_…/table_partd.csv`, `paired_tests.csv`, `metrics_long.csv`, `changed_items.csv` | `8_part_d_prompt_hardening` (analysis) |
| Summary of all results | `7_overview_all_experiments/table2_headlines.csv`, `table1_conditions.csv` | `7_overview_all_experiments` |

(All source paths above are under `results/analysis/` unless written in full.)

---

## Reproducibility controls

- **Pinned:** dataset revision + SHA-256, model revisions, prompts, seed and decoding in `config.yaml`, saved in every record.
- **Deterministic:** greedy decoding; each run generates one item twice and requires identical text and option probabilities.
- **Safe reuse of clean answers:** each Stress-set notebook regenerates three clean items and requires a token-for-token match with Part A.
- **Nothing dropped or edited:** invalid generations are kept; result files are written only by the notebooks.

## Notes

- **Secrets:** no token is stored in this repository — a Kaggle Secret (`HF_TOKEN`) on Kaggle, a local `.env`
  (from `.env.example`, ignored by git) elsewhere.
- **Canaries** are synthetic strings (`MCS2-CANARY-<id>-7F3A9C`); no real identifiers or patient data are used.
