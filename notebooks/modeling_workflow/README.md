# Paper reconstruction notebooks

Run in this order:

1. `01_Cohort_Construction.ipynb` (cohort construction only)
2. `02_Clinical_Risk_Score_Benchmarks.ipynb` (including USPSTF eligibility sensitivity)
3. `03_PheCode_Model.ipynb` (PheCode construction, modeling, and temporal diagnostic-blackout sensitivity)
4. `04_Prepare_MEDS_for_CLMBR.ipynb`
5. `05A_CLMBR_Foundation_Model.ipynb`
6. `05B_LoRA_Finetune_CLMBR.ipynb` (optional GPU experiment)
7. `06_Combined_Model_Evaluation.ipynb`

By default, generated participant-level artifacts are written under `~/lungcancer`. Set `LUNG_CANCER_DATA_DIR` to use another location, but all derived participant-level data must remain inside the controlled Workbench environment.

Notebook 01 saves the cohort-only artifact `paper_cohort.parquet` under the configured data directory. Notebook 02 uses that cohort for USPSTF eligibility sensitivity, clinical risk-score benchmarks, and the shared cross-validation folds. Notebook 03 builds the date-aware PheCode features and saves `demo_baseline_paper_reproduction.parquet` for later modeling notebooks.

All learned models use the same saved cross-validation folds. Only out-of-fold predictions are used for comparison. MEDS events and CLMBR inputs are truncated strictly before each participant's index date.

Notebook 03 trains the EHR-only and smoking-plus-EHR PheCode models and includes a 12-month diagnostic-blackout sensitivity analysis. It reports AUROC/AUPRC and top 5%, 10%, and 20% sensitivity/PPV from out-of-fold predictions.

Notebook 05A evaluates frozen CLMBR representations. Notebook 05B separately performs task-specific LoRA fine-tuning and saves resumable fold adapters and out-of-fold predictions under the configured controlled data directory.


## Synthetic smoke test

Run `scripts/generate_synthetic_data.py`, set `LUNG_CANCER_DATA_DIR` to its
output directory, and set `LUNG_CANCER_SYNTHETIC=1`. Skip notebooks 01 and 04,
which are the controlled Workbench extraction steps. The CPU smoke-test order is
02, 03, 05A, and 06. Notebook 05B is optional and still requires a GPU.
Synthetic outputs are for contract testing only, not scientific inference.
