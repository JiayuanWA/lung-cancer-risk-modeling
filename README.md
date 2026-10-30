# Lung Cancer Risk Modeling

This repository contains the notebooks used for lung cancer risk modeling,
cohort reconstruction, and sensitivity analyses.

## Repository layout

```text
scripts/
└── generate_synthetic_data.py
notebooks/
├── modeling_workflow/
│   ├── 01_Cohort_Construction.ipynb
│   ├── 02_Clinical_Risk_Score_Benchmarks.ipynb
│   ├── 03_PheCode_Model.ipynb
│   ├── 04_Prepare_MEDS_for_CLMBR.ipynb
│   ├── 05A_CLMBR_Foundation_Model.ipynb
│   ├── 05B_LoRA_Finetune_CLMBR.ipynb
│   ├── 06_Combined_Model_Evaluation.ipynb
│   └── README.md
```

## Suggested execution order

1. Run the numbered notebooks in `modeling_workflow` to rebuild the cohort,
   evaluate USPSTF eligibility and clinical risk scores, evaluate PheCode models
   (including the diagnostic-blackout sensitivity analysis), evaluate CLMBR
   models, and produce the combined clinical evaluation tables and figures.

## Environment

Create and activate a virtual environment, then install the core dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

The base requirements cover the CPU models and BigQuery access. For MEDS
conversion in a Workbench Linux app, install the tested MEDS extra:

```bash
python -m pip install -r requirements-meds.txt
```

For notebooks 05A/05B, start a Workbench GPU app that already has a
CUDA-enabled PyTorch build, then install the tested CLMBR stack without
replacing that PyTorch build:

```bash
python -m pip install -r requirements-clmbr.txt
python -m pip install --no-deps hf-ehr==0.1.4
```

The `hf-ehr` wheel declares a large training environment with older,
conflicting scientific-package pins. The notebooks only need its tokenizer and
`Event` class, so it is intentionally installed with `--no-deps`; the
required runtime dependencies are listed explicitly in
`requirements-clmbr.txt`.

## Data configuration

Data are intentionally not committed. Generated artifacts default to
`~/lungcancer`. Set the `LUNG_CANCER_DATA_DIR` environment variable to use
another location within the controlled Workbench environment. The workflow may
also require:

- Google Cloud credentials and access to the source BigQuery datasets;
- the PheCode mapping file `Phecode_map12_filtered.csv`;
- cohort and feature files produced by earlier pipeline steps.

Before running the notebooks, set `LUNG_CANCER_DATA_DIR` if the default location
is unsuitable. All numbered notebooks read the same setting.

Workbench 2 does not inject the legacy `WORKSPACE_CDR` and
`WORKSPACE_BUCKET` variables. Run `wb resource list`, then set the
workspace-specific resource IDs before notebooks 01, 04, or 05B:

```bash
export LUNG_CANCER_CDR_RESOURCE_ID="<CDR resource ID>"
export LUNG_CANCER_BUCKET_RESOURCE_ID="<controlled GCS bucket resource ID>"
```

The `wb` CLI itself is supplied by Workbench; it is not a pip dependency.
The CLMBR checkpoint is also gated. Request and accept access at
[StanfordShahLab/gpt-base-2048-clmbr](https://huggingface.co/StanfordShahLab/gpt-base-2048-clmbr),
then authenticate in the GPU app using an approved `HF_TOKEN` or
`huggingface-cli login`. Package installation alone does not grant model
access.

## Synthetic smoke test

Generate a fully synthetic artifact set before using controlled data:

```bash
python scripts/generate_synthetic_data.py --output-dir synthetic_artifacts
export LUNG_CANCER_DATA_DIR="$(pwd)/synthetic_artifacts"
export LUNG_CANCER_SYNTHETIC=1
```

The generator creates a synthetic cohort, five stratified folds, more than 100
paired PheCode/date features, pre-index MEDS events, CLMBR-like embeddings, and
prediction files for all final model contracts. It validates the files after
writing them and refuses to write into a nonempty directory unless that
directory was previously marked as synthetic.

For a CPU smoke test, skip the controlled-data extraction notebooks 01 and 04,
then run notebooks 02, 03, 05A, and 06. Notebook 05B can also consume the
synthetic MEDS fixture, but LoRA training still requires its GPU and model
dependencies. The generated outcomes and metrics are deliberately artificial
and must not be interpreted scientifically.

## Reproducibility notes

- Run notebooks from their own folder or use absolute/configurable data paths.
- Record the Python and package versions used for a final analysis.
- Keep generated figures, model files, caches, credentials, and patient-level
  data out of Git.
