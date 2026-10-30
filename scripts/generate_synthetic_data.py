#!/usr/bin/env python3
"""Generate clearly labeled synthetic artifacts for pipeline smoke testing.

The output mirrors the participant-level contracts consumed by notebooks 02,
03, 05A, 05B, and 06. It contains no All of Us records and must never be used
for scientific inference.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


FINAL_MODELS = {
    "PLCOm2012_published_predictions.parquet": "PLCOm2012_published",
    "PLCOm2012_refitted_logistic_predictions.parquet": "PLCOm2012_refitted_logistic",
    "Clinical_retrained_predictions.parquet": "Clinical_retrained",
    "Demographic_PheCode_predictions.parquet": "Demographic_PheCode",
    "Demographic_PheCode_Smoking_predictions.parquet": "Demographic_PheCode_Smoking",
    "Demographic_CLMBR_predictions.parquet": "Demographic_CLMBR",
    "Demographic_CLMBR_Smoking_predictions.parquet": "Demographic_CLMBR_Smoking",
    "Demographic_Finetuned_CLMBR_predictions.parquet": "Demographic_Finetuned_CLMBR",
    "Demographic_Finetuned_CLMBR_Smoking_predictions.parquet": (
        "Demographic_Finetuned_CLMBR_Smoking"
    ),
}

MARKER_NAME = "SYNTHETIC_DATA_ONLY.txt"


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), -30, 30)
    return 1.0 / (1.0 + np.exp(-values))


def locate_phecode_map(explicit_path: Path | None) -> Path:
    candidates = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    repository_root = Path(__file__).resolve().parents[1]
    candidates.extend(
        [
            repository_root / "Phecode_map12_filtered.csv",
            Path.cwd() / "Phecode_map12_filtered.csv",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find Phecode_map12_filtered.csv. Run this script from the "
        "repository root or pass --phecode-map."
    )


def load_phecode_names(path: Path, count: int) -> list[str]:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "cp1252", "latin1"):
        for separator in (None, ","):
            try:
                kwargs = {
                    "dtype": str,
                    "encoding": encoding,
                    "engine": "python" if separator is None else "c",
                }
                if separator is None:
                    kwargs["sep"] = None
                frame = pd.read_csv(path, **kwargs)
                normalized = {str(column).strip().lower(): column for column in frame.columns}
                source = normalized.get("phecodestring")
                if source is None:
                    raise ValueError("PheCode map lacks a phecodestring column")
                names = frame[source].astype("string").str.strip().dropna().drop_duplicates()
                lung_pattern = re.compile(
                    r"(lung|bronchus).*(cancer|carcinoma|malignan|neoplasm)|"
                    r"(cancer|carcinoma|malignan|neoplasm).*(lung|bronchus)",
                    re.IGNORECASE,
                )
                reserved = {
                    "person_id",
                    "fold",
                    "has_lungcancer_3Y",
                    "lifestyle_survey_date",
                    "date_of_birth",
                    "lungcancer_date",
                }
                usable = [
                    str(name)
                    for name in names
                    if name not in reserved
                    and not str(name).endswith("_date")
                    and not lung_pattern.search(str(name))
                ]
                if len(usable) < count:
                    raise ValueError(
                        f"Only {len(usable)} non-lung PheCode labels are available; need {count}."
                    )
                return usable[:count]
            except Exception as error:  # try the next encoding/delimiter
                last_error = error
    raise RuntimeError(f"Could not read {path}: {last_error}")


def prepare_output_directory(path: Path) -> Path:
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    marker = path / MARKER_NAME
    existing = [item for item in path.iterdir() if item.name != MARKER_NAME]
    if existing and not marker.exists():
        raise RuntimeError(
            f"Refusing to write synthetic artifacts into nonempty unmarked directory {path}. "
            "Choose an empty directory."
        )
    marker.write_text(
        "SYNTHETIC TEST DATA ONLY\n"
        "Contains no All of Us participant records and is not valid for scientific inference.\n"
    )
    return path


def stratified_folds(outcome: np.ndarray, rng: np.random.Generator, n_splits: int = 5) -> np.ndarray:
    folds = np.full(len(outcome), -1, dtype=np.int8)
    for label in (0, 1):
        rows = np.flatnonzero(outcome == label)
        rng.shuffle(rows)
        folds[rows] = np.arange(len(rows), dtype=np.int64) % n_splits
    if (folds < 0).any():
        raise AssertionError("Every synthetic participant must receive a fold.")
    return folds


def generate_cohort(n: int, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    if n < 100:
        raise ValueError("Use at least 100 participants so five-fold metrics contain both classes.")

    person_id = np.arange(9_000_000_000, 9_000_000_000 + n, dtype=np.int64)
    age = rng.integers(45, 86, size=n)
    survey_date = pd.to_datetime("2019-01-01") + pd.to_timedelta(
        rng.integers(0, 730, size=n), unit="D"
    )
    status = rng.choice(["current", "former"], size=n, p=[0.35, 0.65])
    maximum_smoking_years = np.maximum(age - 18, 5)
    smoking_years = np.minimum(
        rng.integers(5, 51, size=n), maximum_smoking_years
    ).astype(float)
    average_daily = np.clip(rng.normal(18, 7, size=n), 3, 60)
    current_daily = np.where(
        status == "current",
        np.clip(average_daily + rng.normal(0, 3, size=n), 1, 70),
        0.0,
    )
    quit_years = np.where(status == "former", rng.integers(1, 31, size=n), 0).astype(float)
    quit_age = np.where(status == "former", age - quit_years, np.nan)

    sex = rng.choice(["Female", "Male"], size=n, p=[0.52, 0.48])
    race = rng.choice(
        ["White", "Black", "Hispanic", "Asian", "American Indian or Alaskan Native"],
        size=n,
        p=[0.48, 0.20, 0.19, 0.10, 0.03],
    )
    education = rng.choice(
        [
            "Less than high school",
            "High school or GED",
            "Some college",
            "College graduate",
            "Advanced degree",
        ],
        size=n,
        p=[0.10, 0.25, 0.30, 0.23, 0.12],
    )
    bmi = np.clip(rng.normal(28, 5.5, size=n), 16, 55)
    copd_probability = sigmoid(-3.0 + 0.045 * (age - 60) + 0.025 * (smoking_years - 25))
    copd = rng.binomial(1, copd_probability)
    family_lung = rng.binomial(1, 0.13, size=n)
    cancer_history = rng.binomial(1, sigmoid(-2.8 + 0.035 * (age - 60)))

    latent_logit = (
        -4.2
        + 0.055 * (age - 60)
        + 0.025 * (smoking_years - 25)
        + 0.020 * (average_daily - 18)
        + 0.55 * copd
        + 0.38 * family_lung
        + 0.30 * cancer_history
        + 0.20 * (status == "current")
    )
    risk = sigmoid(latent_logit)
    outcome = rng.binomial(1, risk).astype(np.int8)
    minimum_cases = max(10, math.ceil(0.02 * n))
    if int(outcome.sum()) < minimum_cases:
        outcome[np.argsort(latent_logit)[-minimum_cases:]] = 1

    lung_days = rng.integers(30, 1096, size=n)
    lung_date = pd.Series(pd.NaT, index=np.arange(n), dtype="datetime64[ns]")
    lung_date.loc[outcome == 1] = survey_date[outcome == 1] + pd.to_timedelta(
        lung_days[outcome == 1], unit="D"
    )

    date_of_birth = survey_date - pd.to_timedelta((age * 365.25).astype(int), unit="D")
    first_ehr_date = survey_date - pd.to_timedelta(rng.integers(3 * 365, 12 * 365, size=n), unit="D")
    last_ehr_date = survey_date + pd.to_timedelta(rng.integers(3 * 365, 5 * 365, size=n), unit="D")

    cohort = pd.DataFrame(
        {
            "person_id": person_id,
            "date_of_birth": date_of_birth,
            "lifestyle_survey_date": survey_date,
            "lungcancer_date": lung_date,
            "first_ehr_date": first_ehr_date,
            "last_ehr_date": last_ehr_date,
            "age_at_survey": age.astype(float),
            "age_at_survey_int": age.astype(np.int16),
            "sex_at_birth": sex,
            "race_ethnicity": race,
            "education_group": education,
            "bmi": bmi.round(2),
            "smoking_status": status,
            "Smoking: Number Of Years": smoking_years,
            "Smoking: Average Daily Cigarette Number": average_daily.round(2),
            "Smoking: Current Daily Cigarette Number": current_daily.round(2),
            "Attempt Quit Smoking: Completely Quit Age": quit_age,
            "duration_of_quitting": quit_years,
            "COPD_pre": copd.astype(np.int8),
            "family_lung": family_lung.astype(np.int8),
            "cancer_history": cancer_history.astype(np.int8),
            "has_lungcancer_3Y": outcome,
            "synthetic_record": np.ones(n, dtype=np.int8),
        }
    )
    return cohort, latent_logit


def generate_phecodes(
    cohort: pd.DataFrame,
    names: list[str],
    latent_logit: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    columns: dict[str, object] = {
        "person_id": cohort["person_id"].to_numpy(),
    }
    survey = pd.to_datetime(cohort["lifestyle_survey_date"])
    standardized_risk = (latent_logit - latent_logit.mean()) / max(latent_logit.std(), 1e-8)

    for index, name in enumerate(names):
        base_probability = rng.uniform(0.025, 0.18)
        association = rng.uniform(0.15, 0.55) if index < 24 else rng.uniform(-0.08, 0.08)
        base_logit = math.log(base_probability / (1 - base_probability))
        probability = sigmoid(base_logit + association * standardized_risk)
        present = rng.binomial(1, probability).astype(np.int8)
        if present.sum() < 3:
            present[rng.choice(len(present), size=3, replace=False)] = 1

        recent = rng.random(len(present)) < 0.25
        days_before = np.where(
            recent,
            rng.integers(1, 366, size=len(present)),
            rng.integers(366, 3651, size=len(present)),
        )
        dates = pd.Series(pd.NaT, index=cohort.index, dtype="datetime64[ns]")
        active = present == 1
        dates.loc[active] = survey.loc[active] - pd.to_timedelta(days_before[active], unit="D")
        columns[name] = present
        columns[f"{name}_date"] = dates.to_numpy()
    return pd.DataFrame(columns)


def generate_embeddings(
    cohort: pd.DataFrame,
    latent_logit: np.ndarray,
    dimension: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    matrix = rng.normal(0, 1, size=(len(cohort), dimension)).astype(np.float32)
    standardized_risk = (latent_logit - latent_logit.mean()) / max(latent_logit.std(), 1e-8)
    signal_dimensions = min(8, dimension)
    matrix[:, :signal_dimensions] += standardized_risk[:, None] * np.linspace(
        0.9, 0.2, signal_dimensions
    )
    keep = rng.random(len(cohort)) >= 0.03
    embeddings = pd.DataFrame(
        matrix[keep], columns=[f"clmbr_{index:04d}" for index in range(dimension)]
    )
    embeddings.insert(0, "person_id", cohort.loc[keep, "person_id"].to_numpy())
    return embeddings


def generate_meds(cohort: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    codes = np.array(
        [
            "ICD10CM/I10",
            "ICD10CM/J44.9",
            "SNOMED/44054006",
            "SNOMED/195967001",
            "LOINC/8480-6",
            "LOINC/8462-4",
            "RxNorm/860975",
            "CPT/99213",
        ],
        dtype=object,
    )
    rows: list[dict[str, object]] = []
    for record in cohort.itertuples(index=False):
        rows.append(
            {
                "subject_id": int(record.person_id),
                "time": pd.Timestamp(record.date_of_birth),
                "code": "MEDS_BIRTH",
            }
        )
        event_count = int(rng.integers(12, 31))
        days_before = rng.integers(2, 8 * 365, size=event_count)
        selected_codes = rng.choice(codes, size=event_count, replace=True)
        for day, code in zip(days_before, selected_codes):
            rows.append(
                {
                    "subject_id": int(record.person_id),
                    "time": pd.Timestamp(record.lifestyle_survey_date) - pd.Timedelta(days=int(day)),
                    "code": str(code),
                }
            )
    return pd.DataFrame(rows).sort_values(["subject_id", "time"]).reset_index(drop=True)


def generate_predictions(
    cohort: pd.DataFrame,
    latent_logit: np.ndarray,
    prediction_dir: Path,
    rng: np.random.Generator,
) -> None:
    prevalence = float(cohort["has_lungcancer_3Y"].mean())
    intercept = math.log(prevalence / (1 - prevalence))
    standardized_risk = (latent_logit - latent_logit.mean()) / max(latent_logit.std(), 1e-8)
    model_quality = {
        "PLCOm2012_published": (0.75, 1.10),
        "PLCOm2012_refitted_logistic": (0.90, 0.95),
        "Clinical_retrained": (1.00, 0.85),
        "Demographic_PheCode": (1.12, 0.75),
        "Demographic_PheCode_Smoking": (1.20, 0.70),
        "Demographic_CLMBR": (1.05, 0.82),
        "Demographic_CLMBR_Smoking": (1.15, 0.76),
        "Demographic_Finetuned_CLMBR": (1.18, 0.73),
        "Demographic_Finetuned_CLMBR_Smoking": (1.25, 0.68),
    }
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for filename, label in FINAL_MODELS.items():
        strength, noise = model_quality[label]
        probability = sigmoid(
            intercept + strength * standardized_risk + rng.normal(0, noise, size=len(cohort))
        )
        frame = pd.DataFrame(
            {
                "person_id": cohort["person_id"],
                "y_true": cohort["has_lungcancer_3Y"].astype(np.int8),
                "prob": np.clip(probability, 1e-5, 1 - 1e-5),
                "model": label,
            }
        )
        if label == "PLCOm2012_published":
            frame["eligible_1_51"] = frame["prob"].ge(0.0151)
        frame.to_parquet(prediction_dir / filename, index=False)


def validate_outputs(output_dir: Path, phecode_count: int) -> dict[str, int | float]:
    cohort = pd.read_parquet(output_dir / "paper_cohort.parquet")
    modeling = pd.read_parquet(output_dir / "demo_baseline_paper_reproduction.parquet")
    date_aware = pd.read_parquet(output_dir / "per_patient_with_phe_preindex_rebuilt.parquet")
    folds = pd.read_parquet(output_dir / "paper_reproduction" / "cv_folds.parquet")
    embeddings = pd.read_parquet(output_dir / "embeddings_final.parquet")
    meds = pd.read_parquet(output_dir / "EHR_MEDS.parquet")

    if cohort["person_id"].duplicated().any():
        raise AssertionError("Synthetic cohort contains duplicate person IDs.")
    if set(cohort["has_lungcancer_3Y"].unique()) != {0, 1}:
        raise AssertionError("Synthetic outcome must contain both classes.")
    if len(modeling) != len(cohort) or len(date_aware) != len(cohort):
        raise AssertionError("Synthetic cohort artifacts have inconsistent row counts.")
    if set(folds["fold"].unique()) != set(range(5)):
        raise AssertionError("Synthetic folds must contain fold IDs 0 through 4.")
    paired = [
        column
        for column in date_aware.columns
        if column.endswith("_date") and column[:-5] in date_aware.columns
    ]
    if len(paired) < phecode_count:
        raise AssertionError("Synthetic date-aware PheCode pairs are incomplete.")
    if embeddings["person_id"].duplicated().any():
        raise AssertionError("Synthetic embeddings contain duplicate person IDs.")

    maximum_event_time = meds.groupby("subject_id")["time"].max()
    survey_by_id = cohort.set_index("person_id")["lifestyle_survey_date"]
    aligned_survey = survey_by_id.reindex(maximum_event_time.index)
    if not (maximum_event_time < aligned_survey).all():
        raise AssertionError("Synthetic MEDS events must all precede the index date.")

    prediction_dir = output_dir / "paper_reproduction" / "predictions"
    for filename, label in FINAL_MODELS.items():
        frame = pd.read_parquet(prediction_dir / filename)
        required = {"person_id", "y_true", "prob", "model"}
        if not required.issubset(frame.columns):
            raise AssertionError(f"{filename} does not satisfy the prediction contract.")
        if set(frame["model"].unique()) != {label}:
            raise AssertionError(f"{filename} has the wrong model label.")
        if len(frame) != len(cohort) or frame["person_id"].duplicated().any():
            raise AssertionError(f"{filename} does not cover the synthetic cohort once.")
        if not frame["prob"].between(0, 1, inclusive="neither").all():
            raise AssertionError(f"{filename} contains invalid probabilities.")

    return {
        "participants": int(len(cohort)),
        "cases": int(cohort["has_lungcancer_3Y"].sum()),
        "phecode_features": int(len(paired)),
        "embedding_rows": int(len(embeddings)),
        "meds_events": int(len(meds)),
        "prediction_models": int(len(FINAL_MODELS)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("synthetic_artifacts"),
        help="Empty or previously marked synthetic output directory.",
    )
    parser.add_argument("--phecode-map", type=Path, default=None)
    parser.add_argument("--participants", type=int, default=1000)
    parser.add_argument("--phecodes", type=int, default=120)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()

    if args.phecodes < 100:
        raise ValueError("Notebook 03 requires at least 100 PheCode features.")
    if args.embedding_dim < 8:
        raise ValueError("Use at least eight synthetic embedding dimensions.")

    output_dir = prepare_output_directory(args.output_dir)
    mapping_path = locate_phecode_map(args.phecode_map)
    phecode_names = load_phecode_names(mapping_path, args.phecodes)
    rng = np.random.default_rng(args.seed)

    cohort, latent_logit = generate_cohort(args.participants, rng)
    folds = stratified_folds(cohort["has_lungcancer_3Y"].to_numpy(), rng)
    phecodes = generate_phecodes(cohort, phecode_names, latent_logit, rng)
    modeling_cohort = cohort.merge(
        phecodes.drop(columns=[column for column in phecodes if column.endswith("_date")]),
        on="person_id",
        how="left",
        validate="one_to_one",
    )
    embeddings = generate_embeddings(cohort, latent_logit, args.embedding_dim, rng)
    meds = generate_meds(cohort, rng)

    run_dir = output_dir / "paper_reproduction"
    prediction_dir = run_dir / "predictions"
    run_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    cohort.to_parquet(output_dir / "paper_cohort.parquet", index=False)
    phecodes.to_parquet(
        output_dir / "per_patient_with_phe_preindex_rebuilt.parquet", index=False
    )
    modeling_cohort.to_parquet(
        output_dir / "demo_baseline_paper_reproduction.parquet", index=False
    )
    pd.DataFrame({"person_id": cohort["person_id"], "fold": folds}).to_parquet(
        run_dir / "cv_folds.parquet", index=False
    )
    embeddings.to_parquet(output_dir / "embeddings_final.parquet", index=False)
    meds.to_parquet(output_dir / "EHR_MEDS.parquet", index=False)
    generate_predictions(cohort, latent_logit, prediction_dir, rng)

    summary = validate_outputs(output_dir, args.phecodes)
    manifest = {
        "synthetic": True,
        "scientific_use_allowed": False,
        "seed": args.seed,
        "phecode_map": str(mapping_path),
        **summary,
    }
    (output_dir / "synthetic_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    print(json.dumps(manifest, indent=2, sort_keys=True))
    print("\nSynthetic artifacts are ready.")
    print(f"export LUNG_CANCER_DATA_DIR={output_dir}")
    print("export LUNG_CANCER_SYNTHETIC=1")


if __name__ == "__main__":
    main()
