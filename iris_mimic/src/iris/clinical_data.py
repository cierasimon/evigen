import csv
import re
import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal

import polars as pl
from sklearn.model_selection import train_test_split

from iris.config import ClinicalDatasetConfig


class ClinicalDataError(ValueError):
    """Raised when the clinical cohort or its fixed splits are inconsistent."""


def _quote_identifier(identifier: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier):
        raise ClinicalDataError(
            f"Unsafe SQLite identifier in configuration: {identifier}"
        )
    return f'"{identifier}"'


def _read_header(path: Path) -> list[str]:
    if not path.is_file():
        raise ClinicalDataError(f"Required clinical data file does not exist: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        header = next(csv.reader(file), None)

    if not header:
        raise ClinicalDataError(f"Clinical data file is empty: {path}")
    return header


def _require_columns(path: Path, required: list[str]) -> None:
    header = _read_header(path)
    missing = [column for column in required if column not in header]
    if missing:
        raise ClinicalDataError(
            f"{path.name} is missing required columns: {', '.join(missing)}"
        )


def _validate_patient_ids(df: pl.DataFrame, source_name: str) -> None:
    missing_count = df.filter(
        pl.col("patient_id").is_null() | (pl.col("patient_id").str.len_chars() == 0)
    ).height
    if missing_count:
        raise ClinicalDataError(
            f"{source_name} contains {missing_count} missing patient IDs"
        )

    duplicate_count = df.height - df["patient_id"].n_unique()
    if duplicate_count:
        raise ClinicalDataError(
            f"{source_name} contains {duplicate_count} duplicate patient IDs"
        )


def load_cohort(cfg: ClinicalDatasetConfig) -> pl.DataFrame:
    """Read and validate the minimum patient-level fields needed by IRIS."""
    required = [cfg.patient_column, cfg.cutoff_column, cfg.label_column]
    _require_columns(cfg.cohort_path, required)

    cohort = (
        pl.read_csv(
            cfg.cohort_path,
            columns=required,
            schema_overrides={column: pl.String for column in required},
        )
        .with_columns(
            pl.col(cfg.patient_column).str.strip_chars().alias("patient_id"),
            pl.col(cfg.cutoff_column)
            .str.strip_chars()
            .str.to_date(format="%Y-%m-%d", strict=False)
            .alias("anchor_date"),
            pl.col(cfg.label_column)
            .str.strip_chars()
            .cast(pl.Int8, strict=False)
            .alias("label"),
        )
        .select("patient_id", "anchor_date", "label")
    )

    if cohort.is_empty():
        raise ClinicalDataError(f"{cfg.cohort_path.name} contains no patients")

    _validate_patient_ids(cohort, cfg.cohort_path.name)

    invalid_date_count = cohort["anchor_date"].null_count()
    if invalid_date_count:
        raise ClinicalDataError(
            f"{cfg.cohort_path.name} contains {invalid_date_count} missing or invalid "
            "anchor dates"
        )

    invalid_label_count = cohort.filter(
        pl.col("label").is_null() | ~pl.col("label").is_in([0, 1])
    ).height
    if invalid_label_count:
        raise ClinicalDataError(
            f"{cfg.cohort_path.name} contains {invalid_label_count} labels outside 0/1"
        )

    return cohort


def load_split_ids(
    path: Path,
    patient_column: str,
    split: Literal["train", "test"],
) -> pl.DataFrame:
    """Read one fixed split while preserving patient IDs as strings."""
    _require_columns(path, [patient_column])

    split_ids = (
        pl.read_csv(
            path,
            columns=[patient_column],
            schema_overrides={patient_column: pl.String},
        )
        .with_columns(
            pl.col(patient_column).str.strip_chars().alias("patient_id"),
            pl.lit(split).alias("split"),
        )
        .select("patient_id", "split")
    )

    if split_ids.is_empty():
        raise ClinicalDataError(f"{path.name} contains no patient IDs")

    _validate_patient_ids(split_ids, path.name)
    return split_ids


def load_patient_index(
    cfg: ClinicalDatasetConfig,
    seed: int,
) -> pl.DataFrame:
    """Build a validated patient index with train, validation, and test splits."""
    if not 0 < cfg.validation_size < 1:
        raise ClinicalDataError("validation_size must be between 0 and 1")

    cohort = load_cohort(cfg)
    train_ids = load_split_ids(
        cfg.train_ids_path,
        patient_column=cfg.patient_column,
        split="train",
    )
    test_ids = load_split_ids(
        cfg.test_ids_path,
        patient_column=cfg.patient_column,
        split="test",
    )

    overlap_count = train_ids.join(test_ids, on="patient_id", how="inner").height
    if overlap_count:
        raise ClinicalDataError(
            f"Train and test splits overlap for {overlap_count} patients"
        )

    fixed_splits = pl.concat([train_ids, test_ids])
    missing_from_cohort = fixed_splits.join(cohort, on="patient_id", how="anti").height
    if missing_from_cohort:
        raise ClinicalDataError(
            f"The split files contain {missing_from_cohort} patients missing from the cohort"
        )

    unassigned_count = cohort.join(fixed_splits, on="patient_id", how="anti").height
    if unassigned_count:
        raise ClinicalDataError(
            f"The cohort contains {unassigned_count} patients absent from both split files"
        )

    patient_index = cohort.join(fixed_splits, on="patient_id", how="inner")
    train_rows = patient_index.filter(pl.col("split") == "train").sort("patient_id")

    try:
        _, validation_ids = train_test_split(
            train_rows["patient_id"].to_list(),
            test_size=cfg.validation_size,
            random_state=seed,
            shuffle=True,
            stratify=train_rows["label"].to_list(),
        )
    except ValueError as error:
        raise ClinicalDataError(
            "The training split is too small or imbalanced for stratified validation"
        ) from error

    patient_index = (
        patient_index.with_columns(
            pl.when(
                (pl.col("split") == "train")
                & pl.col("patient_id").is_in(validation_ids)
            )
            .then(pl.lit("validation"))
            .otherwise(pl.col("split"))
            .alias("split")
        )
        .sort("patient_id")
        .with_row_index("id")
        .select("id", "patient_id", "label", "anchor_date", "split")
    )

    expected_labels = set(train_rows["label"].to_list())
    for split_name in ("train", "validation"):
        observed_labels = set(
            patient_index.filter(pl.col("split") == split_name)["label"].to_list()
        )
        if observed_labels != expected_labels:
            raise ClinicalDataError(
                f"The {split_name} split does not contain every training label; "
                "increase validation_size or provide more examples per label"
            )

    return patient_index


def summarize_patient_index(patient_index: pl.DataFrame) -> pl.DataFrame:
    """Return aggregate split statistics without exposing patient identifiers."""
    split_order = {"train": 0, "validation": 1, "test": 2}
    return (
        patient_index.group_by("split")
        .agg(
            pl.len().alias("patients"),
            pl.col("label").sum().alias("asd_patients"),
            pl.col("label").mean().alias("asd_prevalence"),
        )
        .with_columns(pl.col("split").replace_strict(split_order).alias("_order"))
        .sort("_order")
        .drop("_order")
    )


def iter_pre_anchor_notes(
    patient_index: pl.DataFrame,
    cfg: ClinicalDatasetConfig,
) -> Iterator[pl.DataFrame]:
    """Yield minimum note fields for cohort patients at or before their anchor."""
    required_index_columns = {"patient_id", "anchor_date"}
    missing_index_columns = required_index_columns - set(patient_index.columns)
    if missing_index_columns:
        raise ClinicalDataError(
            "Patient index is missing required columns: "
            + ", ".join(sorted(missing_index_columns))
        )
    if patient_index.is_empty():
        raise ClinicalDataError("Patient index contains no patients")
    if cfg.sqlite_fetch_size <= 0:
        raise ClinicalDataError("sqlite_fetch_size must be positive")
    if not cfg.notes_db_path.is_file():
        raise ClinicalDataError(
            f"Clinical notes database does not exist: {cfg.notes_db_path}"
        )

    table = _quote_identifier(cfg.notes_table)
    patient_column = _quote_identifier(cfg.note_patient_column)
    date_column = _quote_identifier(cfg.note_date_column)
    text_column = _quote_identifier(cfg.note_text_column)
    cutoff_operator = "<=" if cfg.include_cutoff_date else "<"

    uri = f"{cfg.notes_db_path.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (cfg.notes_table,),
        ).fetchone()
        if table_exists is None:
            raise ClinicalDataError(f"Notes table does not exist: {cfg.notes_table}")

        available_columns = {
            row[1] for row in connection.execute(f"PRAGMA table_info({table})")
        }
        required_note_columns = {
            cfg.note_patient_column,
            cfg.note_date_column,
            cfg.note_text_column,
        }
        missing_note_columns = required_note_columns - available_columns
        if missing_note_columns:
            raise ClinicalDataError(
                f"{cfg.notes_table} is missing required columns: "
                + ", ".join(sorted(missing_note_columns))
            )

        connection.execute(
            "CREATE TEMP TABLE selected_patients "
            "(patient_id TEXT PRIMARY KEY, anchor_date TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO selected_patients(patient_id, anchor_date) VALUES (?, ?)",
            (
                (str(patient_id), anchor_date.isoformat())
                for patient_id, anchor_date in patient_index.select(
                    "patient_id", "anchor_date"
                ).iter_rows()
            ),
        )

        query = f"""
            SELECT
                notes.rowid AS note_rowid,
                notes.{patient_column} AS patient_id,
                DATE(notes.{date_column}) AS note_date,
                notes.{text_column} AS note_text
            FROM {table} AS notes
            INNER JOIN selected_patients AS patients
                ON notes.{patient_column} = patients.patient_id
            WHERE
                notes.{date_column} IS NOT NULL
                AND TRIM(notes.{date_column}) != ''
                AND DATE(notes.{date_column}) {cutoff_operator} DATE(patients.anchor_date)
                AND notes.{text_column} IS NOT NULL
                AND TRIM(notes.{text_column}) != ''
        """
        cursor = connection.execute(query)
        schema = {
            "note_rowid": pl.Int64,
            "patient_id": pl.String,
            "note_date": pl.String,
            "note_text": pl.String,
        }
        while rows := cursor.fetchmany(cfg.sqlite_fetch_size):
            yield pl.DataFrame(rows, schema=schema, orient="row")
    finally:
        connection.close()


def summarize_note_coverage(
    patient_index: pl.DataFrame,
    note_batches: Iterable[pl.DataFrame],
) -> pl.DataFrame:
    """Summarize usable note coverage by split without exposing patient IDs."""
    required_index_columns = {"patient_id", "split"}
    missing_index_columns = required_index_columns - set(patient_index.columns)
    if missing_index_columns:
        raise ClinicalDataError(
            "Patient index is missing required columns: "
            + ", ".join(sorted(missing_index_columns))
        )

    count_frames = []
    for batch in note_batches:
        if "patient_id" not in batch.columns:
            raise ClinicalDataError("Note batch is missing patient_id")
        if not batch.is_empty():
            count_frames.append(
                batch.group_by("patient_id").agg(
                    pl.len().cast(pl.UInt64).alias("note_count")
                )
            )

    if count_frames:
        note_counts = (
            pl.concat(count_frames)
            .group_by("patient_id")
            .agg(pl.col("note_count").sum())
        )
    else:
        note_counts = pl.DataFrame(
            schema={"patient_id": pl.String, "note_count": pl.UInt64}
        )

    unknown_patient_count = note_counts.join(
        patient_index.select("patient_id"), on="patient_id", how="anti"
    ).height
    if unknown_patient_count:
        raise ClinicalDataError(
            f"Note batches contain {unknown_patient_count} patients outside the index"
        )

    split_order = {"train": 0, "validation": 1, "test": 2}
    return (
        patient_index.select("patient_id", "split")
        .join(note_counts, on="patient_id", how="left")
        .with_columns(pl.col("note_count").fill_null(0).cast(pl.UInt64))
        .group_by("split")
        .agg(
            pl.len().alias("patients"),
            (pl.col("note_count") > 0).sum().alias("patients_with_notes"),
            (pl.col("note_count") == 0).sum().alias("patients_without_notes"),
            pl.col("note_count").sum().alias("notes"),
            pl.col("note_count").mean().alias("mean_notes_per_patient"),
        )
        .with_columns(pl.col("split").replace_strict(split_order).alias("_order"))
        .sort("_order")
        .drop("_order")
    )


def make_patient_documents(
    patient_index: pl.DataFrame,
    note_batches: Iterable[pl.DataFrame],
) -> pl.DataFrame:
    """Combine each patient's pre-anchor notes into one IRIS input document."""
    required_index_columns = {"id", "patient_id", "label", "split"}
    missing_index_columns = required_index_columns - set(patient_index.columns)
    if missing_index_columns:
        raise ClinicalDataError(
            "Patient index is missing required columns: "
            + ", ".join(sorted(missing_index_columns))
        )

    required_note_columns = {"note_rowid", "patient_id", "note_date", "note_text"}
    frames = []
    for batch in note_batches:
        missing_note_columns = required_note_columns - set(batch.columns)
        if missing_note_columns:
            raise ClinicalDataError(
                "Note batch is missing required columns: "
                + ", ".join(sorted(missing_note_columns))
            )
        if not batch.is_empty():
            frames.append(batch.select(sorted(required_note_columns)))

    if not frames:
        raise ClinicalDataError("No usable pre-anchor notes were found")

    notes = pl.concat(frames)
    duplicate_note_count = notes.height - notes["note_rowid"].n_unique()
    if duplicate_note_count:
        raise ClinicalDataError(
            f"Clinical notes contain {duplicate_note_count} duplicate SQLite row IDs"
        )

    documents = (
        notes.join(
            patient_index.select("id", "patient_id", "label", "split"),
            on="patient_id",
            how="inner",
        )
        .sort("id", "note_date", "note_rowid")
        .group_by("id", "label", "split", maintain_order=True)
        .agg(pl.col("note_text").str.join("\n\n").alias("text"))
        .select("id", "label", "text", "split")
    )

    missing_document_count = patient_index.join(documents, on="id", how="anti").height
    if missing_document_count:
        raise ClinicalDataError(
            f"{missing_document_count} patients have no usable pre-anchor notes"
        )
    return documents
