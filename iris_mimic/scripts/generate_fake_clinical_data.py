"""Generate deterministic synthetic clinical data for local IRIS tests."""

from __future__ import annotations

import argparse
import csv
import sqlite3
from datetime import date, timedelta
from pathlib import Path

COHORT_COLUMNS = [
    "PATID",
    "ENCOUNTERID",
    "ENC_DATE_D",
    "BIRTH_DATE_D",
    "BIRTH_YEAR",
    "SEX",
    "race_eth",
    "language",
    "payer",
    "birthinfo_flag",
    "BABY_WEIGHT_GMS",
    "BABY_ICU_FLAG",
    "GA_WEEK_DAYS",
    "DELIVERY_TYPE",
    "ADDRESS_COUNTY",
    "DX_DATE_1st",
    "WC_AGE_Days",
    "LAST_DATE_D",
    "LAST_AGE_MONTH",
    "Time2Date",
    "Time2ASD_from_birth",
    "Time2ASD_from_WC",
    "ASD",
]

NOTE_COLUMNS = [
    "Pat_ID",
    "Pat_Name",
    "MRN",
    "Pat_ENC_CSN_ID",
    "ENC_Type",
    "ENC_Begin_DTTM",
    "ENC_End_DTTM",
    "Contact_Date",
    "Pat_Status",
    "ProviderID",
    "PROVIDER_Name",
    "Provider_Type",
    "Title",
    "Department_Name",
    "Specialty",
    "Note_ID",
    "Note_CSN_ID",
    "Note_Status",
    "Note_Type",
    "Note_Purpose",
    "Note_Text",
]

NOTE_OFFSETS_DAYS = (-240, -180, -120, -60, 0, 30)
TRAIN_BIRTH_DAYS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28)
RACES = ("NH White", "NH Black", "NH Asian", "Hispanics", "Other")
LANGUAGES = ("English", "Spanish", "Arabic", "Other")
PAYERS = ("Commercial", "Medicaid", "Medicare", "Other Government", "Uninsured")
COUNTIES = ("DURHAM", "WAKE", "ORANGE", "GRANVILLE", "PERSON")
SPECIALTIES = ("Pediatrics", "Primary Care", "Family Medicine")
DELIVERY_TYPES = ("Vaginal, Spontaneous", "C-Section", "Unknown")
NOTE_WORDS = (
    "synthetic",
    "record",
    "routine",
    "visit",
    "history",
    "reviewed",
    "family",
    "reports",
    "sleep",
    "feeding",
    "growth",
    "development",
    "activity",
    "assessment",
    "plan",
    "followup",
    "care",
    "discussion",
    "screening",
    "education",
    "normal",
    "stable",
    "monitor",
    "return",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/fake/clinical"),
        help="Directory for the generated CSV and SQLite files",
    )
    parser.add_argument(
        "--patients",
        type=int,
        default=1_200,
        help="Number of synthetic patients to generate",
    )
    return parser.parse_args()


def make_note_text(patient_index: int, note_index: int, words: int = 48) -> str:
    start = (patient_index * 7 + note_index * 5) % len(NOTE_WORDS)
    tokens = [NOTE_WORDS[(start + offset) % len(NOTE_WORDS)] for offset in range(words)]
    return "SYNTHETIC RECORD. " + " ".join(tokens) + "."


def build_rows(patient_count: int) -> tuple[list[dict[str, object]], list[str], list[str], list[tuple[object, ...]]]:
    if patient_count < 40:
        raise ValueError("--patients must be at least 40 so every split contains both labels")

    cohort_rows: list[dict[str, object]] = []
    train_ids: list[str] = []
    test_ids: list[str] = []
    note_rows: list[tuple[object, ...]] = []
    censor_date = date(2024, 12, 31)

    for patient_index in range(patient_count):
        patid = str(9_100_001 + patient_index)
        is_test = patient_index % 4 == 0
        asd = int(patient_index % 5 == 0)
        birth_year = 2016 + patient_index % 7
        birth_month = 1 + patient_index * 7 % 12
        if is_test:
            birth_day = 10 + patient_index % 8
            test_ids.append(patid)
        else:
            birth_day = TRAIN_BIRTH_DAYS[patient_index % len(TRAIN_BIRTH_DAYS)]
            train_ids.append(patid)

        birth_date = date(birth_year, birth_month, birth_day)
        wc_age_days = 300 + patient_index % 91
        anchor_date = birth_date + timedelta(days=wc_age_days)
        if asd:
            outcome_date = min(
                anchor_date + timedelta(days=365 + patient_index % 365),
                censor_date,
            )
            dx_date = outcome_date.isoformat()
        else:
            outcome_date = censor_date
            dx_date = ""

        cohort_rows.append(
            {
                "PATID": patid,
                "ENCOUNTERID": str(81_000_001 + patient_index),
                "ENC_DATE_D": anchor_date.isoformat(),
                "BIRTH_DATE_D": birth_date.isoformat(),
                "BIRTH_YEAR": birth_year,
                "SEX": "F" if patient_index % 2 == 0 else "M",
                "race_eth": RACES[patient_index % len(RACES)],
                "language": LANGUAGES[patient_index % len(LANGUAGES)],
                "payer": PAYERS[patient_index % len(PAYERS)],
                "birthinfo_flag": int(patient_index % 6 != 0),
                "BABY_WEIGHT_GMS": 2600 + patient_index * 37 % 1800,
                "BABY_ICU_FLAG": int(patient_index % 11 == 0),
                "GA_WEEK_DAYS": 252 + patient_index % 36,
                "DELIVERY_TYPE": DELIVERY_TYPES[patient_index % len(DELIVERY_TYPES)],
                "ADDRESS_COUNTY": COUNTIES[patient_index % len(COUNTIES)],
                "DX_DATE_1st": dx_date,
                "WC_AGE_Days": wc_age_days,
                "LAST_DATE_D": censor_date.isoformat(),
                "LAST_AGE_MONTH": round((censor_date - birth_date).days / 30.44, 6),
                "Time2Date": outcome_date.isoformat(),
                "Time2ASD_from_birth": (outcome_date - birth_date).days,
                "Time2ASD_from_WC": (outcome_date - anchor_date).days,
                "ASD": asd,
            }
        )

        for note_index, offset_days in enumerate(NOTE_OFFSETS_DAYS):
            note_date = anchor_date + timedelta(days=offset_days)
            encounter_number = 82_000_000 + patient_index * 10 + note_index
            begin = f"{note_date.isoformat()} 09:00:00"
            end = f"{note_date.isoformat()} 09:30:00"
            note_rows.append(
                (
                    patid,
                    f"SYNTHETIC_PATIENT_{patient_index:04d}",
                    f"SYNTHETIC_MRN_{patient_index:04d}",
                    encounter_number,
                    "Office Visit",
                    begin,
                    end,
                    note_date.isoformat(),
                    "Active",
                    f"SYNTHETIC_PROVIDER_{patient_index % 8:02d}",
                    f"Synthetic Provider {patient_index % 8:02d}",
                    "Clinician",
                    "Routine Visit",
                    "Synthetic Department",
                    SPECIALTIES[patient_index % len(SPECIALTIES)],
                    f"SYNTHETIC_NOTE_{patient_index:04d}_{note_index:02d}",
                    83_000_000 + patient_index * 10 + note_index,
                    "Signed",
                    "Progress Notes",
                    "Synthetic testing",
                    make_note_text(patient_index, note_index),
                )
            )

    return cohort_rows, train_ids, test_ids, note_rows


def write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_split(path: Path, patient_ids: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_ALL)
        writer.writerow(["PATID"])
        writer.writerows([[patient_id] for patient_id in patient_ids])


def write_notes(path: Path, rows: list[tuple[object, ...]]) -> None:
    path.unlink(missing_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE clinical_notes (
                Pat_ID TEXT,
                Pat_Name TEXT,
                MRN TEXT,
                Pat_ENC_CSN_ID BIGINT,
                ENC_Type TEXT,
                ENC_Begin_DTTM TEXT,
                ENC_End_DTTM TEXT,
                Contact_Date TEXT,
                Pat_Status TEXT,
                ProviderID TEXT,
                PROVIDER_Name TEXT,
                Provider_Type TEXT,
                Title TEXT,
                Department_Name TEXT,
                Specialty TEXT,
                Note_ID TEXT,
                Note_CSN_ID BIGINT,
                Note_Status TEXT,
                Note_Type TEXT,
                Note_Purpose TEXT,
                Note_Text TEXT
            )
            """
        )
        placeholders = ", ".join("?" for _ in NOTE_COLUMNS)
        connection.executemany(
            f"INSERT INTO clinical_notes VALUES ({placeholders})",
            rows,
        )
        connection.execute("CREATE INDEX clinical_notes_patient_idx ON clinical_notes(Pat_ID)")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cohort_rows, train_ids, test_ids, note_rows = build_rows(args.patients)

    write_csv(output_dir / "cohort_wc_asd.csv", COHORT_COLUMNS, cohort_rows)
    write_split(output_dir / "train_patid.csv", train_ids)
    write_split(output_dir / "test_patid.csv", test_ids)
    write_notes(output_dir / "clinical_notes.sqlite", note_rows)

    asd_count = sum(int(row["ASD"]) for row in cohort_rows)
    eligible_notes = args.patients * sum(offset <= 0 for offset in NOTE_OFFSETS_DAYS)
    print(f"output_dir={output_dir}")
    print(f"patients={args.patients} train={len(train_ids)} test={len(test_ids)} asd={asd_count}")
    print(f"notes={len(note_rows)} eligible_pre_anchor={eligible_notes}")


if __name__ == "__main__":
    main()
