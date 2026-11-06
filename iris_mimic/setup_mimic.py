
from pathlib import Path

import re

import tomli as tomllib



p = Path("config.mimic.toml")

if not p.exists():

    p.write_text(Path("config.prcc.example.toml").read_text())



s = p.read_text()



def set_key(section, key, value):

    global s

    pattern = rf'(?ms)(^\[{re.escape(section)}\]\s*\n)(.*?)(?=^\[|\Z)'

    match = re.search(pattern, s)

    if not match:

        raise RuntimeError(f"Missing section: {section}")

    body = match.group(2)

    line = f'{key} = {value}'

    body, n = re.subn(

        rf'(?m)^{re.escape(key)}\s*=.*$',

        lambda _: line,

        body

    )

    if n != 1:

        raise RuntimeError(f"Missing or duplicate key: {section}.{key}")

    s = s[:match.start(2)] + body + s[match.end(2):]



def set_top(key, value):

    global s

    prefix, rest = s.split("[dataset]", 1)

    prefix, n = re.subn(

        rf'(?m)^{re.escape(key)}\s*=.*$',

        lambda _: f"{key} = {value}",

        prefix

    )

    if n != 1:

        raise RuntimeError(f"Missing key: {key}")

    s = prefix + "[dataset]" + rest



for key, value in {

    "make_dataset": "true",

    "make_database": "true",

    "train": "true",

    "evaluate": "false",

    "tune": "false",

    "fast_dev_run": "false",

    "distributed": "false",

}.items():

    set_top(key, value)



for key, value in {

    "source": '"clinical_notes"',

    "name": '"mimic_htn_pilot"',

    "table_name": '"mimic_htn_pilot"',

    "out_path": '"data/mimic/processed/"',

}.items():

    set_key("dataset", key, value)



for key, value in {

    "cohort_path": '"data/mimic/cohort.csv"',

    "train_ids_path": '"data/mimic/train_patids.csv"',

    "test_ids_path": '"data/mimic/test_patids.csv"',

    "notes_db_path": '"data/mimic/clinical_notes.sqlite"',

    "notes_table": '"clinical_notes"',

    "patient_column": '"PATID"',

    "label_column": '"HTN"',

    "cutoff_column": '"ANCHOR_DATE"',

    "note_patient_column": '"patient_id"',

    "note_date_column": '"note_date"',

    "note_text_column": '"note_text"',

}.items():

    set_key("clinical", key, value)



set_key("database", "path", '"data/mimic/db/"')

set_key("model", "query_ini_random", "true")

set_key("trainer", "max_epochs", "3")

set_key("trainer", "checkpoint", '"data/mimic/baseline.pt"')

set_key("dataloader", "batch_size", "8")



p.write_text(s)

tomllib.loads(s)



for f in (

    "data/mimic/cohort.csv",

    "data/mimic/train_patids.csv",

    "data/mimic/test_patids.csv",

    "data/mimic/clinical_notes.sqlite",

):

    if not Path(f).is_file():

        raise RuntimeError(f"Missing input: {f}")



print("MIMIC config and inputs verified")
