# IRIS clinical EHR adaptation

This project adapts IRIS to build a retrieval-based classifier from
longitudinal clinical notes while preserving patient-level train, validation,
and test separation.

## Installation

```bash
git clone https://github.com/cierasimon/evigen.git
cd evigen/iris_prcc

conda env create -f environment.yml
conda activate iris
pip install -e .
```

## Clinical-data pipeline

The clinical pipeline:

1. reads the cohort CSV;
2. reads the predefined training and test patient IDs;
3. creates a validation split from training patients;
4. reads notes from SQLite;
5. excludes notes recorded after each patient's prediction anchor;
6. combines eligible notes into patient documents;
7. tokenizes and chunks the documents;
8. embeds the chunks and stores them in LanceDB;
9. retains `chunk_id` and `chunk_text` for evidence traceback.

The split is performed at the patient level, so chunks belonging to the same
patient cannot appear in multiple splits.

## Synthetic local test data

The repository contains a generator for synthetic cohort, split, and clinical
note data. The generated records contain no patient information.

Generate 1,200 synthetic patients:

```bash
python scripts/generate_fake_clinical_data.py \
  --output-dir data/fake/clinical \
  --patients 1200
```

For the initial build, set the following values in `config.fake.toml`:

```toml
make_dataset = true
make_database = true
train = false
```

Then build the chunk dataset and vector database:

```bash
PYTHONPATH=src python -m iris \
  --config config.fake.toml
```

After the database has been built, display top-k chunks and their stored source
text:

```bash
PYTHONPATH=src python scripts/show_fake_topk.py \
  --config config.fake.toml \
  --split test
```

## PRCC configuration

`config.prcc.example.toml` documents the expected real-data configuration.
Replace `YOUR_PROJECT_ID` only inside the approved PRCC environment.

Real clinical data, patient identifiers, model outputs, and vector databases
must remain in the approved computing environment and must never be committed
to Git.

All clinical files committed under `tests/fixtures/clinical` are synthetic.
