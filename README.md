# evigen

This repository contains experimental variants of IRIS for retrieval-based
document classification and interpretable top-k evidence retrieval.

## Projects

### `iris_arxiv`

An arXiv classification pipeline that:

- downloads selected arXiv11 categories;
- tokenizes and chunks full research articles;
- embeds chunks with ModernBERT;
- stores vectors, source paths, and exact source text in LanceDB;
- returns top-k chunks with traceable original text;
- includes local synthetic-data and Duke Compute Cluster workflows.

### `iris_prcc`

A clinical-note adaptation of IRIS that:

- reads a patient cohort and predefined train/test patient IDs;
- reads clinical notes from SQLite;
- excludes notes recorded after each patient's prediction anchor;
- assigns train, validation, and test splits at the patient level;
- stores chunk vectors together with the corresponding source text;
- includes synthetic clinical fixtures for local development.

## Data and privacy

No arXiv corpus, model weights, generated vector databases, or real clinical
data are stored in this repository.

All files under `iris_prcc/tests/fixtures/clinical` are synthetic and contain
no patient information. Real EHR data must remain inside the institutionally
approved computing environment and must never be committed to Git.

DCC and PRCC configuration templates contain placeholders such as
`YOUR_NETID` and `YOUR_PROJECT_ID`. Replace them only in a local checkout for
the target environment, and do not commit the resulting personal values.

## Installation and usage

Each project is independently installable and contains its own environment,
configuration, scripts, README, and license. Enter the relevant project
directory before following its instructions.

## Provenance and licensing

Both variants are derived from the IRIS project:

https://github.com/engelhard-lab/iris

See the `LICENSE` file inside each project directory for licensing terms.
