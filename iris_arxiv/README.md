# IRIS arXiv

This project adapts IRIS for retrieval-based classification of full arXiv
articles while preserving traceability from every retrieved vector to its
original source text.

It currently uses three arXiv11 categories:

- `cs.AI`
- `cs.DS`
- `cs.PL`

## Installation

```bash
git clone https://github.com/cierasimon/evigen.git
cd evigen/iris_arxiv

conda env create -f environment.yml
conda activate iris
pip install -e .
```

## Source-text traceback

Each chunk stored in LanceDB contains:

- `id`: internal document ID;
- `chunk_id`: chunk number within the document;
- `offset`: starting token position;
- `label`: document category;
- `source_path`: path to the original article;
- `chunk_text`: exact text span from the article;
- `vector`: ModernBERT embedding.

`chunk_text` is obtained from tokenizer character offsets when the document is
chunked. It is stored together with the vector, so top-k retrieval can return
the original evidence without decoding token IDs or reopening the source file.

## Local synthetic-data test

Generate 60 synthetic articles:

```bash
python scripts/generate_fake_arxiv.py
```

Build the chunk dataset and LanceDB vector database:

```bash
PYTHONPATH=src python -m iris \
  --config config.arxiv.fake.toml
```

Display top-k chunks with their source paths and original text:

```bash
PYTHONPATH=src python scripts/show_arxiv_topk.py \
  --config config.arxiv.fake.toml \
  --split test
```

## Download the selected real arXiv11 data

```bash
python scripts/download_arxiv11.py \
  --output data/raw/arxiv11/data \
  --dataset-config default
```

The downloader exports each article as a text file and creates
`data/raw/arxiv11/manifest.csv`.

## Duke Compute Cluster

DCC configuration and Slurm templates are provided in:

- `config.arxiv.dcc.build.toml`
- `config.arxiv.dcc.smoke.toml`
- `environment.dcc.yml`
- `requirements.dcc.txt`
- `slurm/`

Replace `YOUR_NETID` with the appropriate DCC NetID in a local checkout before
submitting a job. Do not commit personal paths, downloaded datasets, model
weights, vector databases, or credentials.

The generated datasets, model checkpoints, logs, and vector databases are
excluded from Git.
