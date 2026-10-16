"""Inspect initial-query top-k chunks from an existing IRIS vector database."""

from __future__ import annotations

import argparse

import lancedb
import numpy as np

from iris.config import get_config
from iris.dataset import make_loaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config.fake.toml",
        help="Path to the TOML configuration file",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="test",
        help="Patient split to inspect",
    )
    parser.add_argument(
        "--document-id",
        type=int,
        default=None,
        help="Internal IRIS document ID; defaults to the first ID in the split",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config(args.config)

    np.random.seed(cfg.seed)

    db = lancedb.connect(cfg.database.path)
    table = db.open_table(cfg.dataset.table_name)

    vector_type = table.schema.field("vector").type
    cfg.model.d_model = vector_type.list_size

    loaders = make_loaders(cfg=cfg, tbl=table)
    datasets = {
        "train": loaders.train.dataset,
        "validation": loaders.validation.dataset,
        "test": loaders.test.dataset,
    }
    dataset = datasets[args.split]

    if args.document_id is None:
        document_id = int(dataset.ids[0])
    else:
        document_id = args.document_id

    if document_id not in dataset.ids:
        raise ValueError(
            f"Document ID {document_id} is not present in the {args.split} split"
        )

    result = dataset.search_chunks(
    document_id,
    include_text=True,
    )

    if result.is_empty():
        print(f"No chunks found for document ID {document_id}")
        return

    metadata_columns = [
        "id",
        "chunk_id",
        "offset",
        "query_id",
        "rank",
        "_distance",
        "label",
        "split",
    ]

    print("query_source=untrained_vector_index_centroids")
    print(f"split={args.split}")
    print(f"document_id={document_id}")
    print(result.select(metadata_columns))

    text_rows = result.select(
        "query_id",
        "rank",
        "id",
        "chunk_id",
        "chunk_text",
    ).iter_rows(named=True)

    for row in text_rows:
        print()
        print(
            f"query_id={row['query_id']} "
            f"rank={row['rank']} "
            f"id={row['id']} "
            f"chunk_id={row['chunk_id']}"
        )
        print(row["chunk_text"])


if __name__ == "__main__":
    main()