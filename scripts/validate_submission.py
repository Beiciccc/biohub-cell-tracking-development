#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


COLUMNS = [
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
]

INT_COLUMNS = ["id", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


def fail(message: str) -> None:
    raise SystemExit(f"validation failed: {message}")


def validate(path: Path, test_dir: Path | None) -> None:
    df = pd.read_csv(path)
    if list(df.columns) != COLUMNS:
        fail(f"unexpected columns: {list(df.columns)}")
    if len(df) == 0:
        fail("submission is empty")
    if df["id"].tolist() != list(range(len(df))):
        fail("id column must be a continuous zero-based index")
    if set(df["row_type"]) - {"node", "edge"}:
        fail("row_type must be node or edge")
    for col in INT_COLUMNS:
        if not pd.api.types.is_integer_dtype(df[col]):
            fail(f"{col} must be integer typed")

    node_rows = df[df["row_type"] == "node"]
    edge_rows = df[df["row_type"] == "edge"]
    if len(node_rows) == 0:
        fail("no node rows found")

    bad_nodes = node_rows[(node_rows["source_id"] != -1) | (node_rows["target_id"] != -1)]
    if len(bad_nodes):
        fail("node rows must have source_id=target_id=-1")

    bad_edges = edge_rows[
        (edge_rows["node_id"] != -1)
        | (edge_rows["t"] != -1)
        | (edge_rows["z"] != -1)
        | (edge_rows["y"] != -1)
        | (edge_rows["x"] != -1)
    ]
    if len(bad_edges):
        fail("edge rows must have node_id,t,z,y,x all set to -1")

    for dataset, group in df.groupby("dataset"):
        nodes = set(group.loc[group["row_type"] == "node", "node_id"].astype(int))
        if len(nodes) != len(group.loc[group["row_type"] == "node"]):
            fail(f"duplicate node_id in {dataset}")
        for col in ("source_id", "target_id"):
            missing = set(group.loc[group["row_type"] == "edge", col].astype(int)) - nodes
            if missing:
                fail(f"{dataset} has {len(missing)} edge references missing from {col}")

    if test_dir is not None:
        expected = sorted(p.name[:-5] for p in test_dir.iterdir() if p.name.endswith(".zarr"))
        actual = sorted(df["dataset"].unique())
        if expected and expected != actual:
            fail(f"dataset coverage mismatch: expected {len(expected)}, got {len(actual)}")

    print(f"OK: {len(df)} rows, {len(node_rows)} nodes, {len(edge_rows)} edges, {df['dataset'].nunique()} datasets")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--test-dir", type=Path)
    args = parser.parse_args()
    validate(args.submission, args.test_dir)


if __name__ == "__main__":
    main()
