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
    if (node_rows[["t", "z", "y", "x"]] < 0).any().any():
        fail("node rows must have non-negative time and coordinates")

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
        dataset_nodes = group.loc[group["row_type"] == "node", ["node_id", "t"]].copy()
        nodes = set(dataset_nodes["node_id"].astype(int))
        if len(nodes) != len(dataset_nodes):
            fail(f"duplicate node_id in {dataset}")
        edges = group.loc[group["row_type"] == "edge", ["source_id", "target_id"]].copy()
        for col in ("source_id", "target_id"):
            missing = set(edges[col].astype(int)) - nodes
            if missing:
                fail(f"{dataset} has {len(missing)} edge references missing from {col}")
        if edges.duplicated().any():
            fail(f"{dataset} has duplicate source-target edges")
        if (edges["source_id"] == edges["target_id"]).any():
            fail(f"{dataset} has self-loop edges")
        if len(edges):
            node_times = dataset_nodes.set_index("node_id")["t"]
            source_times = edges["source_id"].map(node_times)
            target_times = edges["target_id"].map(node_times)
            if not (target_times == source_times + 1).all():
                fail(f"{dataset} has edges that do not connect consecutive frames")
            if (edges.groupby("target_id").size() > 1).any():
                fail(f"{dataset} has nodes with indegree greater than one")
            if (edges.groupby("source_id").size() > 2).any():
                fail(f"{dataset} has nodes with outdegree greater than two")

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
