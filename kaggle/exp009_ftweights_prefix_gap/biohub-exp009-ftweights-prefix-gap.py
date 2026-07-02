from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


COMPETITION = "biohub-cell-tracking-during-development"
COMP_DIR_CANDIDATES = [
    Path(f"/kaggle/input/competitions/{COMPETITION}"),
    Path(f"/kaggle/input/{COMPETITION}"),
]
COMP_DIR = next((path for path in COMP_DIR_CANDIDATES if path.exists()), COMP_DIR_CANDIDATES[0])
TEST_DIR = COMP_DIR / "test"

WORKING_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
REPO_DIR = WORKING_DIR / "cellmot_repo"
SUBMISSION_PATH = WORKING_DIR / "submission.csv"
RUN_STATS_PATH = WORKING_DIR / "run_stats.csv"

METHOD = "finetune_all_public_train"
PRED_METHOD = "ftweights_det0995_adaptivegap_cpu"

DET_THRESHOLD = float(os.environ.get("BIOHUB_DET_THRESHOLD", "0.995"))
UNET_BATCH_SIZE = int(os.environ.get("BIOHUB_UNET_BATCH_SIZE", "4"))
USE_ILP = os.environ.get("BIOHUB_USE_ILP", "1") != "0"
ILP_EDGE_WEIGHT = float(os.environ.get("BIOHUB_ILP_EDGE_WEIGHT", "-1.0"))
ILP_APPEARANCE_WEIGHT = float(os.environ.get("BIOHUB_ILP_APPEARANCE_WEIGHT", "0.1"))
ILP_DISAPPEARANCE_WEIGHT = float(os.environ.get("BIOHUB_ILP_DISAPPEARANCE_WEIGHT", "0.1"))
ILP_DIVISION_WEIGHT = float(os.environ.get("BIOHUB_ILP_DIVISION_WEIGHT", "1.0"))
SLICE = os.environ.get("BIOHUB_SLICE", "").strip()
DEFAULT_GAP_MAX = int(os.environ.get("BIOHUB_GAP_MAX", "1"))
DEFAULT_GAP_DIST_UM = float(os.environ.get("BIOHUB_GAP_DIST_UM", "3.25"))
PREFIX_GAP_POLICY = {
    "44b6": (2, 2.5),
    "6bba": (0, 0.0),
}
SIZE_GATED_GAP_POLICY = [
    {
        "prefix": "6bba",
        "min_raw_nodes": 10000,
        "max_gap": 2,
        "gap_dist_um": 2.75,
        "name": "6bba_nodes_gt10000_gap2_dist2p75",
    }
]
VOXEL_SCALE_UM = np.array([1.625, 0.40625, 0.40625], dtype=float)

print("Biohub fine-tuned CellMOT weight inference")
print("COMP_DIR:", COMP_DIR, "exists:", COMP_DIR.exists())
print("TEST_DIR:", TEST_DIR, "exists:", TEST_DIR.exists())
print(
    "Config:",
    json.dumps(
        {
            "method": METHOD,
            "pred_method": PRED_METHOD,
            "det_threshold": DET_THRESHOLD,
            "unet_batch_size": UNET_BATCH_SIZE,
            "use_ilp": USE_ILP,
            "ilp_edge_weight": ILP_EDGE_WEIGHT,
            "ilp_appearance_weight": ILP_APPEARANCE_WEIGHT,
            "ilp_disappearance_weight": ILP_DISAPPEARANCE_WEIGHT,
            "ilp_division_weight": ILP_DIVISION_WEIGHT,
            "slice": SLICE,
            "default_gap_max": DEFAULT_GAP_MAX,
            "default_gap_dist_um": DEFAULT_GAP_DIST_UM,
            "prefix_gap_policy": PREFIX_GAP_POLICY,
            "size_gated_gap_policy": SIZE_GATED_GAP_POLICY,
        },
        indent=2,
        sort_keys=True,
    ),
)


def find_artifacts_root() -> Path:
    explicit = os.environ.get("BIOHUB_ARTIFACTS", "").strip()
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))

    candidates.extend(
        [
            Path("/kaggle/input/datasets/thibautgoldsborough/cellmot-baseline-artifacts/cellmot-baseline-artifacts"),
            Path("/kaggle/input/cellmot-baseline-artifacts"),
            Path("/kaggle/input/cellmot-baseline-artifacts/cellmot-baseline-artifacts"),
        ]
    )

    input_root = Path("/kaggle/input")
    if input_root.exists():
        for child in input_root.iterdir():
            if not child.is_dir():
                continue
            if "cellmot-baseline-artifacts" in child.name:
                candidates.append(child)
                candidates.append(child / "cellmot-baseline-artifacts")
            if child.name == "datasets":
                candidates.append(
                    child
                    / "thibautgoldsborough"
                    / "cellmot-baseline-artifacts"
                    / "cellmot-baseline-artifacts"
                )

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "repo").exists() and (candidate / "weights").exists() and (candidate / "wheels").exists():
            return candidate

    searched = "\n".join(str(path) for path in seen)
    raise FileNotFoundError(f"Could not find cellmot-baseline-artifacts. Searched:\n{searched}")


def find_finetuned_weight() -> Path:
    candidates = [
        Path("/kaggle/input/biohub-finetuned-cellmot-weights/edge_predictor_best.pth"),
        Path("/kaggle/input/datasets/ladyfaye/biohub-finetuned-cellmot-weights/edge_predictor_best.pth"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    input_root = Path("/kaggle/input")
    if input_root.exists():
        for path in input_root.rglob("edge_predictor_best.pth"):
            if "biohub-finetuned-cellmot-weights" in str(path):
                return path

    raise FileNotFoundError("Could not find public fine-tuned edge_predictor_best.pth")


def list_test_stems() -> list[str]:
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test directory does not exist: {TEST_DIR}")
    stems = sorted(path.name[:-5] for path in TEST_DIR.iterdir() if path.name.endswith(".zarr"))
    if not stems:
        raise FileNotFoundError(f"No test .zarr files found in {TEST_DIR}")
    return stems


def close_gap_dataframes(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    max_gap: int,
    gap_dist_um: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    if max_gap <= 0 or len(nodes) == 0 or len(edges) == 0:
        return nodes, edges, {"gap_bridges": 0, "gap_added_nodes": 0, "gap_added_edges": 0}

    coords = {
        int(row.node_id): (int(row.t), float(row.z), float(row.y), float(row.x))
        for row in nodes.itertuples(index=False)
    }
    has_out = set(edges["source_id"].astype(int))
    has_in = set(edges["target_id"].astype(int))

    ends_by_t: dict[int, list[int]] = {}
    starts_by_t: dict[int, list[int]] = {}
    for node_id, (t, *_xyz) in coords.items():
        if node_id not in has_out:
            ends_by_t.setdefault(t, []).append(node_id)
        if node_id not in has_in:
            starts_by_t.setdefault(t, []).append(node_id)

    next_id = int(nodes["node_id"].max()) + 1
    new_nodes: list[dict[str, float | int]] = []
    new_edges: list[dict[str, int]] = []
    bridges = 0

    for gap in range(1, max_gap + 1):
        for t, ends in sorted(ends_by_t.items()):
            starts = starts_by_t.get(t + gap + 1, [])
            if not ends or not starts:
                continue
            end_xyz = np.array(
                [[coords[node_id][1], coords[node_id][2], coords[node_id][3]] for node_id in ends],
                dtype=float,
            ) * VOXEL_SCALE_UM
            start_xyz = np.array(
                [[coords[node_id][1], coords[node_id][2], coords[node_id][3]] for node_id in starts],
                dtype=float,
            ) * VOXEL_SCALE_UM
            d = np.sqrt(((end_xyz[:, None, :] - start_xyz[None, :, :]) ** 2).sum(axis=2))
            threshold = gap_dist_um * (gap + 1)
            big = threshold * 1000.0 + 1.0
            cost = np.where(d <= threshold, d, big)
            row_ind, col_ind = linear_sum_assignment(cost)

            used_starts: set[int] = set()
            for r, c in zip(row_ind, col_ind):
                if d[r, c] > threshold:
                    continue
                end_id = int(ends[int(r)])
                start_id = int(starts[int(c)])
                if end_id in has_out or start_id in used_starts:
                    continue
                te, ze, ye, xe = coords[end_id]
                _ts, zs, ys, xs = coords[start_id]
                previous = end_id
                for k in range(1, gap + 1):
                    frac = k / (gap + 1)
                    node_id = next_id
                    next_id += 1
                    new_nodes.append(
                        {
                            "node_id": node_id,
                            "t": te + k,
                            "z": ze + (zs - ze) * frac,
                            "y": ye + (ys - ye) * frac,
                            "x": xe + (xs - xe) * frac,
                        }
                    )
                    new_edges.append({"source_id": previous, "target_id": node_id})
                    previous = node_id
                new_edges.append({"source_id": previous, "target_id": start_id})
                has_out.add(end_id)
                used_starts.add(start_id)
                bridges += 1

    if not new_nodes:
        return nodes, edges, {"gap_bridges": 0, "gap_added_nodes": 0, "gap_added_edges": 0}

    closed_nodes = pd.concat([nodes, pd.DataFrame(new_nodes)], ignore_index=True)
    closed_edges = pd.concat([edges, pd.DataFrame(new_edges)], ignore_index=True)
    return closed_nodes, closed_edges, {
        "gap_bridges": int(bridges),
        "gap_added_nodes": int(len(new_nodes)),
        "gap_added_edges": int(len(new_edges)),
    }


def gap_params_for_dataset(dataset: str, raw_node_count: int) -> tuple[int, float, str]:
    for rule in SIZE_GATED_GAP_POLICY:
        prefix = str(rule["prefix"])
        if dataset.startswith(prefix) and raw_node_count > int(rule["min_raw_nodes"]):
            return int(rule["max_gap"]), float(rule["gap_dist_um"]), str(rule["name"])
    for prefix, (max_gap, gap_dist_um) in PREFIX_GAP_POLICY.items():
        if dataset.startswith(prefix):
            return int(max_gap), float(gap_dist_um), prefix
    return DEFAULT_GAP_MAX, DEFAULT_GAP_DIST_UM, "default"


def graph_from_geff(path: Path):
    import tracksdata as td

    graph = td.graph.IndexedRXGraph.from_geff(path)
    return graph[0] if isinstance(graph, tuple) else graph


ARTIFACTS = find_artifacts_root()
FT_WEIGHT = find_finetuned_weight()
print("ARTIFACTS:", ARTIFACTS)
print("FT_WEIGHT:", FT_WEIGHT)

install_cmd = [
    sys.executable,
    "-m",
    "pip",
    "install",
    "--no-index",
    "--find-links",
    str(ARTIFACTS / "wheels"),
    "tracksdata",
    "zarr>=3.0.10",
    "pyscipopt",
]
print(" ".join(install_cmd))
subprocess.run(install_cmd, check=True)

if REPO_DIR.exists():
    shutil.rmtree(REPO_DIR)
shutil.copytree(ARTIFACTS / "repo", REPO_DIR)
shutil.copytree(ARTIFACTS / "weights", REPO_DIR / "weights", dirs_exist_ok=True)

weights_dir = REPO_DIR / "weights" / METHOD / "split_0"
weights_dir.mkdir(parents=True, exist_ok=True)
weights_path = weights_dir / "edge_predictor_best.pth"
shutil.copy2(FT_WEIGHT, weights_path)
sys.path.insert(0, str(REPO_DIR / "src"))

test_stems = list_test_stems()
print(f"Found {len(test_stems)} test videos")
print(test_stems[:10])

splits_path = REPO_DIR / "kaggle_test_splits.json"
splits_path.write_text(json.dumps([{"split": 0, "train": [], "test": test_stems}], indent=2))

predict_cmd = [
    sys.executable,
    "scripts/predict_unet_transformer.py",
    "--data-dir",
    str(TEST_DIR),
    "--splits",
    str(splits_path.name),
    "--split",
    "0",
    "--method",
    PRED_METHOD,
    "--weights",
    str(weights_path),
    "--unet-batch-size",
    str(UNET_BATCH_SIZE),
    "--det-threshold",
    str(DET_THRESHOLD),
    "--ilp-edge-weight",
    str(ILP_EDGE_WEIGHT),
    "--ilp-appearance-weight",
    str(ILP_APPEARANCE_WEIGHT),
    "--ilp-disappearance-weight",
    str(ILP_DISAPPEARANCE_WEIGHT),
    "--ilp-division-weight",
    str(ILP_DIVISION_WEIGHT),
]
if USE_ILP:
    predict_cmd.append("--use-ilp")
if SLICE:
    predict_cmd.extend(["--slice", SLICE])

start_time = time.time()
print(" ".join(predict_cmd))
subprocess.run(predict_cmd, cwd=REPO_DIR, env={**os.environ, "PYTHONPATH": "src"}, check=True)
predict_seconds = time.time() - start_time
print(f"Prediction completed in {predict_seconds / 60:.2f} minutes")

SUBMISSION_COLUMNS = ["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]

geffs = sorted((REPO_DIR / "predictions").glob(f"*/{PRED_METHOD}/split_0/*.geff"))
print(f"Found {len(geffs)} prediction graphs")
if len(geffs) != len(test_stems):
    found = {path.stem for path in geffs}
    missing = sorted(set(test_stems) - found)
    raise RuntimeError(f"Expected {len(test_stems)} graphs, found {len(geffs)}. Missing: {missing[:10]}")

rows: list[dict[str, object]] = []
stats_rows: list[dict[str, object]] = []

for geff_path in geffs:
    dataset = geff_path.stem
    graph = graph_from_geff(geff_path)
    node_records: list[dict[str, object]] = []
    edge_records: list[dict[str, object]] = []

    for row in graph.node_attrs().iter_rows(named=True):
        node_records.append(
            {
                "node_id": int(row["node_id"]),
                "t": int(row["t"]),
                "z": float(row["z"]),
                "y": float(row["y"]),
                "x": float(row["x"]),
            }
        )

    for row in graph.edge_attrs().iter_rows(named=True):
        edge_records.append(
            {
                "source_id": int(row["source_id"]),
                "target_id": int(row["target_id"]),
            }
        )

    node_df = pd.DataFrame(node_records, columns=["node_id", "t", "z", "y", "x"])
    edge_df = pd.DataFrame(edge_records, columns=["source_id", "target_id"])
    raw_node_count = len(node_df)
    raw_edge_count = len(edge_df)
    gap_max, gap_dist_um, gap_policy = gap_params_for_dataset(dataset, raw_node_count)
    node_df, edge_df, gap_stats = close_gap_dataframes(
        node_df,
        edge_df,
        max_gap=gap_max,
        gap_dist_um=gap_dist_um,
    )

    node_count = len(node_df)
    edge_count = len(edge_df)
    division_sources: dict[int, int] = {}

    for row in node_df.itertuples(index=False):
        rows.append(
            {
                "dataset": dataset,
                "row_type": "node",
                "node_id": int(row.node_id),
                "t": int(row.t),
                "z": int(round(float(row.z))),
                "y": int(round(float(row.y))),
                "x": int(round(float(row.x))),
                "source_id": -1,
                "target_id": -1,
            }
        )

    for row in edge_df.itertuples(index=False):
        source_id = int(row.source_id)
        target_id = int(row.target_id)
        rows.append(
            {
                "dataset": dataset,
                "row_type": "edge",
                "node_id": -1,
                "t": -1,
                "z": -1,
                "y": -1,
                "x": -1,
                "source_id": source_id,
                "target_id": target_id,
            }
        )
        division_sources[source_id] = division_sources.get(source_id, 0) + 1

    stats_rows.append(
        {
            "dataset": dataset,
            "raw_nodes": raw_node_count,
            "raw_edges": raw_edge_count,
            "nodes": node_count,
            "edges": edge_count,
            "division_like_sources": sum(1 for count in division_sources.values() if count >= 2),
            "gap_policy": gap_policy,
            "gap_max": gap_max,
            "gap_dist_um": gap_dist_um,
            **gap_stats,
        }
    )

submission = pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)
submission.index.name = "id"

expected_datasets = set(test_stems)
actual_datasets = set(submission["dataset"].unique())
missing_datasets = sorted(expected_datasets - actual_datasets)
if missing_datasets:
    raise AssertionError(f"Missing datasets in submission: {missing_datasets[:10]}")

nodes = submission[submission["row_type"] == "node"]
edges = submission[submission["row_type"] == "edge"]
assert len(nodes) > 0, "No node rows produced"
assert set(submission["row_type"].unique()).issubset({"node", "edge"}), "Invalid row_type"
assert not submission.isna().any().any(), "NaN values found"
assert (nodes[["node_id", "t", "z", "y", "x"]] >= 0).all().all(), "Node fields must be non-negative"
assert (nodes[["source_id", "target_id"]] == -1).all().all(), "Node edge sentinels must be -1"
if len(edges):
    assert (edges[["node_id", "t", "z", "y", "x"]] == -1).all().all(), "Edge sentinel fields must be -1"
    assert (edges[["source_id", "target_id"]] >= 0).all().all(), "Edge endpoints must be non-negative node ids"

for dataset, group in submission.groupby("dataset"):
    ds_nodes = group[group["row_type"] == "node"]
    ds_edges = group[group["row_type"] == "edge"]
    node_ids = set(ds_nodes["node_id"])
    assert ds_nodes["node_id"].is_unique, f"Duplicate node_id in {dataset}"
    assert ds_edges["source_id"].isin(node_ids).all(), f"Dangling source_id in {dataset}"
    assert ds_edges["target_id"].isin(node_ids).all(), f"Dangling target_id in {dataset}"

submission.to_csv(SUBMISSION_PATH)
stats = pd.DataFrame(stats_rows).sort_values("dataset").reset_index(drop=True)
stats["predict_minutes_total"] = predict_seconds / 60.0
stats["det_threshold"] = DET_THRESHOLD
stats.to_csv(RUN_STATS_PATH, index=False)

print(f"Wrote {SUBMISSION_PATH} with {len(submission):,} rows")
print(f"Node rows: {len(nodes):,} | edge rows: {len(edges):,}")
print(f"Wrote {RUN_STATS_PATH}")
print(stats.to_string(index=False))

if os.environ.get("BIOHUB_KEEP_INTERMEDIATES", "0") == "0" and REPO_DIR.exists():
    shutil.rmtree(REPO_DIR)
    print(f"Removed intermediate repo/output tree: {REPO_DIR}")
