
from __future__ import annotations

import gc
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

try:
    import zarr
    ZARR_AVAILABLE = True
except Exception:
    zarr = None
    ZARR_AVAILABLE = False

try:
    import blosc2
    BLOSC2_AVAILABLE = True
except Exception:
    blosc2 = None
    BLOSC2_AVAILABLE = False

try:
    from skimage.feature import peak_local_max
    SKIMAGE_AVAILABLE = True
except Exception:
    peak_local_max = None
    SKIMAGE_AVAILABLE = False

SEED = 42
np.random.seed(SEED)

COMPETITION = "biohub-cell-tracking-during-development"
COMP_DIR_CANDIDATES = [
    Path(f"/kaggle/input/{COMPETITION}"),
    Path(f"/kaggle/input/competitions/{COMPETITION}"),
]
COMP_DIR = next((p for p in COMP_DIR_CANDIDATES if p.exists()), COMP_DIR_CANDIDATES[0])
TRAIN_DIR = COMP_DIR / "train"
TEST_DIR = COMP_DIR / "test"
OUTPUT_PATH = Path("/kaggle/working/submission.csv")
RUN_SUMMARY_JSON = Path("/kaggle/working/run_summary.json")
RUN_SUMMARY_MD = Path("/kaggle/working/run_summary.md")

SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)  # z, y, x in micrometers per voxel

PRESET = os.environ.get("BIOHUB_PRESET", "safe_plus").strip().lower()

BASE_CFG = {
    "experiment": "032_safe_division_recovery_plus",
    "base_source": "lb-0-835-biohub-rule-based-baseline.ipynb",
    "score_up_axis": "safe_division_recovery_plus",
    "detector": "blob",
    "dog_scales": [(1.5, 4.0), (2.2, 5.5)],
    "rel_threshold": 0.045,
    "min_distance_um": 4.0,
    "max_peaks": 40000,
    "max_link_um": 8.0,
    "close_gaps": True,
    "max_gap": 1,
    "gap_dist_um": 6.0,
    "allow_divisions": False,
    "safe_divisions": True,
    "safe_division_max_um": 5.0,
    "safe_division_sibling_max_um": 8.0,
    "safe_division_existing_child_max_um": 8.5,
    "safe_division_frame_frac_cap": 0.012,
    "safe_division_global_frac_cap": 0.006,
    "debug_max_datasets": int(os.environ.get("BIOHUB_DEBUG_MAX_DATASETS", "0")),
    "debug_max_t": int(os.environ.get("BIOHUB_DEBUG_MAX_T", "0")),
}

if PRESET == "lb839":
    CFG = dict(BASE_CFG)
elif PRESET == "safe_plus":
    CFG = dict(BASE_CFG)
    CFG.update({
        "experiment": "032_safe_plus",
        "safe_division_max_um": 5.25,
        "safe_division_sibling_max_um": 8.5,
        "safe_division_existing_child_max_um": 9.0,
        "safe_division_frame_frac_cap": 0.014,
        "safe_division_global_frac_cap": 0.007,
    })
elif PRESET == "safe_tiny":
    CFG = dict(BASE_CFG)
    CFG.update({
        "experiment": "032_safe_tiny",
        "safe_division_max_um": 5.1,
        "safe_division_sibling_max_um": 8.25,
        "safe_division_existing_child_max_um": 8.75,
        "safe_division_frame_frac_cap": 0.013,
        "safe_division_global_frac_cap": 0.0065,
    })
else:
    raise ValueError(f"Unknown BIOHUB_PRESET={PRESET!r}; use safe_plus, safe_tiny, or lb839")

print("032_safe_division_recovery_plus")
print(f"COMP_DIR : {COMP_DIR} (exists={COMP_DIR.exists()})")
print(f"TEST_DIR : {TEST_DIR} (exists={TEST_DIR.exists()})")
print(f"TRAIN_DIR: {TRAIN_DIR} (exists={TRAIN_DIR.exists()})")
print(f"zarr={ZARR_AVAILABLE} | blosc2={BLOSC2_AVAILABLE} | skimage={SKIMAGE_AVAILABLE}")
print("Preset:", PRESET)
print("Config:")
print(json.dumps(CFG, indent=2))



def list_dataset_names(data_dir: Path) -> List[str]:
    names = sorted(p.name[:-5] for p in data_dir.iterdir() if p.name.endswith(".zarr"))
    if CFG["debug_max_datasets"] > 0:
        names = names[:CFG["debug_max_datasets"]]
    if not names:
        raise FileNotFoundError(f"No .zarr datasets found in {data_dir}")
    return names


def read_array_meta(zarr_path: Path) -> Tuple[Path, Tuple[int, ...], np.dtype, Tuple[int, ...]]:
    candidates = [zarr_path / "0" / "zarr.json", zarr_path / "0" / ".zarray"]
    candidates += list(zarr_path.rglob("zarr.json"))[:16]
    candidates += list(zarr_path.rglob(".zarray"))[:16]
    seen = set()
    for meta_path in candidates:
        if meta_path in seen or not meta_path.exists():
            continue
        seen.add(meta_path)
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        shape = tuple(meta.get("shape", ()))
        if len(shape) != 4:
            continue
        dtype = np.dtype(meta.get("data_type", meta.get("dtype")))
        if "chunk_grid" in meta:
            chunk_shape = tuple(meta["chunk_grid"]["configuration"]["chunk_shape"])
        else:
            chunk_shape = tuple(meta.get("chunks", shape))
        return meta_path.parent, shape, dtype, chunk_shape
    raise FileNotFoundError(f"Could not find 4D zarr metadata under {zarr_path}")


def _try_decompress_chunk(raw: bytes, dtype: np.dtype, shape: Tuple[int, ...]) -> np.ndarray:
    if BLOSC2_AVAILABLE:
        try:
            out = blosc2.decompress(raw)
            arr = np.frombuffer(out, dtype=dtype).reshape(shape)
            return arr
        except Exception:
            pass
    arr = np.frombuffer(raw, dtype=dtype)
    if arr.size == int(np.prod(shape)):
        return arr.reshape(shape)
    raise RuntimeError("Could not decode zarr chunk; add zarr/numcodecs support or use Kaggle default environment.")


def load_volume(zarr_path: Path, t: int) -> np.ndarray:
    """Load a single timepoint as a (Z, Y, X) uint16/float array."""
    if ZARR_AVAILABLE:
        try:
            arr = zarr.open(str(zarr_path / "0"), mode="r")
            return np.asarray(arr[t])
        except Exception:
            pass

    array_dir, shape, dtype, chunk_shape = read_array_meta(zarr_path)
    _, Z, Y, X = shape
    chunk_path = array_dir / "c" / str(t) / "0" / "0" / "0"
    if not chunk_path.exists():
        chunk_path = array_dir / str(t) / "0" / "0" / "0"
    if not chunk_path.exists():
        raise FileNotFoundError(f"Chunk not found for t={t}: {zarr_path}")
    raw = chunk_path.read_bytes()
    arr = _try_decompress_chunk(raw, dtype, tuple(chunk_shape))
    return np.asarray(arr).reshape(chunk_shape)[0]


def physical_nms(coords_zyx: np.ndarray, scores: np.ndarray, radius_um: float) -> Tuple[np.ndarray, np.ndarray]:
    if len(coords_zyx) == 0:
        return coords_zyx.astype(np.int32), scores.astype(np.float32)
    order = np.argsort(-scores)
    pts = coords_zyx.astype(np.float64) * SCALE[None, :]
    keep = []
    suppressed = np.zeros(len(coords_zyx), dtype=bool)
    tree = cKDTree(pts)
    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(idx)
        for j in tree.query_ball_point(pts[idx], r=radius_um):
            suppressed[j] = True
    keep = np.asarray(keep, dtype=int)
    return coords_zyx[keep].astype(np.int32), scores[keep].astype(np.float32)


def detect_blobs(vol: np.ndarray, T: int) -> Tuple[np.ndarray, np.ndarray]:
    """Detect bright blob centers using max of multi-scale DoG responses."""
    x = vol.astype(np.float32, copy=False)
    lo, hi = np.percentile(x, [1.0, 99.9])
    x = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)

    responses = []
    for s1_xy, s2_xy in CFG["dog_scales"]:
        # Convert XY sigma to a rough z sigma using the physical anisotropy.
        s1 = (max(0.35, s1_xy * SCALE[1] / SCALE[0]), s1_xy, s1_xy)
        s2 = (max(0.60, s2_xy * SCALE[1] / SCALE[0]), s2_xy, s2_xy)
        r = gaussian_filter(x, s1) - gaussian_filter(x, s2)
        responses.append(r)
    resp = np.maximum.reduce(responses)

    bg = float(np.median(resp))
    top = float(np.percentile(resp, 99.9))
    thr = bg + CFG["rel_threshold"] * max(top - bg, 1e-6)

    min_dist_vox = max(1, int(round(CFG["min_distance_um"] / SCALE[1])))
    if SKIMAGE_AVAILABLE:
        coords = peak_local_max(resp, min_distance=min_dist_vox, threshold_abs=thr, exclude_border=False)
    else:
        mx = maximum_filter(resp, size=(3, min_dist_vox, min_dist_vox), mode="nearest")
        coords = np.argwhere((resp >= mx) & (resp > thr))

    if len(coords) == 0:
        return coords.astype(np.int32), np.empty(0, dtype=np.float32)

    scores = resp[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.float32)
    # Cap peaks per frame to avoid excessive false positives on noisy test volumes.
    max_per_frame = max(64, int(np.ceil(CFG["max_peaks"] / max(T, 1))))
    if len(coords) > max_per_frame:
        order = np.argsort(-scores)[:max_per_frame]
        coords, scores = coords[order], scores[order]

    coords, scores = physical_nms(coords.astype(np.int32), scores, CFG["min_distance_um"])
    return coords, scores


def link_adjacent(prev_ids: List[int], prev_xyz: np.ndarray, curr_ids: List[int], curr_xyz: np.ndarray, max_um: float) -> List[Tuple[int, int]]:
    if len(prev_ids) == 0 or len(curr_ids) == 0:
        return []
    P = prev_xyz.astype(np.float64) * SCALE[None, :]
    C = curr_xyz.astype(np.float64) * SCALE[None, :]
    D = np.sqrt(((P[:, None, :] - C[None, :, :]) ** 2).sum(axis=2))
    BIG = 1e6
    cost = np.where(D <= max_um, D, BIG)
    ri, ci = linear_sum_assignment(cost)
    edges = []
    for r, c in zip(ri, ci):
        if cost[r, c] < BIG:
            edges.append((int(prev_ids[r]), int(curr_ids[c])))
    return edges


def edge_row(dataset: str, source: int, target: int) -> dict:
    return {"dataset": dataset, "row_type": "edge", "node_id": -1, "t": -1, "z": -1, "y": -1, "x": -1, "source_id": int(source), "target_id": int(target)}


def node_row(dataset: str, node_id: int, t: int, zyx: Sequence[int]) -> dict:
    z, y, x = [int(v) for v in zyx]
    return {"dataset": dataset, "row_type": "node", "node_id": int(node_id), "t": int(t), "z": z, "y": y, "x": x, "source_id": -1, "target_id": -1}



def add_gap_edges(node_info: Dict[int, Tuple[int, np.ndarray]], edges: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], int]:
    if not CFG["close_gaps"] or CFG["max_gap"] < 1:
        return edges, 0
    outgoing = {s for s, _ in edges}
    incoming = {t for _, t in edges}
    by_t = defaultdict(list)
    for nid, (t, zyx) in node_info.items():
        by_t[int(t)].append((nid, zyx))

    added = []
    for t in sorted(by_t):
        srcs = [(nid, zyx) for nid, zyx in by_t[t] if nid not in outgoing]
        tgts = [(nid, zyx) for nid, zyx in by_t.get(t + 2, []) if nid not in incoming]
        if not srcs or not tgts:
            continue
        S = np.stack([z for _, z in srcs]).astype(np.float64) * SCALE[None, :]
        Tgt = np.stack([z for _, z in tgts]).astype(np.float64) * SCALE[None, :]
        tree = cKDTree(Tgt)
        cand = []
        for i, p in enumerate(S):
            d, j = tree.query(p, k=1)
            if np.isfinite(d) and d <= CFG["gap_dist_um"]:
                cand.append((float(d), i, int(j)))
        cand.sort(key=lambda x: x[0])
        used_s, used_t = set(), set()
        for d, i, j in cand:
            if i in used_s or j in used_t:
                continue
            s_id = int(srcs[i][0])
            t_id = int(tgts[j][0])
            added.append((s_id, t_id))
            used_s.add(i); used_t.add(j)
            outgoing.add(s_id); incoming.add(t_id)
    return edges + added, len(added)


def add_safe_divisions(node_info: Dict[int, Tuple[int, np.ndarray]], edges: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], int, Dict[int, int]]:
    if not CFG["safe_divisions"]:
        return edges, 0, {}

    children = defaultdict(list)
    parents = defaultdict(list)
    for s, t in edges:
        children[int(s)].append(int(t))
        parents[int(t)].append(int(s))

    by_t = defaultdict(list)
    for nid, (t, zyx) in node_info.items():
        by_t[int(t)].append((nid, zyx))

    total_nodes = len(node_info)
    global_cap = int(np.ceil(CFG["safe_division_global_frac_cap"] * max(total_nodes, 1)))
    added = []
    per_frame_added = defaultdict(int)

    for s, existing_children in list(children.items()):
        if len(added) >= global_cap:
            break
        if len(existing_children) != 1:
            continue
        st, sxyz = node_info[s]
        child = existing_children[0]
        if child not in node_info:
            continue
        ct, cxyz = node_info[child]
        if ct != st + 1:
            continue

        frame_cap = int(np.ceil(CFG["safe_division_frame_frac_cap"] * max(len(by_t.get(ct, [])), 1)))
        if per_frame_added[ct] >= frame_cap:
            continue

        parent_um = sxyz.astype(np.float64) * SCALE
        existing_um = cxyz.astype(np.float64) * SCALE
        existing_dist = float(np.linalg.norm(parent_um - existing_um))
        if existing_dist > CFG["safe_division_existing_child_max_um"]:
            continue

        # Candidate second daughter: unmatched node in the next frame, close to parent and sibling.
        candidates = []
        for cand_id, cand_xyz in by_t.get(ct, []):
            cand_id = int(cand_id)
            if cand_id == child or cand_id in parents:
                continue
            cand_um = cand_xyz.astype(np.float64) * SCALE
            d_parent = float(np.linalg.norm(parent_um - cand_um))
            d_sibling = float(np.linalg.norm(existing_um - cand_um))
            if d_parent <= CFG["safe_division_max_um"] and d_sibling <= CFG["safe_division_sibling_max_um"]:
                # Prefer compact Y-shaped splits. Lower score is safer.
                score = d_parent + 0.5 * d_sibling
                candidates.append((score, cand_id, d_parent, d_sibling))
        if not candidates:
            continue
        candidates.sort(key=lambda x: x[0])
        _, cand_id, _, _ = candidates[0]
        added.append((int(s), int(cand_id)))
        parents[int(cand_id)].append(int(s))
        children[int(s)].append(int(cand_id))
        per_frame_added[ct] += 1

    return edges + added, len(added), dict(per_frame_added)


def process_dataset(dataset: str) -> Tuple[List[dict], dict]:
    zarr_path = TEST_DIR / f"{dataset}.zarr"
    _, shape, dtype, chunk_shape = read_array_meta(zarr_path)
    T, Z, Y, X = shape
    if CFG["debug_max_t"] > 0:
        T = min(T, CFG["debug_max_t"])
    print(f"\n[{dataset}] shape={shape}, dtype={dtype}, chunk={chunk_shape}, run_T={T}")
    start = time.time()

    rows = []
    node_info: Dict[int, Tuple[int, np.ndarray]] = {}
    frame_ids = []
    next_id = 1
    frame_counts = []

    for t in range(T):
        vol = load_volume(zarr_path, t)
        coords, scores = detect_blobs(vol, T=T)
        ids = []
        for zyx in coords:
            nid = next_id
            next_id += 1
            ids.append(nid)
            node_info[nid] = (t, zyx.astype(np.int32))
            rows.append(node_row(dataset, nid, t, zyx))
        frame_ids.append((ids, coords.astype(np.int32)))
        frame_counts.append(len(ids))
        if (t + 1) % 20 == 0 or (t + 1) == T:
            print(f"  frame {t+1:3d}/{T}: nodes={len(ids):4d}, mean20={np.mean(frame_counts[-20:]):.1f}")
        del vol
        gc.collect()

    edges: List[Tuple[int, int]] = []
    for t in range(T - 1):
        prev_ids, prev_xyz = frame_ids[t]
        curr_ids, curr_xyz = frame_ids[t + 1]
        edges.extend(link_adjacent(prev_ids, prev_xyz, curr_ids, curr_xyz, CFG["max_link_um"]))

    raw_edges = len(edges)
    edges, gap_edges = add_gap_edges(node_info, edges)
    edges, safe_div_edges, per_frame_safe = add_safe_divisions(node_info, edges)

    # Deduplicate while preserving order.
    seen = set()
    dedup = []
    for e in edges:
        if e not in seen:
            seen.add(e)
            dedup.append(e)
    edges = dedup

    for s, t in edges:
        rows.append(edge_row(dataset, s, t))

    elapsed = time.time() - start
    stats = {
        "dataset": dataset,
        "nodes": len(node_info),
        "edges": len(edges),
        "raw_adjacent_edges": raw_edges,
        "gap_edges_added": gap_edges,
        "safe_division_edges_added": safe_div_edges,
        "division_like_sources": sum(1 for v in defaultdict(list, {}).values() if len(v) >= 2),
        "mean_nodes_per_frame": float(np.mean(frame_counts)) if frame_counts else 0.0,
        "min_nodes_per_frame": int(np.min(frame_counts)) if frame_counts else 0,
        "max_nodes_per_frame": int(np.max(frame_counts)) if frame_counts else 0,
        "seconds": elapsed,
    }
    # Compute actual division-like sources from final edge set.
    ch = defaultdict(list)
    for s, t in edges:
        ch[s].append(t)
    stats["division_like_sources"] = sum(1 for v in ch.values() if len(v) >= 2)

    print(
        f"[{dataset}] nodes={stats['nodes']:,} edges={stats['edges']:,} "
        f"gap={gap_edges:,} safe_div={safe_div_edges:,} "
        f"div_sources={stats['division_like_sources']:,} sec={elapsed:.1f}"
    )
    return rows, stats


def validate_submission(sub: pd.DataFrame) -> None:
    expected = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
    assert list(sub.columns) == expected, list(sub.columns)
    assert sub["id"].tolist() == list(range(len(sub))), "id must be consecutive"
    assert set(sub["row_type"].unique()).issubset({"node", "edge"})
    nodes = sub[sub.row_type == "node"]
    edges = sub[sub.row_type == "edge"]
    assert (nodes[["node_id", "t", "z", "y", "x"]] >= 0).all().all()
    assert (nodes[["source_id", "target_id"]] == -1).all().all()
    assert (edges[["node_id", "t", "z", "y", "x"]] == -1).all().all()
    assert (edges[["source_id", "target_id"]] >= 0).all().all()
    for ds, g in sub.groupby("dataset"):
        node_ids = set(g.loc[g.row_type == "node", "node_id"].astype(int))
        e = g[g.row_type == "edge"]
        assert set(e["source_id"].astype(int)).issubset(node_ids)
        assert set(e["target_id"].astype(int)).issubset(node_ids)



def build_submission() -> Tuple[pd.DataFrame, pd.DataFrame]:
    datasets = list_dataset_names(TEST_DIR)
    print(f"Found {len(datasets)} test datasets: {datasets[:6]}{'...' if len(datasets) > 6 else ''}")
    all_rows = []
    stats = []
    t0 = time.time()
    for i, ds in enumerate(datasets, 1):
        rows, st = process_dataset(ds)
        all_rows.extend(rows)
        stats.append(st)
        print(f"Progress {i}/{len(datasets)} | rows={len(all_rows):,} | elapsed={(time.time()-t0)/60:.1f} min")

    sub = pd.DataFrame(all_rows)
    sub = sub[["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]]
    sub.insert(0, "id", np.arange(len(sub), dtype=np.int64))
    sub.to_csv(OUTPUT_PATH, index=False)
    return sub, pd.DataFrame(stats)


def write_run_summary(sub: pd.DataFrame, stats_df: pd.DataFrame) -> None:
    nodes = int((sub.row_type == "node").sum())
    edges = int((sub.row_type == "edge").sum())
    total = int(len(sub))
    div_sources = int(stats_df["division_like_sources"].sum()) if len(stats_df) else 0
    gap_edges = int(stats_df["gap_edges_added"].sum()) if len(stats_df) else 0
    safe_div_edges = int(stats_df["safe_division_edges_added"].sum()) if len(stats_df) else 0
    elapsed = float(stats_df["seconds"].sum()) if len(stats_df) else 0.0

    summary = {
        "experiment": "exp007_rule_safe_plus",
        "base_source": CFG["base_source"],
        "score_up_axis": CFG["score_up_axis"],
        "preset": PRESET,
        "final_output": str(OUTPUT_PATH),
        "rows": total,
        "nodes": nodes,
        "edges": edges,
        "edges_per_node": edges / max(nodes, 1),
        "gap_edges_added": gap_edges,
        "safe_division_edges_added": safe_div_edges,
        "division_like_sources": div_sources,
        "runtime_sec": elapsed,
        "config": CFG,
        "dataset_summary": stats_df.to_dict(orient="records"),
            }
    RUN_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = []
    md.append("# 032 Run Summary")
    md.append(f"- Base source: `{CFG['base_source']}`")
    md.append(f"- Score-up axis: `{CFG['score_up_axis']}`")
    md.append(f"- Preset: `{PRESET}`")
    md.append(f"- Final output: `{OUTPUT_PATH}`")
    md.append(f"- Rows: `{total}`")
    md.append(f"- Nodes: `{nodes}`")
    md.append(f"- Edges: `{edges}`")
    md.append(f"- Edges per node: `{edges / max(nodes, 1):.4f}`")
    md.append(f"- Gap edges added: `{gap_edges}`")
    md.append(f"- Safe division edges added: `{safe_div_edges}`")
    md.append(f"- Division-like sources: `{div_sources}`")
    md.append("- Config:")
    md.append("```json")
    md.append(json.dumps(CFG, indent=2))
    md.append("```")
    RUN_SUMMARY_MD.write_text("\n".join(md), encoding="utf-8")

    print("\n" + "=" * 88)
    print("FINAL RUN SUMMARY")
    print("=" * 88)
    print(f"Notebook          : 032_lb839_plus_safe_division_recovery")
    print(f"Preset            : {PRESET}")
    print(f"Output file       : {OUTPUT_PATH}")
    print(f"Total rows        : {total:,}")
    print(f"Node rows         : {nodes:,}")
    print(f"Edge rows         : {edges:,}")
    print(f"Edges per node    : {edges / max(nodes, 1):.4f}")
    print(f"Gap edges added   : {gap_edges:,}")
    print(f"Safe div added    : {safe_div_edges:,}")
    print(f"Div-like sources  : {div_sources:,}")
    print(f"Runtime           : {elapsed/60:.1f} min")
    print("-" * 88)
    print("Per-dataset summary")
    print("-" * 88)
    if len(stats_df):
        cols = ["dataset", "nodes", "edges", "gap_edges_added", "safe_division_edges_added", "division_like_sources", "mean_nodes_per_frame", "seconds"]
        print(stats_df[cols].to_string(index=False))
    print("-" * 88)
    print("RUN SUMMARY")
    print("-" * 88)
    print("===== RUN SUMMARY START =====")
    print("# 032 Run Summary")
    print(f"- Base source: `{CFG['base_source']}`")
    print(f"- Score-up axis: `{CFG['score_up_axis']}`")
    print(f"- Preset: `{PRESET}`")
    print(f"- Final output: `{OUTPUT_PATH}`")
    print(f"- Rows: `{total}`")
    print(f"- Nodes: `{nodes}`")
    print(f"- Edges: `{edges}`")
    print(f"- Edges per node: `{edges / max(nodes, 1):.4f}`")
    print(f"- Gap edges added: `{gap_edges}`")
    print(f"- Safe division edges added: `{safe_div_edges}`")
    print(f"- Division-like sources: `{div_sources}`")
    print("- Config:")
    print(json.dumps(CFG, indent=2))
    print("===== RUN SUMMARY END =====")
    print("=" * 88)
    print(f"Run summary files: {RUN_SUMMARY_JSON} | {RUN_SUMMARY_MD}")

submission, stats_df = build_submission()
validate_submission(submission)
print("All submission checks passed.")
write_run_summary(submission, stats_df)
print("Ready to submit:", OUTPUT_PATH)
