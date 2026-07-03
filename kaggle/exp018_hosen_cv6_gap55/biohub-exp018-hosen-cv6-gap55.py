from __future__ import annotations
import os, sys, json, time
import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass, field
try:
    import blosc2
except Exception:
    blosc2 = None

from dataclasses import dataclass


SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)


# =========================================================================
# biohub.io  (unchanged)
# =========================================================================
@dataclass
class ImageVolume:
    path: str
    shape: tuple  # (T, Z, Y, X)
    dtype: np.dtype
    chunk: tuple

    @property
    def n_t(self) -> int:
        return int(self.shape[0])

    def frame(self, t: int) -> np.ndarray:
        return _read_chunk(self.path, t, self.shape, self.dtype)


def open_image(zarr_path: str) -> ImageVolume:
    with open(os.path.join(zarr_path, "0", "zarr.json")) as f:
        meta = json.load(f)
    shape = tuple(int(s) for s in meta["shape"])
    dtype = np.dtype(meta["data_type"])
    chunk = None
    cg = meta.get("chunk_grid", {})
    conf = cg.get("configuration", {})
    if "chunk_shape" in conf:
        chunk = tuple(int(s) for s in conf["chunk_shape"])
    return ImageVolume(path=zarr_path, shape=shape, dtype=dtype, chunk=chunk)


_BLOSC2 = None


def _blosc2():
    global _BLOSC2
    if _BLOSC2 is None:
        import blosc2
        _BLOSC2 = blosc2
    return _BLOSC2


def _read_chunk(zarr_path: str, t: int, shape: tuple, dtype: np.dtype) -> np.ndarray:
    frame_shape = shape[1:]
    chunk_path = os.path.join(zarr_path, "0", "c", str(t), "0", "0", "0")
    with open(chunk_path, "rb") as f:
        raw = f.read()
    try:
        dec = _blosc2().decompress(raw)
        arr = np.frombuffer(dec, dtype=dtype)
        if arr.size == int(np.prod(frame_shape)):
            return arr.reshape(frame_shape).copy()
    except Exception:
        pass
    import zarr
    z = zarr.open(os.path.join(zarr_path, "0"), mode="r")
    return np.asarray(z[t])


@dataclass
class TrackGraph:
    node_t: np.ndarray
    node_z: np.ndarray
    node_y: np.ndarray
    node_x: np.ndarray
    node_ids: np.ndarray
    edges: np.ndarray
    meta: dict

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def coords_by_id(self) -> dict:
        out = {}
        for i, nid in enumerate(self.node_ids):
            out[int(nid)] = (int(self.node_t[i]), float(self.node_z[i]),
                             float(self.node_y[i]), float(self.node_x[i]))
        return out


def read_geff(geff_path: str) -> TrackGraph:
    import zarr
    g = zarr.open(geff_path, mode="r")

    def _arr(path):
        return np.asarray(g[path][:])

    node_ids = _arr("nodes/ids").astype(np.int64)
    t = _arr("nodes/props/t/values").astype(np.int64)
    z = _arr("nodes/props/z/values").astype(np.float64)
    y = _arr("nodes/props/y/values").astype(np.float64)
    x = _arr("nodes/props/x/values").astype(np.float64)
    edges = _arr("edges/ids").astype(np.int64)
    if edges.ndim == 1:
        edges = edges.reshape(-1, 2)

    meta = {}
    try:
        with open(os.path.join(geff_path, "zarr.json")) as f:
            zj = json.load(f)
        geff_meta = zj.get("attributes", {}).get("geff", {})
        extra = geff_meta.get("extra", {}) or {}
        meta = dict(geff_meta)
        if "estimated_number_of_nodes" in extra:
            meta["estimated_number_of_nodes"] = extra["estimated_number_of_nodes"]
    except Exception:
        pass

    return TrackGraph(node_t=t, node_z=z, node_y=y, node_x=x,
                      node_ids=node_ids, edges=edges, meta=meta)


def list_datasets(root: str, kind: str = "train") -> list:
    d = os.path.join(root, kind)
    if not os.path.isdir(d):
        d = root
    names = sorted(n[:-5] for n in os.listdir(d) if n.endswith(".zarr"))
    return names


def embryo_id(dataset_name: str) -> str:
    return dataset_name.split("_")[0]


# =========================================================================
# biohub.metric  (unchanged local re-implementation, used for validation only)
# =========================================================================
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment

MAX_MATCH_UM = 7.0


def match_per_timepoint(pred_coords: dict, gt_coords: dict,
                        max_dist: float = MAX_MATCH_UM) -> dict:
    pred_by_t: dict = {}
    gt_by_t: dict = {}
    for nid, (t, z, y, x) in pred_coords.items():
        pred_by_t.setdefault(t, []).append((nid, z, y, x))
    for nid, (t, z, y, x) in gt_coords.items():
        gt_by_t.setdefault(t, []).append((nid, z, y, x))

    matches: dict = {}
    for t, plist in pred_by_t.items():
        glist = gt_by_t.get(t)
        if not glist:
            continue
        pid = [p[0] for p in plist]
        gid = [g[0] for g in glist]
        pc = np.array([[p[1], p[2], p[3]] for p in plist], dtype=np.float64) * SCALE
        gc = np.array([[g[1], g[2], g[3]] for g in glist], dtype=np.float64) * SCALE
        d = np.sqrt(((pc[:, None, :] - gc[None, :, :]) ** 2).sum(axis=2))
        big = max_dist * 1000.0 + 1.0
        cost = np.where(d <= max_dist, d, big)
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci):
            if d[r, c] <= max_dist:
                matches[pid[r]] = gid[c]
    return matches


@dataclass
class EdgeScore:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    n_pred_nodes: int = 0
    n_est_nodes: float = 0.0
    raw_jaccard: float = 0.0
    adj_jaccard: float = 0.0
    weight: float = 0.0


def edge_jaccard(pred_edges, gt_edges, matches, n_pred_nodes, n_est_nodes,
                 over_pred_penalty: bool = True) -> EdgeScore:
    gt_edge_set = set((int(a), int(b)) for a, b in gt_edges)
    tp = 0
    fp = 0
    covered = set()
    for u, v in pred_edges:
        mu = matches.get(u)
        mv = matches.get(v)
        if mu is not None and mv is not None:
            ge = (int(mu), int(mv))
            if ge in gt_edge_set:
                tp += 1
                covered.add(ge)
            else:
                fp += 1
    fn = len(gt_edge_set - covered)
    denom = tp + fp + fn
    raw = tp / denom if denom > 0 else 0.0
    adj = raw
    if over_pred_penalty and n_est_nodes and n_pred_nodes > n_est_nodes:
        adj = raw * (n_est_nodes / n_pred_nodes)
    return EdgeScore(tp=tp, fp=fp, fn=fn, n_pred_nodes=n_pred_nodes,
                     n_est_nodes=float(n_est_nodes or 0.0),
                     raw_jaccard=raw, adj_jaccard=adj, weight=float(denom))


def _out_adj(edges):
    adj: dict = {}
    for u, v in edges:
        adj.setdefault(int(u), []).append(int(v))
    return adj


@dataclass
class DivScore:
    tp: int = 0
    fp: int = 0
    fn: int = 0


def division_score(pred_edges, gt_edges, matches) -> DivScore:
    pred_out = _out_adj(pred_edges)
    gt_out = _out_adj(gt_edges)
    inv: dict = {}
    for p, g in matches.items():
        inv.setdefault(int(g), int(p))
    gt_divs = {a: ds for a, ds in gt_out.items() if len(ds) >= 2}
    pred_divs = {u: ds for u, ds in pred_out.items() if len(ds) >= 2}
    tp = 0
    matched_pred = set()
    for a, daughters in gt_divs.items():
        p = inv.get(a)
        if p is None or len(pred_out.get(p, [])) < 2:
            continue
        pred_daughter_gts = set()
        for pv in pred_out.get(p, []):
            g = matches.get(pv)
            if g is not None:
                pred_daughter_gts.add(g)
        if len(set(daughters) & pred_daughter_gts) >= 2:
            tp += 1
            matched_pred.add(p)
    fn = len(gt_divs) - tp
    fp = len([u for u in pred_divs if u not in matched_pred])
    return DivScore(tp=tp, fp=fp, fn=fn)


@dataclass
class SampleResult:
    name: str
    edge: EdgeScore
    div: DivScore


@dataclass
class Aggregate:
    edge_jaccard: float = 0.0
    division_jaccard: float = 0.0
    combined: float = 0.0
    per_sample: list = field(default_factory=list)
    extras: dict = field(default_factory=dict)


def aggregate(results: list, div_weight: float = 0.5) -> Aggregate:
    w = np.array([r.edge.weight for r in results], dtype=np.float64)
    aj = np.array([r.edge.adj_jaccard for r in results], dtype=np.float64)
    edge_j = float((aj * w).sum() / w.sum()) if w.sum() > 0 else 0.0
    dtp = sum(r.div.tp for r in results)
    dfp = sum(r.div.fp for r in results)
    dfn = sum(r.div.fn for r in results)
    ddenom = dtp + dfp + dfn
    div_j = dtp / ddenom if ddenom > 0 else 0.0
    # NOTE: official combined is additive (edge + div); this convex form is only
    # a convenience for local ranking. Track edge_j and div_j separately.
    combined = (1 - div_weight) * edge_j + div_weight * div_j
    return Aggregate(edge_jaccard=edge_j, division_jaccard=div_j,
                     combined=combined, per_sample=results,
                     extras={"div_tp": dtp, "div_fp": dfp, "div_fn": dfn})


def score_sample(name, pred_graph, gt_graph, n_est_nodes=None,
                 over_pred_penalty=True) -> SampleResult:
    pred_coords = pred_graph.coords_by_id()
    gt_coords = gt_graph.coords_by_id()
    matches = match_per_timepoint(pred_coords, gt_coords)
    if n_est_nodes is None:
        n_est_nodes = gt_graph.meta.get("estimated_number_of_nodes") if gt_graph.meta else None
    if not n_est_nodes:
        n_est_nodes = gt_graph.n_nodes
    es = edge_jaccard(
        [(int(a), int(b)) for a, b in pred_graph.edges],
        [(int(a), int(b)) for a, b in gt_graph.edges],
        matches, pred_graph.n_nodes, n_est_nodes, over_pred_penalty,
    )
    ds = division_score(
        [(int(a), int(b)) for a, b in pred_graph.edges],
        [(int(a), int(b)) for a, b in gt_graph.edges],
        matches,
    )
    return SampleResult(name=name, edge=es, div=ds)


# =========================================================================
# biohub.detect  (unchanged detectors + stable sort for determinism)
# =========================================================================
from scipy.ndimage import (gaussian_filter, maximum_filter, grey_erosion,
                           grey_dilation)


def detect_blobs(vol: np.ndarray,
                 xy_downsample: int = 4,
                 dog_small_um: float = 2.0,
                 dog_large_um: float = 6.0,
                 min_distance_um: float = 3.0,
                 rel_threshold: float = 0.04,
                 abs_percentile: float = 50.0,
                 max_peaks: int | None = 30000,
                 dog_scales: list | None = None) -> np.ndarray:
    vf = vol.astype(np.float32)
    ds = vf[:, ::xy_downsample, ::xy_downsample]
    eff = np.array([SCALE[0], SCALE[1] * xy_downsample, SCALE[2] * xy_downsample])
    lo, hi = np.percentile(ds, [1.0, 99.7])
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((ds - lo) / (hi - lo), 0, None)

    if dog_scales:
        dog = None
        for (s_um, l_um) in dog_scales:
            resp = (gaussian_filter(norm, sigma=s_um / eff)
                    - gaussian_filter(norm, sigma=l_um / eff))
            dog = resp if dog is None else np.maximum(dog, resp)
    else:
        g1 = gaussian_filter(norm, sigma=dog_small_um / eff)
        g2 = gaussian_filter(norm, sigma=dog_large_um / eff)
        dog = g1 - g2

    footprint = _ball_footprint(min_distance_um, eff)
    mx = maximum_filter(dog, footprint=footprint, mode="nearest")
    thr = max(rel_threshold, 0.0)
    abs_thr = np.percentile(norm, abs_percentile)
    peaks = (dog == mx) & (dog >= thr) & (norm >= abs_thr)
    coords = np.argwhere(peaks)
    if coords.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    vals = dog[peaks]
    order = np.argsort(vals, kind="stable")[::-1]
    coords = coords[order]
    if max_peaks is not None and len(coords) > max_peaks:
        coords = coords[:max_peaks]
    out = coords.astype(np.float64)
    out[:, 1] *= xy_downsample
    out[:, 2] *= xy_downsample
    return out


def detect_centroids(vol: np.ndarray,
                     xy_downsample: int = 4,
                     sigma: float = 1.0,
                     percentile: float = 99.0,
                     min_distance_um: float = 5.0,
                     max_peaks: int | None = None) -> np.ndarray:
    vf = vol.astype(np.float32)
    ds = vf[:, ::xy_downsample, ::xy_downsample]
    eff = np.array([SCALE[0], SCALE[1] * xy_downsample, SCALE[2] * xy_downsample])
    sig = sigma / (eff / eff.min())
    sm = gaussian_filter(ds, sigma=sig)
    thr = np.percentile(sm, percentile)
    footprint = _ball_footprint(min_distance_um, eff)
    mx = maximum_filter(sm, footprint=footprint, mode="nearest")
    peaks = (sm == mx) & (sm >= thr)
    coords = np.argwhere(peaks)
    if coords.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    vals = sm[peaks]
    order = np.argsort(vals, kind="stable")[::-1]
    coords = coords[order]
    if max_peaks is not None and len(coords) > max_peaks:
        coords = coords[:max_peaks]
    out = coords.astype(np.float64)
    out[:, 1] *= xy_downsample
    out[:, 2] *= xy_downsample
    return out


def _ball_footprint(radius_um: float, eff_spacing: np.ndarray) -> np.ndarray:
    rad_vox = np.maximum(1, np.round(radius_um / eff_spacing).astype(int))
    zz, yy, xx = np.ogrid[-rad_vox[0]:rad_vox[0] + 1,
                          -rad_vox[1]:rad_vox[1] + 1,
                          -rad_vox[2]:rad_vox[2] + 1]
    d = ((zz * eff_spacing[0]) ** 2 + (yy * eff_spacing[1]) ** 2 +
         (xx * eff_spacing[2]) ** 2)
    return d <= radius_um ** 2


def refine_centroids(vol: np.ndarray, coords: np.ndarray, win=(1, 3, 3)) -> np.ndarray:
    if len(coords) == 0:
        return coords
    Z, Y, X = vol.shape
    out = coords.copy().astype(np.float64)
    wz, wy, wx = win
    for i, (z, y, x) in enumerate(coords):
        z, y, x = int(round(z)), int(round(y)), int(round(x))
        z0, z1 = max(0, z - wz), min(Z, z + wz + 1)
        y0, y1 = max(0, y - wy), min(Y, y + wy + 1)
        x0, x1 = max(0, x - wx), min(X, x + wx + 1)
        patch = vol[z0:z1, y0:y1, x0:x1].astype(np.float64)
        s = patch.sum()
        if s <= 0:
            continue
        zz = np.arange(z0, z1)[:, None, None]
        yy = np.arange(y0, y1)[None, :, None]
        xx = np.arange(x0, x1)[None, None, :]
        out[i, 0] = (patch * zz).sum() / s
        out[i, 1] = (patch * yy).sum() / s
        out[i, 2] = (patch * xx).sum() / s
    return out


# =========================================================================
# biohub.link
# =========================================================================
from scipy.optimize import linear_sum_assignment


def close_gaps(frames: list, g: TrackGraph, max_gap: int = 1,
               gap_dist_um: float = 8.0) -> TrackGraph:
    """Insert interpolated nodes to bridge single-frame detection gaps."""
    if g.n_edges == 0:
        return g
    coords = {int(nid): (int(g.node_t[i]), g.node_z[i], g.node_y[i], g.node_x[i])
              for i, nid in enumerate(g.node_ids)}
    has_out = set(int(s) for s, _ in g.edges)
    has_in = set(int(t) for _, t in g.edges)
    ends_by_t = {}
    starts_by_t = {}
    for nid, (t, z, y, x) in coords.items():
        if nid not in has_out:
            ends_by_t.setdefault(t, []).append(nid)
        if nid not in has_in:
            starts_by_t.setdefault(t, []).append(nid)
    new_nodes = []
    new_edges = []
    next_id = int(g.node_ids.max()) + 1 if g.n_nodes else 1
    for gap in range(1, max_gap + 1):
        for t, ends in ends_by_t.items():
            starts = starts_by_t.get(t + gap + 1)
            if not starts:
                continue
            ec = np.array([[coords[e][1], coords[e][2], coords[e][3]] for e in ends]) * SCALE
            sc = np.array([[coords[s][1], coords[s][2], coords[s][3]] for s in starts]) * SCALE
            d = np.sqrt(((ec[:, None, :] - sc[None, :, :]) ** 2).sum(axis=2))
            thr = gap_dist_um * (gap + 1)
            big = thr * 1000 + 1
            cost = np.where(d <= thr, d, big)
            ri, ci = linear_sum_assignment(cost)
            used_s = set()
            for r, c in zip(ri, ci):
                if d[r, c] > thr or ends[r] in has_out or starts[c] in used_s:
                    continue
                e_id, s_id = ends[r], starts[c]
                te, ze, ye, xe = coords[e_id]
                ts, zs, ys, xs = coords[s_id]
                prev = e_id
                for k in range(1, gap + 1):
                    frac = k / (gap + 1)
                    zi = ze + (zs - ze) * frac
                    yi = ye + (ys - ye) * frac
                    xi = xe + (xs - xe) * frac
                    nid = next_id
                    next_id += 1
                    new_nodes.append((te + k, zi, yi, xi, nid))
                    new_edges.append((prev, nid))
                    prev = nid
                new_edges.append((prev, s_id))
                has_out.add(e_id)
                used_s.add(c)
    if not new_nodes:
        return g
    nt = np.concatenate([g.node_t, np.array([n[0] for n in new_nodes], dtype=np.int64)])
    nz = np.concatenate([g.node_z, np.array([n[1] for n in new_nodes])])
    ny = np.concatenate([g.node_y, np.array([n[2] for n in new_nodes])])
    nx = np.concatenate([g.node_x, np.array([n[3] for n in new_nodes])])
    nid = np.concatenate([g.node_ids, np.array([n[4] for n in new_nodes], dtype=np.int64)])
    edges = np.concatenate([g.edges, np.array(new_edges, dtype=np.int64).reshape(-1, 2)])
    return TrackGraph(node_t=nt, node_z=nz, node_y=ny, node_x=nx, node_ids=nid,
                      edges=edges, meta=g.meta)


def prune_isolated(g: TrackGraph) -> TrackGraph:
    if g.n_edges == 0:
        return g
    used = set(int(x) for x in g.edges.reshape(-1))
    keep = np.array([i for i, nid in enumerate(g.node_ids) if int(nid) in used])
    if len(keep) == len(g.node_ids):
        return g
    return TrackGraph(
        node_t=g.node_t[keep], node_z=g.node_z[keep], node_y=g.node_y[keep],
        node_x=g.node_x[keep], node_ids=g.node_ids[keep], edges=g.edges, meta=g.meta,
    )


def link_twopass(frames, tight_um=6.0, loose_um=8.0, vel_blend=0.5):
    """Original baseline two-pass velocity Hungarian linker (1:1, NO divisions).
    This keeps the public two-pass velocity linker unchanged for a baseline run."""
    node_ids = []; node_t = []; node_z = []; node_y = []; node_x = []; frame_ids = []; nid = 1
    for t, coords in enumerate(frames):
        ids = []
        for (z, y, x) in coords:
            node_ids.append(nid); node_t.append(t); node_z.append(z); node_y.append(y); node_x.append(x)
            ids.append(nid); nid += 1
        frame_ids.append(ids)
    def _hun(P, C, pred, pi, ci, gate):
        if len(pi) == 0 or len(ci) == 0:
            return []
        BIG = 1e9
        Draw = np.sqrt(((P[pi][:, None] - C[ci][None]) ** 2).sum(2))
        D = np.sqrt(((pred[pi][:, None] - C[ci][None]) ** 2).sum(2))
        cost = np.where(Draw > gate, BIG, D)
        ri, rc = linear_sum_assignment(cost)
        return [(int(pi[r]), int(ci[c])) for r, c in zip(ri, rc) if cost[r, c] < BIG]
    edges = []; prev_xyz = np.zeros((0, 3)); prev_ids = []; prev_vel = None
    for t in range(len(frames)):
        curr = np.asarray(frames[t], float).reshape(-1, 3); curr_ids = frame_ids[t]
        if t > 0 and len(prev_ids) and len(curr):
            P = prev_xyz * SCALE[None, :]; C = curr * SCALE[None, :]
            pred = P + (vel_blend * prev_vel if prev_vel is not None else 0.0)
            N, M = len(P), len(C)
            links = _hun(P, C, pred, np.arange(N), np.arange(M), min(tight_um, loose_um))
            up = {p for p, _ in links}; uc = {c for _, c in links}
            fp = np.array([i for i in range(N) if i not in up], int)
            fc = np.array([j for j in range(M) if j not in uc], int)
            links += _hun(P, C, pred, fp, fc, loose_um)
            vel = np.zeros((N, 3)); nv = np.zeros((M, 3))
            for p, c in links:
                edges.append((prev_ids[p], curr_ids[c])); vel[p] = (curr[c] - prev_xyz[p]) * SCALE
            for p, c in links:
                nv[c] = vel[p]
            prev_vel = nv
        else:
            prev_vel = None
        prev_xyz, prev_ids = curr, curr_ids
    return TrackGraph(node_t=np.array(node_t, np.int64), node_z=np.array(node_z, float),
                      node_y=np.array(node_y, float), node_x=np.array(node_x, float),
                      node_ids=np.array(node_ids, np.int64),
                      edges=np.array(edges, np.int64).reshape(-1, 2), meta={})


# >>> NEW: division-aware two-pass velocity linker. This is the main upgrade.
def link_twopass_div(frames, tight_um=6.0, loose_um=8.0, vel_blend=0.5,
                     allow_divisions=True, division_max_um=5.0,
                     sibling_max_um=6.0, div_parent_speed_um=6.0):
    """Two-pass velocity-aware Hungarian linking WITH mitosis detection.

    Pass 1/2 do the usual 1:1 assignment (tight gate, then loose gate on the
    residual). Then, from any current-frame detections still unmatched, we try to
    attach a SECOND daughter to a nearby matched parent -> creates a division node
    (parent with 2 outgoing edges). Gates keep division precision reasonable:
      - parent must be near-stationary (speed <= div_parent_speed_um um/frame)
      - both daughters within division_max_um of the parent
      - the two daughters within sibling_max_um of each other
    Everything is in physical (um) units. Deterministic.
    """
    node_ids = []; node_t = []; node_z = []; node_y = []; node_x = []; frame_ids = []; nid = 1
    for t, coords in enumerate(frames):
        ids = []
        for (z, y, x) in coords:
            node_ids.append(nid); node_t.append(t); node_z.append(z); node_y.append(y); node_x.append(x)
            ids.append(nid); nid += 1
        frame_ids.append(ids)

    def _hun(P, C, pred, pi, ci, gate):
        if len(pi) == 0 or len(ci) == 0:
            return []
        BIG = 1e9
        Draw = np.sqrt(((P[pi][:, None] - C[ci][None]) ** 2).sum(2))
        D = np.sqrt(((pred[pi][:, None] - C[ci][None]) ** 2).sum(2))
        cost = np.where(Draw > gate, BIG, D)
        ri, rc = linear_sum_assignment(cost)
        return [(int(pi[r]), int(ci[c])) for r, c in zip(ri, rc) if cost[r, c] < BIG]

    edges = []; prev_xyz = np.zeros((0, 3)); prev_ids = []; prev_vel = None
    for t in range(len(frames)):
        curr = np.asarray(frames[t], float).reshape(-1, 3); curr_ids = frame_ids[t]
        if t > 0 and len(prev_ids) and len(curr):
            P = prev_xyz * SCALE[None, :]; C = curr * SCALE[None, :]
            pred = P + (vel_blend * prev_vel if prev_vel is not None else 0.0)
            N, M = len(P), len(C)
            links = _hun(P, C, pred, np.arange(N), np.arange(M), min(tight_um, loose_um))
            up = {p for p, _ in links}; uc = {c for _, c in links}
            fp = np.array([i for i in range(N) if i not in up], int)
            fc = np.array([j for j in range(M) if j not in uc], int)
            links += _hun(P, C, pred, fp, fc, loose_um)

            vel = np.zeros((N, 3)); nv = np.zeros((M, 3))
            child_of = {}
            for p, c in links:
                edges.append((prev_ids[p], curr_ids[c]))
                vel[p] = (curr[c] - prev_xyz[p]) * SCALE
                child_of[p] = c

            # ---- division detection from leftover current detections ----
            if allow_divisions:
                assigned_c = {c for _, c in links}
                free_c = [j for j in range(M) if j not in assigned_c]
                parents = list(child_of.keys())
                for c2 in free_c:
                    best_p = -1; best_score = 1e18
                    for p in parents:
                        if np.linalg.norm(vel[p]) > div_parent_speed_um:
                            continue
                        d_par2 = np.linalg.norm(P[p] - C[c2])
                        if d_par2 > division_max_um:
                            continue
                        c1 = child_of[p]
                        d_par1 = np.linalg.norm(P[p] - C[c1])
                        if d_par1 > division_max_um:
                            continue
                        d_sib = np.linalg.norm(C[c1] - C[c2])
                        if d_sib > sibling_max_um:
                            continue
                        score = d_par2 + 0.5 * d_sib
                        if score < best_score:
                            best_score = score; best_p = p
                    if best_p >= 0:
                        edges.append((prev_ids[best_p], curr_ids[c2]))
                        nv[c2] = 0.0                 # daughter starts with no velocity
                        parents.remove(best_p)       # binary division only

            for p, c in links:
                nv[c] = vel[p]
            prev_vel = nv
        else:
            prev_vel = None
        prev_xyz, prev_ids = curr, curr_ids

    return TrackGraph(node_t=np.array(node_t, np.int64), node_z=np.array(node_z, float),
                      node_y=np.array(node_y, float), node_x=np.array(node_x, float),
                      node_ids=np.array(node_ids, np.int64),
                      edges=np.array(edges, np.int64).reshape(-1, 2), meta={})


# >>> NEW: prune divisions whose daughters don't continue -> cuts division FPs.
def validate_divisions(g: TrackGraph, min_daughter_len: int = 2) -> TrackGraph:
    """For every division (node with >=2 outgoing edges), require each daughter
    lineage to survive at least `min_daughter_len` nodes forward. Daughters that
    die immediately are almost always spurious detections; drop the weakest
    surplus daughters so the node keeps its two best (or degrades to a normal
    1:1 link). Reduces division false positives without touching real mitoses.
    """
    if g.n_edges == 0:
        return g
    out_adj = {}
    for s, tt in g.edges.reshape(-1, 2):
        out_adj.setdefault(int(s), []).append(int(tt))

    # forward chain length starting from a node (counts nodes, capped for speed)
    def chain_len(n, cap=8):
        seen = 0; cur = n
        while cur is not None and seen < cap:
            seen += 1
            nxt = out_adj.get(cur, [])
            cur = nxt[0] if nxt else None
        return seen

    drop_edges = set()
    for parent, kids in out_adj.items():
        if len(kids) < 2:
            continue
        scored = sorted(kids, key=lambda k: chain_len(k), reverse=True)
        # keep the two longest-surviving daughters; drop the rest
        keep = set(scored[:2])
        # if the 2nd-best daughter is too short, this "division" is unreliable ->
        # keep only the single best daughter (degrade to a normal link)
        if chain_len(scored[1]) < min_daughter_len:
            keep = {scored[0]}
        for k in kids:
            if k not in keep:
                drop_edges.add((parent, k))

    if not drop_edges:
        return g
    new_edges = np.array([(int(s), int(t)) for s, t in g.edges.reshape(-1, 2)
                          if (int(s), int(t)) not in drop_edges],
                         dtype=np.int64).reshape(-1, 2)
    return TrackGraph(node_t=g.node_t, node_z=g.node_z, node_y=g.node_y,
                      node_x=g.node_x, node_ids=g.node_ids, edges=new_edges, meta=g.meta)


# =========================================================================
# biohub.pipeline
# =========================================================================
from dataclasses import dataclass, asdict


@dataclass
class Config:
    detector: str = "blob"
    xy_downsample: int = 4
    # -- blob (DoG) detector params --
    dog_small_um: float = 1.5
    dog_large_um: float = 4.0
    dog_scales: list | None = None
    rel_threshold: float = 0.02
    abs_percentile: float = 50.0
    min_distance_um: float = 2.5
    max_peaks: int | None = 40000
    # -- legacy peak detector params --
    sigma: float = 1.0
    percentile: float = 99.0
    refine: bool = True
    # -- linking --
    linker: str = "hungarian"            # BASELINE (no divisions)
    tight_um: float = 6.0                 # baseline gate
    vel_blend: float = 0.5                # >>> NEW
    max_link_um: float = 10.0
    # -- divisions --  >>> NEW block
    link_divisions: bool = False
    division_max_um: float = 5.0
    sibling_max_um: float = 6.0
    div_parent_speed_um: float = 6.0
    validate_div: bool = False
    min_daughter_len: int = 2
    # -- gap closing / pruning --
    close_gaps: bool = False
    max_gap: int = 1
    gap_dist_um: float = 8.0
    prune_isolated: bool = True
    min_track_len: int = 4                # >>> NEW (was hard-coded 4)
    keep_div_components: bool = False     # baseline behavior


# >>> CHANGED: keep divisions when filtering short tracks
def filter_short_tracks(g, min_len, keep_divisions=True):
    if g.n_edges == 0 or min_len <= 1:
        return g
    out_count = {}
    for s, t in g.edges.reshape(-1, 2):
        out_count[int(s)] = out_count.get(int(s), 0) + 1
    div_nodes = {n for n, c in out_count.items() if c >= 2}

    parent = {int(n): int(n) for n in g.node_ids}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for s, t in g.edges.reshape(-1, 2):
        ra, rb = find(int(s)), find(int(t))
        if ra != rb:
            parent[ra] = rb
    comp = {}
    for n in g.node_ids:
        comp.setdefault(find(int(n)), []).append(int(n))
    keep = set()
    for members in comp.values():
        long_enough = len(members) >= min_len
        has_div = keep_divisions and any(m in div_nodes for m in members)
        if long_enough or has_div:
            keep.update(members)
    idx = [i for i, n in enumerate(g.node_ids) if int(n) in keep]
    keepset = set(int(g.node_ids[i]) for i in idx)
    edges = np.array([(int(s), int(t)) for s, t in g.edges.reshape(-1, 2)
                      if int(s) in keepset and int(t) in keepset], dtype=np.int64).reshape(-1, 2)
    return TrackGraph(node_t=g.node_t[idx], node_z=g.node_z[idx], node_y=g.node_y[idx],
                      node_x=g.node_x[idx], node_ids=g.node_ids[idx], edges=edges, meta=g.meta)


def run_one(zarr_path: str, cfg: Config, t_limit: int | None = None) -> TrackGraph:
    vol_meta = open_image(zarr_path)
    n_t = vol_meta.n_t if t_limit is None else min(t_limit, vol_meta.n_t)
    frames = []
    for t in range(n_t):
        vol = vol_meta.frame(t)
        if cfg.detector == "blob":
            coords = detect_blobs(
                vol, xy_downsample=cfg.xy_downsample,
                dog_small_um=cfg.dog_small_um, dog_large_um=cfg.dog_large_um,
                min_distance_um=cfg.min_distance_um, rel_threshold=cfg.rel_threshold,
                abs_percentile=cfg.abs_percentile, max_peaks=cfg.max_peaks,
                dog_scales=cfg.dog_scales,
            )
        else:
            coords = detect_centroids(
                vol, xy_downsample=cfg.xy_downsample, sigma=cfg.sigma,
                percentile=cfg.percentile, min_distance_um=cfg.min_distance_um,
                max_peaks=cfg.max_peaks,
            )
        if cfg.refine and len(coords) > 0:
            _o = (cfg.xy_downsample - 1) / 2.0
            coords[:, 1] += _o; coords[:, 2] += _o
            coords = refine_centroids(vol, coords, win=(3, 9, 9))
        frames.append(coords)

    # ---- linking ----
    if cfg.linker == "twopass_div":
        # EXPERIMENTAL: adds divisions. Only enable after local validation shows a
        # net gain (edge + division) on a holdout. Divisions regressed the LB in V5.
        g = link_twopass_div(
            frames, tight_um=cfg.tight_um, loose_um=cfg.max_link_um,
            vel_blend=cfg.vel_blend,
            allow_divisions=cfg.link_divisions,
            division_max_um=cfg.division_max_um,
            sibling_max_um=cfg.sibling_max_um,
            div_parent_speed_um=cfg.div_parent_speed_um,
        )
    else:
        # BASELINE (default): exact 0.842/0.847 linker, no divisions.
        g = link_twopass(frames, tight_um=cfg.tight_um,
                         loose_um=cfg.max_link_um, vel_blend=cfg.vel_blend)

    # ---- post-processing ----
    if cfg.close_gaps:
        g = close_gaps(frames, g, max_gap=cfg.max_gap, gap_dist_um=cfg.gap_dist_um)
    if cfg.validate_div:
        g = validate_divisions(g, min_daughter_len=cfg.min_daughter_len)
    if cfg.prune_isolated:
        g = prune_isolated(g)
    g = filter_short_tracks(g, cfg.min_track_len, keep_divisions=cfg.keep_div_components)
    return g


def graph_to_rows(name: str, g: TrackGraph) -> list:
    rows = []
    for i in range(g.n_nodes):
        rows.append({
            "dataset": name, "row_type": "node", "node_id": int(g.node_ids[i]),
            "t": int(g.node_t[i]), "z": int(round(g.node_z[i])),
            "y": int(round(g.node_y[i])), "x": int(round(g.node_x[i])),
            "source_id": -1, "target_id": -1,
        })
    for (s, t) in g.edges:
        rows.append({
            "dataset": name, "row_type": "edge", "node_id": -1, "t": -1,
            "z": -1, "y": -1, "x": -1, "source_id": int(s), "target_id": int(t),
        })
    return rows


def write_submission(all_rows: list, path: str) -> pd.DataFrame:
    df = pd.DataFrame(all_rows, columns=["dataset", "row_type", "node_id", "t",
                                         "z", "y", "x", "source_id", "target_id"])
    df.index.name = "id"
    df.to_csv(path)
    return df


import sys as _sys
io = _sys.modules[__name__]


# ============================================================================
# PUBLIC CV6 BASELINE CONFIG  (divisions OFF)
# ============================================================================
CONFIG_OVERRIDE = {
    'detector': 'blob',
    'dog_scales': [[1.5, 4.0], [2.2, 5.5]],
    'rel_threshold': 0.045,
    'min_distance_um': 4.0,
    'max_peaks': 40000,
    'max_link_um': 8.0,
    'tight_um': 6.0,
    'vel_blend': 0.5,
    'close_gaps': True,
    'max_gap': 1,
    'gap_dist_um': 5.5,
    'min_track_len': 4,
    # divisions OFF (they regressed the LB; only re-enable after offline proof)
    'linker': 'hungarian',
    'link_divisions': False,
    'validate_div': False,
    'keep_div_components': False,
}


# ============================ inference driver ============================
def find_test_dir():
    env = os.environ.get("TEST_DIR")
    cands = [
        env,
        "/kaggle/input/biohub-cell-tracking-during-development/test",
        "/kaggle/input/competitions/biohub-cell-tracking-during-development/test",
    ]
    for c in cands:
        if c and os.path.isdir(c):
            return c
    base = "/kaggle/input"
    if os.path.isdir(base):
        for root, dirs, files in os.walk(base):
            if os.path.basename(root) == "test" and any(d.endswith(".zarr") for d in dirs):
                return root
    raise FileNotFoundError("test dir not found")


def main():
    test_dir = find_test_dir()
    names = sorted(d[:-5] for d in os.listdir(test_dir) if d.endswith(".zarr"))
    print(f"test dir: {test_dir}; {len(names)} datasets", flush=True)
    cfg = Config(**CONFIG_OVERRIDE)
    all_rows = []
    t0 = time.time()
    for i, name in enumerate(names):
        zp = os.path.join(test_dir, name + ".zarr")
        g = run_one(zp, cfg)
        n_div = 0
        if g.n_edges:
            oc = {}
            for s, _ in g.edges.reshape(-1, 2):
                oc[int(s)] = oc.get(int(s), 0) + 1
            n_div = sum(1 for v in oc.values() if v >= 2)
        all_rows.extend(graph_to_rows(name, g))
        print(f"[{i+1}/{len(names)}] {name}: nodes={g.n_nodes} edges={g.n_edges} "
              f"divisions={n_div} ({time.time()-t0:.1f}s)", flush=True)
    out = "submission.csv"
    write_submission(all_rows, out)
    print(f"wrote {out}: {len(all_rows)} rows in {time.time()-t0:.1f}s", flush=True)
    summary = {
        "experiment": "exp018_hosen_cv6_gap55",
        "base": "hosen42_cv6_public_rule",
        "change": "cv6_gap_dist_5p5",
        "config": CONFIG_OVERRIDE,
        "rows": int(len(all_rows)),
        "elapsed_sec": float(time.time() - t0),
    }
    with open("run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("run summary written to run_summary.json", flush=True)


if __name__ == "__main__":
    main()
