# src/qt/quotient.py
from typing import Dict, Optional
import numpy as np
from ..types import Grid, QtSpec, Classes
from ..kernel.grid import (
    residues_rc, diagonal_residues, box_count_per_color, components4, dims
)

# Optional refinement toggles (used by WO-07 ladder; default False keeps v2.3 base)
class _ExtraFlags:
    use_border_distance: bool = False            # S4
    use_centroid_parity: bool = False            # S5a
    use_component_scan_index: bool = False       # S5b (exclusive with S5a)


# ========== feature construction ==========

def make_initial_signature(
    x: Grid,
    spec: QtSpec,
    extra: Optional[_ExtraFlags] = None
) -> Dict[str, np.ndarray]:
    """
    Build per-pixel feature dict from input grid and spec.
    Returns dict mapping feature names to arrays.
    """
    h, w = dims(x)
    sig = {}

    # 1. Base color
    sig["color"] = x.astype(np.int16)

    # 2. Residues (for each k in spec.residues, ascending)
    for k in spec.residues:
        rr, cc = residues_rc(h, w, k)
        sig[f"r{k}"] = rr
        sig[f"c{k}"] = cc

    # 3. Diagonal residues (if enabled, k in [2,3,4,5])
    if spec.use_diagonals:
        for k in [2, 3, 4, 5]:
            anti_d, d = diagonal_residues(h, w, k)
            sig[f"anti_diag{k}"] = anti_d
            sig[f"diag{k}"] = d

    # 4. Local counts (for each radius in spec.radii, ascending)
    for radius in spec.radii:
        sig[f"cnt_r{radius}"] = box_count_per_color(x, radius)

    # 5. Components
    labels, comps = components4(x)
    comp_shape = np.zeros((h, w, 4), dtype=np.int16)
    for cid, summary in comps.items():
        area = summary["area"]
        sh, sw = summary["shape"]
        border = summary["border_contact"]
        mask = (labels == cid)
        comp_shape[mask] = (area, sh, sw, border)
    sig["comp_shape"] = comp_shape
    sig["comp_id"] = labels.astype(np.int32)

    # 6. Optional refinement features (S4-S5)
    if extra:
        if extra.use_border_distance:
            # Chebyshev distance to border
            rr = np.arange(h)[:, None]
            cc = np.arange(w)[None, :]
            border_dist = np.maximum(
                np.minimum(rr, h - 1 - rr),
                np.minimum(cc, w - 1 - cc)
            ).astype(np.int16)
            sig["border_dist"] = border_dist

        if extra.use_centroid_parity:
            # Centroid parity per component
            centroid_parity = np.zeros((h, w, 2), dtype=np.int8)
            for cid, summary in comps.items():
                mask = (labels == cid)
                rs, cs = np.where(mask)
                if len(rs) > 0:
                    r_mean = int(np.floor(rs.mean()))
                    c_mean = int(np.floor(cs.mean()))
                    centroid_parity[mask] = (r_mean % 2, c_mean % 2)
            sig["centroid_parity"] = centroid_parity

        elif extra.use_component_scan_index:
            # Component scan index (discovery order in row-major)
            comp_scan_idx = np.zeros((h, w), dtype=np.int32)
            for cid in comps.keys():
                mask = (labels == cid)
                comp_scan_idx[mask] = cid
            sig["comp_scan_idx"] = comp_scan_idx

    return sig


def pack_signature(sig: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Pack signature features into (h,w,F) array in deterministic order.
    Returns int32 array.
    """
    h, w = sig["color"].shape
    parts = []

    # Fixed order (binding):
    # 1. color
    parts.append(np.ascontiguousarray(sig["color"])[..., None])

    # 2. wl_embed (if present)
    if "wl_embed" in sig:
        parts.append(np.ascontiguousarray(sig["wl_embed"])[..., None])

    # 3. Residues (sorted keys)
    for key in sorted(sig.keys()):
        if key.startswith("r") or key.startswith("c"):
            # Check if it's a residue key (e.g., r2, c3)
            suffix = key[1:]
            if suffix.replace("_", "").isdigit():
                parts.append(np.ascontiguousarray(sig[key])[..., None])

    # 4. Diagonal residues (k in [2,3,4,5])
    for k in [2, 3, 4, 5]:
        if f"anti_diag{k}" in sig:
            parts.append(np.ascontiguousarray(sig[f"anti_diag{k}"])[..., None])
        if f"diag{k}" in sig:
            parts.append(np.ascontiguousarray(sig[f"diag{k}"])[..., None])

    # 5. Local counts (sorted radii)
    for key in sorted(sig.keys()):
        if key.startswith("cnt_r"):
            parts.append(np.ascontiguousarray(sig[key]))

    # 6. Components
    if "comp_shape" in sig:
        parts.append(np.ascontiguousarray(sig["comp_shape"]))
    if "comp_id" in sig:
        parts.append(np.ascontiguousarray(sig["comp_id"])[..., None])

    # 7. Optional features (S4-S5)
    if "border_dist" in sig:
        parts.append(np.ascontiguousarray(sig["border_dist"])[..., None])
    if "centroid_parity" in sig:
        parts.append(np.ascontiguousarray(sig["centroid_parity"]))
    if "comp_scan_idx" in sig:
        parts.append(np.ascontiguousarray(sig["comp_scan_idx"])[..., None])

    # Concatenate and cast to int32 (platform-independent)
    cat = np.concatenate(parts, axis=2)
    return cat.astype(np.int32, copy=False)


# ========== relabeling with stable keys ==========

def relabel_classes(packed: np.ndarray) -> Classes:
    """
    Relabel packed features via lexicographic sort.
    Returns Classes with local ids and stable bytes keys.
    """
    h, w, F = packed.shape
    M = packed.reshape(h * w, F)

    # Create byte view for lexicographic comparison
    view = M.view(np.uint8)

    # Lexicographic sort (rightmost key primary)
    order = np.lexsort(view.T[::-1])

    # Compute uniqueness flags
    uniq = np.ones(h * w, dtype=bool)
    uniq[1:] = (view[order][1:] != view[order][:-1]).any(axis=1)

    # Assign consecutive local ids
    ids = np.empty(h * w, dtype=np.int32)
    cid = -1
    first_idx = {}

    for idx, is_uniq in zip(order, uniq):
        if is_uniq:
            cid += 1
            first_idx[cid] = idx
        ids[idx] = cid

    # Build stable keys (intrinsic packed-feature bytes)
    key_for = {cid: bytes(view[first_idx[cid]]) for cid in first_idx}

    return Classes(ids=ids.reshape(h, w), key_for=key_for)


# ========== WL refinement (channel-separated) ==========

def wl_refine(x: Grid, spec: QtSpec, extra: Optional[_ExtraFlags] = None) -> Classes:
    """
    Apply WL refinement rounds with channel separation.
    WL adds wl_embed channel, never overwrites base color.
    """
    sig = make_initial_signature(x, spec, extra)
    packed = pack_signature(sig)
    classes = relabel_classes(packed)

    h, w = x.shape

    for round_num in range(spec.wl_rounds):
        # Build 5-neighbor stack (up, down, left, right, self)
        neigh = np.zeros((h, w, 5), dtype=np.int32)
        neigh[:, :, 4] = classes.ids  # self

        # Neighbors (edges implicitly 0 from initialization)
        if h > 1:
            neigh[1:, :, 0] = classes.ids[:-1, :]  # up
            neigh[:-1, :, 1] = classes.ids[1:, :]  # down
        if w > 1:
            neigh[:, 1:, 2] = classes.ids[:, :-1]  # left
            neigh[:, :-1, 3] = classes.ids[:, 1:]  # right

        # Sort and take median (WL channel separation)
        neigh_sorted = np.sort(neigh, axis=2)
        sig["wl_embed"] = neigh_sorted[:, :, 2]

        # Repack and relabel
        packed = pack_signature(sig)
        classes = relabel_classes(packed)

    return classes


# ========== top-level API ==========

def classes_for(x: Grid, spec: QtSpec, extra: Optional[_ExtraFlags] = None) -> Classes:
    """
    Compute equivalence classes for grid x using spec.
    Returns Classes with local ids and stable keys.
    """
    return wl_refine(x, spec, extra)
