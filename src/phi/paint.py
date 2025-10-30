# src/phi/paint.py
from typing import Dict, List, Tuple
import numpy as np

from ..types import Grid, Boundary, QtSpec, ShapeLaw, ShapeLawKind, CanonMeta
from ..qt.quotient import classes_for
from ..present.pi import uncanonize


def select_size_writer(
    canon_train_pairs: List[Tuple[Grid, Grid]],  # (X_canon, Y_canon)
    metas_train: List[CanonMeta],
    spec: QtSpec,
    bt: Boundary,
    delta: ShapeLaw
) -> str:
    """
    Return 'blowup' or 'tiling' for kh×kw size-change tasks.

    Deterministic: evaluate both writers on train inputs and compare to train outputs.
    No changes to Qt or Δ. Uses Y only to decide which Φ law reproduces the training.

    Algorithm:
    1. For each train pair, compute predictions with both writers
    2. Un-canonize predictions and compare to actual train outputs
    3. Count mismatches for each writer
    4. Pick writer with fewer errors (tie-break: 'blowup')
    """
    if delta.kind != ShapeLawKind.BLOW_UP:
        return 'identity'

    if delta.kh == 1 and delta.kw == 1:
        return 'identity'

    kh, kw = delta.kh, delta.kw
    err_bu = 0  # blow-up errors
    err_ti = 0  # tiling errors

    for (cx, cy_canon), meta in zip(canon_train_pairs, metas_train):
        # Compute both predictions on canonized input
        pred_bu_canon = _paint_blowup_internal(cx, spec, bt, kh, kw)
        pred_ti_canon = _paint_tiling_internal(cx, spec, bt, kh, kw)

        # Un-canonize predictions to match original train output
        pred_bu = uncanonize(pred_bu_canon, meta)
        pred_ti = uncanonize(pred_ti_canon, meta)

        # Un-canonize the canonized train output to get original Y
        y_orig = uncanonize(cy_canon, meta)

        # Count mismatches
        err_bu += np.sum(pred_bu != y_orig)
        err_ti += np.sum(pred_ti != y_orig)

    # Decision logic
    if err_ti == 0 and err_bu > 0:
        return 'tiling'
    elif err_bu == 0 and err_ti > 0:
        return 'blowup'
    elif err_ti < err_bu:
        return 'tiling'
    else:
        # Tie-break: prefer blowup (stable default)
        return 'blowup'


def _build_color_map(x: Grid, cls, bt: Boundary) -> Dict[int, np.int8]:
    """
    Build per-local-id color map with identity-guard.

    For each class:
    - If key in Bt.forced_color: use that
    - Else: use input color at first pixel of that class (identity-guard)
    """
    col_of_local: Dict[int, np.int8] = {}

    for local_id, key in cls.key_for.items():
        if key in bt.forced_color:
            # Use forced color from Bt
            v = bt.forced_color[key]
        else:
            # Bt-empty guard: use input color (input-only fallback)
            # Find first pixel with this local_id
            positions = np.argwhere(cls.ids == local_id)
            if len(positions) > 0:
                r0, c0 = positions[0]
                v = int(x[r0, c0])
            else:
                # Shouldn't happen (class exists in key_for), but guard anyway
                v = 0

        col_of_local[local_id] = np.int8(v)

    return col_of_local


def paint_phi(
    x: Grid,                # canonized test input
    spec: QtSpec,           # input-only feature family
    bt: Boundary,           # ρ: stable_key -> color
    delta: ShapeLaw,        # Δ: IDENTITY / BLOW_UP(kh,kw) / FRAME / TILING
    *,
    enable_frame: bool = False,
    enable_tiling: bool = False
) -> Grid:
    """
    Deterministic, Δ-aware, one-stroke painting.

    Guards:
      - identity-guard: if key not in ρ, color = class's input color in x
      - blow-up guard: same as above (input-only)

    FRAME border rule:
      - if enabled and delta.kind == FRAME: border color = mode input color on perimeter

    TILING:
      - if enabled and delta.kind == TILING: build identity-painted patch Z, then tile to canvas

    Returns:
      Output grid as np.int8, C-contiguous
    """
    h, w = x.shape

    # Step 1: Compute classes for input x
    cls = classes_for(x, spec)

    # Step 2: Build per-local-id color map with guards
    col_of_local = _build_color_map(x, cls, bt)

    # Step 3: Paint based on Δ (canvas size) and writer mode (from training)
    if delta.kind == ShapeLawKind.IDENTITY:
        return _paint_identity(x, cls, col_of_local)

    # Size-changed canvas: choose writer by training-selected flag
    if delta.kind == ShapeLawKind.BLOW_UP:
        if enable_tiling:
            return _paint_tiling(x, cls, col_of_local, bt, delta.kh, delta.kw)
        else:
            return _paint_blowup(x, cls, col_of_local, delta.kh, delta.kw)

    # Optional frame (unchanged)
    if delta.kind == ShapeLawKind.FRAME and enable_frame:
        return _paint_frame(x, cls, col_of_local, delta.kh)  # t stored in kh

    # Optional explicit TILING-kind path (if ever emitted from Δ)
    if delta.kind == ShapeLawKind.TILING and enable_tiling:
        return _paint_tiling(x, cls, col_of_local, bt, delta.kh, delta.kw)

    # Conservative fallback
    return _paint_identity(x, cls, col_of_local)


def _paint_identity(x: Grid, cls, col_of_local: Dict[int, np.int8]) -> Grid:
    """Paint IDENTITY case: same shape, fill each class region."""
    h, w = x.shape
    out = np.zeros((h, w), dtype=np.int8)

    # One stroke per class
    for local_id, v in col_of_local.items():
        mask = (cls.ids == local_id)
        out[mask] = v

    return np.ascontiguousarray(out)


def _paint_blowup_internal(x: Grid, spec: QtSpec, bt: Boundary, kh: int, kw: int) -> Grid:
    """Internal: Paint BLOW_UP case with full setup."""
    cls = classes_for(x, spec)
    col_of_local = _build_color_map(x, cls, bt)
    return _paint_blowup(x, cls, col_of_local, kh, kw)


def _paint_blowup(x: Grid, cls, col_of_local: Dict[int, np.int8], kh: int, kw: int) -> Grid:
    """Paint BLOW_UP case: each pixel becomes kh×kw block."""
    h, w = x.shape
    H, W = h * kh, w * kw
    out = np.zeros((H, W), dtype=np.int8)

    # One stroke per pixel-class: paint constant block
    for local_id, v in col_of_local.items():
        ys, xs = np.where(cls.ids == local_id)
        for r, c in zip(ys, xs):
            out[r*kh:(r+1)*kh, c*kw:(c+1)*kw] = v

    return np.ascontiguousarray(out)


def _paint_frame(x: Grid, cls, col_of_local: Dict[int, np.int8], t: int) -> Grid:
    """Paint FRAME case: border + interior identity."""
    h, w = x.shape
    H, W = h + 2*t, w + 2*t
    out = np.zeros((H, W), dtype=np.int8)

    # Border color: mode input color on perimeter (input-only)
    perimeter_mask = np.zeros((h, w), dtype=bool)
    perimeter_mask[0, :] = True   # Top row
    perimeter_mask[-1, :] = True  # Bottom row
    perimeter_mask[:, 0] = True   # Left column
    perimeter_mask[:, -1] = True  # Right column

    perimeter_colors = x[perimeter_mask]
    if len(perimeter_colors) > 0:
        # Mode: most frequent color, ties broken by smallest color
        counts = np.bincount(perimeter_colors, minlength=10)
        v_border = np.argmax(counts)
    else:
        v_border = 0

    v_border = np.int8(v_border)

    # Fill border
    out[:t, :] = v_border      # Top
    out[-t:, :] = v_border     # Bottom
    out[:, :t] = v_border      # Left
    out[:, -t:] = v_border     # Right

    # Interior: identity painting
    Z = _paint_identity(x, cls, col_of_local)
    out[t:H-t, t:W-t] = Z

    return np.ascontiguousarray(out)


def _paint_tiling_internal(x: Grid, spec: QtSpec, bt: Boundary, kh: int, kw: int) -> Grid:
    """Internal: Paint TILING case with full setup."""
    cls = classes_for(x, spec)
    col_of_local = _build_color_map(x, cls, bt)
    return _paint_tiling(x, cls, col_of_local, bt, kh, kw)


def _paint_tiling(
    x: Grid,
    cls,
    col_of_local: Dict[int, np.int8],
    bt: Boundary,
    kh: int,
    kw: int
) -> Grid:
    """
    Paint TILING case: selective stamping based on input classes.

    For each pane (pr, pc):
      - Map to input anchor via modulo: ar = pr % h, ac = pc % w
      - Derive "used color" same as Φ does: ρ(key) if forced, else input color
      - Place identity patch Z in pane iff used != 0

    This is input-only + Bt, no target peeking, and handles selective tiling.
    """
    h, w = x.shape
    H, W = h * kh, w * kw

    # Step 1: Build identity patch Z (same rule as IDENTITY: use ρ when present, else input color)
    Z = _paint_identity(x, cls, col_of_local)

    # Step 2: Pane mask via modulo anchors and the SAME "used-color" rule
    mask = np.zeros((kh, kw), dtype=bool)
    for pr in range(kh):
        for pc in range(kw):
            ar, ac = pr % h, pc % w  # Modulo mapping (binding)
            lid = int(cls.ids[ar, ac])
            key = cls.key_for[lid]
            forced = bt.forced_color.get(key, None)

            # Derive the "used color" exactly like Φ would do
            used = forced if forced is not None else int(x[ar, ac])
            mask[pr, pc] = (used != 0)  # Selective stamping

    # Step 3: Tile Z selectively
    out = np.zeros((H, W), dtype=np.int8)
    for pr in range(kh):
        for pc in range(kw):
            if mask[pr, pc]:
                out[pr*h:(pr+1)*h, pc*w:(pc+1)*w] = Z

    return np.ascontiguousarray(out)
