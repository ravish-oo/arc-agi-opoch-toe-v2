# src/phi/paint.py
from typing import Dict
import numpy as np

from ..types import Grid, Boundary, QtSpec, ShapeLaw, ShapeLawKind
from ..qt.quotient import classes_for


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

    # Step 3: Paint based on Δ
    if delta.kind == ShapeLawKind.IDENTITY:
        return _paint_identity(x, cls, col_of_local)

    elif delta.kind == ShapeLawKind.BLOW_UP:
        return _paint_blowup(x, cls, col_of_local, delta.kh, delta.kw)

    elif delta.kind == ShapeLawKind.FRAME and enable_frame:
        return _paint_frame(x, cls, col_of_local, delta.kh)  # t stored in kh

    elif delta.kind == ShapeLawKind.TILING and enable_tiling:
        return _paint_tiling(x, cls, col_of_local, delta.kh, delta.kw)

    else:
        # Fallback to identity if flags not enabled
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


def _paint_tiling(x: Grid, cls, col_of_local: Dict[int, np.int8], kh: int, kw: int) -> Grid:
    """Paint TILING case: tile identity patch kh×kw times."""
    h, w = x.shape
    H, W = h * kh, w * kw

    # First compute identity patch
    Z = _paint_identity(x, cls, col_of_local)

    # Tile it
    out = np.zeros((H, W), dtype=np.int8)
    for i in range(kh):
        for j in range(kw):
            out[i*h:(i+1)*h, j*w:(j+1)*w] = Z

    return np.ascontiguousarray(out)
