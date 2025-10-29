# src/kernel/grid.py
from typing import Tuple, Dict, List
import numpy as np
from ..types import Grid

# ========== validation / shape ==========

def assert_grid(g: Grid) -> None:
    """
    Validate grid constraints per math anchor (finite palette, finite shapes).
    - g is np.ndarray with ndim==2
    - 1 <= h,w <= 30
    - dtype is np.int8
    - palette in 0..9
    Raises AssertionError with precise message if violated.
    """
    assert isinstance(g, np.ndarray), f"grid must be np.ndarray, got {type(g)}"
    assert g.ndim == 2, f"grid must be 2D, got ndim={g.ndim}"
    h, w = g.shape
    assert 1 <= h <= 30 and 1 <= w <= 30, f"grid shape out of bounds: ({h},{w})"
    assert g.dtype == np.int8, f"grid dtype must be np.int8, got {g.dtype}"
    if g.size > 0:
        mn = int(g.min())
        mx = int(g.max())
        assert 0 <= mn and mx <= 9, f"grid palette outside 0..9: min={mn} max={mx}"


def dims(g: Grid) -> Tuple[int, int]:
    """
    Return (h, w) as int tuple from g.shape.
    No validation; caller may call assert_grid beforehand.
    """
    h, w = g.shape
    return int(h), int(w)


def divisors(n: int, max_div: int = 10) -> List[int]:
    """
    Return list of divisors of n in range [2, max_div], ascending order.
    Used for computing residue moduli from grid dimensions.
    """
    return [d for d in range(2, min(n + 1, max_div + 1)) if n % d == 0]


# ========== D8 group (pose operations) ==========

def d8_apply(g: Grid, t: int) -> Grid:
    """
    Apply D8 transform t in 0..7 to grid g.
    Encoding:
      k = t & 3         → rotation by k×90°
      f = (t >> 2) & 1  → horizontal flip after rotation
    Returns C-contiguous np.int8 grid.
    """
    assert_grid(g)
    k = t & 3
    f = (t >> 2) & 1
    out = np.ascontiguousarray(np.rot90(g, k))
    if f:
        out = np.ascontiguousarray(np.fliplr(out))
    return out


def d8_inv(t: int) -> int:
    """
    Inverse of D8 transform t.
    For rotation-only (f=0): inverse is rot90(_, -k)
    For rotation+flip (f=1): fliplr·rot90(_,k) is self-inverse due to:
        fliplr · rot90(_, -k) = rot90(_, k) · fliplr
    Ensures d8_apply(d8_apply(g, t), d8_inv(t)) == g.
    """
    k = t & 3
    f = (t >> 2) & 1
    if f == 0:
        # Rotation only: inverse is -k
        invk = (-k) & 3
        return invk
    else:
        # Rotation + flip: self-inverse
        return t


# ========== residues and diagonals ==========

def residues_rc(h: int, w: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute row/column residues modulo k.
    rr[r,c] = r % k
    cc[r,c] = c % k
    Returns (rr, cc) as np.int16, C-contiguous.
    """
    rr = np.fromfunction(lambda r, c: r % k, (h, w), dtype=int)
    cc = np.fromfunction(lambda r, c: c % k, (h, w), dtype=int)
    return rr.astype(np.int16), cc.astype(np.int16)


def diagonal_residues(h: int, w: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute diagonal residues modulo k (strictly non-negative).
    anti_diag[r,c] = (r + c) % k
    diag[r,c]      = ((r - c) % k + k) % k
    Returns (anti_diag, diag) as np.int16, C-contiguous.
    """
    anti_diag = np.fromfunction(lambda r, c: (r + c) % k, (h, w), dtype=int)
    diag = np.fromfunction(lambda r, c: ((r - c) % k + k) % k, (h, w), dtype=int)
    return anti_diag.astype(np.int16), diag.astype(np.int16)


# ========== local counts (integral images) ==========

def box_count_per_color(g: Grid, radius: int) -> np.ndarray:
    """
    Count occurrences of each color in Chebyshev ball (square window) around each pixel.
    Returns C[h,w,10] as np.int16, where C[r,c,col] = count of color col in window.
    Window is (2*radius+1)×(2*radius+1) centered at (r,c), clipped to grid bounds.
    Fully vectorized via integral images (no per-pixel Python loops).
    """
    assert_grid(g)
    h, w = g.shape
    out = np.zeros((h, w, 10), dtype=np.int16)

    # Precompute clipped window bounds for all pixels
    rr0 = np.arange(h)[:, None]
    cc0 = np.arange(w)[None, :]
    rmin = np.clip(rr0 - radius, 0, h - 1)
    rmax = np.clip(rr0 + radius, 0, h - 1)
    cmin = np.clip(cc0 - radius, 0, w - 1)
    cmax = np.clip(cc0 + radius, 0, w - 1)

    # For each color, build integral image and query vectorized
    for color in range(10):
        mask = (g == color).astype(np.int32)
        S = np.zeros((h + 1, w + 1), dtype=np.int32)
        S[1:, 1:] = mask.cumsum(axis=0).cumsum(axis=1)
        out[:, :, color] = (
            S[rmax + 1, cmax + 1] - S[rmin, cmax + 1] -
            S[rmax + 1, cmin] + S[rmin, cmin]
        ).astype(np.int16)

    return out


# ========== components (4-connected, same color) ==========

def components4(g: Grid) -> Tuple[np.ndarray, Dict[int, Dict]]:
    """
    4-connected components of equal color, enumerated in row-major discovery order.
    Returns (labels, summaries):
      labels: (h,w) np.int32, values 0..C-1
      summaries: Dict[int, Dict] with keys:
        "area": int
        "bbox": (minr, minc, maxr, maxc)
        "color": int in 0..9
        "shape": (height, width) from bbox
        "border_contact": 0/1 if any pixel touches image border
    Algorithm: DFS with Python stack (no recursion), deterministic row-major order.
    """
    assert_grid(g)
    h, w = g.shape
    labels = -np.ones((h, w), dtype=np.int32)
    next_id = 0
    summaries = {}

    for r in range(h):
        for c in range(w):
            if labels[r, c] != -1:
                continue

            color = int(g[r, c])
            stack = [(r, c)]
            labels[r, c] = next_id

            area = 0
            minr = maxr = r
            minc = maxc = c
            border_contact = False

            while stack:
                y, x = stack.pop()
                area += 1

                # Update bbox
                if y < minr:
                    minr = y
                if y > maxr:
                    maxr = y
                if x < minc:
                    minc = x
                if x > maxc:
                    maxc = x

                # Check border
                if y == 0 or y == h - 1 or x == 0 or x == w - 1:
                    border_contact = True

                # Visit 4 neighbors
                for ny, nx in [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]:
                    if 0 <= ny < h and 0 <= nx < w:
                        if labels[ny, nx] == -1 and int(g[ny, nx]) == color:
                            labels[ny, nx] = next_id
                            stack.append((ny, nx))

            # Save summary
            bbox = (minr, minc, maxr, maxc)
            summaries[next_id] = {
                "area": area,
                "bbox": bbox,
                "color": color,
                "shape": (maxr - minr + 1, maxc - minc + 1),
                "border_contact": 1 if border_contact else 0
            }
            next_id += 1

    return labels, summaries
