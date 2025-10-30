# src/present/pi.py
from typing import List, Tuple
import numpy as np
from ..types import Grid, CanonMeta, Canonized
from ..kernel.grid import assert_grid, d8_apply, d8_inv

# Global DEBUG flag (off by default)
DEBUG: bool = False

# ========== rank/keys ==========

def color_counts(g: Grid) -> np.ndarray:
    """
    Count occurrences of each color (0..9) in grid.
    Returns length-10 np.int64 array.
    """
    flat = g.flatten()
    cnt = np.bincount(flat, minlength=10)
    return cnt.astype(np.int64)


def rank_order_from_counts(counts: np.ndarray) -> np.ndarray:
    """
    Given color counts, return permutation of 0..9 listing colors in:
      - descending frequency (primary)
      - ascending color ID (tie-break)
    Returns np.ndarray of int, shape (10,).
    """
    # Create array of (count, color_id) tuples for sorting
    # Sort by (-count, color_id) to get desired order
    colors = np.arange(10, dtype=int)
    # Use lexsort: rightmost key is primary
    order = np.lexsort((colors, -counts))
    return order


def rank_view_from_order(g: Grid, order: np.ndarray) -> np.ndarray:
    """
    Build rank-view: replace each color with its rank according to order.
    order[i] = color with rank i
    Returns np.int8, C-contiguous.
    """
    # Build inverse: rank[color] = position in order
    rank = np.empty(10, dtype=np.int8)
    for i, color in enumerate(order):
        rank[color] = i

    # Apply to grid
    rv = rank[g]
    return np.ascontiguousarray(rv, dtype=np.int8)


def rank_view_per_grid(g: Grid) -> np.ndarray:
    """
    Compute per-grid rank-view using its own color frequency.
    Returns np.int8, C-contiguous.
    """
    cnt = color_counts(g)
    order = rank_order_from_counts(cnt)
    return rank_view_from_order(g, order)


def rank_view_union(g: Grid, union_order: np.ndarray) -> np.ndarray:
    """
    Compute rank-view using precomputed union order from batch.
    Returns np.int8, C-contiguous.
    """
    return rank_view_from_order(g, union_order)


# ========== canon for one grid (requires a union order) ==========

def canon_one_with_union(g: Grid, union_order: np.ndarray) -> Tuple[Grid, CanonMeta]:
    """
    Choose canonical D8 pose for grid using three-level lexicographic key:
      1. Per-grid rank-view bytes (primary)
      2. Union rank-view bytes (tie-break)
      3. Raw grid bytes (final tie-break)

    Returns (canon_grid, CanonMeta) where canon_grid is C-contiguous np.int8.
    """
    global DEBUG

    assert_grid(g)

    best_t = 0
    best_key = None

    for t in range(8):
        cand = d8_apply(g, t)

        # Primary key: per-grid rank-view
        rv_local = rank_view_per_grid(cand)
        key1 = rv_local.tobytes()

        # Secondary key: union rank-view
        rv_union = rank_view_union(cand, union_order)
        key2 = rv_union.tobytes()

        # Tertiary key: raw bytes
        key3 = cand.tobytes()

        # Lexicographic tuple
        key = (key1, key2, key3)

        if best_key is None or key < best_key:
            best_key = key
            best_t = t

    cx = d8_apply(g, best_t)

    # DEBUG: check idempotence (avoid recursion by temporarily disabling DEBUG)
    if DEBUG:
        DEBUG_saved = DEBUG
        DEBUG = False
        try:
            cx_again, meta_again = canon_one_with_union(cx, union_order)
            assert meta_again.transform_id == 0, \
                f"Idempotence failed: canon(canon(g)) gave transform {meta_again.transform_id} != 0"
            assert np.array_equal(cx_again, cx), \
                "Idempotence failed: canon(canon(g)) changed grid"
        finally:
            DEBUG = DEBUG_saved

    return cx, CanonMeta(transform_id=best_t, original_shape=g.shape)


# ========== batch API (computes union order once) ==========

def canonize_inputs(xs: List[Grid]) -> Canonized:
    """
    Batch canonization: compute union palette order once, apply to all grids.
    Returns Canonized(grids, metas).
    """
    # Validate all inputs
    for x in xs:
        assert_grid(x)

    # Compute union counts
    U = np.zeros(10, dtype=np.int64)
    for x in xs:
        U += color_counts(x)

    # Union order
    union_order = rank_order_from_counts(U)

    # Canonize each grid
    grids = []
    metas = []
    for x in xs:
        cx, meta = canon_one_with_union(x, union_order)
        grids.append(cx)
        metas.append(meta)

    return Canonized(grids=grids, metas=metas)


def compute_union_order(train_xs: List[Grid], test_xs: List[Grid]) -> np.ndarray:
    """
    Inputs-only: sum color counts across all grids (train+test) to build a single union palette order.
    Deterministic for the same multiset of inputs.
    Returns np.ndarray of int, shape (10,).
    """
    U = np.zeros(10, dtype=np.int64)
    for x in train_xs:
        U += color_counts(x)
    for x in test_xs:
        U += color_counts(x)
    return rank_order_from_counts(U)


def canonize_with_union(xs: List[Grid], union_order: np.ndarray) -> Canonized:
    """
    Canonize a list of grids using a precomputed union palette order (inputs-only).
    Returns Canonized(grids, metas).
    """
    grids = []
    metas = []
    for x in xs:
        cx, meta = canon_one_with_union(x, union_order)
        grids.append(cx)
        metas.append(meta)
    return Canonized(grids=grids, metas=metas)


def canonize_task(train_xs: List[Grid], test_xs: List[Grid]) -> Tuple[Canonized, Canonized, np.ndarray]:
    """
    Task-level Π: compute a single union order over train+test inputs and canonize both sets with it.
    Returns (c_train, c_test, union_order).

    This ensures train and test use the same union palette order, so D8 tie-breaks are consistent
    and class keys learned on train apply to test.
    """
    union_order = compute_union_order(train_xs, test_xs)
    c_train = canonize_with_union(train_xs, union_order)
    c_test = canonize_with_union(test_xs, union_order)
    return c_train, c_test, union_order


def uncanonize(g: Grid, meta: CanonMeta) -> Grid:
    """
    Invert canonization: apply inverse D8 transform to restore original pose.
    Returns C-contiguous np.int8.
    """
    inv_t = d8_inv(meta.transform_id)
    return d8_apply(g, inv_t)
