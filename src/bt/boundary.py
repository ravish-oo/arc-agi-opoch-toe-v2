# src/bt/boundary.py
from typing import List, Tuple, Dict
from collections import defaultdict
import numpy as np

from ..types import Grid, Boundary, QtSpec
from ..qt.quotient import classes_for, _ExtraFlags
from ..qt.spec import MAX_RESIDUES


def check_boundary_forced(
    train_pairs: List[Tuple[Grid, Grid]],
    spec: QtSpec,
    extra: _ExtraFlags | None = None,
) -> Tuple[Boundary, bool]:
    """
    Aggregate evidence on identity-shaped pairs only.

    For each (X,Y) with X.shape == Y.shape:
      cls = classes_for(X, spec, extra)
      for each pixel p: key = cls.key_for[cls.ids[p]]; add Y[p] to bucket[key]

    Returns:
      Boundary(forced_color=dict, unforced=list), all_forced = (len(unforced) == 0)

    Deterministic, no side effects.
    Uses stable bytes keys for cross-grid class identity.
    """
    # Initialize bucket: bytes key -> set of colors
    bucket: Dict[bytes, set] = defaultdict(set)

    # Aggregate evidence from identity-shaped pairs only
    for X, Y in train_pairs:
        # Skip size-changed pairs (Δ handled later in Φ)
        if X.shape != Y.shape:
            continue

        # Compute classes for input X
        cls = classes_for(X, spec, extra)

        # Flatten for vectorized processing
        ids = cls.ids.ravel(order="C")
        ys = Y.ravel(order="C")

        # Accumulate colors per class key
        for i in range(ids.size):
            local_id = int(ids[i])
            key = cls.key_for[local_id]  # Stable bytes key
            color = int(ys[i])
            bucket[key].add(color)

    # Build forced and unforced maps
    forced_color: Dict[bytes, int] = {}
    unforced: List[bytes] = []

    for key, colors in bucket.items():
        if len(colors) == 1:
            # Single color → forced
            forced_color[key] = next(iter(colors))
        elif len(colors) > 1:
            # Multiple colors → unforced (collision)
            unforced.append(key)

    # Check if all keys are forced
    all_forced = (len(unforced) == 0)

    return Boundary(forced_color=forced_color, unforced=unforced), all_forced


def extract_bt_force_until_forced(
    train_pairs: List[Tuple[Grid, Grid]],
    initial_spec: QtSpec,
) -> Tuple[Boundary, QtSpec, _ExtraFlags]:
    """
    Deterministic refinement ladder S0..S6.

    Returns:
      - final Boundary (may still have unforced in worst case)
      - final QtSpec (feature family after refinements)
      - final _ExtraFlags (which extra channels got enabled)

    Stops at first step that achieves all_forced=True.
    Each step strictly increases discriminative power (input-only).
    """
    # Initialize mutable state
    spec = initial_spec
    extra = _ExtraFlags()

    # S0: Base spec (from WO-04)
    bt, all_forced = check_boundary_forced(train_pairs, spec, extra)
    if all_forced:
        return bt, spec, extra

    # S1: Extend residues to all k in [2..min(max_dim, MAX_RESIDUES)]
    # Compute max dimension from train inputs
    max_dim = 0
    for X, _ in train_pairs:
        h, w = X.shape
        max_dim = max(max_dim, h, w)

    # Build extended residue set
    new_res_set = set(spec.residues)
    for k in range(2, min(max_dim, MAX_RESIDUES) + 1):
        new_res_set.add(k)

    # Sort and cap at MAX_RESIDUES
    new_res = sorted(new_res_set)[:MAX_RESIDUES]

    # Only update if residues changed
    if tuple(new_res) != spec.residues:
        spec = QtSpec(
            radii=spec.radii,
            residues=tuple(new_res),
            use_diagonals=spec.use_diagonals,
            wl_rounds=spec.wl_rounds
        )
        bt, all_forced = check_boundary_forced(train_pairs, spec, extra)
        if all_forced:
            return bt, spec, extra

    # S2: Add radius = 3
    if 3 not in spec.radii:
        spec = QtSpec(
            radii=tuple(sorted(spec.radii + (3,))),
            residues=spec.residues,
            use_diagonals=spec.use_diagonals,
            wl_rounds=spec.wl_rounds
        )
        bt, all_forced = check_boundary_forced(train_pairs, spec, extra)
        if all_forced:
            return bt, spec, extra

    # S3: WL rounds = 4
    if spec.wl_rounds < 4:
        spec = QtSpec(
            radii=spec.radii,
            residues=spec.residues,
            use_diagonals=spec.use_diagonals,
            wl_rounds=4
        )
        bt, all_forced = check_boundary_forced(train_pairs, spec, extra)
        if all_forced:
            return bt, spec, extra

    # S4: Add distance-to-border channel (input-only)
    if not extra.use_border_distance:
        extra.use_border_distance = True
        bt, all_forced = check_boundary_forced(train_pairs, spec, extra)
        if all_forced:
            return bt, spec, extra

    # S5a: Component centroid parity
    if not extra.use_centroid_parity and not extra.use_component_scan_index:
        extra.use_centroid_parity = True
        bt, all_forced = check_boundary_forced(train_pairs, spec, extra)
        if all_forced:
            return bt, spec, extra

    # S5b: Component scan index (try alternative if centroid parity didn't force)
    if not all_forced:
        extra.use_centroid_parity = False
        extra.use_component_scan_index = True
        bt, all_forced = check_boundary_forced(train_pairs, spec, extra)
        if all_forced:
            return bt, spec, extra

    # S6: WL rounds = 5 (final step)
    if spec.wl_rounds < 5:
        spec = QtSpec(
            radii=spec.radii,
            residues=spec.residues,
            use_diagonals=spec.use_diagonals,
            wl_rounds=5
        )
        bt, all_forced = check_boundary_forced(train_pairs, spec, extra)

    # Return final result (may still have unforced keys)
    return bt, spec, extra
