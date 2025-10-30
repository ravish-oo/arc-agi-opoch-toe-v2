# src/bt/boundary.py
from typing import List, Tuple, Dict
from collections import defaultdict
import numpy as np

from ..types import Grid, Boundary, QtSpec, ShapeLaw, ShapeLawKind
from ..qt.quotient import classes_for, _ExtraFlags
from ..qt.spec import MAX_RESIDUES


def probe_writer_mode(
    canon_train_pairs: List[Tuple[Grid, Grid]],  # (X_canon, Y_canon)
    delta: ShapeLaw
) -> str:
    """
    Deterministic writer probe: decide 'blowup' vs 'tiling' for size-changed tasks.

    Checks training outputs:
    - BLOW_UP: each kh×kw block is constant color
    - TILING: each h×w pane is either all zeros or equals X

    Returns 'blowup', 'tiling', or 'identity'
    """
    if delta.kind != ShapeLawKind.BLOW_UP:
        return 'identity'

    if delta.kh == 1 and delta.kw == 1:
        return 'identity'

    kh, kw = delta.kh, delta.kw

    blowup_votes = 0
    tiling_votes = 0

    for cx, cy in canon_train_pairs:
        h, w = cx.shape
        H, W = cy.shape

        # Skip if not size-changed
        if (h, w) == (H, W):
            continue

        # Check BLOW_UP: all kh×kw blocks constant?
        all_blocks_constant = True
        for pr in range(h):
            for pc in range(w):
                block = cy[pr*kh:(pr+1)*kh, pc*kw:(pc+1)*kw]
                if len(np.unique(block)) > 1:
                    all_blocks_constant = False
                    break
            if not all_blocks_constant:
                break

        # Check TILING: h×w panes are zeros or equal to X?
        all_panes_valid = True
        for pr in range(kh):
            for pc in range(kw):
                pane = cy[pr*h:(pr+1)*h, pc*w:(pc+1)*w]
                is_zero = np.all(pane == 0)
                is_input = np.array_equal(pane, cx)
                if not (is_zero or is_input):
                    all_panes_valid = False
                    break
            if not all_panes_valid:
                break

        if all_blocks_constant:
            blowup_votes += 1
        if all_panes_valid:
            tiling_votes += 1

    # Decision: majority vote, tie-break 'blowup'
    if tiling_votes > blowup_votes:
        return 'tiling'
    else:
        return 'blowup'


def check_boundary_forced(
    train_pairs: List[Tuple[Grid, Grid]],
    spec: QtSpec,
    delta: ShapeLaw,
    writer_mode: str,
    extra: _ExtraFlags | None = None,
) -> Tuple[Boundary, bool]:
    """
    Aggregate evidence with Δ-aware pullback on ALL pairs (including size-changed).

    For each (X_canon, Y_canon):
      cls = classes_for(X_canon, spec, extra)
      for each output pixel Y[R,C]:
        - pull back to input: (r,c) = pullback(R,C) based on delta and writer_mode
        - key = cls.key_for[cls.ids[r,c]]
        - add Y[R,C] to bucket[key]

    Pullback mappings:
      - IDENTITY: (r,c) = (R,C)
      - BLOW_UP: (r,c) = (R//kh, C//kw)
      - TILING: (r,c) = (R%h, C%w)

    Returns:
      Boundary(forced_color=dict, unforced=list), all_forced = (len(unforced) == 0)

    Deterministic, no side effects.
    Uses stable bytes keys for cross-grid class identity.
    """
    # Initialize bucket: bytes key -> set of colors
    bucket: Dict[bytes, set] = defaultdict(set)

    # Aggregate evidence from ALL pairs with Δ-aware pullback
    for X, Y in train_pairs:
        h, w = X.shape
        H, W = Y.shape

        # Compute classes for input X
        cls = classes_for(X, spec, extra)

        # Determine pullback mapping
        if (h, w) == (H, W):
            # IDENTITY: direct mapping
            for R in range(H):
                for C in range(W):
                    r, c = R, C
                    local_id = int(cls.ids[r, c])
                    key = cls.key_for[local_id]
                    color = int(Y[R, C])
                    bucket[key].add(color)
        else:
            # Size-changed: use Δ-aware pullback
            kh, kw = delta.kh, delta.kw

            if writer_mode == 'blowup':
                # BLOW_UP: (r,c) = (R//kh, C//kw)
                for R in range(H):
                    for C in range(W):
                        r = R // kh
                        c = C // kw
                        if r < h and c < w:
                            local_id = int(cls.ids[r, c])
                            key = cls.key_for[local_id]
                            color = int(Y[R, C])
                            bucket[key].add(color)
            elif writer_mode == 'tiling':
                # TILING: (r,c) = (R%h, C%w)
                for R in range(H):
                    for C in range(W):
                        r = R % h
                        c = C % w
                        local_id = int(cls.ids[r, c])
                        key = cls.key_for[local_id]
                        color = int(Y[R, C])
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
    delta: ShapeLaw,
    writer_mode: str,
) -> Tuple[Boundary, QtSpec, _ExtraFlags]:
    """
    Deterministic refinement ladder S0..S6 with Δ-aware pullback.

    Args:
        train_pairs: canonized (X_canon, Y_canon) pairs
        initial_spec: initial QtSpec
        delta: ShapeLaw for canvas size
        writer_mode: 'identity', 'blowup', or 'tiling'

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
    bt, all_forced = check_boundary_forced(train_pairs, spec, delta, writer_mode, extra)
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
        bt, all_forced = check_boundary_forced(train_pairs, spec, delta, writer_mode, extra)
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
        bt, all_forced = check_boundary_forced(train_pairs, spec, delta, writer_mode, extra)
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
        bt, all_forced = check_boundary_forced(train_pairs, spec, delta, writer_mode, extra)
        if all_forced:
            return bt, spec, extra

    # S4: Add distance-to-border channel (input-only)
    if not extra.use_border_distance:
        extra.use_border_distance = True
        bt, all_forced = check_boundary_forced(train_pairs, spec, delta, writer_mode, extra)
        if all_forced:
            return bt, spec, extra

    # S5a: Component centroid parity
    if not extra.use_centroid_parity and not extra.use_component_scan_index:
        extra.use_centroid_parity = True
        bt, all_forced = check_boundary_forced(train_pairs, spec, delta, writer_mode, extra)
        if all_forced:
            return bt, spec, extra

    # S5b: Component scan index (try alternative if centroid parity didn't force)
    if not all_forced:
        extra.use_centroid_parity = False
        extra.use_component_scan_index = True
        bt, all_forced = check_boundary_forced(train_pairs, spec, delta, writer_mode, extra)
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
        bt, all_forced = check_boundary_forced(train_pairs, spec, delta, writer_mode, extra)

    # Return final result (may still have unforced keys)
    return bt, spec, extra
