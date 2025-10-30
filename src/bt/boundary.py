# src/bt/boundary.py
from typing import List, Tuple, Dict
from collections import defaultdict
import numpy as np

from ..types import Grid, Boundary, QtSpec, ShapeLaw, ShapeLawKind
from ..qt.quotient import classes_for, _ExtraFlags
from ..qt.spec import MAX_RESIDUES


def apply_pane_transform(grid: Grid, pr: int, pc: int, policy: str) -> Grid:
    """
    Apply pane transform based on policy.

    Policies:
    - 'uniform': no transform
    - 'row_FH': horizontal flip on odd pane rows (pr % 2 == 1)
    - 'col_FH': horizontal flip on odd pane cols (pc % 2 == 1)
    - 'checker_FH': horizontal flip when (pr + pc) % 2 == 1
    """
    if policy == 'uniform':
        return grid
    elif policy == 'row_FH':
        if pr % 2 == 1:
            return np.fliplr(grid)
        return grid
    elif policy == 'col_FH':
        if pc % 2 == 1:
            return np.fliplr(grid)
        return grid
    elif policy == 'checker_FH':
        if (pr + pc) % 2 == 1:
            return np.fliplr(grid)
        return grid
    else:
        return grid


def inverse_pane_coords(r: int, c: int, w: int, pr: int, pc: int, policy: str) -> Tuple[int, int]:
    """
    Apply inverse transform to coordinates within a pane.

    Returns source coordinates (r', c') in the original motif.
    """
    if policy == 'uniform':
        return r, c
    elif policy == 'row_FH':
        if pr % 2 == 1:
            return r, w - 1 - c
        return r, c
    elif policy == 'col_FH':
        if pc % 2 == 1:
            return r, w - 1 - c
        return r, c
    elif policy == 'checker_FH':
        if (pr + pc) % 2 == 1:
            return r, w - 1 - c
        return r, c
    else:
        return r, c




def probe_writer_mode(
    canon_train_pairs: List[Tuple[Grid, Grid]],  # (X_canon, Y_canon)
    kh: int,
    kw: int,
    policies: Tuple[str, ...] = ("uniform", "row_FH", "col_FH", "checker_FH")
) -> Tuple[str, str | None]:
    """
    Return (writer_mode, tiling_policy_or_None).

    Deterministically tests blow-up vs tiling(+policy) on TRAIN PAIRS ONLY.
    Uses apply_pane_transform for motif transforms.

    - BLOW_UP: each kh×kw block is constant color
    - TILING: each h×w pane equals transformed input or is all zeros

    Returns:
        ('blowup', None) or ('tiling', policy_string)
    """
    if not canon_train_pairs:
        return ('identity', None)

    h, w = canon_train_pairs[0][0].shape
    H, W = canon_train_pairs[0][1].shape

    # Skip if not size-changed
    if (h, w) == (H, W):
        return ('identity', None)

    if kh == 1 and kw == 1:
        return ('identity', None)

    # 1) Blow-up error: sum of non-constant kh×kw blocks
    err_bu = 0
    for X_c, Y_c in canon_train_pairs:
        for R in range(0, H, kh):
            for C in range(0, W, kw):
                block = Y_c[R:R+kh, C:C+kw]
                if not (block == block[0, 0]).all():
                    err_bu += int(np.count_nonzero(block != block[0, 0]))

    # 2) Tiling error per policy: ON if pane == T(X_c); OFF if pane==0; otherwise penalize pixel diffs
    err_ti = {p: 0 for p in policies}
    for policy in policies:
        total = 0
        for X_c, Y_c in canon_train_pairs:
            for pr in range(kh):
                for pc in range(kw):
                    pane = Y_c[pr*h:(pr+1)*h, pc*w:(pc+1)*w]
                    motif = apply_pane_transform(X_c, pr, pc, policy)
                    if (pane == motif).all() or (pane == 0).all():
                        continue
                    total += int(np.count_nonzero(pane != motif))
        err_ti[policy] = total

    # 3) Decide writer + policy (deterministic tie-break)
    best_policy, best_err_ti = min(err_ti.items(), key=lambda kv: (kv[1], kv[0]))
    if best_err_ti < err_bu:
        return ('tiling', best_policy)
    return ('blowup', None)


def check_boundary_forced(
    train_pairs: List[Tuple[Grid, Grid]],
    spec: QtSpec,
    delta: ShapeLaw,
    writer_mode: str,
    tiling_policy: str = 'uniform',
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
                # TILING: pane-aware learning with policy transform (only from ON panes)
                # Build ON/OFF mask for each pane
                kh, kw = delta.kh, delta.kw
                pane_mask = np.zeros((kh, kw), dtype=bool)

                for pr in range(kh):
                    for pc in range(kw):
                        pane = Y[pr*h:(pr+1)*h, pc*w:(pc+1)*w]

                        # Apply pane transform to input
                        transformed_input = apply_pane_transform(X, pr, pc, tiling_policy)

                        # ON pane: pane equals transformed input motif
                        is_on = np.array_equal(pane, transformed_input)

                        # OFF pane: all zeros (ignore these)
                        is_off = np.all(pane == 0)

                        # Only mark as ON if it matches transformed motif
                        # Skip ambiguous panes (neither exact match nor all zeros)
                        if is_on:
                            pane_mask[pr, pc] = True

                # Only bucket from ON panes
                for pr in range(kh):
                    for pc in range(kw):
                        if not pane_mask[pr, pc]:
                            continue  # Skip OFF panes

                        # Bucket from this ON pane with inverse transform
                        for r in range(h):
                            for c in range(w):
                                R = pr * h + r
                                C = pc * w + c

                                # Apply inverse transform to get source coords in motif
                                r_src, c_src = inverse_pane_coords(r, c, w, pr, pc, tiling_policy)

                                local_id = int(cls.ids[r_src, c_src])
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
    tiling_policy: str = 'uniform',
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
    bt, all_forced = check_boundary_forced(train_pairs, spec, delta, writer_mode, tiling_policy, extra)
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
        bt, all_forced = check_boundary_forced(train_pairs, spec, delta, writer_mode, tiling_policy, extra)
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
        bt, all_forced = check_boundary_forced(train_pairs, spec, delta, writer_mode, tiling_policy, extra)
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
        bt, all_forced = check_boundary_forced(train_pairs, spec, delta, writer_mode, tiling_policy, extra)
        if all_forced:
            return bt, spec, extra

    # S4: Add distance-to-border channel (input-only)
    if not extra.use_border_distance:
        extra.use_border_distance = True
        bt, all_forced = check_boundary_forced(train_pairs, spec, delta, writer_mode, tiling_policy, extra)
        if all_forced:
            return bt, spec, extra

    # S5a: Component centroid parity
    if not extra.use_centroid_parity and not extra.use_component_scan_index:
        extra.use_centroid_parity = True
        bt, all_forced = check_boundary_forced(train_pairs, spec, delta, writer_mode, tiling_policy, extra)
        if all_forced:
            return bt, spec, extra

    # S5b: Component scan index (try alternative if centroid parity didn't force)
    if not all_forced:
        extra.use_centroid_parity = False
        extra.use_component_scan_index = True
        bt, all_forced = check_boundary_forced(train_pairs, spec, delta, writer_mode, tiling_policy, extra)
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
        bt, all_forced = check_boundary_forced(train_pairs, spec, delta, writer_mode, tiling_policy, extra)

    # Return final result (may still have unforced keys)
    return bt, spec, extra
