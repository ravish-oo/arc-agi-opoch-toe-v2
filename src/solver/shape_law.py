# src/solver/shape_law.py
from typing import List, Tuple
import numpy as np
from ..types import Grid, ShapeLaw, ShapeLawKind


def _minimal_period_axis(g: Grid, axis: int) -> int:
    """
    Returns the smallest p >= 1 that divides length L along `axis`,
    such that g equals g rolled by p repeatedly to cover L.
    If none (other than L), returns L (no strict periodicity).

    This is an input-only check (no output content consulted).
    """
    L = g.shape[axis]

    # Check divisors in ascending order (excluding L itself)
    for p in range(1, L):
        if L % p == 0:
            # Check if rolling by p gives the same grid
            rolled = np.roll(g, shift=p, axis=axis)
            if np.array_equal(g, rolled):
                return p

    # No proper divisor qualifies - not strictly periodic
    return L


def _is_strongly_periodic(g: Grid) -> bool:
    """
    Check if grid g is strictly periodic along both axes.
    Returns True if minimal period < full length for both axes.
    """
    h, w = g.shape
    period_h = _minimal_period_axis(g, axis=0)
    period_w = _minimal_period_axis(g, axis=1)

    return period_h < h and period_w < w


def infer_shape_law(
    train_pairs: List[Tuple[Grid, Grid]],
    *,
    enable_frame: bool = False,
    enable_tiling: bool = False,
    periodicity_check: bool = False
) -> ShapeLaw:
    """
    Deterministic Δ inference from train pair dimensions only.

    Always supported:
      - IDENTITY: output shape == input shape
      - BLOW_UP(kh, kw): integer scale (kh*h, kw*w), rectangular allowed

    Optional (when enabled via flags):
      - FRAME(t): add border (dimensions only: H-h == W-w == 2t)
      - TILING(kh, kw): same ratios as BLOW_UP, selected by tie-break if periodicity holds

    Tie-break: when both BLOW_UP and TILING are arithmetically possible,
      prefer BLOW_UP unless (enable_tiling and periodicity_check and all inputs are periodic).

    Parameters:
      train_pairs: list of (input_grid, output_grid) from training
      enable_frame: allow FRAME detection (default False)
      enable_tiling: allow TILING detection (default False)
      periodicity_check: perform input-only periodicity validation for TILING (default False)

    Returns:
      ShapeLaw object with kind and scale factors
    """
    # Edge case: empty train
    if not train_pairs:
        return ShapeLaw(kind=ShapeLawKind.IDENTITY)

    # 4.1 Gather dimensions
    HX = []  # input heights
    WX = []  # input widths
    HY = []  # output heights
    WY = []  # output widths

    for x, y in train_pairs:
        hx, wx = x.shape
        hy, wy = y.shape
        HX.append(hx)
        WX.append(wx)
        HY.append(hy)
        WY.append(wy)

    n = len(train_pairs)

    # 4.2 Identity test (first rule)
    if all(HY[i] == HX[i] and WY[i] == WX[i] for i in range(n)):
        return ShapeLaw(kind=ShapeLawKind.IDENTITY)

    # 4.3 Integer ratio test (candidate blow-up / tiling)
    kh_list = []
    kw_list = []
    ratios_valid = True

    for i in range(n):
        # Check divisibility
        if HX[i] > 0 and WX[i] > 0 and HY[i] % HX[i] == 0 and WY[i] % WX[i] == 0:
            kh_i = HY[i] // HX[i]
            kw_i = WY[i] // WX[i]
            kh_list.append(kh_i)
            kw_list.append(kw_i)
        else:
            ratios_valid = False
            break

    # Check if ratios are constant (singleton sets)
    blow_up_candidate = None
    if ratios_valid and len(set(kh_list)) == 1 and len(set(kw_list)) == 1:
        kh = kh_list[0]
        kw = kw_list[0]
        # At least one must be > 1 for true blow-up (otherwise it's identity)
        if kh > 1 or kw > 1:
            blow_up_candidate = (kh, kw)

    # 4.3.1 Frame test (optional and purely dimensional)
    frame_candidate = None
    if enable_frame:
        # Check if all pairs have HY - HX == WY - WX == 2*t for same t >= 1
        t_list = []
        frame_valid = True

        for i in range(n):
            diff_h = HY[i] - HX[i]
            diff_w = WY[i] - WX[i]

            # Must be same difference and even
            if diff_h == diff_w and diff_h > 0 and diff_h % 2 == 0:
                t_i = diff_h // 2
                t_list.append(t_i)
            else:
                frame_valid = False
                break

        # Check if t is constant across all pairs
        if frame_valid and len(set(t_list)) == 1:
            t = t_list[0]
            if t >= 1:
                frame_candidate = t

    # 4.4 Candidate set - we have blow_up_candidate and frame_candidate
    # TILING has same arithmetic as BLOW_UP but different semantics

    # 4.5 Deterministic tie-break

    # Rule 1: If only one candidate exists, pick it
    has_blow = blow_up_candidate is not None
    has_frame = frame_candidate is not None

    # Rule 2: If BLOW_UP exists
    if has_blow:
        kh, kw = blow_up_candidate

        # Check if TILING should be considered
        tiling_selected = False
        if enable_tiling and periodicity_check:
            # Check strong periodicity on all train inputs
            all_periodic = all(
                _is_strongly_periodic(x)
                for x, _ in train_pairs
            )
            if all_periodic:
                tiling_selected = True

        if tiling_selected:
            return ShapeLaw(kind=ShapeLawKind.TILING, kh=kh, kw=kw)
        else:
            return ShapeLaw(kind=ShapeLawKind.BLOW_UP, kh=kh, kw=kw)

    # Rule 3: Else if FRAME exists
    if has_frame:
        t = frame_candidate
        # Store t in both kh and kw fields for FRAME
        return ShapeLaw(kind=ShapeLawKind.FRAME, kh=t, kw=t)

    # Rule 4: Else pick IDENTITY
    return ShapeLaw(kind=ShapeLawKind.IDENTITY)
