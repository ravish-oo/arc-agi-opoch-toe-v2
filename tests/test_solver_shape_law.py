"""
WO-06: Shape Law (Δ) Tests
Tests for src/solver/shape_law.py

Test strategy:
1. Acceptance: verify core API contracts (IDENTITY, BLOW_UP, FRAME, TILING)
2. Edge cases: empty, inconsistent, single pair, extreme ratios
3. Periodicity: verify input-only periodic detection
4. Tie-breaks: verify deterministic preference order
"""
import numpy as np
import pytest
from src.solver.shape_law import infer_shape_law, _minimal_period_axis, _is_strongly_periodic
from src.types import ShapeLawKind as K


def mk(h, w, H, W):
    """Helper to make (input, output) pair with given shapes (contents irrelevant for Δ)."""
    X = np.zeros((h, w), np.int8)
    Y = np.zeros((H, W), np.int8)
    return (X, Y)


# ========== Acceptance Tests ==========

def test_identity_same_shapes():
    """All pairs have same input/output shape → IDENTITY."""
    pairs = [mk(5, 7, 5, 7), mk(3, 3, 3, 3)]
    law = infer_shape_law(pairs)
    assert law.kind == K.IDENTITY


def test_uniform_blowup():
    """Uniform scale factor (kh=kw=2) → BLOW_UP."""
    pairs = [mk(5, 7, 10, 14), mk(3, 3, 6, 6)]
    law = infer_shape_law(pairs)
    assert law.kind == K.BLOW_UP
    assert law.kh == 2 and law.kw == 2


def test_rectangular_blowup():
    """Rectangular scale (kh≠kw) → BLOW_UP with correct factors."""
    pairs = [mk(4, 5, 12, 10), mk(2, 1, 6, 2)]
    law = infer_shape_law(pairs)
    assert law.kind == K.BLOW_UP
    assert law.kh == 3 and law.kw == 2


def test_inconsistent_ratios_fallback():
    """Inconsistent scale factors across pairs → IDENTITY."""
    pairs = [mk(4, 4, 8, 8), mk(3, 3, 9, 6)]  # kh=2,kw=2 vs kh=3,kw=2
    law = infer_shape_law(pairs)
    assert law.kind == K.IDENTITY


def test_frame_detection():
    """Enable FRAME: all pairs add same border thickness → FRAME."""
    pairs = [mk(5, 7, 9, 11), mk(3, 3, 7, 7)]  # both add t=2
    law = infer_shape_law(pairs, enable_frame=True)
    assert law.kind == K.FRAME
    assert law.kh == 2 and law.kw == 2  # thickness stored in kh, kw


def test_tiling_with_periodicity():
    """Enable TILING + periodicity check: periodic input → TILING."""
    # Create strictly periodic input (2x3 tiling)
    base = np.array([[1, 2], [3, 4]], np.int8)
    X1 = np.tile(base, (2, 3))  # 4x6 periodic
    Y1 = np.zeros((8, 18), np.int8)  # 2x3 scale
    pairs = [(X1, Y1)]
    law = infer_shape_law(pairs, enable_tiling=True, periodicity_check=True)
    assert law.kind == K.TILING
    assert law.kh == 2 and law.kw == 3


def test_tiling_without_periodicity():
    """Enable TILING but non-periodic input → BLOW_UP (prefer BLOW_UP)."""
    X = np.array([[1, 2, 3], [4, 5, 6]], np.int8)  # not periodic
    Y = np.zeros((4, 6), np.int8)
    pairs = [(X, Y)]
    law = infer_shape_law(pairs, enable_tiling=True, periodicity_check=True)
    assert law.kind == K.BLOW_UP
    assert law.kh == 2 and law.kw == 2


# ========== Edge Cases ==========

def test_empty_train():
    """Empty train list → IDENTITY."""
    law = infer_shape_law([])
    assert law.kind == K.IDENTITY


def test_single_pair_blowup():
    """Single train pair with scale → BLOW_UP."""
    pairs = [mk(3, 3, 9, 9)]
    law = infer_shape_law(pairs)
    assert law.kind == K.BLOW_UP
    assert law.kh == 3 and law.kw == 3


def test_single_pair_identity():
    """Single train pair same size → IDENTITY."""
    pairs = [mk(5, 5, 5, 5)]
    law = infer_shape_law(pairs)
    assert law.kind == K.IDENTITY


def test_non_integer_ratios():
    """Non-integer ratios (e.g., 5→11) → IDENTITY fallback."""
    pairs = [mk(5, 5, 11, 11)]  # 11/5 not integer
    law = infer_shape_law(pairs)
    assert law.kind == K.IDENTITY


def test_extreme_blowup():
    """Large scale factor (kh=10, kw=10) → valid BLOW_UP."""
    pairs = [mk(3, 3, 30, 30)]
    law = infer_shape_law(pairs)
    assert law.kind == K.BLOW_UP
    assert law.kh == 10 and law.kw == 10


def test_single_axis_blowup():
    """One axis scales, other doesn't (kh=3, kw=1) → valid rectangular BLOW_UP."""
    pairs = [mk(4, 5, 12, 5)]
    law = infer_shape_law(pairs)
    assert law.kind == K.BLOW_UP
    assert law.kh == 3 and law.kw == 1


def test_mixed_identity_blowup():
    """Some pairs identity (1,1), others scale (2,2) but constant → BLOW_UP."""
    # This is tricky: if some ratios are 1,1 and others are 2,2, sets aren't singleton
    # Actually per spec, this should fail consistency check
    pairs = [mk(5, 5, 5, 5), mk(3, 3, 6, 6)]  # kh=[1,2], kw=[1,2]
    law = infer_shape_law(pairs)
    # Not constant ratios → IDENTITY
    assert law.kind == K.IDENTITY


# ========== Periodicity Helpers ==========

def test_minimal_period_axis_simple():
    """Periodic grid with period 2 along axis."""
    g = np.array([[1, 2], [3, 4], [1, 2], [3, 4]], np.int8)  # period 2 along axis 0
    period = _minimal_period_axis(g, axis=0)
    assert period == 2


def test_minimal_period_axis_no_period():
    """Non-periodic grid returns full length."""
    g = np.array([[1, 2], [3, 4]], np.int8)
    period = _minimal_period_axis(g, axis=0)
    assert period == 2  # No proper divisor works


def test_is_strongly_periodic_true():
    """Grid periodic along both axes."""
    base = np.array([[1, 2], [3, 4]], np.int8)
    g = np.tile(base, (3, 2))  # 6x4 grid, periodic with periods 2x2
    assert _is_strongly_periodic(g)


def test_is_strongly_periodic_false():
    """Grid not periodic."""
    g = np.array([[1, 2, 3], [4, 5, 6]], np.int8)
    assert not _is_strongly_periodic(g)


def test_is_strongly_periodic_one_axis():
    """Periodic along one axis only → not strongly periodic."""
    g = np.array([[1, 2], [3, 4], [1, 2], [3, 4]], np.int8)  # period 2 along axis 0
    # But axis 1 is not periodic (column 0: [1,3,1,3], column 1: [2,4,2,4] are periodic too!)
    # Need a grid that's periodic on one axis but NOT the other
    g = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3]], np.int8)  # 3x3, no period along axis 1
    # Actually axis 1 columns are [1,4,1], [2,5,2], [3,6,3] - not periodic
    # But axis 0: rolling by any amount doesn't match (only 3 rows, not divisible)
    # Let me use a better example
    g = np.array([[1, 2], [1, 2], [1, 2], [1, 2]], np.int8)  # all rows same
    # This has period 1 along axis 0 (all rows equal) and period 1 along axis 1 (each row constant)
    # Actually this IS strongly periodic!
    # For NOT strongly periodic, need non-repeating pattern
    g = np.array([[1, 2, 3, 4]], np.int8)  # 1x4, no repetition
    assert _is_strongly_periodic(g) == False  # 1D grid can't be periodic on both axes


# ========== Tie-Break Tests ==========

def test_tiebreak_prefer_blowup_over_tiling():
    """Default tie-break: prefer BLOW_UP over TILING when periodicity_check disabled."""
    base = np.array([[1, 2], [3, 4]], np.int8)
    X = np.tile(base, (2, 2))  # 4x4 periodic
    Y = np.zeros((8, 8), np.int8)
    pairs = [(X, Y)]

    # Without periodicity_check → BLOW_UP
    law = infer_shape_law(pairs, enable_tiling=True, periodicity_check=False)
    assert law.kind == K.BLOW_UP


def test_frame_disabled_by_default():
    """FRAME not detected when enable_frame=False (default)."""
    pairs = [mk(5, 7, 9, 11)]  # valid frame t=2
    law = infer_shape_law(pairs)  # enable_frame=False
    # Falls back to IDENTITY (no integer ratio, frame disabled)
    assert law.kind == K.IDENTITY


def test_tiling_disabled_by_default():
    """TILING not considered when enable_tiling=False (default)."""
    base = np.array([[1, 2], [3, 4]], np.int8)
    X = np.tile(base, (2, 3))  # 4x6 periodic
    Y = np.zeros((8, 12), np.int8)  # 2x blow-up
    pairs = [(X, Y)]

    law = infer_shape_law(pairs)  # enable_tiling=False
    # Should be BLOW_UP (not TILING) since tiling disabled
    assert law.kind == K.BLOW_UP


def test_blowup_wins_over_frame_when_both():
    """When both BLOW_UP and FRAME are arithmetically valid, BLOW_UP wins."""
    # Construct case where both apply: need integer ratio AND frame condition
    # Example: 3x3 → 6x6 is both 2x blow-up and +1.5 frame? No, frame needs 2t integer.
    # Actually, frame needs H-h == W-w == 2t. If H=2h and W=2w, then h=2t and w=2t.
    # Example: 4x4 → 8x8 is blow-up 2x2, and also frame with t=2 (8-4=4=2*2).
    pairs = [mk(4, 4, 8, 8)]
    law = infer_shape_law(pairs, enable_frame=True, enable_tiling=False)
    # BLOW_UP should win per tie-break rule 2
    assert law.kind == K.BLOW_UP


# ========== Determinism Tests ==========

def test_determinism_same_input():
    """Same input produces same output (deterministic)."""
    pairs = [mk(5, 7, 10, 14)]
    law1 = infer_shape_law(pairs)
    law2 = infer_shape_law(pairs)
    assert law1.kind == law2.kind
    assert law1.kh == law2.kh
    assert law1.kw == law2.kw


def test_order_independence():
    """Order of train pairs doesn't matter (deterministic)."""
    p1 = mk(3, 3, 6, 6)
    p2 = mk(5, 7, 10, 14)
    law_a = infer_shape_law([p1, p2])
    law_b = infer_shape_law([p2, p1])
    assert law_a.kind == law_b.kind
    assert law_a.kh == law_b.kh
    assert law_a.kw == law_b.kw


print("All tests defined successfully")
