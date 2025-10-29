"""
WO-06 Test Suite: Shape Law (Δ) Inference
Focus: BUG-CATCHING (critical bugs only)

Critical bugs to catch:
1. Reading Y pixel content (dimensions-only violation)
2. Non-deterministic inference
3. Rectangular blow-up not supported (kh ≠ kw)
4. Wrong tie-break order (FRAME/BLOW_UP/TILING)
5. Periodicity check edge cases (1xN, wrapped edges)
6. Frame detection off-by-one
7. Empty train not handled
8. Inconsistent ratios not handled properly
"""

import pytest
import numpy as np
from src.solver.shape_law import infer_shape_law, _minimal_period_axis, _is_strongly_periodic
from src.types import ShapeLaw, ShapeLawKind


# ========== DIMENSIONS-ONLY (CONTRACT) ==========

def test_dimensions_only_no_pixel_reads():
    """CRITICAL: Must never read Y pixel content (dimensions-only core)"""
    # Same dimensions, different pixel content in Y
    x1 = np.ones((3, 4), dtype=np.int8)
    y1_all_ones = np.ones((6, 8), dtype=np.int8)
    y1_all_zeros = np.zeros((6, 8), dtype=np.int8)
    y1_random = np.random.randint(0, 10, (6, 8), dtype=np.int8)

    # All should give identical shape law (only dimensions matter)
    law1 = infer_shape_law([(x1, y1_all_ones)])
    law2 = infer_shape_law([(x1, y1_all_zeros)])
    law3 = infer_shape_law([(x1, y1_random)])

    assert law1.kind == law2.kind == law3.kind
    assert law1.kh == law2.kh == law3.kh
    assert law1.kw == law2.kw == law3.kw


def test_dimensions_only_x_content_irrelevant():
    """CRITICAL: X pixel content must also be irrelevant (dimensions-only)"""
    # Same X dimensions, different content
    x_ones = np.ones((3, 4), dtype=np.int8)
    x_zeros = np.zeros((3, 4), dtype=np.int8)
    x_random = np.random.randint(0, 10, (3, 4), dtype=np.int8)

    y = np.zeros((6, 8), dtype=np.int8)

    law1 = infer_shape_law([(x_ones, y)])
    law2 = infer_shape_law([(x_zeros, y)])
    law3 = infer_shape_law([(x_random, y)])

    # All should be identical
    assert law1 == law2 == law3


# ========== IDENTITY ==========

def test_identity_same_shapes():
    """Bug: Not detecting identity"""
    x = np.zeros((5, 7), dtype=np.int8)
    y = np.zeros((5, 7), dtype=np.int8)

    law = infer_shape_law([(x, y)])
    assert law.kind == ShapeLawKind.IDENTITY


def test_identity_multiple_pairs():
    """Bug: Identity check across multiple pairs"""
    pairs = [
        (np.zeros((3, 4), dtype=np.int8), np.zeros((3, 4), dtype=np.int8)),
        (np.zeros((5, 7), dtype=np.int8), np.zeros((5, 7), dtype=np.int8)),
    ]
    law = infer_shape_law(pairs)
    assert law.kind == ShapeLawKind.IDENTITY


# ========== BLOW_UP (RECTANGULAR) ==========

def test_blow_up_square_scaling():
    """Bug: Square blow-up (kh == kw)"""
    x = np.zeros((3, 4), dtype=np.int8)
    y = np.zeros((6, 8), dtype=np.int8)  # kh=2, kw=2

    law = infer_shape_law([(x, y)])
    assert law.kind == ShapeLawKind.BLOW_UP
    assert law.kh == 2
    assert law.kw == 2


def test_blow_up_rectangular():
    """CRITICAL: Rectangular blow-up (kh ≠ kw)"""
    x = np.zeros((3, 4), dtype=np.int8)
    y = np.zeros((9, 8), dtype=np.int8)  # kh=3, kw=2

    law = infer_shape_law([(x, y)])
    assert law.kind == ShapeLawKind.BLOW_UP
    assert law.kh == 3
    assert law.kw == 2


def test_blow_up_only_height():
    """Bug: Blow-up only in one dimension"""
    x = np.zeros((3, 5), dtype=np.int8)
    y = np.zeros((6, 5), dtype=np.int8)  # kh=2, kw=1

    law = infer_shape_law([(x, y)])
    assert law.kind == ShapeLawKind.BLOW_UP
    assert law.kh == 2
    assert law.kw == 1


def test_blow_up_only_width():
    """Bug: Blow-up only in width"""
    x = np.zeros((5, 3), dtype=np.int8)
    y = np.zeros((5, 9), dtype=np.int8)  # kh=1, kw=3

    law = infer_shape_law([(x, y)])
    assert law.kind == ShapeLawKind.BLOW_UP
    assert law.kh == 1
    assert law.kw == 3


def test_blow_up_consistent_ratios():
    """CRITICAL: All pairs must have same ratios"""
    pairs = [
        (np.zeros((2, 3), dtype=np.int8), np.zeros((4, 9), dtype=np.int8)),  # kh=2, kw=3
        (np.zeros((5, 7), dtype=np.int8), np.zeros((10, 21), dtype=np.int8)),  # kh=2, kw=3
    ]
    law = infer_shape_law(pairs)
    assert law.kind == ShapeLawKind.BLOW_UP
    assert law.kh == 2
    assert law.kw == 3


def test_blow_up_inconsistent_ratios_fallback():
    """Bug: Inconsistent ratios should fall back to identity"""
    pairs = [
        (np.zeros((3, 4), dtype=np.int8), np.zeros((6, 8), dtype=np.int8)),  # kh=2, kw=2
        (np.zeros((5, 7), dtype=np.int8), np.zeros((10, 14), dtype=np.int8)),  # kh=2, kw=2
        (np.zeros((2, 3), dtype=np.int8), np.zeros((6, 6), dtype=np.int8)),  # kh=3, kw=2 (inconsistent!)
    ]
    law = infer_shape_law(pairs)
    # Should fallback to identity when ratios inconsistent
    assert law.kind == ShapeLawKind.IDENTITY


# ========== FRAME ==========

def test_frame_detection():
    """Bug: Frame detection with even border"""
    x = np.zeros((3, 4), dtype=np.int8)
    y = np.zeros((7, 8), dtype=np.int8)  # +4 height, +4 width → t=2

    law = infer_shape_law([(x, y)], enable_frame=True)
    assert law.kind == ShapeLawKind.FRAME
    assert law.kh == 2  # t stored in kh
    assert law.kw == 2  # t stored in kw


def test_frame_t1():
    """Bug: Single-pixel frame"""
    x = np.zeros((5, 5), dtype=np.int8)
    y = np.zeros((7, 7), dtype=np.int8)  # +2 height, +2 width → t=1

    law = infer_shape_law([(x, y)], enable_frame=True)
    assert law.kind == ShapeLawKind.FRAME
    assert law.kh == 1
    assert law.kw == 1


def test_frame_disabled_by_default():
    """CRITICAL: Frame must be disabled without flag"""
    x = np.zeros((3, 4), dtype=np.int8)
    y = np.zeros((7, 8), dtype=np.int8)

    law = infer_shape_law([(x, y)])  # No enable_frame
    # Should fallback to identity (frame not detected)
    assert law.kind == ShapeLawKind.IDENTITY


def test_frame_odd_difference_rejected():
    """Bug: Frame requires even difference"""
    x = np.zeros((3, 4), dtype=np.int8)
    y = np.zeros((6, 7), dtype=np.int8)  # +3 height, +3 width (odd)

    law = infer_shape_law([(x, y)], enable_frame=True)
    # Should not be frame (odd difference)
    assert law.kind == ShapeLawKind.IDENTITY


def test_frame_different_h_w_diff_rejected():
    """Bug: Frame requires HY-HX == WY-WX"""
    x = np.zeros((3, 4), dtype=np.int8)
    y = np.zeros((7, 10), dtype=np.int8)  # +4 height, +6 width (different)

    law = infer_shape_law([(x, y)], enable_frame=True)
    # Should not be frame
    assert law.kind == ShapeLawKind.IDENTITY


# ========== TIE-BREAK (BLOW_UP vs FRAME) ==========

def test_tiebreak_blowup_vs_frame():
    """CRITICAL: When both BLOW_UP and FRAME match, prefer BLOW_UP"""
    # Edge case: (3,4) → (6,8) matches both BLOW_UP(2,2) and FRAME(t=1.5 invalid)
    # Actually, let's create a case where both are valid

    # (3,3) → (6,6) is BLOW_UP(2,2) AND FRAME(t=1.5 invalid, so not frame)
    # Let me construct proper case:
    # (2,2) → (4,4) could be BLOW_UP(2,2) or FRAME(t=1)

    x = np.zeros((2, 2), dtype=np.int8)
    y = np.zeros((4, 4), dtype=np.int8)

    law = infer_shape_law([(x, y)], enable_frame=True)
    # Both blow-up(2,2) and frame(t=1) are valid
    # Tie-break: prefer BLOW_UP
    assert law.kind == ShapeLawKind.BLOW_UP
    assert law.kh == 2
    assert law.kw == 2


# ========== TILING ==========

def test_tiling_disabled_by_default():
    """CRITICAL: Tiling requires enable_tiling flag"""
    # Periodic input
    x = np.array([[1, 2, 1, 2]], dtype=np.int8)
    y = np.zeros((1, 8), dtype=np.int8)  # kw=2

    law = infer_shape_law([(x, y)])  # No enable_tiling
    assert law.kind == ShapeLawKind.BLOW_UP  # Not tiling


def test_tiling_requires_periodicity_check():
    """CRITICAL: Tiling requires both flag AND periodicity check"""
    # Strongly periodic in BOTH axes
    x = np.array([[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]], dtype=np.int8)
    y = np.zeros((8, 8), dtype=np.int8)

    # With enable_tiling but no periodicity_check
    law1 = infer_shape_law([(x, y)], enable_tiling=True, periodicity_check=False)
    assert law1.kind == ShapeLawKind.BLOW_UP

    # With both flags
    law2 = infer_shape_law([(x, y)], enable_tiling=True, periodicity_check=True)
    assert law2.kind == ShapeLawKind.TILING


def test_tiling_non_periodic_fallback():
    """Bug: Non-periodic input should not select tiling"""
    # Non-periodic
    x = np.array([[1, 2, 3, 4]], dtype=np.int8)
    y = np.zeros((1, 8), dtype=np.int8)

    law = infer_shape_law([(x, y)], enable_tiling=True, periodicity_check=True)
    # Not periodic → fallback to BLOW_UP
    assert law.kind == ShapeLawKind.BLOW_UP


def test_tiling_all_inputs_must_be_periodic():
    """CRITICAL: All train inputs must be periodic for tiling"""
    pairs = [
        (np.array([[1, 2, 1, 2]], dtype=np.int8), np.zeros((1, 8), dtype=np.int8)),  # periodic
        (np.array([[1, 2, 3, 4]], dtype=np.int8), np.zeros((1, 8), dtype=np.int8)),  # NOT periodic
    ]
    law = infer_shape_law(pairs, enable_tiling=True, periodicity_check=True)
    # Not all periodic → BLOW_UP
    assert law.kind == ShapeLawKind.BLOW_UP


# ========== PERIODICITY CHECK ==========

def test_periodic_1d_horizontal():
    """Bug: 1D grids cannot be strongly periodic (requires both axes)"""
    x = np.array([[1, 2, 1, 2]], dtype=np.int8)
    # 1xN grid: height=1, so period_h=1 (full length), not < h
    assert _is_strongly_periodic(x) is False


def test_periodic_1d_vertical():
    """Bug: 1D grids cannot be strongly periodic (requires both axes)"""
    x = np.array([[1], [2], [1], [2]], dtype=np.int8)
    # Nx1 grid: width=1, so period_w=1 (full length), not < w
    assert _is_strongly_periodic(x) is False


def test_periodic_2d_grid():
    """Bug: 2D periodic pattern"""
    x = np.array([
        [1, 2, 1, 2],
        [3, 4, 3, 4],
        [1, 2, 1, 2],
        [3, 4, 3, 4]
    ], dtype=np.int8)
    assert _is_strongly_periodic(x) is True


def test_non_periodic():
    """Bug: Non-periodic should return False"""
    x = np.array([[1, 2, 3]], dtype=np.int8)
    assert _is_strongly_periodic(x) is False


def test_periodic_single_value():
    """Edge case: All same value is periodic"""
    x = np.full((3, 3), 5, dtype=np.int8)
    assert _is_strongly_periodic(x) is True


def test_minimal_period_axis():
    """Bug: Minimal period computation"""
    # Pattern [1,2,1,2] has period 2
    x = np.array([[1, 2, 1, 2]], dtype=np.int8)
    p = _minimal_period_axis(x, axis=1)
    assert p == 2


def test_minimal_period_no_periodicity():
    """Bug: Non-periodic returns full length"""
    x = np.array([[1, 2, 3, 4]], dtype=np.int8)
    p = _minimal_period_axis(x, axis=1)
    assert p == 4  # No proper divisor works


def test_periodic_edge_case_1xN():
    """Bug: 1xN periodic edge case"""
    x = np.array([[1, 2, 1, 2, 1, 2]], dtype=np.int8)
    # Period 2 horizontally, but vertically?
    # 1xN has height 1, so period_h = 1 (full length, not < h)
    # So not strongly periodic
    assert _is_strongly_periodic(x) is False


def test_periodic_requires_both_axes():
    """CRITICAL: Strong periodicity requires BOTH axes"""
    # Periodic horizontally but not vertically
    x = np.array([
        [1, 2, 1, 2],
        [3, 4, 5, 6]  # Different row
    ], dtype=np.int8)
    assert _is_strongly_periodic(x) is False


# ========== EDGE CASES ==========

def test_empty_train():
    """CRITICAL: Empty train should return identity"""
    law = infer_shape_law([])
    assert law.kind == ShapeLawKind.IDENTITY


def test_single_pair():
    """Edge case: Single train pair"""
    x = np.zeros((3, 4), dtype=np.int8)
    y = np.zeros((6, 8), dtype=np.int8)

    law = infer_shape_law([(x, y)])
    assert law.kind == ShapeLawKind.BLOW_UP
    assert law.kh == 2
    assert law.kw == 2


def test_1x1_identity():
    """Edge case: Tiny grids"""
    x = np.zeros((1, 1), dtype=np.int8)
    y = np.zeros((1, 1), dtype=np.int8)

    law = infer_shape_law([(x, y)])
    assert law.kind == ShapeLawKind.IDENTITY


def test_non_integer_ratio():
    """Bug: Non-divisible dimensions"""
    x = np.zeros((3, 4), dtype=np.int8)
    y = np.zeros((7, 9), dtype=np.int8)  # Not divisible

    law = infer_shape_law([(x, y)])
    # Should fallback to identity
    assert law.kind == ShapeLawKind.IDENTITY


# ========== DETERMINISM ==========

def test_determinism_same_dimensions():
    """CRITICAL: Same dimensions produce same law"""
    pairs1 = [
        (np.zeros((3, 4), dtype=np.int8), np.zeros((6, 8), dtype=np.int8)),
        (np.zeros((5, 7), dtype=np.int8), np.zeros((10, 14), dtype=np.int8)),
    ]
    pairs2 = [
        (np.ones((3, 4), dtype=np.int8), np.ones((6, 8), dtype=np.int8)),
        (np.full((5, 7), 9, dtype=np.int8), np.full((10, 14), 9, dtype=np.int8)),
    ]

    law1 = infer_shape_law(pairs1)
    law2 = infer_shape_law(pairs2)

    assert law1 == law2


def test_determinism_repeated_calls():
    """Bug: Repeated calls must be identical"""
    pairs = [
        (np.zeros((3, 4), dtype=np.int8), np.zeros((9, 8), dtype=np.int8))
    ]

    law1 = infer_shape_law(pairs)
    law2 = infer_shape_law(pairs)
    law3 = infer_shape_law(pairs)

    assert law1 == law2 == law3


# ========== ACCEPTANCE TESTS ==========

def test_acceptance_identity():
    """WO-06 Acceptance: Identity detection"""
    pairs = [
        (np.zeros((5, 7), dtype=np.int8), np.zeros((5, 7), dtype=np.int8)),
        (np.zeros((3, 4), dtype=np.int8), np.zeros((3, 4), dtype=np.int8)),
    ]
    law = infer_shape_law(pairs)
    assert law.kind == ShapeLawKind.IDENTITY


def test_acceptance_blow_up():
    """WO-06 Acceptance: Blow-up with rectangular"""
    pairs = [
        (np.zeros((3, 4), dtype=np.int8), np.zeros((9, 8), dtype=np.int8)),  # kh=3, kw=2
        (np.zeros((2, 5), dtype=np.int8), np.zeros((6, 10), dtype=np.int8)),  # kh=3, kw=2
    ]
    law = infer_shape_law(pairs)
    assert law.kind == ShapeLawKind.BLOW_UP
    assert law.kh == 3
    assert law.kw == 2


def test_acceptance_frame():
    """WO-06 Acceptance: Frame detection"""
    pairs = [
        (np.zeros((3, 4), dtype=np.int8), np.zeros((7, 8), dtype=np.int8)),  # t=2
        (np.zeros((5, 5), dtype=np.int8), np.zeros((9, 9), dtype=np.int8)),  # t=2
    ]
    law = infer_shape_law(pairs, enable_frame=True)
    assert law.kind == ShapeLawKind.FRAME
    assert law.kh == 2


def test_acceptance_tiling():
    """WO-06 Acceptance: Tiling with periodicity"""
    # Strongly periodic in both axes
    x1 = np.array([[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]], dtype=np.int8)
    y1 = np.zeros((8, 8), dtype=np.int8)

    law = infer_shape_law([(x1, y1)], enable_tiling=True, periodicity_check=True)
    assert law.kind == ShapeLawKind.TILING
    assert law.kh == 2
    assert law.kw == 2


def test_acceptance_dimensions_only():
    """WO-06 Acceptance: Dimensions-only (never reads pixels)"""
    x = np.ones((3, 4), dtype=np.int8)
    y1 = np.zeros((6, 8), dtype=np.int8)
    y2 = np.full((6, 8), 9, dtype=np.int8)

    law1 = infer_shape_law([(x, y1)])
    law2 = infer_shape_law([(x, y2)])

    # Must be identical (Y pixels never read)
    assert law1 == law2


# ========== ANCHOR COMPLIANCE ==========

def test_math_anchor_delta_dimensions_only():
    """Math Anchor: Δ uses dimensions only, never Y content"""
    # Already tested in test_dimensions_only_no_pixel_reads
    pass


def test_production_spec_rectangular_support():
    """Production Spec: Rectangular blow-up supported"""
    x = np.zeros((3, 4), dtype=np.int8)
    y = np.zeros((9, 8), dtype=np.int8)

    law = infer_shape_law([(x, y)])
    assert law.kh == 3
    assert law.kw == 2  # kh ≠ kw allowed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
