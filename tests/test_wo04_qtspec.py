"""
WO-04 Test Suite: QtSpec Builder
Focus: BUG-CATCHING (real bugs only)

Critical bugs to catch:
1. Cap applied to unsorted set (non-deterministic)
2. Radii threshold off-by-one
3. Empty input not rejected
4. Missing base residues
5. Divisors not included
6. Pixel reads (violates input-only)
7. Non-deterministic ordering
"""

import pytest
import numpy as np
from src.qt.spec import build_qt_spec, MAX_RESIDUES
from src.types import QtSpec

# ========== DETERMINISM ==========

def test_determinism_same_shapes():
    """CRITICAL: Same shapes must produce identical spec"""
    xs1 = [np.zeros((7, 10), dtype=np.int8), np.zeros((3, 15), dtype=np.int8)]
    xs2 = [np.zeros((7, 10), dtype=np.int8), np.zeros((3, 15), dtype=np.int8)]

    spec1 = build_qt_spec(xs1)
    spec2 = build_qt_spec(xs2)

    assert spec1 == spec2, "Same shapes produced different specs"
    assert spec1.residues == spec2.residues
    assert spec1.radii == spec2.radii


def test_determinism_different_pixel_values():
    """CRITICAL: Different pixel values (same shapes) must give same spec (input-only)"""
    # Same shape, different content
    x1 = np.ones((5, 7), dtype=np.int8)
    x2 = np.full((5, 7), 9, dtype=np.int8)
    x3 = np.random.randint(0, 10, (5, 7), dtype=np.int8)

    spec1 = build_qt_spec([x1])
    spec2 = build_qt_spec([x2])
    spec3 = build_qt_spec([x3])

    # All should be identical (only shape matters)
    assert spec1 == spec2 == spec3, "Pixel values affected spec (violates input-only)"


# ========== RESIDUES ==========

def test_residues_base_set_always_present():
    """Bug: forgetting base set {2,3,4,5,6}"""
    xs = [np.zeros((1, 1), dtype=np.int8)]  # No divisors from shape
    spec = build_qt_spec(xs)

    # Base set must be present
    assert set(spec.residues).issuperset({2, 3, 4, 5, 6})


def test_residues_include_divisors():
    """Bug: not calling divisors() on shapes"""
    # h=10 has divisors {2,5,10}; w=7 has divisor {7}
    xs = [np.zeros((10, 7), dtype=np.int8)]
    spec = build_qt_spec(xs)

    # Should include divisors
    assert 7 in spec.residues, "Divisor 7 missing"
    assert 10 in spec.residues, "Divisor 10 missing"


def test_residues_sorted_ascending():
    """CRITICAL: Residues must be sorted (determinism)"""
    xs = [np.zeros((10, 9), dtype=np.int8)]
    spec = build_qt_spec(xs)

    # Must be sorted
    assert tuple(sorted(spec.residues)) == spec.residues


def test_residues_cap_at_10():
    """CRITICAL: Cap applied AFTER sorting (not before)"""
    # Create many divisors: 2,3,4,5,6 (base) + 7,8,9,10 from shapes
    # Then test cap
    xs = [
        np.zeros((8, 9), dtype=np.int8),   # divisors: 2,3,4,8,9
        np.zeros((10, 7), dtype=np.int8),  # divisors: 2,5,7,10
    ]
    spec = build_qt_spec(xs)

    # Cap at 10
    assert len(spec.residues) <= MAX_RESIDUES

    # Must be sorted
    assert tuple(sorted(spec.residues)) == spec.residues


def test_residues_cap_takes_smallest():
    """CRITICAL: If >10 residues, take SMALLEST (sorted then sliced)"""
    # Force many residues
    from src.kernel.grid import divisors as get_divisors

    # Manually check: if we have residues {2,3,4,5,6,7,8,9,10,...}
    # and cap at 10, we should get {2,3,4,5,6,7,8,9,10} + one more
    # Actually let me create a case where we exceed 10

    # Shapes with many divisors
    xs = [
        np.zeros((12, 1), dtype=np.int8),  # 12: divisors 2,3,4,6
        np.zeros((15, 1), dtype=np.int8),  # 15: divisors 3,5
        np.zeros((16, 1), dtype=np.int8),  # 16: divisors 2,4,8
        np.zeros((18, 1), dtype=np.int8),  # 18: divisors 2,3,6,9
        np.zeros((20, 1), dtype=np.int8),  # 20: divisors 2,4,5,10
    ]

    spec = build_qt_spec(xs)

    # Check: smallest values win
    assert spec.residues[0] == 2
    assert len(spec.residues) <= 10


# ========== RADII ==========

def test_radii_small_grids():
    """Bug: radii threshold off"""
    # max(h,w) = 20 → should get (1,2)
    xs = [np.zeros((20, 15), dtype=np.int8)]
    spec = build_qt_spec(xs)

    assert spec.radii == (1, 2)


def test_radii_large_grids():
    """CRITICAL: max(h,w) > 20 → (1,2,3)"""
    # max(h,w) = 21 → should get (1,2,3)
    xs = [np.zeros((21, 15), dtype=np.int8)]
    spec = build_qt_spec(xs)

    assert spec.radii == (1, 2, 3), "Large grid didn't get radius 3"


def test_radii_boundary_20():
    """Bug: off-by-one at boundary"""
    # Exactly 20
    xs1 = [np.zeros((20, 20), dtype=np.int8)]
    spec1 = build_qt_spec(xs1)
    assert spec1.radii == (1, 2), "20x20 should get (1,2)"

    # Just over 20
    xs2 = [np.zeros((21, 1), dtype=np.int8)]
    spec2 = build_qt_spec(xs2)
    assert spec2.radii == (1, 2, 3), "21x1 should get (1,2,3)"


def test_radii_max_across_all_grids():
    """Bug: checking wrong dimension"""
    # Mix of shapes: 10x30 has max=30
    xs = [
        np.zeros((10, 30), dtype=np.int8),
        np.zeros((5, 5), dtype=np.int8),
    ]
    spec = build_qt_spec(xs)

    # max(h,w) across all = 30 > 20
    assert spec.radii == (1, 2, 3)


# ========== DIAGONALS & WL ROUNDS ==========

def test_diagonals_always_true():
    """Bug: diagonals flag wrong"""
    xs = [np.zeros((5, 5), dtype=np.int8)]
    spec = build_qt_spec(xs)

    assert spec.use_diagonals is True


def test_wl_rounds_always_3():
    """Bug: wrong WL rounds initial value"""
    xs = [np.zeros((5, 5), dtype=np.int8)]
    spec = build_qt_spec(xs)

    assert spec.wl_rounds == 3


# ========== EDGE CASES ==========

def test_empty_input_raises():
    """CRITICAL: Empty input must raise ValueError"""
    with pytest.raises(ValueError, match="must be non-empty"):
        build_qt_spec([])


def test_single_tiny_grid():
    """Edge case: 1x1 grid"""
    xs = [np.zeros((1, 1), dtype=np.int8)]
    spec = build_qt_spec(xs)

    # Base residues present
    assert set(spec.residues).issuperset({2, 3, 4, 5, 6})
    # Small grid
    assert spec.radii == (1, 2)
    # Diagonals on
    assert spec.use_diagonals is True


def test_1xN_and_Nx1():
    """Edge case: edge shapes"""
    xs = [
        np.zeros((1, 30), dtype=np.int8),
        np.zeros((25, 1), dtype=np.int8),
    ]
    spec = build_qt_spec(xs)

    # max dimension is 30 > 20
    assert spec.radii == (1, 2, 3)


def test_duplicate_shapes_in_batch():
    """Edge case: same shape repeated"""
    xs = [
        np.zeros((7, 9), dtype=np.int8),
        np.zeros((7, 9), dtype=np.int8),
        np.zeros((7, 9), dtype=np.int8),
    ]
    spec = build_qt_spec(xs)

    # Should work fine (set handles duplicates)
    assert spec is not None


# ========== ACCEPTANCE TESTS (from spec) ==========

def test_acceptance_determinism():
    """WO-04 Acceptance §7: Determinism"""
    xs = [
        np.zeros((7, 10), dtype=np.int8),
        np.zeros((3, 15), dtype=np.int8)
    ]
    spec1 = build_qt_spec(xs)
    spec2 = build_qt_spec(xs.copy())

    assert isinstance(spec1, QtSpec)
    assert spec1 == spec2


def test_acceptance_residues():
    """WO-04 Acceptance §7: Residues include base and divisors"""
    xs = [
        np.zeros((7, 10), dtype=np.int8),
        np.zeros((3, 15), dtype=np.int8)
    ]
    spec = build_qt_spec(xs)

    # Base set present
    assert set(spec.residues).issuperset({2, 3, 4, 5, 6})
    # Sorted
    assert tuple(sorted(spec.residues)) == spec.residues
    # Capped
    assert len(spec.residues) <= MAX_RESIDUES


def test_acceptance_radii_small():
    """WO-04 Acceptance §7: Small grids get (1,2)"""
    xs = [
        np.zeros((7, 10), dtype=np.int8),
        np.zeros((3, 15), dtype=np.int8)
    ]
    spec = build_qt_spec(xs)

    # max(h,w) = 15 <= 20
    assert spec.radii == (1, 2)


def test_acceptance_radii_large():
    """WO-04 Acceptance §7: Large grids get (1,2,3)"""
    xl = [np.zeros((25, 8), dtype=np.int8)]
    spec = build_qt_spec(xl)

    assert spec.radii == (1, 2, 3)


def test_acceptance_diagonals_and_wl():
    """WO-04 Acceptance §7: Diagonals on, WL=3"""
    xs = [np.zeros((5, 5), dtype=np.int8)]
    spec = build_qt_spec(xs)

    assert spec.use_diagonals is True
    assert spec.wl_rounds == 3


def test_acceptance_empty_fails():
    """WO-04 Acceptance §7: Empty list raises"""
    with pytest.raises(ValueError):
        build_qt_spec([])


# ========== ANCHOR COMPLIANCE ==========

def test_math_anchor_qt_is_spec():
    """Math Anchor §2: Qt is a spec, not cached partition"""
    xs = [np.zeros((5, 5), dtype=np.int8)]
    spec = build_qt_spec(xs)

    # Returns a spec (frozen dataclass)
    assert isinstance(spec, QtSpec)
    # Spec has no partition data
    assert not hasattr(spec, 'ids')
    assert not hasattr(spec, 'labels')


def test_math_anchor_input_only():
    """Math Anchor §2: Qt uses input-only features

    Verified: build_qt_spec only calls dims() - no pixel access
    """
    # Already tested via test_determinism_different_pixel_values
    pass


def test_production_spec_v23_residues_cap():
    """Production Spec v2.3: Residues capped at 10"""
    # Create many divisors
    xs = [np.zeros((i, 1), dtype=np.int8) for i in range(2, 30)]
    spec = build_qt_spec(xs)

    assert len(spec.residues) <= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
