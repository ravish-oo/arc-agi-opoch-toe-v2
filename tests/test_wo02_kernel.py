"""
WO-02 Test Suite: Kernel Grid Primitives
Focus: BUG-CATCHING (no fluff tests)

Critical bugs to catch:
1. D8 inverse incorrect (group theory violation)
2. Diagonal residues negative (spec violation)
3. Box counts not vectorized / wrong edge clipping
4. Components4 non-deterministic / wrong border_contact
5. 1xN/Nx1 edge cases fail
6. Wrong dtypes (int16/int32)
7. Non-contiguous arrays
"""

import pytest
import numpy as np
from src.kernel.grid import (
    assert_grid, dims,
    d8_apply, d8_inv,
    residues_rc, diagonal_residues,
    box_count_per_color,
    components4
)

# ========== VALIDATION TESTS ==========

def test_assert_grid_rejects_non_ndarray():
    """Bug: accepting lists would break downstream"""
    with pytest.raises(AssertionError, match="must be np.ndarray"):
        assert_grid([[1, 2]])


def test_assert_grid_rejects_wrong_dtype():
    """Bug: float32 grids would corrupt palette"""
    g = np.array([[1, 2]], dtype=np.float32)
    with pytest.raises(AssertionError, match="must be np.int8"):
        assert_grid(g)


def test_assert_grid_shape_boundaries():
    """Bug: 0x0 or 31x31 grids would break"""
    # Too large
    with pytest.raises(AssertionError, match="out of bounds"):
        assert_grid(np.zeros((31, 5), dtype=np.int8))

    # 1x1 valid (boundary)
    assert_grid(np.zeros((1, 1), dtype=np.int8))

    # 30x30 valid (boundary)
    assert_grid(np.zeros((30, 30), dtype=np.int8))


def test_dims_returns_ints():
    """Bug: returning np.int64 could cause type issues"""
    g = np.zeros((3, 5), dtype=np.int8)
    h, w = dims(g)
    assert isinstance(h, int) and isinstance(w, int)
    assert (h, w) == (3, 5)


# ========== D8 GROUP TESTS ==========

def test_d8_apply_all_transforms_invertible():
    """CRITICAL: D8 must be a proper group (all t have inverse)"""
    g = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int8)

    for t in range(8):
        inv = d8_inv(t)
        restored = d8_apply(d8_apply(g, t), inv)
        assert np.array_equal(restored, g), f"D8 inverse broken for t={t}"


def test_d8_apply_1xN_and_Nx1():
    """Bug: edge shapes could break rot90/fliplr"""
    # 1xN
    g1 = np.array([[0, 1, 2, 3, 4]], dtype=np.int8)
    for t in range(8):
        out = d8_apply(g1, t)
        assert out.dtype == np.int8
        inv = d8_inv(t)
        restored = d8_apply(out, inv)
        assert np.array_equal(restored, g1), f"1xN broken for t={t}"

    # Nx1
    g2 = np.array([[0], [1], [2]], dtype=np.int8)
    for t in range(8):
        out = d8_apply(g2, t)
        inv = d8_inv(t)
        restored = d8_apply(out, inv)
        assert np.array_equal(restored, g2), f"Nx1 broken for t={t}"


def test_d8_apply_c_contiguous():
    """Bug: non-contiguous arrays break downstream"""
    g = np.array([[1, 2], [3, 4]], dtype=np.int8)
    for t in range(8):
        out = d8_apply(g, t)
        assert out.flags['C_CONTIGUOUS'], f"t={t} not C-contiguous"


def test_d8_inverse_specific_cases():
    """Bug: d8_inv logic could be wrong for flip cases"""
    # t=0: identity → inv=0
    assert d8_inv(0) == 0

    # t=1: rot90 → inv=3 (rot270)
    assert d8_inv(1) == 3

    # t=2: rot180 → inv=2 (self-inverse)
    assert d8_inv(2) == 2

    # t=4: flip → inv=4 (self-inverse)
    assert d8_inv(4) == 4

    # t=5: rot90+flip → inv=5 (self-inverse)
    assert d8_inv(5) == 5


# ========== RESIDUES TESTS ==========

def test_residues_rc_dtype_int16():
    """Bug: wrong dtype would corrupt features"""
    rr, cc = residues_rc(5, 7, 3)
    assert rr.dtype == np.int16
    assert cc.dtype == np.int16


def test_residues_rc_correctness():
    """Bug: wrong modulo arithmetic"""
    rr, cc = residues_rc(4, 5, 3)

    # Check specific values
    assert rr[0, 0] == 0  # 0 % 3
    assert rr[3, 2] == 0  # 3 % 3
    assert cc[1, 4] == 1  # 4 % 3


def test_residues_rc_1xN_Nx1():
    """Bug: edge shapes could fail fromfunction"""
    rr1, cc1 = residues_rc(1, 10, 5)
    assert rr1.shape == (1, 10)

    rr2, cc2 = residues_rc(10, 1, 4)
    assert rr2.shape == (10, 1)


def test_diagonal_residues_NON_NEGATIVE():
    """CRITICAL: Spec requires strictly non-negative diagonals"""
    # Test many shapes and moduli
    for h in [1, 5, 10, 20]:
        for w in [1, 5, 10, 20]:
            for k in [2, 3, 5, 7]:
                ad, dg = diagonal_residues(h, w, k)

                assert ad.min() >= 0, f"anti_diag negative at ({h},{w},k={k}): min={ad.min()}"
                assert dg.min() >= 0, f"diag negative at ({h},{w},k={k}): min={dg.min()}"


def test_diagonal_residues_dtype_int16():
    """Bug: wrong dtype corrupts features"""
    ad, dg = diagonal_residues(5, 7, 3)
    assert ad.dtype == np.int16
    assert dg.dtype == np.int16


def test_diagonal_residues_correctness():
    """Bug: wrong formula"""
    ad, dg = diagonal_residues(4, 5, 3)

    # anti_diag[r,c] = (r+c) % k
    assert ad[0, 0] == 0  # (0+0) % 3
    assert ad[2, 1] == 0  # (2+1) % 3
    assert ad[1, 1] == 2  # (1+1) % 3

    # diag[r,c] = ((r-c) % k + k) % k (non-negative)
    assert dg[0, 0] == 0  # (0-0) = 0
    assert dg[0, 2] == 1  # (0-2) = -2 → (-2%3+3)%3 = 1
    assert dg[3, 1] == 2  # (3-1) = 2


# ========== BOX COUNT TESTS ==========

def test_box_count_dtype_shape():
    """Bug: wrong dtype/shape breaks Qt"""
    g = np.array([[1, 2], [3, 4]], dtype=np.int8)
    C = box_count_per_color(g, 1)

    assert C.shape == (2, 2, 10)
    assert C.dtype == np.int16


def test_box_count_radius_0():
    """CRITICAL: radius=0 must count only center pixel"""
    g = np.array([[1, 2, 3], [4, 1, 6], [7, 8, 9]], dtype=np.int8)
    C = box_count_per_color(g, 0)

    # Center (1,1) has color 1
    assert C[1, 1, 1] == 1  # Only sees itself
    assert C[1, 1].sum() == 1  # Total count is 1

    # Corner (0,0) has color 1
    assert C[0, 0, 1] == 1
    assert C[0, 0].sum() == 1


def test_box_count_edge_clipping():
    """CRITICAL: edges must clip window, not wrap"""
    g = np.array([[1, 1, 0], [0, 1, 0], [2, 2, 2]], dtype=np.int8)
    C = box_count_per_color(g, 1)

    # Top-left (0,0), radius 1 → window clips to rows [0,1], cols [0,1]
    # Contains: [1,1], [0,1] → two 1s, one 0
    assert C[0, 0, 1] == 3, "Top-left edge clipping broken"
    assert C[0, 0, 0] == 1


def test_box_count_center_pixel():
    """Bug: center pixel could be miscounted"""
    g = np.array([[0, 0, 0], [0, 5, 0], [0, 0, 0]], dtype=np.int8)
    C = box_count_per_color(g, 1)

    # Center (1,1) radius 1 → 3x3 window, all visible
    # Contains: 8 zeros and 1 five
    assert C[1, 1, 0] == 8
    assert C[1, 1, 5] == 1
    assert C[1, 1].sum() == 9


def test_box_count_1xN_Nx1():
    """Bug: edge shapes could break integral image"""
    # 1xN
    g1 = np.array([[0, 1, 1, 0, 2]], dtype=np.int8)
    C1 = box_count_per_color(g1, 2)
    assert C1.shape == (1, 5, 10)
    assert C1.dtype == np.int16

    # Nx1
    g2 = np.array([[0], [1], [1], [0]], dtype=np.int8)
    C2 = box_count_per_color(g2, 1)
    assert C2.shape == (4, 1, 10)


def test_box_count_no_pixel_loops():
    """CRITICAL: must be vectorized (performance test)"""
    # Large grid - would timeout if pixel loops
    g = np.random.randint(0, 10, (30, 30), dtype=np.int8)

    import time
    start = time.time()
    C = box_count_per_color(g, 3)
    elapsed = time.time() - start

    assert C.shape == (30, 30, 10)
    assert elapsed < 1.0, f"Too slow ({elapsed}s) - likely has pixel loops"


def test_box_count_c_contiguous():
    """Bug: non-contiguous breaks downstream"""
    g = np.array([[1, 2]], dtype=np.int8)
    C = box_count_per_color(g, 1)
    assert C.flags['C_CONTIGUOUS']


# ========== COMPONENTS4 TESTS ==========

def test_components4_dtype_shape():
    """Bug: wrong dtype breaks Qt"""
    g = np.array([[1, 2], [3, 4]], dtype=np.int8)
    labels, info = components4(g)

    assert labels.shape == (2, 2)
    assert labels.dtype == np.int32


def test_components4_summary_keys():
    """Bug: missing keys would break Qt"""
    g = np.array([[1, 1]], dtype=np.int8)
    labels, info = components4(g)

    required_keys = {"area", "bbox", "color", "shape", "border_contact"}
    for cid, summary in info.items():
        assert set(summary.keys()) == required_keys


def test_components4_border_contact_values():
    """Bug: border_contact must be 0 or 1 (not bool)"""
    g = np.array([[1, 2, 1], [1, 2, 1]], dtype=np.int8)
    labels, info = components4(g)

    for summary in info.values():
        bc = summary["border_contact"]
        assert bc in (0, 1), f"border_contact must be 0/1, got {bc}"
        assert isinstance(bc, int)


def test_components4_border_contact_edges():
    """CRITICAL: touching any edge sets border_contact=1"""
    # Interior component (no border)
    g = np.array([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0]
    ], dtype=np.int8)
    labels, info = components4(g)

    # Find component with color 1
    comp1 = [s for s in info.values() if s["color"] == 1][0]
    assert comp1["border_contact"] == 0, "Interior component should not touch border"

    # Border component
    comp0 = [s for s in info.values() if s["color"] == 0][0]
    assert comp0["border_contact"] == 1, "Perimeter component must touch border"


def test_components4_deterministic_ordering():
    """CRITICAL: row-major discovery order (determinism)"""
    g = np.array([
        [1, 0, 2],
        [1, 0, 2],
        [3, 3, 3]
    ], dtype=np.int8)

    # Run twice
    labels1, info1 = components4(g)
    labels2, info2 = components4(g)

    # Must be identical
    assert np.array_equal(labels1, labels2)
    assert info1 == info2

    # IDs should be 0, 1, 2, 3 (4 components)
    assert set(info1.keys()) == {0, 1, 2, 3}


def test_components4_area_bbox_correctness():
    """Bug: wrong area/bbox calculation"""
    g = np.array([
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 0]
    ], dtype=np.int8)
    labels, info = components4(g)

    # Component with color 1
    comp1 = [s for s in info.values() if s["color"] == 1][0]
    assert comp1["area"] == 3
    assert comp1["bbox"] == (0, 0, 1, 1)  # minr, minc, maxr, maxc
    assert comp1["shape"] == (2, 2)  # (maxr-minr+1, maxc-minc+1)


def test_components4_single_pixel():
    """Bug: single-pixel component edge case"""
    g = np.array([[5]], dtype=np.int8)
    labels, info = components4(g)

    assert labels[0, 0] == 0
    assert len(info) == 1
    assert info[0]["area"] == 1
    assert info[0]["bbox"] == (0, 0, 0, 0)
    assert info[0]["shape"] == (1, 1)
    assert info[0]["border_contact"] == 1  # touches border


def test_components4_all_same_color():
    """Bug: entire grid same color → 1 component"""
    g = np.full((3, 4), 7, dtype=np.int8)
    labels, info = components4(g)

    assert len(info) == 1
    assert info[0]["area"] == 12
    assert info[0]["color"] == 7
    assert info[0]["border_contact"] == 1


def test_components4_1xN():
    """CRITICAL: 1xN must work (components are runs)"""
    g = np.array([[0, 0, 1, 1, 1, 0]], dtype=np.int8)
    labels, info = components4(g)

    # Should have 3 components: [0,0], [1,1,1], [0]
    assert len(info) == 3

    # All touch border (1xN grid)
    for s in info.values():
        assert s["border_contact"] == 1


def test_components4_Nx1():
    """CRITICAL: Nx1 must work"""
    g = np.array([[1], [1], [2], [1]], dtype=np.int8)
    labels, info = components4(g)

    # Components: [1,1], [2], [1]
    assert len(info) == 3

    for s in info.values():
        assert s["border_contact"] == 1  # All touch border


def test_components4_labels_contiguous():
    """Bug: labels IDs must be 0..C-1"""
    g = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
    labels, info = components4(g)

    C = len(info)
    assert set(info.keys()) == set(range(C))
    assert labels.max() == C - 1
    assert labels.min() == 0


# ========== ANCHOR COMPLIANCE ==========

def test_math_anchor_qt_features():
    """Qt features: residues, diagonals, counts, components

    Math Anchor §2: Feature family F includes:
    - Residues (r mod k, c mod k)
    - Diagonal residues ((r+c) mod k, (r-c) mod k)
    - Local counts (Chebyshev balls)
    - Component summaries
    """
    # All primitives exist and work
    g = np.array([[1, 2], [3, 4]], dtype=np.int8)

    rr, cc = residues_rc(2, 2, 3)
    assert rr is not None

    ad, dg = diagonal_residues(2, 2, 3)
    assert ad is not None

    C = box_count_per_color(g, 1)
    assert C is not None

    labels, info = components4(g)
    assert labels is not None


def test_production_spec_v23_vectorized_counts():
    """Production Spec v2.3 §2: Vectorized integral-image counts

    Quote: "Fully vectorized integral images (no Python loops over pixels)"
    """
    # Already tested in test_box_count_no_pixel_loops
    pass


def test_production_spec_v23_non_negative_diagonals():
    """Production Spec v2.3 §4: Non-negative diagonal residues

    Quote: "diagonal_residues...strictly non-negative"
    """
    # Already tested in test_diagonal_residues_NON_NEGATIVE
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
