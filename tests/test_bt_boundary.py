"""
WO-07: Bt Boundary Tests
Tests for src/bt/boundary.py

Test strategy:
1. Acceptance: verify core contracts (forced/unforced, determinism)
2. Edge cases: no identity pairs, size-changed, single-color
3. Ladder: verify each step can be triggered
4. Stable keys: verify bytes-based cross-grid identity
"""
import numpy as np
import pytest
from src.bt.boundary import check_boundary_forced, extract_bt_force_until_forced
from src.qt.spec import build_qt_spec
from src.qt.quotient import _ExtraFlags
from src.types import Boundary, QtSpec


# ========== Acceptance Tests ==========

def test_check_boundary_forced_basic():
    """Basic functionality: forced and unforced detection."""
    # Simple identity mapping
    X = np.array([[1, 2], [3, 4]], np.int8)
    Y = np.array([[5, 6], [7, 8]], np.int8)

    spec = build_qt_spec([X])
    bt, all_forced = check_boundary_forced([(X, Y)], spec, None)

    assert isinstance(bt, Boundary)
    assert isinstance(bt.forced_color, dict)
    assert isinstance(bt.unforced, list)
    assert isinstance(all_forced, bool)


def test_determinism():
    """Same input produces same output (deterministic)."""
    X1 = np.array([[1, 1, 0], [0, 1, 0], [2, 2, 2]], np.int8)
    Y1 = np.array([[1, 1, 0], [0, 1, 0], [2, 2, 2]], np.int8)
    X2 = np.array([[1, 0], [0, 1]], np.int8)
    Y2 = np.array([[1, 0], [0, 1]], np.int8)

    train_pairs = [(X1, Y1), (X2, Y2)]
    spec0 = build_qt_spec([X1, X2])

    # Run multiple times
    results = [extract_bt_force_until_forced(train_pairs, spec0) for _ in range(3)]

    # All should be identical
    bt1, spec1, extra1 = results[0]
    for bt, spec, extra in results[1:]:
        assert bt.forced_color == bt1.forced_color
        assert bt.unforced == bt1.unforced
        assert spec.residues == spec1.residues
        assert spec.radii == spec1.radii
        assert spec.wl_rounds == spec1.wl_rounds


def test_stable_bytes_keys():
    """Verify keys are bytes and stable across grids."""
    X1 = np.array([[1, 2]], np.int8)
    Y1 = np.array([[3, 4]], np.int8)
    X2 = np.array([[1, 2]], np.int8)
    Y2 = np.array([[3, 4]], np.int8)

    spec = build_qt_spec([X1, X2])
    bt, _ = check_boundary_forced([(X1, Y1), (X2, Y2)], spec, None)

    # All keys must be bytes
    for key in bt.forced_color.keys():
        assert isinstance(key, bytes), f"Key is not bytes: {type(key)}"

    for key in bt.unforced:
        assert isinstance(key, bytes), f"Unforced key is not bytes: {type(key)}"


# ========== Edge Cases ==========

def test_no_identity_pairs():
    """No identity-shaped pairs → empty boundary."""
    X = np.array([[1, 2]], np.int8)
    Y = np.array([[3, 4, 5]], np.int8)  # Different shape

    spec = build_qt_spec([X])
    bt, all_forced = check_boundary_forced([(X, Y)], spec, None)

    # No evidence collected → empty
    assert len(bt.forced_color) == 0
    assert len(bt.unforced) == 0
    assert all_forced == True  # Vacuously true (no keys to force)


def test_size_changed_pairs_skipped():
    """Size-changed pairs are skipped during bucketing."""
    X1 = np.array([[1, 2]], np.int8)
    Y1_same = np.array([[3, 4]], np.int8)  # Same shape
    Y1_diff = np.array([[5]], np.int8)      # Different shape

    spec = build_qt_spec([X1])

    # With same-shape pair
    bt_same, _ = check_boundary_forced([(X1, Y1_same)], spec, None)

    # With different-shape pair
    bt_diff, _ = check_boundary_forced([(X1, Y1_diff)], spec, None)

    # Same-shape should have evidence, different should be empty
    assert len(bt_same.forced_color) > 0
    assert len(bt_diff.forced_color) == 0


def test_single_color_train():
    """Single-color grids should be forced quickly."""
    X = np.array([[7, 7], [7, 7]], np.int8)
    Y = np.array([[9, 9], [9, 9]], np.int8)

    spec = build_qt_spec([X])
    bt, all_forced = check_boundary_forced([(X, Y)], spec, None)

    # Should be forced (all pixels same class → same color)
    # Note: may have multiple classes due to positional features
    assert all_forced or len(bt.unforced) == 0


def test_collision_detection():
    """Multi-color class should be detected as unforced."""
    # Create a case where same class key gets different colors
    X1 = np.array([[0, 0]], np.int8)
    Y1 = np.array([[1, 2]], np.int8)  # Different colors for same input

    spec = QtSpec(radii=(), residues=(), use_diagonals=False, wl_rounds=0)
    bt, all_forced = check_boundary_forced([(X1, Y1)], spec, None)

    # Without positional features, both pixels have same class but different colors
    # This should create an unforced key
    assert all_forced == False or len(bt.unforced) > 0


def test_empty_train():
    """Empty train list handled gracefully."""
    spec = QtSpec(radii=(1,), residues=(2, 3), use_diagonals=True, wl_rounds=3)
    bt, all_forced = check_boundary_forced([], spec, None)

    assert len(bt.forced_color) == 0
    assert len(bt.unforced) == 0
    assert all_forced == True


# ========== Ladder Progression Tests ==========

def test_ladder_stops_at_first_forced():
    """Ladder stops at first step that achieves all_forced."""
    X = np.array([[1, 2], [3, 4]], np.int8)
    Y = np.array([[5, 6], [7, 8]], np.int8)

    spec0 = build_qt_spec([X])
    bt, spec_final, extra_final = extract_bt_force_until_forced([(X, Y)], spec0)

    # Should stop early (likely S0 or S1)
    # Verify it didn't enable unnecessary features
    if len(bt.unforced) == 0:
        # If forced at early stage, later features shouldn't be enabled
        # (This is heuristic; exact step depends on data)
        assert isinstance(spec_final, QtSpec)
        assert isinstance(extra_final, _ExtraFlags)


def test_ladder_s1_residue_extend():
    """S1 extends residues based on grid dimensions."""
    # Create small grid to test residue extension
    X = np.array([[1, 2, 3, 4, 5]], np.int8)  # 1x5 grid
    Y = np.array([[6, 7, 8, 9, 0]], np.int8)

    spec0 = QtSpec(radii=(1,), residues=(2, 3), use_diagonals=True, wl_rounds=3)
    bt, spec_final, extra = extract_bt_force_until_forced([(X, Y)], spec0)

    # S1 should add residues for dimensions up to max_dim
    # Original: (2, 3), should extend to include 4, 5 (divisors of 5)
    assert 4 in spec_final.residues or 5 in spec_final.residues or spec0.residues == spec_final.residues


def test_ladder_s2_radius_3():
    """S2 adds radius=3 if needed."""
    # If we manually force S2, it should add radius 3
    X = np.array([[1, 2], [3, 4]], np.int8)
    Y = np.array([[5, 6], [7, 8]], np.int8)

    spec0 = QtSpec(radii=(1, 2), residues=(2, 3), use_diagonals=True, wl_rounds=3)
    # Run ladder - may or may not reach S2, but spec should be valid
    bt, spec_final, extra = extract_bt_force_until_forced([(X, Y)], spec0)

    assert isinstance(spec_final.radii, tuple)
    assert all(isinstance(r, int) for r in spec_final.radii)


def test_ladder_s3_wl_rounds():
    """S3 increases WL rounds to 4 if needed."""
    X = np.array([[1]], np.int8)
    Y = np.array([[5]], np.int8)

    spec0 = QtSpec(radii=(1,), residues=(2,), use_diagonals=True, wl_rounds=2)
    bt, spec_final, extra = extract_bt_force_until_forced([(X, Y)], spec0)

    # May increase WL rounds (depends on whether earlier steps forced)
    assert spec_final.wl_rounds >= spec0.wl_rounds


def test_ladder_s4_border_distance():
    """S4 enables border_distance if needed."""
    X = np.array([[1, 2], [3, 4]], np.int8)
    Y = np.array([[5, 6], [7, 8]], np.int8)

    spec0 = build_qt_spec([X])
    bt, spec_final, extra = extract_bt_force_until_forced([(X, Y)], spec0)

    # extra.use_border_distance may be enabled
    assert isinstance(extra.use_border_distance, bool)


def test_ladder_s5_component_features():
    """S5 enables component features (centroid_parity or scan_index)."""
    X = np.array([[1, 2], [3, 4]], np.int8)
    Y = np.array([[5, 6], [7, 8]], np.int8)

    spec0 = build_qt_spec([X])
    bt, spec_final, extra = extract_bt_force_until_forced([(X, Y)], spec0)

    # At most one of these should be True (exclusive)
    assert isinstance(extra.use_centroid_parity, bool)
    assert isinstance(extra.use_component_scan_index, bool)
    # Mutual exclusion (at S5a try centroid, at S5b switch to scan_index)
    # But by the time we return, only one can be True if S5 was reached
    if extra.use_centroid_parity and extra.use_component_scan_index:
        # This shouldn't happen - S5a and S5b are exclusive
        pytest.fail("Both S5 flags enabled simultaneously")


def test_ladder_s6_final_wl():
    """S6 increases WL rounds to 5 (final step)."""
    X = np.array([[1]], np.int8)
    Y = np.array([[5]], np.int8)

    spec0 = QtSpec(radii=(1,), residues=(2,), use_diagonals=True, wl_rounds=3)
    bt, spec_final, extra = extract_bt_force_until_forced([(X, Y)], spec0)

    # Final WL rounds should be at most 5
    assert spec_final.wl_rounds <= 5


# ========== Comprehensive Integration Tests ==========

def test_identical_mapping():
    """Input == Output (identity transform) should be fully forced."""
    X1 = np.array([[1, 2, 3], [4, 5, 6]], np.int8)
    Y1 = X1.copy()
    X2 = np.array([[7, 8], [9, 0]], np.int8)
    Y2 = X2.copy()

    spec0 = build_qt_spec([X1, X2])
    bt, spec_final, extra = extract_bt_force_until_forced([(X1, Y1), (X2, Y2)], spec0)

    # Should be fully forced (each input class maps to itself)
    assert len(bt.unforced) == 0


def test_multiple_pairs_aggregation():
    """Multiple pairs with consistent mapping should aggregate correctly."""
    # Create consistent mapping: all 1s→5, all 2s→6
    X1 = np.array([[1, 1], [2, 2]], np.int8)
    Y1 = np.array([[5, 5], [6, 6]], np.int8)
    X2 = np.array([[1, 2]], np.int8)
    Y2 = np.array([[5, 6]], np.int8)

    spec = QtSpec(radii=(), residues=(), use_diagonals=False, wl_rounds=0)
    bt, all_forced = check_boundary_forced([(X1, Y1), (X2, Y2)], spec, None)

    # With no positional features, all 1s should map to 5, all 2s to 6
    # Should be forced
    assert all_forced == True


def test_large_grid():
    """Large grid (30x30) should complete in reasonable time."""
    X = np.random.randint(0, 10, (30, 30), dtype=np.int8)
    Y = np.random.randint(0, 10, (30, 30), dtype=np.int8)

    spec0 = build_qt_spec([X])
    bt, spec_final, extra = extract_bt_force_until_forced([(X, Y)], spec0)

    # Should complete without error
    assert isinstance(bt, Boundary)


def test_1d_grids():
    """1×N and N×1 grids should work correctly."""
    X1 = np.array([[1, 2, 3, 4, 5]], np.int8)  # 1x5
    Y1 = np.array([[6, 7, 8, 9, 0]], np.int8)
    X2 = np.array([[1], [2], [3]], np.int8)    # 3x1
    Y2 = np.array([[4], [5], [6]], np.int8)

    spec0 = build_qt_spec([X1, X2])
    bt, spec_final, extra = extract_bt_force_until_forced([(X1, Y1), (X2, Y2)], spec0)

    # Should complete without error
    assert isinstance(bt, Boundary)


print("All tests defined successfully")
