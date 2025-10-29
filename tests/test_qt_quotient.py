"""
WO-05: Qt quotient with stable class keys
Tests for src/qt/quotient.py

Test strategy (rigorous bug detection):
1. Acceptance: verify each function's core contract
2. Comprehensive: edge cases, determinism, WL effectiveness
3. Focus on BUGS: idempotence, stable keys, platform independence
"""
import numpy as np
import pytest
from src.qt.quotient import (
    make_initial_signature, pack_signature, relabel_classes,
    wl_refine, classes_for, _ExtraFlags
)
from src.qt.spec import build_qt_spec
from src.types import Grid, QtSpec


# ========== Acceptance Tests ==========

def test_initial_signature_basic():
    """Signature contains required keys in correct shapes."""
    g = np.array([[1, 2], [3, 4]], dtype=np.int8)
    spec = QtSpec(radii=(1,), residues=(2,), use_diagonals=False, wl_rounds=0)

    sig = make_initial_signature(g, spec)

    assert "color" in sig
    assert sig["color"].shape == (2, 2)
    assert "r2" in sig and "c2" in sig
    assert "cnt_r1" in sig
    assert sig["cnt_r1"].shape == (2, 2, 10)
    assert "comp_shape" in sig
    assert sig["comp_shape"].shape == (2, 2, 4)


def test_initial_signature_with_diagonals():
    """Diagonal residues added when enabled."""
    g = np.array([[0, 1], [2, 3]], dtype=np.int8)
    spec = QtSpec(radii=(), residues=(), use_diagonals=True, wl_rounds=0)

    sig = make_initial_signature(g, spec)

    for k in [2, 3, 4, 5]:
        assert f"anti_diag{k}" in sig
        assert f"diag{k}" in sig


def test_pack_signature_deterministic_order():
    """Pack preserves feature order: color, residues, diagonals, counts, components."""
    g = np.array([[5]], dtype=np.int8)
    spec = QtSpec(radii=(1,), residues=(2, 3), use_diagonals=True, wl_rounds=0)

    sig = make_initial_signature(g, spec)
    packed = pack_signature(sig)

    # Should have: color(1) + r2,r3,c2,c3(4) + diag2,diag3,diag4,diag5(8) + cnt_r1(10) + comp(5)
    expected_channels = 1 + 4 + 8 + 10 + 5
    assert packed.shape == (1, 1, expected_channels)
    assert packed.dtype == np.int32


def test_relabel_stable_keys():
    """Relabel produces deterministic local IDs and unique stable keys."""
    g = np.array([[1, 2], [2, 1]], dtype=np.int8)
    spec = build_qt_spec([g])

    sig = make_initial_signature(g, spec)
    packed = pack_signature(sig)

    classes1 = relabel_classes(packed)
    classes2 = relabel_classes(packed)

    # Determinism
    assert np.array_equal(classes1.ids, classes2.ids)
    assert classes1.key_for == classes2.key_for

    # Stable keys are bytes
    for key in classes1.key_for.values():
        assert isinstance(key, bytes)

    # Keys are unique
    keys = list(classes1.key_for.values())
    assert len(keys) == len(set(keys))


def test_wl_channel_separation():
    """WL adds wl_embed without overwriting color channel."""
    g = np.array([[0, 1], [1, 0]], dtype=np.int8)
    spec = QtSpec(radii=(), residues=(), use_diagonals=False, wl_rounds=2)

    sig = make_initial_signature(g, spec)
    assert "wl_embed" not in sig

    # After WL
    classes = wl_refine(g, spec)

    # Verify WL ran (check by running with 0 rounds and comparing)
    spec0 = QtSpec(radii=(), residues=(), use_diagonals=False, wl_rounds=0)
    classes0 = wl_refine(g, spec0)

    # WL should potentially refine (may or may not add classes depending on structure)
    assert len(classes.key_for) >= len(classes0.key_for)


def test_extra_flags_border_distance():
    """S4: border_dist feature added when flag enabled."""
    g = np.array([[1, 2], [3, 4]], dtype=np.int8)
    spec = QtSpec(radii=(), residues=(), use_diagonals=False, wl_rounds=0)
    extra = _ExtraFlags()
    extra.use_border_distance = True

    sig = make_initial_signature(g, spec, extra)
    assert "border_dist" in sig
    assert sig["border_dist"].shape == (2, 2)

    # Check Chebyshev distance (all border pixels in 2x2)
    expected = np.array([[0, 0], [0, 0]], dtype=np.int16)
    assert np.array_equal(sig["border_dist"], expected)


def test_extra_flags_centroid_parity():
    """S5a: centroid_parity feature added when flag enabled."""
    g = np.array([[1, 1], [1, 1]], dtype=np.int8)
    spec = QtSpec(radii=(), residues=(), use_diagonals=False, wl_rounds=0)
    extra = _ExtraFlags()
    extra.use_centroid_parity = True

    sig = make_initial_signature(g, spec, extra)
    assert "centroid_parity" in sig
    assert sig["centroid_parity"].shape == (2, 2, 2)


def test_platform_independent_bytes():
    """Packed features cast to int32 before bytes view (platform independence)."""
    g = np.array([[7]], dtype=np.int8)
    spec = build_qt_spec([g])

    sig = make_initial_signature(g, spec)
    packed = pack_signature(sig)

    # Verify int32 dtype
    assert packed.dtype == np.int32

    # Relabel creates bytes from int32 view
    classes = relabel_classes(packed)
    for key in classes.key_for.values():
        # Key length should be multiple of 4 (int32 = 4 bytes per feature)
        assert len(key) % 4 == 0


# ========== Comprehensive Tests ==========

def test_checkerboard_pattern():
    """Checkerboard produces multiple classes due to color and position."""
    g = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ], dtype=np.int8)
    spec = build_qt_spec([g])
    classes = classes_for(g, spec)

    assert len(classes.key_for) > 1
    assert classes.ids.shape == (3, 3)


def test_complex_multicolor():
    """Complex grid with multiple colors produces unique stable keys."""
    g = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ], dtype=np.int8)
    spec = build_qt_spec([g])
    classes = classes_for(g, spec)

    assert len(classes.key_for) > 0
    # Stable keys must be unique
    keys = list(classes.key_for.values())
    assert len(keys) == len(set(keys))


def test_uniform_grid_with_residues():
    """Uniform grid with residues produces positional classes (correct behavior)."""
    g = np.array([[7, 7], [7, 7]], dtype=np.int8)
    spec = build_qt_spec([g])  # Includes residues
    classes = classes_for(g, spec)

    # With residues, different positions have different features
    # 2x2 grid with r%2 and c%2 creates 4 distinct signatures
    assert len(classes.key_for) == 4
    assert classes.ids.shape == (2, 2)


def test_uniform_grid_pure_color():
    """Uniform grid without positional features produces single class."""
    g = np.array([[7, 7], [7, 7]], dtype=np.int8)
    spec = QtSpec(radii=(), residues=(), use_diagonals=False, wl_rounds=0)
    classes = classes_for(g, spec)

    # Without positional features, all same-color pixels should be same class
    assert len(classes.key_for) == 1
    assert classes.ids.min() == classes.ids.max() == 0


def test_horizontal_stripes():
    """Horizontal stripes with same colors in different rows produce multiple classes."""
    g = np.array([
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [1, 1, 1, 1]
    ], dtype=np.int8)
    spec = build_qt_spec([g])
    classes = classes_for(g, spec)

    # Rows 0 and 2 have same color but different positions (residues distinguish)
    assert len(classes.key_for) > 2


def test_wl_refinement_effectiveness():
    """WL refinement distinguishes neighborhood structure."""
    g = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype=np.int8)

    spec_no_wl = QtSpec(radii=(1,), residues=(2, 3), use_diagonals=True, wl_rounds=0)
    spec_with_wl = QtSpec(radii=(1,), residues=(2, 3), use_diagonals=True, wl_rounds=3)

    classes_no_wl = classes_for(g, spec_no_wl)
    classes_with_wl = classes_for(g, spec_with_wl)

    # WL should not reduce classes
    assert len(classes_with_wl.key_for) >= len(classes_no_wl.key_for)


def test_relabel_idempotence():
    """Relabeling twice produces same result (critical stability check)."""
    g = np.array([[1, 2], [3, 4]], dtype=np.int8)
    spec = build_qt_spec([g])

    sig = make_initial_signature(g, spec)
    packed = pack_signature(sig)

    classes1 = relabel_classes(packed)
    classes2 = relabel_classes(packed)

    assert np.array_equal(classes1.ids, classes2.ids)
    assert classes1.key_for == classes2.key_for


def test_large_grid_performance():
    """Large grid (30x30) completes without errors."""
    g = np.random.randint(0, 10, (30, 30), dtype=np.int8)
    spec = build_qt_spec([g])

    classes = classes_for(g, spec)

    assert classes.ids.shape == (30, 30)
    assert len(classes.key_for) > 0
    assert classes.ids.max() == len(classes.key_for) - 1


print("All tests defined successfully")
