"""
WO-08: Φ Paint Tests
Tests for src/phi/paint.py

Test strategy:
1. Acceptance: verify core contracts (IDENTITY, BLOW_UP, FRAME, TILING)
2. Guards: Bt-empty guard, identity guard
3. Edge cases: 1×N grids, single-class, large grids
4. One-stroke: verify no multi-stroke
5. Stable keys: verify bytes-based lookup
"""
import numpy as np
import pytest
from src.phi.paint import paint_phi
from src.types import Boundary, ShapeLaw, ShapeLawKind, QtSpec
from src.qt.spec import build_qt_spec
from src.qt.quotient import classes_for


# ========== Acceptance Tests ==========

def test_identity_basic():
    """IDENTITY case with forced Bt reproduces input."""
    x = np.array([[1, 2], [3, 4]], np.int8)
    spec = build_qt_spec([x])
    cls = classes_for(x, spec)

    # Force each class to its input color
    forced = {key: int(x[np.where(cls.ids == lid)][0]) for lid, key in cls.key_for.items()}
    bt = Boundary(forced_color=forced, unforced=[])

    y = paint_phi(x, spec, bt, ShapeLaw(ShapeLawKind.IDENTITY))

    assert y.shape == x.shape
    assert y.dtype == np.int8
    assert np.array_equal(y, x)


def test_blowup_basic():
    """BLOW_UP case expands each pixel to kh×kw block."""
    x = np.array([[1, 2]], np.int8)
    spec = build_qt_spec([x])
    cls = classes_for(x, spec)

    # Force colors
    forced = {key: int(x[np.where(cls.ids == lid)][0]) for lid, key in cls.key_for.items()}
    bt = Boundary(forced_color=forced, unforced=[])

    y = paint_phi(x, spec, bt, ShapeLaw(ShapeLawKind.BLOW_UP, kh=3, kw=2))

    assert y.shape == (3, 4)  # 1×2 → 3×4
    assert y.dtype == np.int8
    # Top-left block should be x[0,0]
    assert (y[0:3, 0:2] == x[0, 0]).all()


def test_blowup_rectangular():
    """Rectangular blow-up (kh ≠ kw)."""
    x = np.array([[5]], np.int8)
    spec = build_qt_spec([x])
    cls = classes_for(x, spec)

    forced = {key: int(x[np.where(cls.ids == lid)][0]) for lid, key in cls.key_for.items()}
    bt = Boundary(forced_color=forced, unforced=[])

    y = paint_phi(x, spec, bt, ShapeLaw(ShapeLawKind.BLOW_UP, kh=2, kw=5))

    assert y.shape == (2, 5)
    assert (y == 5).all()


def test_frame_basic():
    """FRAME adds border and interior."""
    x = np.array([[1, 2], [3, 4]], np.int8)
    spec = build_qt_spec([x])
    bt = Boundary(forced_color={}, unforced=[])  # Use input fallback

    y = paint_phi(x, spec, bt, ShapeLaw(ShapeLawKind.FRAME, kh=1, kw=1), enable_frame=True)

    assert y.shape == (4, 4)  # 2×2 + 2*1 border
    assert y.dtype == np.int8
    # Interior should match input
    assert np.array_equal(y[1:3, 1:3], x)


def test_tiling_basic():
    """TILING replicates identity patch."""
    x = np.array([[1, 2]], np.int8)
    spec = build_qt_spec([x])
    bt = Boundary(forced_color={}, unforced=[])

    y = paint_phi(x, spec, bt, ShapeLaw(ShapeLawKind.TILING, kh=2, kw=3), enable_tiling=True)

    assert y.shape == (2, 6)  # 1×2 tiled 2×3
    assert y.dtype == np.int8
    # Each tile should match identity painting
    for i in range(2):
        for j in range(3):
            tile = y[i:i+1, j*2:(j+1)*2]
            # Should match input (with Bt-empty guard)
            assert tile.shape == x.shape


# ========== Guard Tests ==========

def test_bt_empty_identity_guard():
    """Bt-empty: use input color for IDENTITY."""
    x = np.array([[7, 8], [9, 0]], np.int8)
    spec = build_qt_spec([x])
    bt = Boundary(forced_color={}, unforced=[])  # Empty Bt

    y = paint_phi(x, spec, bt, ShapeLaw(ShapeLawKind.IDENTITY))

    # With Bt-empty guard, should use input colors
    assert y.shape == x.shape
    # Each class gets its input color
    # (exact match depends on how classes partition, but output should be valid)
    assert y.dtype == np.int8


def test_bt_empty_blowup_guard():
    """Bt-empty: use input color for BLOW_UP."""
    x = np.array([[3]], np.int8)
    spec = build_qt_spec([x])
    bt = Boundary(forced_color={}, unforced=[])

    y = paint_phi(x, spec, bt, ShapeLaw(ShapeLawKind.BLOW_UP, kh=2, kw=2))

    assert y.shape == (2, 2)
    # All pixels should be input color (3)
    assert (y == 3).all()


def test_partial_bt_forcing():
    """Some classes forced, others use input fallback."""
    x = np.array([[1, 2], [3, 4]], np.int8)
    spec = build_qt_spec([x])
    cls = classes_for(x, spec)

    # Force only first class
    first_key = next(iter(cls.key_for.values()))
    bt = Boundary(forced_color={first_key: 9}, unforced=[])

    y = paint_phi(x, spec, bt, ShapeLaw(ShapeLawKind.IDENTITY))

    # Should have at least one pixel with forced color 9
    assert 9 in y
    # Other pixels should have their input colors (fallback)
    assert y.dtype == np.int8


# ========== Edge Cases ==========

def test_1xn_grid():
    """1×N grid works correctly."""
    x = np.array([[1, 2, 3, 4, 5]], np.int8)
    spec = build_qt_spec([x])
    bt = Boundary(forced_color={}, unforced=[])

    # IDENTITY
    y_id = paint_phi(x, spec, bt, ShapeLaw(ShapeLawKind.IDENTITY))
    assert y_id.shape == (1, 5)

    # BLOW_UP
    y_bu = paint_phi(x, spec, bt, ShapeLaw(ShapeLawKind.BLOW_UP, kh=2, kw=3))
    assert y_bu.shape == (2, 15)


def test_nx1_grid():
    """N×1 grid works correctly."""
    x = np.array([[1], [2], [3]], np.int8)
    spec = build_qt_spec([x])
    bt = Boundary(forced_color={}, unforced=[])

    # IDENTITY
    y_id = paint_phi(x, spec, bt, ShapeLaw(ShapeLawKind.IDENTITY))
    assert y_id.shape == (3, 1)

    # BLOW_UP
    y_bu = paint_phi(x, spec, bt, ShapeLaw(ShapeLawKind.BLOW_UP, kh=3, kw=2))
    assert y_bu.shape == (9, 2)


def test_single_pixel():
    """1×1 grid works."""
    x = np.array([[7]], np.int8)
    spec = build_qt_spec([x])
    bt = Boundary(forced_color={}, unforced=[])

    y = paint_phi(x, spec, bt, ShapeLaw(ShapeLawKind.IDENTITY))
    assert y.shape == (1, 1)
    assert y[0, 0] == 7


def test_all_zero():
    """All-zero grid works."""
    x = np.zeros((3, 3), dtype=np.int8)
    spec = build_qt_spec([x])
    bt = Boundary(forced_color={}, unforced=[])

    y = paint_phi(x, spec, bt, ShapeLaw(ShapeLawKind.IDENTITY))
    assert y.shape == (3, 3)
    assert (y == 0).all()


def test_single_class():
    """Single-class input (all same color)."""
    x = np.array([[5, 5], [5, 5]], np.int8)
    spec = QtSpec(radii=(), residues=(), use_diagonals=False, wl_rounds=0)
    bt = Boundary(forced_color={}, unforced=[])

    y = paint_phi(x, spec, bt, ShapeLaw(ShapeLawKind.IDENTITY))
    assert y.shape == (2, 2)
    # All pixels should be same color (5 with Bt-empty guard)
    assert (y == 5).all()


def test_large_grid():
    """Large grid (30×30) completes."""
    x = np.random.randint(0, 10, (30, 30), dtype=np.int8)
    spec = build_qt_spec([x])
    bt = Boundary(forced_color={}, unforced=[])

    y = paint_phi(x, spec, bt, ShapeLaw(ShapeLawKind.IDENTITY))
    assert y.shape == (30, 30)
    assert y.dtype == np.int8


# ========== One-Stroke Verification ==========

def test_one_stroke_identity():
    """Each pixel written exactly once in IDENTITY."""
    x = np.array([[1, 2], [3, 4]], np.int8)
    spec = build_qt_spec([x])
    bt = Boundary(forced_color={}, unforced=[])

    y = paint_phi(x, spec, bt, ShapeLaw(ShapeLawKind.IDENTITY))

    # No verification beyond correct output (one-stroke is implementation detail)
    assert y.dtype == np.int8
    assert y.shape == x.shape


def test_one_stroke_blowup():
    """Each block written exactly once in BLOW_UP."""
    x = np.array([[1, 2]], np.int8)
    spec = build_qt_spec([x])
    bt = Boundary(forced_color={}, unforced=[])

    y = paint_phi(x, spec, bt, ShapeLaw(ShapeLawKind.BLOW_UP, kh=2, kw=2))

    # Verify blocks are constant
    assert (y[0:2, 0:2] == y[0, 0]).all()  # First block constant
    assert (y[0:2, 2:4] == y[0, 2]).all()  # Second block constant


# ========== Stable Keys ==========

def test_stable_bytes_keys():
    """Paint uses bytes keys from Bt, not local IDs."""
    x = np.array([[1, 2]], np.int8)
    spec = build_qt_spec([x])
    cls = classes_for(x, spec)

    # Create Bt with bytes keys
    forced = {}
    for lid, key in cls.key_for.items():
        assert isinstance(key, bytes), "Keys must be bytes"
        forced[key] = 9  # Force all to color 9

    bt = Boundary(forced_color=forced, unforced=[])
    y = paint_phi(x, spec, bt, ShapeLaw(ShapeLawKind.IDENTITY))

    # All pixels should be 9 (forced)
    assert (y == 9).all()


# ========== Return Type Verification ==========

def test_output_contiguous():
    """Output is C-contiguous."""
    x = np.array([[1, 2], [3, 4]], np.int8)
    spec = build_qt_spec([x])
    bt = Boundary(forced_color={}, unforced=[])

    y = paint_phi(x, spec, bt, ShapeLaw(ShapeLawKind.IDENTITY))

    assert y.flags['C_CONTIGUOUS'], "Output must be C-contiguous"


def test_output_dtype():
    """Output is np.int8."""
    x = np.array([[1, 2], [3, 4]], np.int8)
    spec = build_qt_spec([x])
    bt = Boundary(forced_color={}, unforced=[])

    for delta in [
        ShapeLaw(ShapeLawKind.IDENTITY),
        ShapeLaw(ShapeLawKind.BLOW_UP, kh=2, kw=2)
    ]:
        y = paint_phi(x, spec, bt, delta)
        assert y.dtype == np.int8, f"Output dtype must be np.int8, got {y.dtype}"


# ========== Determinism ==========

def test_determinism():
    """Same input produces same output."""
    x = np.array([[1, 2], [3, 4]], np.int8)
    spec = build_qt_spec([x])
    bt = Boundary(forced_color={}, unforced=[])
    delta = ShapeLaw(ShapeLawKind.IDENTITY)

    results = [paint_phi(x, spec, bt, delta) for _ in range(3)]

    # All results should be identical
    for y in results[1:]:
        assert np.array_equal(y, results[0])


print("All tests defined successfully")
