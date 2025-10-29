# src/types.py
from dataclasses import dataclass
from typing import List, Tuple, Dict
from enum import Enum, auto
import numpy as np

# ========== Grid ==========
Grid = np.ndarray  # MUST be dtype np.int8, shape (h, w), values in 0..9

# ========== Task ==========
@dataclass
class Task:
    """
    A single ARC task.
    train: list of (X_i, Y_i) pairs, both as int8 Grids.
    test:  list of test inputs X_te_j as int8 Grids.
    """
    train: List[Tuple[Grid, Grid]]
    test:  List[Grid]

# ========== Present canon meta ==========
@dataclass
class CanonMeta:
    """
    Present canon metadata per grid.
    transform_id: integer in [0..7] enumerating D8 action used by Π.
    original_shape: (h, w) of the original, pre-canon grid.
    """
    transform_id: int
    original_shape: Tuple[int, int]

@dataclass
class Canonized:
    """
    Batch canonization result used by Π: list of canon grids + metas.
    """
    grids: List[Grid]
    metas: List[CanonMeta]

# ========== Qt spec (feature family), NOT a cached partition ==========
@dataclass(frozen=True)
class QtSpec:
    """
    Input-only feature family (Qt-as-spec) for building per-grid classes.
    - radii: neighborhood radii (e.g., (1,2) or (1,2,3))
    - residues: tuple of moduli (cap at 10), deterministically sorted
    - use_diagonals: include diagonal residues channels
    - wl_rounds: (3..5) per refinement ladder
    """
    radii: Tuple[int, ...]
    residues: Tuple[int, ...]
    use_diagonals: bool
    wl_rounds: int

# ========== Per-grid classes with stable keys ==========
@dataclass
class Classes:
    """
    Per-grid class ids with stable class signatures (bytes).
    ids:     (h, w) int32 local ids
    key_for: local_id -> bytes stable signature (intrinsic packed features)
    """
    ids: np.ndarray
    key_for: Dict[int, bytes]

# ========== Boundary (Bt) keyed by stable signatures ==========
@dataclass
class Boundary:
    """
    Forced color map keyed by stable class signature bytes.
    forced_color: signature -> color in 0..9
    unforced:     list of signatures that collided in train (should be empty after refinement)
    """
    forced_color: Dict[bytes, int]
    unforced: List[bytes]

# ========== Shape law Δ (dimensions-only) ==========
class ShapeLawKind(Enum):
    IDENTITY = auto()
    BLOW_UP = auto()   # rectangular allowed (kh, kw)
    FRAME = auto()     # add border (kh=kw=t for thickness t)
    TILING = auto()    # periodic copy (kh, kw factors)

@dataclass(frozen=True)
class ShapeLaw:
    """
    Dimensions-only shape law.
    kind: one of ShapeLawKind
    kh, kw: vertical/horizontal integer scale (for BLOW_UP); both =1 for IDENTITY
    """
    kind: ShapeLawKind
    kh: int = 1
    kw: int = 1
