# src/qt/spec.py
from typing import List, Set
from ..types import Grid, QtSpec
from ..kernel.grid import dims, divisors

MAX_RESIDUES: int = 10  # binding cap

def build_qt_spec(inputs_canon: List[Grid]) -> QtSpec:
    """
    Build an input-only QtSpec from canonized inputs.
    - residues: start with {2,3,4,5,6}, extend with divisors of all seen heights/widths, cap=10, ascending sort
    - radii: (1,2) if max(h,w) <= 20 else (1,2,3)
    - use_diagonals: True
    - wl_rounds: 3
    Deterministic for the same list of input shapes.
    """
    # 1. Validate non-empty
    if not inputs_canon:
        raise ValueError("inputs_canon must be non-empty")

    # 2. Extract shapes (input-only: only dimensions, no pixel reads)
    shapes = [dims(x) for x in inputs_canon]
    all_h = [h for h, w in shapes]
    all_w = [w for h, w in shapes]

    # 3. Build residues: base set + divisors from all shapes
    residue_set: Set[int] = {2, 3, 4, 5, 6}
    for h in all_h:
        residue_set.update(divisors(h, max_div=MAX_RESIDUES))
    for w in all_w:
        residue_set.update(divisors(w, max_div=MAX_RESIDUES))

    # Sort ascending and cap at MAX_RESIDUES
    residues = tuple(sorted(residue_set)[:MAX_RESIDUES])

    # 4. Determine radii based on max dimension
    max_h = max(all_h)
    max_w = max(all_w)
    radii = (1, 2) if max_h <= 20 and max_w <= 20 else (1, 2, 3)

    # 5. Return spec (diagonals always on, WL starts at 3)
    return QtSpec(
        radii=radii,
        residues=residues,
        use_diagonals=True,
        wl_rounds=3
    )
