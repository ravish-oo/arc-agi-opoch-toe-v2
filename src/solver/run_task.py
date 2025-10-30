# src/solver/run_task.py
from typing import Dict, List
import numpy as np

from ..types import Task, Grid, ShapeLawKind
from ..present.pi import canonize_task, uncanonize
from ..qt.spec import build_qt_spec
from ..solver.shape_law import infer_shape_law
from ..bt.boundary import extract_bt_force_until_forced, probe_writer_mode
from ..phi.paint import paint_phi
from ..kernel.grid import d8_apply


def solve_task(
    task: Task,
    *,
    enable_frame: bool = False,
    enable_tiling: bool = False,
    periodicity_check: bool = False
) -> List[Grid]:
    """
    Solve a single ARC task end-to-end.

    Pipeline: Π(train) -> QtSpec -> Δ(original sizes) -> Bt(force-until-forced) -> Φ(test) -> un-Π

    Args:
        task: ARC task with train pairs and test inputs
        enable_frame: Enable FRAME shape law detection
        enable_tiling: Enable TILING shape law detection
        periodicity_check: Enable periodicity check for TILING

    Returns:
        List of output grids (one per test input), in index order.
        Each grid is np.int8, C-contiguous.

    Deterministic. No prints/logging/IO.
    """
    # Step 1: Δ from original (non-canonized) train sizes
    # Must be computed BEFORE any canonization to use original dimensions
    delta = infer_shape_law(
        task.train,
        enable_frame=enable_frame,
        enable_tiling=enable_tiling,
        periodicity_check=periodicity_check
    )

    # Step 2: Π task-level (train+test with single union order)
    train_Xs = [x for x, _ in task.train]
    c_train, c_test, union_order = canonize_task(train_Xs, task.test)

    # Step 3: QtSpec from canonized train inputs (content-blind, input-only)
    spec0 = build_qt_spec(c_train.grids)

    # Step 3.5: Canonize outputs with same D8 as inputs
    canon_train_pairs = [
        (cx, d8_apply(y, meta.transform_id))
        for cx, meta, (_, y) in zip(c_train.grids, c_train.metas, task.train)
    ]

    # Step 3.6: Probe write-law for size-change tasks (BEFORE Bt)
    kh, kw = (1, 1)
    if delta.kind == ShapeLawKind.BLOW_UP:
        kh, kw = delta.kh, delta.kw

    writer_mode, tiling_policy = ('identity', None)
    if kh > 1 or kw > 1:
        writer_mode, tiling_policy = probe_writer_mode(canon_train_pairs, kh, kw)

    # Step 4: Bt via ladder (force-until-forced, Δ-aware pullback)
    bt, specF, extraF = extract_bt_force_until_forced(
        canon_train_pairs,
        spec0,
        delta,
        writer_mode,
        tiling_policy if tiling_policy is not None else 'uniform'
    )

    # Step 5: Φ on canonized tests (Δ-aware, guards)
    # Test inputs already canonized in Step 2 with same union order as train

    # Paint each canonized test input using determined writer_mode
    outs_canon = []
    for cx in c_test.grids:
        out_canon = paint_phi(
            cx,
            specF,
            bt,
            delta,
            enable_frame=enable_frame,
            enable_tiling=(writer_mode == 'tiling'),
            tiling_policy=tiling_policy if tiling_policy is not None else 'uniform'
        )
        outs_canon.append(out_canon)

    # Step 6: un-Π (restore original pose)
    outs = [uncanonize(oy, meta) for oy, meta in zip(outs_canon, c_test.metas)]

    # Ensure all outputs are int8 and C-contiguous
    outs = [np.ascontiguousarray(o, dtype=np.int8) for o in outs]

    return outs


def solve_all(
    tasks: Dict[str, Task],
    *,
    enable_frame: bool = False,
    enable_tiling: bool = False,
    periodicity_check: bool = False
) -> Dict[str, List[Grid]]:
    """
    Convenience: run solve_task over a mapping of {task_id: Task}.

    Args:
        tasks: Dictionary mapping task_id -> Task
        enable_frame: Enable FRAME shape law detection
        enable_tiling: Enable TILING shape law detection
        periodicity_check: Enable periodicity check for TILING

    Returns:
        Dictionary mapping task_id -> [outputs...]

    Task IDs are iterated in sorted order for determinism.
    Deterministic. No prints/logging/IO.
    """
    preds = {}

    # Iterate in sorted order for determinism
    for tid in sorted(tasks.keys()):
        preds[tid] = solve_task(
            tasks[tid],
            enable_frame=enable_frame,
            enable_tiling=enable_tiling,
            periodicity_check=periodicity_check
        )

    return preds
