#!/usr/bin/env python3
"""
scripts/diagnostic_single_task.py

Detailed diagnostic for a single ARC task showing 1:1 mapping between:
- Hand-solve steps (what Claude does by inspection)
- Code execution steps (what the solver produces)

Usage:
    python scripts/diagnostic_single_task.py <task_id>
"""
import sys
import json
from pathlib import Path
import numpy as np

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.types import Task, Grid, ShapeLaw, ShapeLawKind
from src.solver.run_task import solve_task
from src.solver.shape_law import infer_shape_law
from src.present.pi import canonize_task, uncanonize
from src.qt.spec import build_qt_spec
from src.bt.boundary import extract_bt_force_until_forced, probe_writer_mode
from src.phi.paint import paint_phi
from src.kernel.grid import d8_apply


def load_task(task_id: str) -> tuple[Task, list[Grid]]:
    """Load task and ground truth."""
    challenges_path = repo_root / "data" / "arc-agi_training_challenges.json"
    solutions_path = repo_root / "data" / "arc-agi_training_solutions.json"

    with open(challenges_path) as f:
        challenges = json.load(f)
    with open(solutions_path) as f:
        solutions = json.load(f)

    if task_id not in challenges:
        raise ValueError(f"Task {task_id} not found")

    data = challenges[task_id]

    # Parse train
    train = []
    for pair in data['train']:
        x = np.array(pair['input'], dtype=np.int8)
        y = np.array(pair['output'], dtype=np.int8)
        train.append((x, y))

    # Parse test
    test = []
    for pair in data['test']:
        x = np.array(pair['input'], dtype=np.int8)
        test.append(x)

    # Parse ground truth
    ground_truth = []
    if task_id in solutions:
        for output in solutions[task_id]:
            y = np.array(output, dtype=np.int8)
            ground_truth.append(y)

    return Task(train=train, test=test), ground_truth


def print_grid(g: Grid, label: str = ""):
    """Pretty print a grid."""
    if label:
        print(f"\n{label}:")
    print(f"  Shape: {g.shape}")
    print(f"  Colors: {sorted(set(g.flat))}")
    for row in g:
        print("  ", "".join(str(c) for c in row))


def print_section(title: str):
    """Print section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def diagnostic(task_id: str):
    """Run detailed diagnostic on a single task."""
    print(f"\nDIAGNOSTIC FOR TASK: {task_id}")
    print("=" * 80)

    # Load task
    task, ground_truth = load_task(task_id)

    print(f"\nTask structure:")
    print(f"  Training pairs: {len(task.train)}")
    print(f"  Test inputs: {len(task.test)}")
    print(f"  Ground truth outputs: {len(ground_truth)}")

    # Show train pairs
    print_section("TRAIN PAIRS (Original)")
    for i, (x, y) in enumerate(task.train):
        print(f"\nPair {i}:")
        print_grid(x, f"  Input X{i}")
        print_grid(y, f"  Output Y{i}")

    # Show test inputs
    print_section("TEST INPUTS (Original)")
    for i, x in enumerate(task.test):
        print_grid(x, f"  Test {i}")

    # Show ground truth
    print_section("GROUND TRUTH")
    for i, y in enumerate(ground_truth):
        print_grid(y, f"  Expected output {i}")

    # ========== STEP 1: Δ (Shape Law) ==========
    print_section("STEP 1: Δ (Shape Law Inference)")
    print("\nDimensions (original, before canonization):")
    for i, (x, y) in enumerate(task.train):
        hx, wx = x.shape
        hy, wy = y.shape
        print(f"  Pair {i}: X{hx}×{wx} → Y{hy}×{wy}")

    delta = infer_shape_law(task.train)
    print(f"\nInferred Δ:")
    print(f"  Kind: {delta.kind.name}")
    print(f"  kh: {delta.kh}, kw: {delta.kw}")

    print("\n[HAND-SOLVE CHECK]")
    print("  Look at dimensions: do all pairs have same shape transformation?")
    print("  IDENTITY: h_out = h_in, w_out = w_in")
    print("  BLOW_UP(kh,kw): h_out = kh*h_in, w_out = kw*w_in")
    print("  FRAME(t): h_out = h_in+2t, w_out = w_in+2t")
    print("  TILING(kh,kw): same as BLOW_UP but inputs are periodic")

    # ========== STEP 2: Π (Canonization) ==========
    print_section("STEP 2: Π (Task-level Canonization)")

    train_Xs = [x for x, _ in task.train]
    c_train, c_test, union_order = canonize_task(train_Xs, task.test)

    print(f"\nCanonized train inputs (pose-normalized with task-level union order):")
    for i, (cx, meta) in enumerate(zip(c_train.grids, c_train.metas)):
        print(f"\nTrain X{i} (canonized):")
        print(f"  Transform ID: {meta.transform_id} (D8 action)")
        print(f"  Original shape: {meta.original_shape}")
        print_grid(cx, f"  Canonized grid")

    print("\n[HAND-SOLVE CHECK]")
    print("  Task-level canonization: compute single union order from train+test inputs")
    print("  This ensures D8 tie-breaks are consistent across train and test")
    print("  D8 = {e, r90, r180, r270, fh, fv, fd, fa} (8 symmetries)")
    print("  Also does rank-view palette normalization (not shown here)")

    # ========== STEP 3: Qt Spec ==========
    print_section("STEP 3: Qt Spec (Feature Family)")

    spec0 = build_qt_spec(c_train.grids)

    print(f"\nQt Spec (input-only, from canonized train inputs):")
    print(f"  Radii: {spec0.radii}")
    print(f"  Residues: {spec0.residues}")
    print(f"  Use diagonals: {spec0.use_diagonals}")
    print(f"  WL rounds: {spec0.wl_rounds}")

    print("\n[HAND-SOLVE CHECK]")
    print("  QtSpec is built from input dimensions only (no pixel content)")
    print("  Radii: (1,2) if max(h,w)<=20, else (1,2,3)")
    print("  Residues: {2,3,4,5,6} + divisors of all h,w, capped at 10")

    # ========== STEP 4: Bt (Boundary via Ladder) ==========
    print_section("STEP 4: Bt (Boundary Learning)")

    print("\nCanonize train outputs (same D8 as paired input):")
    canon_train_pairs = []
    for i, (cx, meta, (_, y)) in enumerate(zip(c_train.grids, c_train.metas, task.train)):
        cy = d8_apply(y, meta.transform_id)
        canon_train_pairs.append((cx, cy))
        print(f"\nTrain Y{i} (canonized with transform_id={meta.transform_id}):")
        print_grid(cy, "  Canonized output")

    print("\nProbing write-law (blowup vs tiling with policy)...")
    kh, kw = (1, 1)
    if delta.kind.name == 'BLOW_UP':
        kh, kw = delta.kh, delta.kw

    writer_mode, tiling_policy = ('identity', None)
    if kh > 1 or kw > 1:
        writer_mode, tiling_policy = probe_writer_mode(canon_train_pairs, kh, kw)

    print(f"  Writer mode: {writer_mode}")
    print(f"  Tiling policy: {tiling_policy if tiling_policy is not None else 'N/A'}")

    print("\nRunning ladder refinement (force-until-forced)...")
    bt, specF, extraF = extract_bt_force_until_forced(
        canon_train_pairs,
        spec0,
        delta,
        writer_mode,
        tiling_policy if tiling_policy is not None else 'uniform'
    )

    print(f"\nFinal Bt (Boundary):")
    print(f"  Forced colors: {len(bt.forced_color)} class signatures")
    print(f"  Unforced: {len(bt.unforced)} signatures")
    print(f"  Final QtSpec wl_rounds: {specF.wl_rounds}")
    if extraF:
        print(f"  Extra refinement flags used: {extraF.__dict__}")

    print("\n[HAND-SOLVE CHECK]")
    print("  Ladder tries S0, S1, S2, ... until all classes are forced")
    print("  S0: color only")
    print("  S1: color + residues + local counts + components")
    print("  S2+: add WL refinement rounds")
    print("  For each signature (bytes), learn forced color from train")

    # ========== STEP 5: Φ (Paint Test) ==========
    print_section("STEP 5: Φ (Paint Test Outputs)")

    print(f"\nCanonized test inputs (using same union order from Step 2):")
    for i, (cx, meta) in enumerate(zip(c_test.grids, c_test.metas)):
        print(f"\nTest {i} (canonized):")
        print(f"  Transform ID: {meta.transform_id}")
        print_grid(cx, "  Canonized test input")

    print("\nPainting each test (Φ with Bt and Δ)...")
    outs_canon = []
    for i, cx in enumerate(c_test.grids):
        print(f"\nPainting test {i}...")
        out_canon = paint_phi(
            cx,
            specF,
            bt,
            delta,
            enable_tiling=(writer_mode == 'tiling'),
            tiling_policy=tiling_policy if tiling_policy is not None else 'uniform'
        )
        outs_canon.append(out_canon)
        print_grid(out_canon, f"  Painted output (canonized)")

    print("\n[HAND-SOLVE CHECK]")
    print("  Φ creates output grid using Δ for dimensions")
    print("  For each output pixel, compute Qt signature")
    print("  Look up signature in Bt to get forced color")
    print("  Δ-aware: map output coords back to input coords via Δ inverse")

    # ========== STEP 6: un-Π (Restore Pose) ==========
    print_section("STEP 6: un-Π (Restore Original Pose)")

    outs = [uncanonize(oy, meta) for oy, meta in zip(outs_canon, c_test.metas)]

    print("\nFinal outputs (un-canonized):")
    for i, out in enumerate(outs):
        print_grid(out, f"\nPredicted output {i}")

    # ========== COMPARISON ==========
    print_section("COMPARISON: Predicted vs Ground Truth")

    for i, (pred, true) in enumerate(zip(outs, ground_truth)):
        print(f"\nTest {i}:")
        match = np.array_equal(pred, true)
        print(f"  Shape match: {pred.shape} vs {true.shape} → {pred.shape == true.shape}")
        print(f"  Content match: {match}")

        if not match:
            print("\n  DIFF (positions where pred != true):")
            if pred.shape == true.shape:
                diff = (pred != true)
                diff_positions = np.argwhere(diff)
                print(f"    Mismatched pixels: {len(diff_positions)}/{pred.size}")
                if len(diff_positions) <= 20:
                    for r, c in diff_positions:
                        print(f"      [{r},{c}]: pred={pred[r,c]}, true={true[r,c]}")
                else:
                    print(f"    (showing first 20)")
                    for r, c in diff_positions[:20]:
                        print(f"      [{r},{c}]: pred={pred[r,c]}, true={true[r,c]}")

    # ========== VERDICT ==========
    print_section("VERDICT")

    success = all(np.array_equal(p, t) for p, t in zip(outs, ground_truth))

    if success:
        print("\n✓ PASS - Solver produces correct output!")
    else:
        print("\n✗ FAIL - Solver output differs from ground truth")
        print("\nNext steps:")
        print("  1. Review each step above")
        print("  2. Identify which step produces unexpected result")
        print("  3. Compare code behavior vs hand-solve expectation")
        print("  4. Isolate minimal fix")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/diagnostic_single_task.py <task_id>")
        print("Example: python scripts/diagnostic_single_task.py 007bbfb7")
        sys.exit(1)

    task_id = sys.argv[1]
    diagnostic(task_id)
