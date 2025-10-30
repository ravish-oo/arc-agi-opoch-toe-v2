#!/usr/bin/env python3
"""
scripts/scan_pi_drift.py

Scan for tasks with "task-level Π" drift symptom:
- Bt has many forced keys from training (good learning)
- But test classes rarely hit forced_color (forced-hit rate ≈ 0%)

This indicates canonization mismatch between train and test.
"""
import sys
import json
from pathlib import Path
import numpy as np

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.types import Task, Grid, ShapeLawKind
from src.solver.shape_law import infer_shape_law
from src.present.pi import canonize_inputs
from src.qt.spec import build_qt_spec
from src.bt.boundary import extract_bt_force_until_forced, probe_writer_mode
from src.qt.quotient import classes_for
from src.kernel.grid import d8_apply


def load_tasks(num_tasks=50):
    """Load first N training tasks."""
    challenges_path = repo_root / "data" / "arc-agi_training_challenges.json"

    with open(challenges_path) as f:
        challenges = json.load(f)

    tasks = {}
    for i, (task_id, data) in enumerate(challenges.items()):
        if i >= num_tasks:
            break

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

        tasks[task_id] = Task(train=train, test=test)

    return tasks


def compute_forced_hit_rate(task_id, task):
    """
    Compute forced-hit rate for a task.

    Returns:
        forced_count: number of forced keys in Bt
        hit_rate: percentage of test classes that hit forced_color
    """
    # Step 1: Δ
    delta = infer_shape_law(task.train)

    # Step 2: Π train
    train_Xs = [x for x, _ in task.train]
    c_train = canonize_inputs(train_Xs)

    # Step 3: Qt spec
    spec0 = build_qt_spec(c_train.grids)

    # Step 3.5: Canonize outputs
    canon_train_pairs = [
        (cx, d8_apply(y, meta.transform_id))
        for cx, meta, (_, y) in zip(c_train.grids, c_train.metas, task.train)
    ]

    # Step 3.6: Probe writer
    kh, kw = (1, 1)
    if delta.kind == ShapeLawKind.BLOW_UP:
        kh, kw = delta.kh, delta.kw

    writer_mode, tiling_policy = ('identity', None)
    if kh > 1 or kw > 1:
        writer_mode, tiling_policy = probe_writer_mode(canon_train_pairs, kh, kw)

    # Step 4: Bt
    bt, specF, extraF = extract_bt_force_until_forced(
        canon_train_pairs,
        spec0,
        delta,
        writer_mode,
        tiling_policy if tiling_policy is not None else 'uniform'
    )

    forced_count = len(bt.forced_color)

    # If no forced keys, can't compute hit rate
    if forced_count == 0:
        return forced_count, 0.0

    # Step 5: Compute hit rate on test
    c_test = canonize_inputs(task.test)

    total_classes = 0
    hit_classes = 0

    for cx in c_test.grids:
        cls = classes_for(cx, specF)

        for local_id, key in cls.key_for.items():
            total_classes += 1
            if key in bt.forced_color:
                hit_classes += 1

    hit_rate = (hit_classes / total_classes * 100) if total_classes > 0 else 0.0

    return forced_count, hit_rate


def main():
    print("Scanning for task-level Π drift (good Bt, low test hit rate)...")
    print("=" * 80)

    tasks = load_tasks(num_tasks=100)

    candidates = []

    for task_id, task in tasks.items():
        try:
            forced_count, hit_rate = compute_forced_hit_rate(task_id, task)

            # Symptom: good Bt learning (>= 4 forced keys) but low test hit rate (< 20%)
            if forced_count >= 4 and hit_rate < 20.0:
                candidates.append((task_id, forced_count, hit_rate))
                print(f"✓ {task_id}  forced={forced_count:2d}  hit_rate={hit_rate:5.1f}%")
        except Exception as e:
            # Skip tasks that error
            continue

    print("=" * 80)
    print(f"\nFound {len(candidates)} candidate tasks with Π drift symptom")

    if candidates:
        # Sort by lowest hit rate (worst drift)
        candidates.sort(key=lambda x: x[2])

        print("\nTop 5 candidates (lowest hit rate):")
        for task_id, forced, hit_rate in candidates[:5]:
            print(f"  {task_id}  forced={forced:2d}  hit_rate={hit_rate:5.1f}%")


if __name__ == "__main__":
    main()
