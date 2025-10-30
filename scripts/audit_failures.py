#!/usr/bin/env python3
"""
scripts/audit_failures.py

Comprehensive audit to categorize failure modes and identify systemic drifts:
1. Π drift: Good Bt but low test forced-hit rate (canonization mismatch)
2. Shape mismatch: Δ fails to predict correct dimensions
3. Empty Bt: No forced keys learned
4. Content mismatch: Good Bt and hit rate, but painting is wrong
5. Transform ID variation: Different D8 between train and test
"""
import sys
import json
from pathlib import Path
import numpy as np
from collections import defaultdict

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.types import Task, Grid, ShapeLawKind
from src.solver.run_task import solve_task
from src.solver.shape_law import infer_shape_law
from src.present.pi import canonize_task
from src.qt.spec import build_qt_spec
from src.bt.boundary import extract_bt_force_until_forced, probe_writer_mode
from src.qt.quotient import classes_for
from src.kernel.grid import d8_apply


def load_tasks(num_tasks=200):
    """Load first N training tasks with ground truth."""
    challenges_path = repo_root / "data" / "arc-agi_training_challenges.json"
    solutions_path = repo_root / "data" / "arc-agi_training_solutions.json"

    with open(challenges_path) as f:
        challenges = json.load(f)

    with open(solutions_path) as f:
        solutions = json.load(f)

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

        # Parse test inputs
        test = []
        for pair in data['test']:
            x = np.array(pair['input'], dtype=np.int8)
            test.append(x)

        # Parse ground truth outputs from solutions
        ground_truth = []
        if task_id in solutions:
            for output_data in solutions[task_id]:
                y = np.array(output_data, dtype=np.int8)
                ground_truth.append(y)

        tasks[task_id] = {
            'task': Task(train=train, test=test),
            'ground_truth': ground_truth
        }

    return tasks


def audit_task(task_id, task_obj, ground_truth):
    """
    Audit a single task and return detailed metrics.

    Returns dict with:
        - status: 'pass' or 'fail'
        - delta_kind: IDENTITY, BLOW_UP, etc.
        - writer_mode: identity, blowup, tiling
        - tiling_policy: uniform, row_FH, etc. (if tiling)
        - forced_count: number of forced keys in Bt
        - test_hit_rate: percentage of test classes that hit forced_color
        - shape_match: bool, does predicted shape match expected?
        - content_match: bool, does predicted content match expected?
        - train_transforms: list of D8 transform IDs from train
        - test_transforms: list of D8 transform IDs from test
        - transform_variation: bool, do train/test have different transforms?
    """
    task = task_obj

    result = {
        'status': 'fail',
        'delta_kind': None,
        'writer_mode': None,
        'tiling_policy': None,
        'forced_count': 0,
        'test_hit_rate': 0.0,
        'shape_match': False,
        'content_match': False,
        'train_transforms': [],
        'test_transforms': [],
        'transform_variation': False,
        'error': None
    }

    try:
        # Step 1: Δ
        delta = infer_shape_law(task.train, enable_tiling=True, enable_frame=True)
        result['delta_kind'] = delta.kind.name

        # Step 2: Π task-level
        train_Xs = [x for x, _ in task.train]
        c_train, c_test, union_order = canonize_task(train_Xs, task.test)
        result['train_transforms'] = [meta.transform_id for meta in c_train.metas]
        result['test_transforms'] = [meta.transform_id for meta in c_test.metas]

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

        result['writer_mode'] = writer_mode
        result['tiling_policy'] = tiling_policy

        # Step 4: Bt
        bt, specF, extraF = extract_bt_force_until_forced(
            canon_train_pairs,
            spec0,
            delta,
            writer_mode,
            tiling_policy if tiling_policy is not None else 'uniform'
        )

        result['forced_count'] = len(bt.forced_color)

        # Step 5: Compute hit rate on test

        # Check transform variation
        train_set = set(result['train_transforms'])
        test_set = set(result['test_transforms'])
        result['transform_variation'] = (train_set != test_set)

        # Compute forced-hit rate on test
        total_classes = 0
        hit_classes = 0

        for cx in c_test.grids:
            cls = classes_for(cx, specF)
            for local_id, key in cls.key_for.items():
                total_classes += 1
                if key in bt.forced_color:
                    hit_classes += 1

        result['test_hit_rate'] = (hit_classes / total_classes * 100) if total_classes > 0 else 0.0

        # Step 6: Run solver and check output
        predictions = solve_task(task, enable_tiling=True, enable_frame=True)

        # Compare with ground truth
        if len(predictions) != len(ground_truth):
            result['shape_match'] = False
            result['content_match'] = False
        else:
            shape_matches = []
            content_matches = []

            for pred, true in zip(predictions, ground_truth):
                shape_match = pred.shape == true.shape
                content_match = shape_match and np.array_equal(pred, true)

                shape_matches.append(shape_match)
                content_matches.append(content_match)

            result['shape_match'] = all(shape_matches)
            result['content_match'] = all(content_matches)

            if result['content_match']:
                result['status'] = 'pass'

    except Exception as e:
        result['error'] = str(e)

    return result


def main():
    print("=" * 80)
    print("COMPREHENSIVE FAILURE AUDIT")
    print("=" * 80)
    print("\nLoading tasks...")

    tasks = load_tasks(num_tasks=200)
    print(f"Loaded {len(tasks)} tasks")

    # Categories
    categories = {
        'pass': [],
        'pi_drift': [],  # Good Bt (>=4 forced), low hit rate (<20%)
        'shape_mismatch': [],  # Shape wrong
        'empty_bt': [],  # No forced keys (0-3)
        'content_mismatch_good_hit': [],  # Good Bt, good hit rate (>=50%), but wrong content
        'content_mismatch_medium_hit': [],  # Good Bt, medium hit rate (20-50%), wrong content
        'transform_variation': [],  # Different D8 transforms between train/test
        'other': []
    }

    print("\nAuditing tasks...\n")

    for i, (task_id, data) in enumerate(tasks.items()):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(tasks)}")

        result = audit_task(task_id, data['task'], data['ground_truth'])

        # Categorize
        if result['status'] == 'pass':
            categories['pass'].append((task_id, result))
        elif result['error']:
            categories['other'].append((task_id, result))
        else:
            # Categorize failures
            forced = result['forced_count']
            hit_rate = result['test_hit_rate']
            shape_ok = result['shape_match']
            content_ok = result['content_match']

            # Check multiple categories (task can be in multiple)
            if forced >= 4 and hit_rate < 20.0:
                categories['pi_drift'].append((task_id, result))

            if not shape_ok:
                categories['shape_mismatch'].append((task_id, result))

            if forced < 4:
                categories['empty_bt'].append((task_id, result))

            if forced >= 4 and hit_rate >= 50.0 and not content_ok:
                categories['content_mismatch_good_hit'].append((task_id, result))
            elif forced >= 4 and 20.0 <= hit_rate < 50.0 and not content_ok:
                categories['content_mismatch_medium_hit'].append((task_id, result))

            if result['transform_variation']:
                categories['transform_variation'].append((task_id, result))

            # Catch-all
            if not any([
                forced >= 4 and hit_rate < 20.0,
                not shape_ok,
                forced < 4,
                result['transform_variation']
            ]):
                categories['other'].append((task_id, result))

    print("\n" + "=" * 80)
    print("AUDIT RESULTS")
    print("=" * 80)

    total = len(tasks)

    print(f"\n✓ PASSING: {len(categories['pass'])}/{total} ({len(categories['pass'])/total*100:.1f}%)")

    print(f"\n✗ FAILURE CATEGORIES:")
    print(f"  1. Π DRIFT (good Bt, low test hit):      {len(categories['pi_drift'])} tasks")
    print(f"  2. SHAPE MISMATCH (Δ wrong):              {len(categories['shape_mismatch'])} tasks")
    print(f"  3. EMPTY Bt (no forced keys):             {len(categories['empty_bt'])} tasks")
    print(f"  4. CONTENT MISMATCH (good hit rate):      {len(categories['content_mismatch_good_hit'])} tasks")
    print(f"  5. CONTENT MISMATCH (medium hit rate):    {len(categories['content_mismatch_medium_hit'])} tasks")
    print(f"  6. TRANSFORM VARIATION (train≠test D8):   {len(categories['transform_variation'])} tasks")
    print(f"  7. OTHER:                                 {len(categories['other'])} tasks")

    # Find biggest category
    failure_cats = [
        ('Π DRIFT', categories['pi_drift']),
        ('SHAPE MISMATCH', categories['shape_mismatch']),
        ('EMPTY Bt', categories['empty_bt']),
        ('CONTENT MISMATCH (good hit)', categories['content_mismatch_good_hit']),
        ('CONTENT MISMATCH (medium hit)', categories['content_mismatch_medium_hit']),
        ('TRANSFORM VARIATION', categories['transform_variation'])
    ]

    failure_cats.sort(key=lambda x: len(x[1]), reverse=True)

    print(f"\n" + "=" * 80)
    print("HIGHEST LEVERAGE FIX")
    print("=" * 80)

    if failure_cats[0][1]:
        biggest_cat = failure_cats[0][0]
        biggest_count = len(failure_cats[0][1])
        print(f"\n🎯 TOP PRIORITY: {biggest_cat}")
        print(f"   Affects {biggest_count} tasks ({biggest_count/total*100:.1f}% of all tasks)")

        # Show examples
        print(f"\n   Example tasks:")
        for task_id, result in failure_cats[0][1][:5]:
            print(f"     {task_id}  forced={result['forced_count']:3d}  hit_rate={result['test_hit_rate']:5.1f}%  " +
                  f"shapes_ok={result['shape_match']}  transforms={result['train_transforms'][:2]}→{result['test_transforms'][:1]}")

    # Π drift deep analysis
    if categories['pi_drift']:
        print(f"\n" + "=" * 80)
        print("Π DRIFT DEEP ANALYSIS")
        print("=" * 80)

        pi_with_transform_var = [
            (tid, r) for tid, r in categories['pi_drift']
            if r['transform_variation']
        ]

        print(f"\nΠ drift tasks with transform variation: {len(pi_with_transform_var)}/{len(categories['pi_drift'])}")

        if pi_with_transform_var:
            print(f"\n🔥 SMOKING GUN: {len(pi_with_transform_var)} tasks have BOTH:")
            print(f"   - Good Bt learning but low test hit rate")
            print(f"   - Different D8 transforms between train and test")
            print(f"\n   This confirms GPT's diagnosis:")
            print(f"   Train and test are canonized separately with different")
            print(f"   union frequency orders, causing D8 tie-breaks to diverge.")
            print(f"\n   FIX: Implement task-level Π with single union order.")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
