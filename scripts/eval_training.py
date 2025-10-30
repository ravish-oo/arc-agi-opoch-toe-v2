#!/usr/bin/env python3
"""
scripts/eval_training.py

Deterministic training-set evaluator for the Opoch ARC-AGI solver.
Runs the full pipeline on 10 real ARC-AGI training tasks and reports results.

Usage:
    python scripts/eval_training.py [--data-dir PATH] [--num-tasks N]

Defaults:
    --data-dir: data/training (relative to repo root)
    --num-tasks: 10
"""
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Add repo root to path for imports
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.types import Task, Grid
from src.solver.run_task import solve_task


def load_tasks_from_consolidated_json(
    challenges_path: Path,
    solutions_path: Path
) -> Dict[str, Tuple[Task, List[Grid]]]:
    """Load all tasks from consolidated JSON files."""
    with open(challenges_path, 'r') as f:
        challenges = json.load(f)

    with open(solutions_path, 'r') as f:
        solutions = json.load(f)

    tasks = {}
    for task_id, challenge_data in challenges.items():
        # Parse train pairs
        train = []
        for pair in challenge_data['train']:
            x = np.array(pair['input'], dtype=np.int8)
            y = np.array(pair['output'], dtype=np.int8)
            train.append((x, y))

        # Parse test inputs
        test = []
        for pair in challenge_data['test']:
            x = np.array(pair['input'], dtype=np.int8)
            test.append(x)

        task = Task(train=train, test=test)

        # Parse ground truth from solutions
        ground_truth = []
        if task_id in solutions:
            for output in solutions[task_id]:
                y = np.array(output, dtype=np.int8)
                ground_truth.append(y)

        tasks[task_id] = (task, ground_truth)

    return tasks


def grids_equal(pred: Grid, true: Grid) -> bool:
    """Check if two grids are exactly equal."""
    if pred.shape != true.shape:
        return False
    return np.array_equal(pred, true)


def evaluate_task(
    task_id: str,
    task: Task,
    ground_truth: List[Grid],
    *,
    enable_frame: bool = False,
    enable_tiling: bool = False,
    periodicity_check: bool = False
) -> Tuple[bool, str]:
    """
    Evaluate solver on a single task.

    Returns:
        (success: bool, message: str)
    """
    try:
        # Run solver
        predictions = solve_task(
            task,
            enable_frame=enable_frame,
            enable_tiling=enable_tiling,
            periodicity_check=periodicity_check
        )

        # Check number of outputs
        if len(predictions) != len(ground_truth):
            return False, f"Output count mismatch: got {len(predictions)}, expected {len(ground_truth)}"

        # Check each output
        for i, (pred, true) in enumerate(zip(predictions, ground_truth)):
            if not grids_equal(pred, true):
                return False, f"Test output {i} incorrect: shape {pred.shape} vs {true.shape}"

        return True, "PASS"

    except Exception as e:
        return False, f"ERROR: {type(e).__name__}: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="Evaluate Opoch solver on ARC training tasks")
    parser.add_argument(
        "--challenges",
        type=Path,
        default=repo_root / "data" / "arc-agi_training_challenges.json",
        help="Path to ARC training challenges JSON file"
    )
    parser.add_argument(
        "--solutions",
        type=Path,
        default=repo_root / "data" / "arc-agi_training_solutions.json",
        help="Path to ARC training solutions JSON file"
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=10,
        help="Number of tasks to evaluate"
    )
    parser.add_argument(
        "--enable-frame",
        action="store_true",
        help="Enable FRAME shape law detection"
    )
    parser.add_argument(
        "--enable-tiling",
        action="store_true",
        help="Enable TILING shape law detection"
    )
    parser.add_argument(
        "--periodicity-check",
        action="store_true",
        help="Enable periodicity check for TILING"
    )

    args = parser.parse_args()

    # Load all tasks
    if not args.challenges.exists():
        print(f"ERROR: Challenges file not found: {args.challenges}")
        sys.exit(1)

    if not args.solutions.exists():
        print(f"ERROR: Solutions file not found: {args.solutions}")
        sys.exit(1)

    print(f"Loading tasks from {args.challenges.name}...")
    all_tasks = load_tasks_from_consolidated_json(args.challenges, args.solutions)

    # Take first N tasks (deterministic - sorted by task ID)
    task_ids = sorted(all_tasks.keys())[:args.num_tasks]

    print(f"Evaluating Opoch solver on {len(task_ids)} training tasks")
    print(f"Flags: frame={args.enable_frame}, tiling={args.enable_tiling}, periodicity={args.periodicity_check}")
    print("=" * 80)

    # Run evaluation
    results = []
    for task_id in task_ids:
        task, ground_truth = all_tasks[task_id]
        success, message = evaluate_task(
            task_id,
            task,
            ground_truth,
            enable_frame=args.enable_frame,
            enable_tiling=args.enable_tiling,
            periodicity_check=args.periodicity_check
        )
        results.append((task_id, success, message))

        # Print result
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}  {task_id}  {message}")

    # Summary
    print("=" * 80)
    num_pass = sum(1 for _, success, _ in results if success)
    num_fail = len(results) - num_pass
    print(f"SUMMARY: {num_pass}/{len(results)} passed, {num_fail}/{len(results)} failed")
    print(f"Success rate: {100 * num_pass / len(results):.1f}%")

    # Exit code
    sys.exit(0 if num_fail == 0 else 1)


if __name__ == "__main__":
    main()
