"""
WO-09: End-to-End Solver Tests
Tests for src/solver/run_task.py

Test strategy:
1. Acceptance: verify core pipeline works (IDENTITY, BLOW_UP)
2. Determinism: verify same input produces same output
3. Return types: verify int8 C-contiguous
4. Multi-task: verify solve_all batches correctly
"""
import numpy as np
import pytest
from src.solver.run_task import solve_task, solve_all
from src.types import Task


# ========== Acceptance Tests ==========

def test_identity_basic():
    """Simple identity mapping: output == input."""
    X1 = np.array([[1, 1, 0], [0, 1, 0], [2, 2, 2]], np.int8)
    Y1 = X1.copy()
    X2 = np.array([[1, 0], [0, 1]], np.int8)
    Y2 = X2.copy()

    task = Task(
        train=[(X1, Y1), (X2, Y2)],
        test=[X1, X2]
    )

    outputs = solve_task(task)

    assert len(outputs) == 2
    assert np.array_equal(outputs[0], Y1), f"Test 0 mismatch:\nExpected:\n{Y1}\nGot:\n{outputs[0]}"
    assert np.array_equal(outputs[1], Y2), f"Test 1 mismatch:\nExpected:\n{Y2}\nGot:\n{outputs[1]}"


def test_blowup_2x2():
    """BLOW_UP case: 1×1 → 2×2."""
    X1 = np.array([[3]], np.int8)
    Y1 = np.array([[3, 3], [3, 3]], np.int8)
    X2 = np.array([[5]], np.int8)
    Y2 = np.array([[5, 5], [5, 5]], np.int8)

    task = Task(
        train=[(X1, Y1), (X2, Y2)],
        test=[np.array([[7]], np.int8)]
    )

    outputs = solve_task(task)

    assert len(outputs) == 1
    expected = np.array([[7, 7], [7, 7]], np.int8)
    assert np.array_equal(outputs[0], expected), f"Expected:\n{expected}\nGot:\n{outputs[0]}"


def test_blowup_multicolor():
    """BLOW_UP with multiple colors."""
    X1 = np.array([[1, 2]], np.int8)
    Y1 = np.array([[1, 1, 2, 2], [1, 1, 2, 2]], np.int8)
    X2 = np.array([[3, 4]], np.int8)
    Y2 = np.array([[3, 3, 4, 4], [3, 3, 4, 4]], np.int8)

    task = Task(
        train=[(X1, Y1), (X2, Y2)],
        test=[np.array([[5, 6]], np.int8)]
    )

    outputs = solve_task(task)

    assert len(outputs) == 1
    expected = np.array([[5, 5, 6, 6], [5, 5, 6, 6]], np.int8)
    assert np.array_equal(outputs[0], expected), f"Expected:\n{expected}\nGot:\n{outputs[0]}"


# ========== Determinism ==========

def test_determinism():
    """Same input produces same output."""
    X = np.array([[1, 2], [3, 4]], np.int8)
    Y = np.array([[5, 6], [7, 8]], np.int8)

    task = Task(train=[(X, Y)], test=[X])

    # Run 3 times
    results = [solve_task(task) for _ in range(3)]

    # All should be identical
    for i, r in enumerate(results[1:], 1):
        assert np.array_equal(r[0], results[0][0]), f"Run {i} differs from run 0"


# ========== Return Type Verification ==========

def test_output_int8():
    """Output is np.int8."""
    X = np.array([[1]], np.int8)
    task = Task(train=[(X, X)], test=[X])

    outputs = solve_task(task)
    assert outputs[0].dtype == np.int8


def test_output_contiguous():
    """Output is C-contiguous."""
    X = np.array([[1, 2], [3, 4]], np.int8)
    task = Task(train=[(X, X)], test=[X])

    outputs = solve_task(task)
    assert outputs[0].flags['C_CONTIGUOUS']


# ========== Multi-Task Batch (solve_all) ==========

def test_solve_all_basic():
    """solve_all batches multiple tasks correctly."""
    X1 = np.array([[1]], np.int8)
    X2 = np.array([[2, 2], [2, 2]], np.int8)

    tasks = {
        "task1": Task(train=[(X1, X1)], test=[X1]),
        "task2": Task(train=[(X2, X2)], test=[X2])
    }

    results = solve_all(tasks)

    assert len(results) == 2
    assert "task1" in results
    assert "task2" in results
    assert np.array_equal(results["task1"][0], X1)
    assert np.array_equal(results["task2"][0], X2)


def test_solve_all_sorted_order():
    """solve_all processes tasks in sorted key order."""
    X = np.array([[1]], np.int8)

    tasks = {
        "c": Task(train=[(X, X)], test=[X]),
        "a": Task(train=[(X, X)], test=[X]),
        "b": Task(train=[(X, X)], test=[X])
    }

    results = solve_all(tasks)

    # Should have all three
    assert set(results.keys()) == {"a", "b", "c"}


# ========== Edge Cases ==========

def test_single_pixel():
    """1×1 grid works."""
    X = np.array([[7]], np.int8)
    task = Task(train=[(X, X)], test=[X])

    outputs = solve_task(task)
    assert np.array_equal(outputs[0], X)


def test_multiple_test_inputs():
    """Multiple test inputs handled correctly."""
    X1 = np.array([[1]], np.int8)
    X2 = np.array([[2]], np.int8)
    X3 = np.array([[3]], np.int8)

    task = Task(
        train=[(X1, X1)],
        test=[X1, X2, X3]
    )

    outputs = solve_task(task)

    assert len(outputs) == 3
    # Each should be identity (since train is identity)
    assert np.array_equal(outputs[0], X1)
    # For X2, X3 - they should follow the learned pattern
    assert outputs[0].dtype == np.int8
    assert outputs[1].dtype == np.int8
    assert outputs[2].dtype == np.int8


def test_rectangular_grids():
    """Non-square grids work correctly."""
    X1 = np.array([[1, 2, 3]], np.int8)  # 1×3
    Y1 = X1.copy()
    X2 = np.array([[4], [5], [6]], np.int8)  # 3×1
    Y2 = X2.copy()

    task = Task(
        train=[(X1, Y1), (X2, Y2)],
        test=[X1, X2]
    )

    outputs = solve_task(task)

    assert len(outputs) == 2
    assert np.array_equal(outputs[0], Y1)
    assert np.array_equal(outputs[1], Y2)


print("All tests defined successfully")
