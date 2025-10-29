# src/io/arc_io.py
import json
from typing import Dict, List
import numpy as np

from ..types import Task, Grid

# ---------- helpers ----------
def _to_grid(a) -> Grid:
    """
    Convert a nested list to a contiguous np.int8 array (h, w).
    Enforce 2D, finite bounds, and palette 0..9.
    """
    g = np.asarray(a, dtype=np.int8)
    if g.ndim != 2:
        raise ValueError(f"grid must be 2D, got ndim={g.ndim}")
    h, w = g.shape
    if not (1 <= h <= 30 and 1 <= w <= 30):
        raise ValueError(f"grid shape out of bounds: {(h,w)}")
    mn = int(g.min()) if g.size else 0
    mx = int(g.max()) if g.size else 0
    if mn < 0 or mx > 9:
        raise ValueError(f"grid palette outside 0..9: min={mn} max={mx}")
    # Return C-contiguous int8
    return np.ascontiguousarray(g, dtype=np.int8)

# ---------- public API ----------
def load_tasks(path_json: str) -> Dict[str, Task]:
    """
    Load ARC-format JSON into a dict of {task_id: Task}.
    JSON shape:
      {
        "<id>": {
          "train": [{"input":[[...]], "output":[[...]]}, ...],
          "test":  [{"input":[[...]]}, ...]
        },
        ...
      }

    Determinism: ids are returned in sorted key order when iterating via sorted(d.keys()).
    """
    with open(path_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks: Dict[str, Task] = {}
    # We do not rely on dict order here; callers should sort(task_ids) explicitly.
    for tid, obj in data.items():
        train_pairs = []
        for pair in obj.get("train", []):
            xi = _to_grid(pair["input"])
            yi = _to_grid(pair["output"])
            train_pairs.append((xi, yi))
        test_inputs = []
        for pair in obj.get("test", []):
            xt = _to_grid(pair["input"])
            test_inputs.append(xt)
        tasks[tid] = Task(train=train_pairs, test=test_inputs)
    return tasks

def write_submission_csv(path_csv: str,
                         preds: Dict[str, List[Grid]],
                         pipe_terminated: bool = True) -> None:
    """
    Write Kaggle submission CSV:
      header: 'output_id,output'
      rows:   '<taskid>_<index>,<space-separated values>[|]'
    - Values are row-major, int cast, separated by single spaces.
    - If pipe_terminated=True, append a trailing '|' at end of each row.
      (Leave True by default; flip if the host expects no trailing pipe.)
    - Rows written in deterministic order of sorted task ids and index ascending.
    """
    suffix = "|" if pipe_terminated else ""
    with open(path_csv, "w", encoding="utf-8", newline="") as f:
        f.write("output_id,output\n")
        for tid in sorted(preds.keys()):
            outs = preds[tid]
            for j, g in enumerate(outs):
                if not isinstance(g, np.ndarray) or g.dtype != np.int8 or g.ndim != 2:
                    raise TypeError(f"pred grid must be np.int8 2D array, got {type(g)}, dtype={getattr(g,'dtype',None)}, ndim={getattr(g,'ndim',None)}")
                vals = " ".join(str(int(v)) for v in g.ravel(order="C"))
                f.write(f"{tid}_{j},{vals}{suffix}\n")

def write_submission_json(path_json: str,
                          preds: Dict[str, List[Grid]]) -> None:
    """
    Optional helper for local inspection:
      {
        "<id>": [
          {"attempt_1": [[...]], "attempt_2": [[...]]},
          ...
        ]
      }
    Writes two identical attempts per test as plain nested lists of ints.
    Deterministic ordering by sorted task ids then index.
    """
    def _to_pylist(g: Grid) -> List[List[int]]:
        return [[int(x) for x in row] for row in g.tolist()]

    out = {}
    for tid in sorted(preds.keys()):
        rendered = []
        for g in preds[tid]:
            rendered.append({
                "attempt_1": _to_pylist(g),
                "attempt_2": _to_pylist(g),
            })
        out[tid] = rendered
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
