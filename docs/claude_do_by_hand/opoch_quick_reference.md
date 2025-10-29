# Opoch Quick Reference Card

## The 5-Step Method (Do This Every Time)

```
1. NORMALIZE    → Present all inputs (palette, D4, translation)
2. QUOTIENT     → Partition cells by input-only equivalence
3. BOUNDARY     → Extract forced colors from training
4. COLORIZER    → Assign least admissible color per class
5. ONE-STROKE   → Paint test input (one pass, done)
```

---

## Step 2 Deep Dive: Computing the Quotient

**Ask: "Which cells are equivalent using ONLY input features?"**

### Seed Features:
- Color at cell
- 3×3 neighborhood pattern
- Position (row, col)

### Refinement Features:
- Union-neighborhood hash (iterate until stable)
- Row/col residue: `i % k`, `j % k` for k ∈ {2,3,4,5}
- Diagonal residue: `(i+j) % k`, `(i-j) % k`
- Block position: `i // k`, `j // k`
- Connected component ID (per color)
- Distance to borders/holes/special cells
- Pane ID (if repeated structures exist)

### Stabilization:
Run refinement until no cell's signature changes. That's your quotient.

---

## Common Patterns Cheat Sheet

| Pattern | Quotient Structure | Formula |
|---------|-------------------|---------|
| **Tiling** | `(i % k, j % k)` | Tile input with period k |
| **Diagonal stripes** | `(i + j) % k` | Anti-diagonal coloring |
| **Blow-up k×** | `(i // k, j // k)` | Each cell → k×k block |
| **Rotation/reflection** | Pane symmetries | Apply D4 transforms |
| **Local substitution** | 3×3 pattern hash | Pattern → output mapping |
| **Component coloring** | Component ID | Color by component property |

---

## The 3 Solved Examples (Memorize These)

### 1. Tiled Motif (00576224)
- **Input**: 2×2 grid
- **Output**: 6×6 (tile 3× with alternating column swap)
- **Quotient**: `(i//2, i%2, j%2)` + parity
- **Rule**: Even tile-rows normal, odd tile-rows swap columns

### 2. Blow-Up (007bbfb7)
- **Input**: 3×3 grid
- **Output**: 9×9 (each cell → 3×3 block)
- **Quotient**: `(i//3, j//3, i%3, j%3)`
- **Rule**: If input[i//3][j//3] ≠ 0, copy input; else all zeros

### 3. Diagonal Stripes (05269061)
- **Input**: 7×7 with diagonal of colors {1,2,4}
- **Output**: 7×7 diagonal striped pattern
- **Quotient**: `(i + j) % 3`
- **Rule**: `output[i][j] = colors[(i+j) % len(colors)]`

---

## Recognition Speed Patterns

**See this** → **Think this**:

- Input size n×n, output size kn×kn → **Blow-up by k**
- Diagonal line in input → **Diagonal tiling**
- Repeated blocks → **Pane symmetries**
- Small motif → **Tiling/replication**
- Same size I/O, local changes → **Neighborhood rules**
- Border of 1s → **Sectioning/dividing**
- Scattered objects → **Component analysis**
- Shape + color pairing → **Object classification**

---

## The Mental Checklist

When looking at a new problem:

```
□ Normalized inputs?
□ Identified size relationship (I/O)?
□ Found repeated structures?
□ Checked for diagonals/stripes?
□ Looked at color distribution?
□ Noted what changed vs what stayed same?
□ Drew equivalence classes?
□ Verified against ALL training examples?
□ Computed one test cell by hand to verify?
```

---

## Debug When Stuck

1. **Print side-by-side**: Input | Output for each training pair
2. **Count colors**: Do color frequencies change?
3. **Overlay grids**: XOR input and output - what changed?
4. **Check corners**: Often reveal tiling/boundary structure
5. **Look at smallest training example first**
6. **Try the formula on training**: Does it reproduce outputs?

---

## The Formula

For every task t:

```
Yt,∗ = U⁻¹ₜ((Φₜ ∘ qₜ)(ΠGₜ(Xt,∗)))
```

Where:
- `ΠGₜ` = Present (normalize)
- `qₜ` = Quotient (partition)
- `Φₜ` = Colorizer (assign colors)
- `U⁻¹ₜ` = Un-present (denormalize)

---

## Code Structure Template

```python
def solve(task):
    # 1. Normalize
    train_inputs = [normalize(p['input']) for p in task['train']]
    test_input = normalize(task['test'][0]['input'])

    # 2. Compute quotient
    quotient_fn = compute_quotient(train_inputs)

    # 3. Extract boundary
    boundary = {}
    for pair in task['train']:
        inp = normalize(pair['input'])
        out = normalize(pair['output'])
        for i, j in cells(out):
            q_class = quotient_fn(i, j)
            boundary[q_class].add(out[i][j])

    # 4. Build colorizer
    colorizer = {}
    for q_class in quotient_fn.classes:
        if len(boundary[q_class]) == 1:
            colorizer[q_class] = boundary[q_class].pop()
        else:
            colorizer[q_class] = min_admissible(q_class)

    # 5. Paint test
    output = empty_grid(test_input.shape)
    for i, j in cells(test_input):
        q_class = quotient_fn(i, j)
        output[i][j] = colorizer[q_class]

    return denormalize(output)
```

---

## Key Equations

**Quotient construction**:
```
qt(x) = stabilize(refine(seed(x)))
```

**Boundary extraction**:
```
Bt(q) = {c ∈ C : ∀i, ∀x ∈ q ∩ dom(Yt,i), Yt,i(x) = c}
```

**Canonical colorizer**:
```
Φt(q) = {
    c,                      if Bt(q) = {c}
    min{admissible colors}, if Bt(q) = ∅
}
```

**One-stroke solution**:
```
Yt,∗(x) = Φt(qt(x))  ∀x ∈ Ωt
```

---

## The Three Axioms (Never Forget)

**A0**: No minted differences (only use input-present distinctions)
**A1**: Exactness (reproduce training outputs exactly)
**A2**: Composition by equality (enforce constraints to fixed point)

---

## Mantras

- **"Can I compute this by hand?"** → If yes, you understand it
- **"What makes these cells equivalent?"** → That's your quotient
- **"What must be true?"** → Those are constraints
- **"Does this work on ALL training examples?"** → Test it
- **"Is this deterministic?"** → It must be

---

## Success Criteria

You've solved it when:
- ✓ Formula works on all training pairs (exact match)
- ✓ Formula is deterministic (no randomness)
- ✓ Formula uses only input features (no output peeking)
- ✓ You can explain it in one sentence
- ✓ You can compute test output by hand in < 5 minutes

---

*Read `solving_arc_by_hand.md` for detailed examples*
