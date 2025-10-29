# Solving ARC-AGI By Hand: The Opoch Method

## Introduction: You Can Do This By Hand

This document is your complete guide to solving ARC-AGI problems by hand using mathematical first principles. When you read this, the method should **immediately transpire** in your mind. You don't need search, you don't need trial-and-error - just systematic reasoning.

## The Core Insight

Every ARC-AGI transformation can be solved as a **one-stroke painting operation** where:
1. You partition the space into equivalence classes (quotient)
2. You determine what color each class must be (boundary + admissibility)
3. You paint all cells of each class with that color in one pass

This is deterministic. No guessing. Just math.

---

## The Three Axioms (A0-A2)

**A0 (No minted differences)**: Only use distinctions present in the inputs. Normalize once (palette, D4 symmetry, translation), then work in that frame.

**A1 (Exactness; no remainder)**: Your solution must reproduce training outputs exactly. All constraints are monotone - they only eliminate possibilities, never add spurious content.

**A2 (Composition by equality)**: When structures overlap or interface, enforce equality constraints until you reach a unique least fixed point.

---

## The Step-by-Step Method

### Step 1: Normalize (Present)
- Load all training inputs and test input
- Apply palette canonicalization (based on input color frequencies)
- Apply D4 pose selection if needed (pick lexicographically smallest)
- Translate to origin
- **Key**: Only look at INPUTS for normalization, not outputs

### Step 2: Compute Input-Invariance Quotient (Qt)

This is the heart of the method. Ask: **"Which cells are indistinguishable using only input-based features?"**

Practical construction:
1. **Start with local patches**: Seed each cell by its (color, 3×3 neighborhood)
2. **Refine iteratively**: Run union-neighborhood refinement (1-WL style) until stable
3. **Add structure-based features**:
   - Pane symmetries (if input has repeated blocks)
   - Row/column residues (i mod k, j mod k for small k)
   - Connected components per color
   - Holes, outlines, borders
4. **Boolean closure**: Take intersections/unions (depth ≤ 2) to get disjoint atoms
5. **Stop at stability**: When no more refinement happens

**Critical insight**: The quotient classes are your "atomic units". Each class will be painted with a single color.

### Step 3: Extract Training Boundary (Bt)

For each equivalence class q:
- Look at all training outputs
- If every cell in class q (across all training examples) is painted color c, then Bt(q) = {c}
- If class q never appears in training outputs, then Bt(q) = ∅ (unconstrained)
- If class q has conflicting colors in training, flag inconsistency

**Forced classes**: Bt(q) = {c} means class q MUST be color c
**Free classes**: Bt(q) = ∅ means class q can be any admissible color

### Step 4: Construct Canonical Colorizer (Φt)

For each class q, define Φ(q):
```
If Bt(q) = {c}:
    Φ(q) = c                    // forced by training
Else:
    Φ(q) = min admissible color // least color that preserves all equalities
```

Admissibility check: Color c is admissible for class q if painting all cells in q with color c preserves all training equalities under input-preserving mappings.

### Step 5: One-Stroke Solution

For the test input:
```
For each cell (i, j):
    class = qt(i, j)           // which equivalence class?
    output[i, j] = Φ(class)    // paint with that class's color
```

Done. In one pass.

---

## Problem Type 1: Tiled Motif Replication

**Task ID**: 00576224
**Pattern**: Tile a small input multiple times with transformations

### Training Example 1:
```
Input (2×2):
[7, 9]
[4, 3]

Output (6×6):
[7, 9, 7, 9, 7, 9]    ← rows 0-1: normal tile, repeated 3×
[4, 3, 4, 3, 4, 3]
[9, 7, 9, 7, 9, 7]    ← rows 2-3: SWAPPED tile (columns flipped), repeated 3×
[3, 4, 3, 4, 3, 4]
[7, 9, 7, 9, 7, 9]    ← rows 4-5: normal tile, repeated 3×
[4, 3, 4, 3, 4, 3]
```

### Training Example 2:
```
Input (2×2):
[8, 6]
[6, 4]

Output (6×6):
[8, 6, 8, 6, 8, 6]
[6, 4, 6, 4, 6, 4]
[6, 8, 6, 8, 6, 8]    ← swapped!
[4, 6, 4, 6, 4, 6]
[8, 6, 8, 6, 8, 6]
[6, 4, 6, 4, 6, 4]
```

### By-Hand Analysis:

**Quotient Qt**:
Output cells (i, j) are equivalent based on:
- row_tile = i // 2 (which 2-row tile block: 0, 1, or 2)
- row_in_tile = i % 2 (which row within the tile: 0 or 1)
- col_in_pattern = j % 2 (which column in the pattern: 0 or 1)

**The pattern**:
- If row_tile is even (0 or 2): use input as-is
- If row_tile is odd (1): swap columns

**Formula**:
```python
row_in_tile = i % 2
col_in_pattern = j % 2
row_tile = i // 2

if row_tile % 2 == 1:  # odd tile - swap columns
    input_col = 1 - col_in_pattern
else:                   # even tile - normal
    input_col = col_in_pattern

output[i][j] = input[row_in_tile][input_col]
```

### Test Input:
```
[3, 2]
[7, 8]
```

### Computing By Hand:

Row 0 (tile=0, even → no swap):
- (0,0): input[0][0] = 3
- (0,1): input[0][1] = 2
- (0,2): input[0][0] = 3
- (0,3): input[0][1] = 2
- (0,4): input[0][0] = 3
- (0,5): input[0][1] = 2
→ [3, 2, 3, 2, 3, 2] ✓

Row 1 (tile=0, even → no swap):
- (1,0): input[1][0] = 7
- (1,1): input[1][1] = 8
- ...repeated...
→ [7, 8, 7, 8, 7, 8] ✓

Row 2 (tile=1, odd → SWAP):
- (2,0): input[0][1-0] = input[0][1] = 2
- (2,1): input[0][1-1] = input[0][0] = 3
- ...repeated...
→ [2, 3, 2, 3, 2, 3] ✓

Row 3 (tile=1, odd → SWAP):
→ [8, 7, 8, 7, 8, 7] ✓

Row 4 (tile=2, even → no swap):
→ [3, 2, 3, 2, 3, 2] ✓

Row 5 (tile=2, even → no swap):
→ [7, 8, 7, 8, 7, 8] ✓

**Solution**:
```
[3, 2, 3, 2, 3, 2]
[7, 8, 7, 8, 7, 8]
[2, 3, 2, 3, 2, 3]
[8, 7, 8, 7, 8, 7]
[3, 2, 3, 2, 3, 2]
[7, 8, 7, 8, 7, 8]
```

---

## Problem Type 2: Blow-Up with Patch Substitution

**Task ID**: 007bbfb7
**Pattern**: Each input cell becomes a k×k block, filled based on cell's color

### Training Example 1:
```
Input (3×3):
[6, 6, 0]
[6, 0, 0]
[0, 6, 6]

Output (9×9):
[6, 6, 0, 6, 6, 0, 0, 0, 0]    ← top-left cell was 6 → copy input
[6, 0, 0, 6, 0, 0, 0, 0, 0]    ← top-mid cell was 6 → copy input
[0, 6, 6, 0, 6, 6, 0, 0, 0]    ← top-right cell was 0 → all zeros

[6, 6, 0, 0, 0, 0, 0, 0, 0]    ← mid-left cell was 6 → copy input
[6, 0, 0, 0, 0, 0, 0, 0, 0]    ← mid-mid cell was 0 → all zeros
[0, 6, 6, 0, 0, 0, 0, 0, 0]    ← mid-right cell was 0 → all zeros

[0, 0, 0, 6, 6, 0, 6, 6, 0]    ← bottom cells...
[0, 0, 0, 6, 0, 0, 6, 0, 0]
[0, 0, 0, 0, 6, 6, 0, 6, 6]
```

### By-Hand Analysis:

**The Rule**:
- Input is m×n
- Output is 3m×3n (each cell becomes a 3×3 block)
- If input[r][c] is non-zero: replicate the entire input in that block
- If input[r][c] is zero: fill block with all zeros

**Quotient Qt**:
Output cells (i, j) map to:
- input_row = i // 3
- input_col = j // 3
- patch_row = i % 3
- patch_col = j % 3

**Formula**:
```python
input_row = i // 3
input_col = j // 3
patch_row = i % 3
patch_col = j % 3

if input[input_row][input_col] != 0:
    output[i][j] = input[patch_row][patch_col]
else:
    output[i][j] = 0
```

### Test Input:
```
[7, 0, 7]
[7, 0, 7]
[7, 7, 0]
```

### Computing By Hand:

**Block (0,0)** - input[0][0] = 7 → replicate input:
```
Rows 0-2, Cols 0-2:
[7, 0, 7]
[7, 0, 7]
[7, 7, 0]
```

**Block (0,1)** - input[0][1] = 0 → all zeros:
```
Rows 0-2, Cols 3-5:
[0, 0, 0]
[0, 0, 0]
[0, 0, 0]
```

**Block (0,2)** - input[0][2] = 7 → replicate input:
```
Rows 0-2, Cols 6-8:
[7, 0, 7]
[7, 0, 7]
[7, 7, 0]
```

Continue for all 9 blocks...

**Solution**:
```
[7, 0, 7, 0, 0, 0, 7, 0, 7]
[7, 0, 7, 0, 0, 0, 7, 0, 7]
[7, 7, 0, 0, 0, 0, 7, 7, 0]
[7, 0, 7, 0, 0, 0, 7, 0, 7]
[7, 0, 7, 0, 0, 0, 7, 0, 7]
[7, 7, 0, 0, 0, 0, 7, 7, 0]
[7, 0, 7, 7, 0, 7, 0, 0, 0]
[7, 0, 7, 7, 0, 7, 0, 0, 0]
[7, 7, 0, 7, 7, 0, 0, 0, 0]
```

---

## Problem Type 3: Diagonal Striped Tiling

**Task ID**: 05269061
**Pattern**: Extract colors from diagonal pattern, tile entire grid

### Training Example 1:
```
Input (7×7):
[0, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 0, 1]
[0, 0, 0, 0, 0, 1, 2]
[0, 0, 0, 0, 1, 2, 4]
[0, 0, 0, 1, 2, 4, 0]
[0, 0, 1, 2, 4, 0, 0]

Diagonal has colors: 1, 2, 4

Output (7×7):
[2, 4, 1, 2, 4, 1, 2]
[4, 1, 2, 4, 1, 2, 4]
[1, 2, 4, 1, 2, 4, 1]
[2, 4, 1, 2, 4, 1, 2]
[4, 1, 2, 4, 1, 2, 4]
[1, 2, 4, 1, 2, 4, 1]
[2, 4, 1, 2, 4, 1, 2]
```

### By-Hand Analysis:

**Key Observation**:
- Input has a diagonal line of non-zero colors
- Output creates diagonal stripes using those colors
- Cells along the same anti-diagonal (i+j constant) have the same color

**Quotient Qt**:
Cells are equivalent if `(i + j) % num_colors` is the same

**Extract color sequence**:
1. Find all non-zero colors in input: {1, 2, 4}
2. Determine ordering (this requires checking which permutation matches training)

**Formula**:
```python
colors = [c1, c2, c3]  # determined from training
num_colors = len(colors)
output[i][j] = colors[(i + j) % num_colors]
```

### Training Example 2:
```
Input has diagonal with colors: 2, 8, 3
Output shows pattern with color order: [2, 8, 3]
```

### Test Input:
```
[0, 1, 0, 0, 0, 0, 2]
[1, 0, 0, 0, 0, 2, 0]
[0, 0, 0, 0, 2, 0, 0]
[0, 0, 0, 2, 0, 0, 0]
[0, 0, 2, 0, 0, 0, 0]
[0, 2, 0, 0, 0, 0, 4]
[2, 0, 0, 0, 0, 4, 0]

Non-zero colors: {1, 2, 4}
```

### Determining Color Order:

From training patterns, I need to determine the order. Let me check training examples to see which order is used.

After checking: **colors = [2, 1, 4]**

### Computing By Hand:

Row 0:
- (0,0): (0+0)%3 = 0 → colors[0] = 2
- (0,1): (0+1)%3 = 1 → colors[1] = 1
- (0,2): (0+2)%3 = 2 → colors[2] = 4
- (0,3): (0+3)%3 = 0 → colors[0] = 2
- (0,4): (0+4)%3 = 1 → colors[1] = 1
- (0,5): (0+5)%3 = 2 → colors[2] = 4
- (0,6): (0+6)%3 = 0 → colors[0] = 2
→ [2, 1, 4, 2, 1, 4, 2] ✓

Row 1:
- (1,0): (1+0)%3 = 1 → colors[1] = 1
- (1,1): (1+1)%3 = 2 → colors[2] = 4
- (1,2): (1+2)%3 = 0 → colors[0] = 2
- ...
→ [1, 4, 2, 1, 4, 2, 1] ✓

Row 2:
→ [4, 2, 1, 4, 2, 1, 4] ✓

Row 3 (same as row 0, since 3%3 = 0):
→ [2, 1, 4, 2, 1, 4, 2] ✓

Row 4 (same as row 1):
→ [1, 4, 2, 1, 4, 2, 1] ✓

Row 5 (same as row 2):
→ [4, 2, 1, 4, 2, 1, 4] ✓

Row 6 (same as row 0):
→ [2, 1, 4, 2, 1, 4, 2] ✓

**Solution**:
```
[2, 1, 4, 2, 1, 4, 2]
[1, 4, 2, 1, 4, 2, 1]
[4, 2, 1, 4, 2, 1, 4]
[2, 1, 4, 2, 1, 4, 2]
[1, 4, 2, 1, 4, 2, 1]
[4, 2, 1, 4, 2, 1, 4]
[2, 1, 4, 2, 1, 4, 2]
```

---

## Key Insights for Recognition

### Pattern Recognition Tips:

1. **Size Change → Likely blow-up or extraction**
   - If output is k× larger: look for replication/tiling
   - If output is smaller: look for extraction/compression

2. **Diagonal lines in input → Likely diagonal tiling**
   - Extract colors from diagonal
   - Apply modulo-based tiling

3. **Repeated structures → Pane symmetries**
   - Identify repeated blocks
   - Apply transformations between panes

4. **Same size, scattered changes → Local rules**
   - Look at neighborhoods
   - Apply connected component analysis

5. **Color 1 as border/divider → Sectioning**
   - Use 1s as boundaries
   - Process each section independently

### Common Quotient Structures:

1. **Modulo arithmetic**: (i % k, j % k)
2. **Anti-diagonals**: (i + j) % k
3. **Diagonals**: (i - j) % k
4. **Blocks**: (i // k, j // k)
5. **Neighborhood signature**: hash(local_patch)
6. **Connected components**: component_id per color
7. **Distance from feature**: distance_to(special_cells)

### When Stuck:

1. **Print both input and output side by side**
2. **Look for what CHANGED between input and output**
3. **Look for what STAYED THE SAME**
4. **Count cells by color in input vs output**
5. **Look for symmetries, repetitions, patterns**
6. **Check if output size relates to input size mathematically**
7. **Try drawing equivalence classes visually**

---

## The Mental Model

Think of solving an ARC task like this:

1. **You're a detective**: The training pairs are clues. What rule explains ALL of them?

2. **You're partitioning space**: Draw lines around cells that "should be the same". That's your quotient.

3. **You're filling in a coloring book**: Each region (equivalence class) gets exactly one color. Training tells you which.

4. **You're applying a stamp**: Once you know the pattern, apply it in one stroke to the test input.

5. **You're following physics**: The solution is the unique "ground state" that satisfies all constraints.

---

## Implementation Strategy

Once you can do it by hand, coding is straightforward:

```python
def solve_arc_task(task):
    # Step 1: Normalize
    inputs = normalize_inputs(task['train'])

    # Step 2: Compute quotient
    quotient = compute_quotient(inputs)

    # Step 3: Extract boundary
    boundary = extract_boundary(task['train'], quotient)

    # Step 4: Build colorizer
    colorizer = build_colorizer(boundary, quotient)

    # Step 5: Apply to test
    test_input = normalize(task['test'][0]['input'])
    output = apply_colorizer(test_input, quotient, colorizer)

    return output
```

Each step is deterministic. No search. No ML. Just pure reasoning.

---

## Remember This

When you read an ARC problem:

1. **Don't panic** - there IS a pattern
2. **Look at ALL training pairs** - one example might be misleading
3. **Think quotients** - what makes cells equivalent?
4. **Think constraints** - what must be true?
5. **Think determinism** - there's ONE right answer
6. **Compute by hand FIRST** - verify your understanding
7. **Then code** - translate your hand computation

The math works. Trust the process.

---

## Final Thoughts

You are not guessing. You are not searching. You are **discovering the unique mathematical structure** that explains the training data and generalizes to the test case.

The Opoch framework gives you the vocabulary and tools to do this systematically.

When you see a new ARC problem:
- See it as a quotient problem
- See it as a boundary problem
- See it as a colorizer problem
- See it as **solvable by hand**

Because it is.

---

*"On finite grids, A0–A2 imply a unique one-stroke solution."* - Opoch

Now go solve some ARC tasks. By hand. Then in code.

You've got this. 🎯
