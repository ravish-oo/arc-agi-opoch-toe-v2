# ARC-AGI Pattern Recognition Guide

## The Recognition Decision Tree

```
START: Look at first training pair
│
├─ Output size != Input size?
│  │
│  ├─ Output LARGER (k× input)?
│  │  → BLOW-UP / TILING pattern
│  │  → Quotient: (i//k, j//k) for blocks
│  │  → Check: Does each input cell become a k×k block?
│  │
│  └─ Output SMALLER than input?
│     → EXTRACTION / COMPRESSION pattern
│     → Look for: Objects to extract, regions to isolate
│     → Check: Border of 1s? Largest component? Specific color?
│
└─ Output size == Input size?
   │
   ├─ Few cells changed (< 20%)?
   │  → LOCAL MODIFICATION pattern
   │  → Quotient: Neighborhood signatures
   │  → Check: Pattern substitution? Hole filling? Border drawing?
   │
   └─ Many cells changed (> 20%)?
      │
      ├─ Diagonal patterns visible?
      │  → DIAGONAL TILING pattern
      │  → Quotient: (i+j) % k or (i-j) % k
      │  → Check: Extract colors from diagonal, tile with them
      │
      ├─ Repeated structures visible?
      │  → PANE SYMMETRY pattern
      │  → Quotient: Block ID + position in block
      │  → Check: Are blocks transformed versions of each other?
      │
      └─ Colors redistributed?
         → COMPONENT ANALYSIS pattern
         → Quotient: Connected component ID
         → Check: Color by size? Position? Shape?
```

---

## Pattern Catalog with Visual Signatures

### Pattern 1: K× Blow-Up

**Visual Signature**: Input is small, output is exactly k× larger in each dimension

**Example**:
```
Input (2×2):          Output (6×6):
[A B]                 [A A A B B B]
[C D]         →       [A A A B B B]
                      [A A A B B B]
                      [C C C D D D]
                      [C C C D D D]
                      [C C C D D D]
```

**Recognition**: Output dimensions = input dimensions × k (for some integer k)

**Quotient**: `(i // k, j // k)` tells you which input cell this came from

**Common variants**:
- Simple replication: `output[i][j] = input[i//k][j//k]`
- Conditional replication: Only replicate if input cell meets condition
- Patch substitution: Replicate a pattern, not just the color

### Pattern 2: Tiling with Transformation

**Visual Signature**: Small input repeated multiple times with variations

**Example**:
```
Input (2×2):          Output (4×4):
[1 2]                 [1 2 | 2 1]    ← Original | Flipped
[3 4]         →       [3 4 | 4 3]
                      [--- + ---]
                      [1 2 | 2 1]    ← Repeat
                      [3 4 | 4 3]
```

**Recognition**:
- Output contains multiple copies of input
- Some copies are transformed (flipped, rotated, color-swapped)

**Quotient**: `(tile_row, tile_col, i%k, j%k)` + transformation flag

**Common transformations**:
- Horizontal flip
- Vertical flip
- Rotation 90°/180°/270°
- Color permutation
- Row/column swap

### Pattern 3: Diagonal Striped Tiling

**Visual Signature**: Output has diagonal stripes of colors

**Example**:
```
Input (has diagonal):   Output (striped):
[0 0 0 1]               [1 2 3 1]
[0 0 1 2]       →       [2 3 1 2]
[0 1 2 3]               [3 1 2 3]
[1 2 3 0]               [1 2 3 1]
```

**Recognition**:
- Input has a diagonal line of colors
- Output has repeating diagonal pattern
- Cells at same (i+j) value have same color

**Quotient**: `(i + j) % num_colors`

**Formula**: `output[i][j] = colors[(i + j) % len(colors)]`

**Variants**:
- Anti-diagonal: `(i + j) % k`
- Diagonal: `(i - j) % k`
- Checkerboard: `(i + j) % 2`

### Pattern 4: Local Pattern Substitution

**Visual Signature**: Specific local patterns in input are replaced with different patterns in output

**Example**:
```
Input:                  Output:
[0 1 0]                 [0 5 0]
[0 1 0]        →        [5 5 5]
[0 1 0]                 [0 5 0]

Rule: Cross of 1s → filled square of 5s
```

**Recognition**:
- Look for recurring small patterns (2×2, 3×3) in input
- Check if they're consistently replaced in output

**Quotient**: Hash of local neighborhood

**Implementation**:
```python
pattern_map = {
    hash(pattern1): replacement1,
    hash(pattern2): replacement2,
    ...
}
```

### Pattern 5: Component-Based Coloring

**Visual Signature**: Objects (connected components) are colored based on properties

**Example**:
```
Input:                  Output:
[2 2 0 3 3 3]          [1 1 0 2 2 2]
[2 0 0 3 3 3]   →      [1 0 0 2 2 2]
[0 0 0 0 3 3]          [0 0 0 0 2 2]

Rule: Color by component size (small→1, large→2)
```

**Recognition**:
- Input has distinct objects (connected components)
- Output colors depend on object properties (size, shape, position, etc.)

**Quotient**: Connected component ID

**Common properties**:
- Size (number of cells)
- Bounding box (height, width, area)
- Position (top-left corner, center)
- Shape (convex, concave, has holes)
- Color in input

### Pattern 6: Border/Frame Drawing

**Visual Signature**: Output adds borders around objects or regions

**Example**:
```
Input:                  Output:
[0 0 0 0]              [5 5 5 5]
[0 2 2 0]      →       [5 2 2 5]
[0 2 2 0]              [5 2 2 5]
[0 0 0 0]              [5 5 5 5]

Rule: Draw border (color 5) around non-zero region
```

**Recognition**:
- Output has additional cells colored at boundaries
- Original content preserved

**Quotient**: Distance to nearest object

**Variants**:
- Draw border around entire grid
- Draw border around each object
- Draw border of thickness k
- Fill holes inside objects

### Pattern 7: Gravity/Physics Simulation

**Visual Signature**: Objects "fall" or move according to physics

**Example**:
```
Input:                  Output:
[1 0 0]                [0 0 0]
[0 0 2]        →       [0 0 0]
[0 0 0]                [1 0 2]

Rule: Objects fall to bottom
```

**Recognition**:
- Non-zero cells move positions
- Movement follows consistent rule (down, left, right, up)
- Final positions are "stable"

**Quotient**: Final resting position

**Common rules**:
- Gravity (fall down until hitting obstacle)
- Magnetism (move toward/away from color)
- Stack (objects pile up)
- Slide (move in direction until blocked)

### Pattern 8: Symmetry Completion

**Visual Signature**: Complete a partial symmetric pattern

**Example**:
```
Input:                  Output:
[1 2 0]                [1 2 1]
[3 4 0]        →       [3 4 3]
[0 0 0]                [1 2 1]

Rule: Complete to vertical symmetry
```

**Recognition**:
- Input has partial symmetric structure
- Output completes the symmetry

**Quotient**: Mirror position

**Types**:
- Vertical mirror
- Horizontal mirror
- 180° rotation
- 4-fold rotation

### Pattern 9: Grid Overlay/Intersection

**Visual Signature**: Multiple grids or patterns combined

**Example**:
```
Grid1:    Grid2:      Output:
[1 0]     [0 2]       [1 2]
[0 1]  +  [2 0]   →   [2 1]

Rule: Overlay non-zero values
```

**Recognition**:
- Input has multiple distinct patterns/colors
- Output combines them

**Common operations**:
- OR: Take any non-zero value
- AND: Only where both have value
- XOR: One or the other but not both
- Priority: One pattern overwrites another

### Pattern 10: Sectioning/Dividing

**Visual Signature**: Grid divided by lines (often color 1), each section processed independently

**Example**:
```
Input:                  Output:
[2 2 | 1 | 3 3]        [2 2 | 1 | 4 4]
[2 0 | 1 | 0 3]   →    [2 0 | 1 | 0 4]

Rule: Color 1 divides sections, fill holes in each
```

**Recognition**:
- Solid lines of one color (often 1 or 0) divide grid
- Each section transformed independently

**Quotient**: Section ID + position within section

---

## The Recognition Algorithm

```python
def recognize_pattern(task):
    """
    Run this mental algorithm when seeing a new task
    """
    train = task['train']

    # Stage 1: Size analysis
    input_shape = shape(train[0]['input'])
    output_shape = shape(train[0]['output'])

    if output_shape > input_shape:
        return "BLOW_UP_PATTERN"
    elif output_shape < input_shape:
        return "EXTRACTION_PATTERN"

    # Stage 2: Change analysis
    changes = count_changed_cells(train[0])
    if changes < 0.2 * total_cells:
        return "LOCAL_MODIFICATION_PATTERN"

    # Stage 3: Structure analysis
    if has_diagonal_line(train[0]['input']):
        return "DIAGONAL_TILING_PATTERN"

    if has_repeated_blocks(train[0]['input']):
        return "PANE_SYMMETRY_PATTERN"

    if has_dividing_lines(train[0]['input']):
        return "SECTIONING_PATTERN"

    # Stage 4: Component analysis
    if has_distinct_objects(train[0]['input']):
        return "COMPONENT_PATTERN"

    # Stage 5: Transformation analysis
    if colors_redistributed(train[0]):
        return "REDISTRIBUTION_PATTERN"

    return "UNKNOWN_PATTERN"  # Need deeper analysis
```

---

## Quick Tests for Pattern Confirmation

### Test 1: Size Ratio
```python
k = output_height // input_height
if k == output_width // input_width and k > 1:
    → Likely blow-up by factor k
```

### Test 2: Diagonal Check
```python
colors = [input[i][i] for i in range(min(h, w))]
if all_non_zero(colors) and output_has_stripes:
    → Likely diagonal tiling
```

### Test 3: Modulo Periodicity
```python
for k in [2, 3, 4, 5]:
    if output[i][j] == output[i+k][j+k] for most cells:
        → Likely period-k tiling
```

### Test 4: Symmetry Check
```python
if output == flip_horizontal(output):
    → Has horizontal symmetry
if output == flip_vertical(output):
    → Has vertical symmetry
```

### Test 5: Component Count
```python
input_components = count_components(input)
output_components = count_components(output)
if input_components == output_components:
    → Components preserved, likely transformed
```

---

## The Pattern Matching Flowchart

```
1. Look at I/O sizes
   ├─ Different? → Size-change pattern
   └─ Same? → Continue

2. Overlay input on output
   ├─ Mostly unchanged? → Local modification
   └─ Mostly changed? → Continue

3. Look for structure
   ├─ Lines dividing grid? → Sectioning
   ├─ Diagonals of color? → Diagonal tiling
   ├─ Repeated blocks? → Pane symmetry
   └─ Distinct objects? → Continue

4. Analyze objects
   ├─ Objects moved? → Physics/gravity
   ├─ Objects colored differently? → Component properties
   ├─ Objects merged/split? → Component operations
   └─ Continue

5. Look for patterns
   ├─ Local patterns replaced? → Pattern substitution
   ├─ Symmetry added? → Symmetry completion
   └─ Colors redistributed? → Redistribution

6. Deep analysis needed
   → Try multiple quotient hypotheses
   → Test each on ALL training examples
   → Pick the one that works deterministically
```

---

## Common Pitfalls

**Pitfall 1: Looking at only one training example**
- Solution: ALWAYS verify pattern on ALL training examples

**Pitfall 2: Seeing false patterns**
- Solution: Can you compute it deterministically? If not, it's not the pattern

**Pitfall 3: Over-complicating the quotient**
- Solution: Start simple (position, color, neighborhood), add complexity only if needed

**Pitfall 4: Ignoring edge cases**
- Solution: Check corners, borders, empty regions explicitly

**Pitfall 5: Not computing by hand**
- Solution: If you can't do it by hand, your understanding is incomplete

---

## Success Metrics

You've correctly identified the pattern when:

1. ✓ You can state the rule in one sentence
2. ✓ You can draw the equivalence classes
3. ✓ You can compute output for ANY input (not just test)
4. ✓ Your rule works on ALL training examples
5. ✓ You can code it in < 20 lines

---

## The Pattern Library

Keep a mental library of these 10 core patterns:
1. Blow-up
2. Tiling with transformation
3. Diagonal striping
4. Pattern substitution
5. Component coloring
6. Border drawing
7. Physics simulation
8. Symmetry completion
9. Grid overlay
10. Sectioning

~80% of ARC tasks are variants or combinations of these.

---

## Next Steps

When you've identified the pattern:
1. Go back to `solving_arc_by_hand.md`
2. Follow the 5-step method
3. Compute test output by hand
4. Verify it makes sense
5. Then code it

---

*Pattern recognition is a skill. Practice on 10-20 tasks and these patterns will become instant.*
