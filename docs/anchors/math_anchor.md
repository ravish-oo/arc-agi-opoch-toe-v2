# Opoch ARC-AGI — Math Anchor (Π → Qt → Δ → Bt → Φ)

This is the **math-only** contract. No engineering, no heuristics. Claude Code should implement exactly this, nothing else.

## 0) Domain

* A task has train pairs ({(X_i,Y_i)}_{i=1}^m) and test inputs ({X^{\mathrm{te}}*j}*{j=1}^n).
* A grid (X) is a function (X: V\to\Sigma) with (V={0,\dots,h-1}\times{0,\dots,w-1}), (\Sigma={0,\dots,9}).
* All operators are **finite, deterministic**, and respect the constraints (1\le h,w\le 30).

## 1) Π — Present Canonicalization (inputs-only)

**Goal.** Remove “presentation” degrees of freedom (pose) without touching label content.

* Let (\mathcal{G}\subset \mathrm{D8}) be the 8 planar isometries: rotations by (k\cdot 90^\circ) and optional horizontal flip (transpose not needed if D8 is used).
* Define a **rank-view** (R(X)\in{0,\dots,9}^{h\times w}): replace each color (c) by its rank when sorting colors by frequency (desc), ties by color id (asc).
* Define the canonical pose
  [
  \Pi(X):=\arg\min_{g\in\mathcal{G}} \big(R(g!\cdot!X),; g!\cdot!X\big)
  ]
  in lex order on the byte strings of the two tensors (rank-view primary, raw grid secondary).
* **Idempotence.** (\Pi(\Pi(X))=\Pi(X)).
* **Input-only.** (\Pi) uses (X) only, never (Y).

> We will apply (\Pi) to **every** input grid; outputs are never canonized. After solving, invert the chosen (g) to un-present.

## 2) Qt — Input Partition as a **Spec**

**Goal.** Construct an **input-only** equivalence relation (\Psi) on pixels (“classes”), shared across all grids of a shape, that captures structure needed to write correct colors.

* A **feature family** (\mathcal{F}) is fixed, finite, and **shape-aware** (but content-blind). Its channels are:

  1. **Base color:** (f_{\text{col}}(p)=X(p)).
  2. **Residues:** for (k\in K) (finite set), (f_{r,k}(p)=r \bmod k), (f_{c,k}(p)=c \bmod k).
  3. **Diagonal residues:** (f_{+ ,k}(p)=(r+c)\bmod k), (f_{-,k}(p)=(r-c)\bmod k) with strict non-negativity.
  4. **Local counts (Chebyshev balls):** for radii (r\in R) (finite), (f_{\text{cnt},r}(p)\in\mathbb{N}^{10}) counts colors in the ((2r+1)\times(2r+1)) box around (p).
  5. **Component summaries (4-conn on (X)):** per-pixel tuple ((\text{area},\text{height},\text{width},\text{border_contact}\in{0,1})).
  6. **WL refinement channel:** (f_{\text{wl}}^{(t)}(p)) = a deterministic summary of neighbor class ids (median of sorted 5-tuple up,down,left,right,self) at round (t).

* **Initialization.** Two pixels (p,q) are equivalent if their concatenated feature vectors (items 1–5) are equal.

* **WL refinement.** Do (T) rounds (typically (T=3), may increase to (4) or (5) if needed): at each round add channel (f_{\text{wl}}^{(t)}) computed from the previous round’s classes and relabel by equality of the full concatenated feature vector.

* The result is an **equivalence relation** (\Psi_{\mathcal{F}}) and a class-id map (\mathrm{cls}_X:V\to\mathbb{N}) for each grid (X).

**Stable keys across grids.** The identity of a class is the **intrinsic signature**: the packed feature vector bytes of any representative pixel of that class. This key is used to match classes across different grids (no reliance on local integer ids).

**Spec, not state.** Qt is the **family (\mathcal{F})** (choice of (K,R,T) and inclusion of diagonals, components). For any grid (X), classes are recomputed from (\mathcal{F}); no caching of partitions from a single exemplar.

## 3) Δ — Shape Law (dimensions-only)

**Goal.** Predict output canvas size from training **dimensions** (never content). Let (X\in\mathbb{Z}^{h\times w}), (Y\in\mathbb{Z}^{h'\times w'}).

We support these minimal laws:

* **IDENTITY:** (h'=h,; w'=w).
* **BLOW_UP ((k_h,k_w)):** (h'=k_h h,; w'=k_w w), integers (k_h,k_w\ge 1). Includes uniform (k_h=k_w) and rectangular (k_h\ne k_w).
* **(Optional, if enabled) TILING ((k_h,k_w)):** (h'=k_h h,; w'=k_w w), semantics is periodic copy, not pixel blow-up.
* **(Optional, if enabled) FRAME ((t)):** (h'=h+2t,; w'=w+2t), border thickness (t\in\mathbb{N}).

**Inference (deterministic, dimensions-only).** From all train pairs’ ((h,w)\to(h',w')):

* If all pairs share integer ratios ((k_h,k_w)), set (\Delta=) **BLOW_UP** ((k_h,k_w)) (unless TILING is explicitly preferred; see tie-break below).
* Else if all share (h'-h=w'-w=2t) for some (t\ge 1), set (\Delta=)**FRAME**((t)).
* Else set (\Delta=)**IDENTITY**.
  **Tie-break:** if both blow-up and tiling are arithmetically possible, prefer **BLOW_UP**; if a tiling mode is explicitly required by the benchmark, set TILING.

Δ is within contract: it uses **sizes only**.

## 4) Bt — Boundary (train-only color forcing)

**Goal.** For each class (by **stable key**), determine if its written color is uniquely determined by the training evidence.

* For each train pair ((X_i,Y_i)) with **matching shape** under (\Delta=\mathrm{IDENTITY}), compute (\mathrm{cls}*{X_i}) from Qt; for every pixel (p\in V), add (Y_i(p)) to the bucket of the class **key** (k=\mathrm{key}(\mathrm{cls}*{X_i}(p))).
* A class key (k) is **forced** if its bucket has a single color (c\in\Sigma). Collect a map (\rho: k\mapsto c).
* If multiple colors appear for the same key, the partition is **too coarse**. Refine Qt (see §5) and recompute Bt.
* If no same-shape pairs exist (e.g., pure blow-up training), Bt may have **no evidence**. This is allowed (see Φ guards).

Bt is the **first and only** place that reads (Y), and only to check uniqueness of written colors per class key.

## 5) Refinement Ladder (deterministic, input-only)

If Bt finds any multi-color class keys, **increase discriminative power** of Qt in a fixed order, each step strictly input-only:

S0. Baseline spec: residues (K={2,3,4,5,6}), diagonals enabled, radii (R={1,2}), WL rounds (T=3).
S1. Extend residues to include divisors of observed heights/widths up to cap 10.
S2. Add radius (3) counts.
S3. Increase WL rounds to (4).
S4. Add distance-to-border channel (e.g., Chebyshev or Manhattan).
S5. Add component centroid parity ( ((\lfloor \bar r\rfloor \bmod 2,\lfloor \bar c\rfloor \bmod 2)) ) or component scan-order index.
S6. (If needed) WL rounds to (5).

Stop as soon as **all** class keys in Bt are forced. The ladder is finite and deterministic.

## 6) Φ — Write Once (lawful painting)

**Goal.** Produce the output grid(s) by a single write per pixel class, using only forced information and lawful input-only fallbacks where Bt has no evidence.

Let (X) be a (canonized) test input, (\mathrm{cls}_X) its class map, and (\Delta) the shape law.

* **Case Δ = IDENTITY.**
  Create (Y^*\in\Sigma^{h\times w}) initialized to (0).
  For each local class id (c) with key (k):

  * If (k\in \mathrm{dom}(\rho)), set (Y^*[p]=\rho(k)) for all (p) with (\mathrm{cls}_X(p)=c).
  * Else (**Bt-empty identity guard**, rare): set (Y^*[p]= X(p)) for those (p) (input-only fallback; valid because base color is a Qt feature).

* **Case Δ = BLOW_UP ((k_h,k_w)).**
  Create (Y^*\in\Sigma^{(k_h h)\times(k_w w)}) initialized to (0).
  For each local class id (c) with key (k), collect its pixels (p=(r,c)):

  * Determine color (v):

    * If (k\in \mathrm{dom}(\rho)), set (v=\rho(k)).
    * Else (**Bt-empty blow-up guard**): set (v=X(p_0)) for any (p_0) in this class of (X) (legal: input-only).
  * Write the constant block (k_h\times k_w) with color (v) for each (p):
    (Y^*[r k_h : (r+1)k_h,, c k_w : (c+1)k_w]\gets v).

* **Case Δ = TILING ((k_h,k_w)) (optional).**
  First form (Z) by the **IDENTITY** rule above on the input shape (h\times w), then copy-tile (Z) in a (k_h\times k_w) tiling to fill (Y^*).

* **Case Δ = FRAME ((t)) (optional).**
  Create (Y^*\in\Sigma^{(h+2t)\times(w+2t)}).
  Fill border with the **forced** color of the background class key if available, else with the mode input color on perimeter classes (input-only).
  Paint the interior (t!:!-t) window by the **IDENTITY** rule applied to (X).

**One-stroke law.** Φ never composes strokes; each class writes once. No post-hoc edits.

## 7) Un-present

After Φ, invert the canonization transform from Π for each test grid to restore the original pose: (Y=\Pi^{-1}(Y^*)).

## 8) Correctness Invariants

* **Input-only up to Bt.** Π and Qt depend only on inputs; Δ depends only on **dimensions**; Bt is the minimal contact with targets (train (Y)) and only to test uniqueness per class key.
* **Lawfulness.** Φ uses only (\rho) from Bt or **input-only** fallbacks (class’s own input color) when Bt has no evidence; never uses target content to invent colors.
* **Determinism.** All choices have fixed, finite tie-breakers (lex orders, caps, step order).
* **Idempotence of Π.** Canonizing an already canonized input changes nothing.
* **Stable classes.** Class identity across grids is the intrinsic feature signature, not a local integer id.

## 9) Minimal Parameter Set (fixed)

* Residue cap (|K|\le 10); default (K={2,3,4,5,6}) extended by divisors of observed sizes up to 10.
* Radii (R={1,2}) with optional (3) if needed.
* WL rounds (T=3), optionally (4) then (5) per ladder.
* Components are 4-connected. Local counts use Chebyshev balls. All channels are integer-valued.

## 10) Acceptance (what “done” means)

For each test input (X^{\mathrm{te}}),
[
Y^{\mathrm{pred}} = \Pi^{-1}\Big(\Phi\big(\mathrm{cls}_{\Pi(X^{\mathrm{te}})},,\rho,,\Delta\big)\Big)
]
where:

* (\mathrm{cls}) arises from (\Psi_{\mathcal{F}}) built by the ladder-refined feature family,
* (\rho) is the Bt-forced color map on class keys,
* (\Delta) is inferred from train dimensions only.

No other operations are permitted. No stochasticity. No heuristic guessing.

---

### Notes matching hand-solves

* **Tiled motif (00576224):** residues (k=2) + WL separate pane classes; Bt forces one color per class; identity Φ paints.
* **Shift/periodic local move (25ff71a9):** WL neighbor summary captures shift; Bt forces the “new top row = 0” class; identity Φ paints.
* **Blow-up 3× (9172f3a0 / 007bbfb7 style):** Δ=Blow-up ((3,3)); Bt may be empty; Φ uses input-only fallback per class to color blocks.
* **Mirror (67a3c6ac):** Π chooses pose but un-presentation restores orientation; WL captures side structure; identity Φ paints.

This anchor is the **sole authority** for the implementation. If a step is not in this spec, it must not appear in code.
