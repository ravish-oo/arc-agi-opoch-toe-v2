Understood. I’ll fold the last-mile implementation details directly into the Work Orders themselves so Claude has zero degrees of freedom. When you ask for any WO, I’ll return the **fully-expanded, atomic spec** with all math-critical edge cases, prohibitions, and acceptance checks baked in.

Below is the **augmented WO index** (what gets included where), plus **three complete examples** so you can see the level of precision I’ll deliver on request.

# Augmented WO index (what details go where)

* **WO-01 Types & IO:** stable types, CSV writer toggle; no logic. ✅ COMPLETE
* **WO-02 Kernel:** D8, residues, **non-negative diagonals**, integral-image counts, components with border_contact; 1×N and N×1 edge cases. ✅ COMPLETE
* **WO-03 Π Present:** rank-view with **union-of-inputs frequency** tie-break; D8; idempotence (DEBUG off by default). ✅ COMPLETE
* **WO-04 QtSpec:** residues cap=10 (include divisors), radii=(1,2[,3]), diagonals on; deterministic sort.
* **WO-05 Qt classes:** **stable class keys** (bytes), WL channel separation, `int32` before bytes view, deterministic pack order.
* **WO-06 Δ Shape law:** **IDENTITY, BLOW_UP(kh,kw)** and (when applicable) **FRAME(t), TILING(kh,kw)** with **deterministic tie-break** (prefer BLOW_UP; optional periodicity check remains input-only).
* **WO-07 Bt Boundary:** force-until-forced; refinement ladder **S0..S6** (residue extend, r=3, WL=4, **distance-to-border**, **component centroid parity / scan index**, WL=5); keys by bytes.
* **WO-08 Φ Writer:** one-stroke; **identity-guard** and **blow-up Bt-empty guard** (input-only color); **FRAME** border color rule; **TILING** by tiling identity-painted patch; no palette search; no multi-stroke.
* **WO-09 Solver:** Π → QtSpec → Δ → Bt → Φ → un-Π; Δ from **original** sizes; deterministic, no prints.
* **WO-10 Notebook:** inline/import and emit `submission.csv` with pipe toggle.

When you say “WO-0X”, I’ll return the finished spec. Below are **three fully-expanded examples** (you can hand them to Claude as-is):

---