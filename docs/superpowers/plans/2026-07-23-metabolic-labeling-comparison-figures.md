# Metabolic Labeling Comparison Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create two publication-ready English SVG figures that compare how SILAC, uniform ¹³C, and uniform ¹⁵N labeling shift the precursor and fragment ions of the same peptide.

**Architecture:** Both figures contain the same declarative scientific content and reuse the visual language of `silac_adaptation_experiment_workflow.svg`. The matrix figure prioritizes direct row-by-row comparison; the radial figure prioritizes a shared-peptide narrative. Each SVG is a standalone editable vector asset, while a separate PNG is only a rendered comparison preview.

**Tech Stack:** SVG 1.1/XML, Unicode isotope superscripts, `rsvg-convert`, ImageMagick `montage` or an SVG wrapper for preview composition, Python standard-library XML parsing.

## Global Constraints

- Create files only under `analysis/workflow_steps/`.
- Do not modify the existing reference SVG.
- Use a transparent canvas with editable vector elements and text; do not embed raster content.
- Use `YLYEIAR` as the single unmodified example peptide in both layouts.
- Use Light `#4DBBD5`, Heavy `#00A087`, structural blue `#3C5488`, mass-shift purple `#8B68AD`, body text `#2B2B2B`, and the existing pale cyan/green fills.
- Use `Arial, Helvetica, sans-serif`.
- Use `ΔM` for neutral mass and `Δm/z = ΔM / z` for precursor peak separation.
- SILAC Arg10 is `¹³C₆¹⁵N₄`, `ΔM = 10.008275 Da`; b ions not containing R remain aligned, and y ions containing R shift.
- Uniform ¹³C uses `ΔM = n(C) × 1.003355 Da`; `YLYEIAR` contains 44 C, so precursor `ΔM = 44.147620 Da`.
- Uniform ¹⁵N uses `ΔM = n(N) × 0.997035 Da`; `YLYEIAR` contains 10 N, so precursor `ΔM = 9.970350 Da`.
- Every selected ¹³C/¹⁵N fragment must shift by its own elemental count; do not draw a constant fragment gap.
- Label all spectra as `schematic, not to scale`.

---

## File Structure

- Create `analysis/workflow_steps/silac_c13_n15_comparison_matrix.svg`
  - Standalone 1600 × 1000 matrix layout with three scheme columns and four shared comparison rows.
- Create `analysis/workflow_steps/silac_c13_n15_comparison_radial.svg`
  - Standalone 1600 × 1100 radial layout with one central Light peptide and three equal information cards.
- Create `analysis/workflow_steps/silac_c13_n15_layout_comparison.png`
  - Rendered side-by-side preview only; not a source asset.

---

### Task 1: Create the matrix comparison SVG

**Files:**
- Create: `analysis/workflow_steps/silac_c13_n15_comparison_matrix.svg`
- Reference: `analysis/workflow_steps/silac_adaptation_experiment_workflow.svg`
- Reference: `docs/superpowers/specs/2026-07-23-metabolic-labeling-comparison-figure-design.md`

**Interfaces:**
- Consumes: the color, font, axis, arrow, rounded-panel, spectrum, and dashed-shift conventions from the reference SVG.
- Produces: a standalone SVG with root `viewBox="0 0 1600 1000"` and groups `matrix-header`, `silac-column`, `c13-column`, `n15-column`, and `matrix-summary`.

- [ ] **Step 1: Define a failing structural acceptance check**

Run:

```bash
python - <<'PY'
from pathlib import Path
from xml.etree import ElementTree as ET

path = Path("analysis/workflow_steps/silac_c13_n15_comparison_matrix.svg")
root = ET.parse(path).getroot()
text = "".join(root.itertext())
assert root.attrib["viewBox"] == "0 0 1600 1000"
for token in (
    "SILAC vs. ¹³C vs. ¹⁵N Metabolic Labeling",
    "K/R residues only",
    "All carbon atoms",
    "All nitrogen atoms",
    "10.008275 Da",
    "44.147620 Da",
    "9.970350 Da",
    "schematic, not to scale",
):
    assert token in text, token
for group_id in (
    "matrix-header", "silac-column", "c13-column",
    "n15-column", "matrix-summary",
):
    assert root.find(f".//*[@id='{group_id}']") is not None, group_id
PY
```

Expected: FAIL with `FileNotFoundError` before the matrix SVG exists.

- [ ] **Step 2: Draw the shared matrix framework**

Create the SVG root with:

```xml
<svg xmlns="http://www.w3.org/2000/svg"
     width="1600" height="1000" viewBox="0 0 1600 1000">
  <title>SILAC, carbon-13, and nitrogen-15 metabolic labeling comparison</title>
  <desc>Matrix comparison of site-specific SILAC and composition-dependent
        uniform carbon-13 and nitrogen-15 precursor and fragment shifts.</desc>
  <defs>
    <marker id="matrix-axis-arrow" viewBox="0 0 10 10"
            refX="8.4" refY="5" markerWidth="9" markerHeight="9"
            orient="auto" markerUnits="userSpaceOnUse">
      <path d="M0 0 L10 5 L0 10 Z" fill="#2B2B2B"/>
    </marker>
  </defs>
  <g id="matrix-header"/>
  <g id="silac-column"/>
  <g id="c13-column"/>
  <g id="n15-column"/>
  <g id="matrix-summary"/>
</svg>
```

Populate the framework at these fixed positions:

- Title centered at `(800, 55)`, 32 px bold.
- Shared legend at `(665, 76)`.
- Row labels at x = 28: `LABELING RULE` y = 185, `MS1 PRECURSOR` y = 390, `MS2 FRAGMENTS` y = 650, and `MASS-SHIFT RULE` y = 902.
- Column panels at x = 135, 630, and 1125, each width 440, y = 120, height = 805, rx = 26.
- Column headers centered at x = 355, 850, and 1345.
- Use pale fills `#D2EEF4`, `#E8F5F1`, and `#F1EEF7` with fill opacity between 0.38 and 0.55.

- [ ] **Step 3: Add the labeling-rule peptide diagrams**

For each column, draw seven linked residue circles for `Y L Y E I A R`.

- Circle centers use local x positions `72, 118, 164, 210, 256, 302, 348`.
- SILAC: residues Y–A use cyan-outline Light styling; R uses green fill and the label `R*`; place `Arg10 = ¹³C₆¹⁵N₄` below.
- ¹³C: keep the peptide Light-styled, then add a green rounded atom band below it labeled `all 44 C → ¹³C`; do not color every residue as if each received the same increment.
- ¹⁵N: use the same construction with `all 10 N → ¹⁵N`.
- Add the scope labels `K/R residues only`, `All carbon atoms`, and `All nitrogen atoms`.

- [ ] **Step 4: Add the MS1 precursor panels**

In each column, draw:

- One cyan four-peak isotope envelope and one green four-peak isotope envelope.
- Shared horizontal m/z axis with arrow.
- A purple dashed bracket from the first Light M0 peak to the first Heavy M0 peak.
- Scheme-specific formula:
  - SILAC: `Δm/z = 10.008275 / z`
  - ¹³C: `Δm/z = (44 × 1.003355) / z`
  - ¹⁵N: `Δm/z = (10 × 0.997035) / z`
- Scheme-specific neutral mass:
  - `ΔM = 10.008275 Da`
  - `ΔM = 44.147620 Da`
  - `ΔM = 9.970350 Da`

Use a visibly larger schematic Light-to-Heavy separation in the ¹³C column while retaining the `not to scale` qualifier.

- [ ] **Step 5: Add the MS2 mirror spectra**

Use four selected ions per column: `b3`, `b5`, `y3`, and `y5`. Draw Light peaks upward and Heavy peaks downward from a shared axis.

Use the following conceptual pair behavior:

| Scheme | b3 | b5 | y3 | y5 |
|---|---:|---:|---:|---:|
| SILAC | aligned | aligned | shifted by Arg10 | shifted by Arg10 |
| ¹³C | 24 C | 35 C | 15 C | 29 C |
| ¹⁵N | 3 N | 5 N | 6 N | 8 N |

For SILAC, draw Heavy b3/b5 at the identical x values as their Light peaks and Heavy y3/y5 at shifted x values. For ¹³C/¹⁵N, offset all four Heavy peaks, vary the offset in proportion to the listed atom counts, and connect each Light/Heavy pair with a purple dashed leader.

- [ ] **Step 6: Add formulas and the shared takeaway**

Add one compact formula block to each column:

```text
SILAC   ΔMfragment = Σ labeled K/R
¹³C     ΔMfragment = nfragment(C) × 1.003355 Da
¹⁵N     ΔMfragment = nfragment(N) × 0.997035 Da
```

Add `schematic, not to scale` above the shared bottom summary:

```text
SILAC: site-specific fragment shifts  |  ¹³C / ¹⁵N: composition-dependent shifts in every fragment
```

- [ ] **Step 7: Run matrix validation and render inspection output**

Run the structural check from Step 1, then:

```bash
rsvg-convert \
  -w 1600 -h 1000 \
  analysis/workflow_steps/silac_c13_n15_comparison_matrix.svg \
  -o /tmp/silac_c13_n15_matrix.png
```

Expected: structural script exits 0; `rsvg-convert` exits 0 and creates a 1600 × 1000 PNG without warnings.

- [ ] **Step 8: Commit the matrix figure**

```bash
git add analysis/workflow_steps/silac_c13_n15_comparison_matrix.svg
git diff --cached --check
git commit -m "docs: add matrix metabolic labeling comparison"
```

Expected: one SVG is committed and no unrelated untracked files are staged.

---

### Task 2: Create the radial comparison SVG

**Files:**
- Create: `analysis/workflow_steps/silac_c13_n15_comparison_radial.svg`
- Reference: `analysis/workflow_steps/silac_c13_n15_comparison_matrix.svg`

**Interfaces:**
- Consumes: the exact scientific copy, peptide styling, peak colors, formulas, and fragment-pair behavior established by Task 1.
- Produces: a standalone SVG with root `viewBox="0 0 1600 1100"` and groups `radial-header`, `shared-light-peptide`, `silac-card`, `c13-card`, `n15-card`, and `radial-summary`.

- [ ] **Step 1: Define a failing structural acceptance check**

Run:

```bash
python - <<'PY'
from pathlib import Path
from xml.etree import ElementTree as ET

path = Path("analysis/workflow_steps/silac_c13_n15_comparison_radial.svg")
root = ET.parse(path).getroot()
text = "".join(root.itertext())
assert root.attrib["viewBox"] == "0 0 1600 1100"
for token in (
    "One light peptide, three labeling strategies",
    "SILAC", "¹³C", "¹⁵N",
    "K/R residues only", "All carbon atoms", "All nitrogen atoms",
    "schematic, not to scale",
):
    assert token in text, token
for group_id in (
    "radial-header", "shared-light-peptide", "silac-card",
    "c13-card", "n15-card", "radial-summary",
):
    assert root.find(f".//*[@id='{group_id}']") is not None, group_id
PY
```

Expected: FAIL with `FileNotFoundError` before the radial SVG exists.

- [ ] **Step 2: Draw the central source and radial flow**

Create a 1600 × 1100 transparent SVG with:

- Title centered at `(800, 52)`.
- Subtitle `One light peptide, three labeling strategies` centered at `(800, 88)`.
- Central rounded source panel at x = 560, y = 120, width = 480, height = 135.
- A seven-residue Light peptide `YLYEIAR` centered inside the source panel.
- Three dark flow arrows leaving the source panel:
  - to SILAC card centered at x = 305, y = 300;
  - to ¹³C card centered at x = 800, y = 300;
  - to ¹⁵N card centered at x = 1295, y = 300.

Although the geometry fans left/center/right rather than forming a literal circle, all three paths must share one origin and have equal visual weight.

- [ ] **Step 3: Build three equal cards**

Create cards at x = 70, 565, and 1060; y = 315; width = 470; height = 680; rx = 28.

Each card must contain, at identical local y positions:

- scheme header at y = 52;
- labeling scope at y = 84;
- Heavy peptide or atom-band diagram at y = 130;
- MS1 title and mini spectrum at y = 245–390;
- MS2 title and mini mirror spectrum at y = 430–575;
- formula and one-line conclusion at y = 620–655.

Reuse the Task 1 scientific behavior exactly. Reduce MS2 to four ions (`b3`, `b5`, `y3`, `y5`) and omit secondary prose so labels remain readable.

- [ ] **Step 4: Add the radial summary and scale qualifier**

Place a shared summary strip below the cards:

```text
Precursor shift identifies the heavy partner; fragment-shift pattern reveals the labeling strategy.
```

Place `schematic, not to scale` at the lower-right edge of the summary strip.

- [ ] **Step 5: Run radial validation and render inspection output**

Run the structural check from Step 1, then:

```bash
rsvg-convert \
  -w 1600 -h 1100 \
  analysis/workflow_steps/silac_c13_n15_comparison_radial.svg \
  -o /tmp/silac_c13_n15_radial.png
```

Expected: structural script exits 0; `rsvg-convert` exits 0 and creates a 1600 × 1100 PNG without warnings.

- [ ] **Step 6: Commit the radial figure**

```bash
git add analysis/workflow_steps/silac_c13_n15_comparison_radial.svg
git diff --cached --check
git commit -m "docs: add radial metabolic labeling comparison"
```

Expected: one SVG is committed and no unrelated untracked files are staged.

---

### Task 3: Produce the comparison preview and complete validation

**Files:**
- Create: `analysis/workflow_steps/silac_c13_n15_layout_comparison.png`
- Verify: `analysis/workflow_steps/silac_c13_n15_comparison_matrix.svg`
- Verify: `analysis/workflow_steps/silac_c13_n15_comparison_radial.svg`

**Interfaces:**
- Consumes: the two final SVGs from Tasks 1 and 2.
- Produces: one side-by-side rendered preview plus final syntax, content, rendering, and repository checks.

- [ ] **Step 1: Render both SVGs at equal preview height**

```bash
rsvg-convert \
  -h 900 \
  analysis/workflow_steps/silac_c13_n15_comparison_matrix.svg \
  -o /tmp/silac_c13_n15_matrix_preview.png
rsvg-convert \
  -h 900 \
  analysis/workflow_steps/silac_c13_n15_comparison_radial.svg \
  -o /tmp/silac_c13_n15_radial_preview.png
```

Expected: both commands exit 0.

- [ ] **Step 2: Compose the side-by-side preview**

If ImageMagick `montage` is available:

```bash
montage \
  /tmp/silac_c13_n15_matrix_preview.png \
  /tmp/silac_c13_n15_radial_preview.png \
  -tile 2x1 -geometry +40+20 -background white \
  analysis/workflow_steps/silac_c13_n15_layout_comparison.png
```

If `montage` is unavailable, create a temporary wrapper SVG that embeds the two rendered preview PNGs side by side, then render that wrapper with `rsvg-convert`. Do not add the temporary wrapper to the repository.

Expected: the output PNG contains both complete layouts with white comparison background and no clipping.

- [ ] **Step 3: Run final semantic checks**

```bash
python - <<'PY'
from pathlib import Path
from xml.etree import ElementTree as ET

paths = [
    Path("analysis/workflow_steps/silac_c13_n15_comparison_matrix.svg"),
    Path("analysis/workflow_steps/silac_c13_n15_comparison_radial.svg"),
]
required = {
    "YLYEIAR",
    "10.008275 Da",
    "44.147620 Da",
    "9.970350 Da",
    "n(C)",
    "n(N)",
    "schematic, not to scale",
}
for path in paths:
    root = ET.parse(path).getroot()
    text = "".join(root.itertext())
    missing = sorted(required - {token for token in required if token in text})
    assert not missing, f"{path}: missing {missing}"
    assert text.count("b3") >= 1
    assert text.count("b5") >= 1
    assert text.count("y3") >= 1
    assert text.count("y5") >= 1
print("semantic SVG checks passed")
PY
```

Expected: `semantic SVG checks passed`.

- [ ] **Step 4: Inspect both rendered figures**

Open the two full-resolution `/tmp` renderings and the repository preview. Confirm:

- no title, formula, card, or spectrum is clipped;
- text remains legible at common slide size;
- Light and Heavy colors match the legend;
- SILAC b3/b5 are aligned while y3/y5 shift;
- every ¹³C/¹⁵N Heavy fragment shifts;
- ¹³C/¹⁵N fragment gaps vary by atom count;
- the two layouts contain the same formulas and conclusions.

- [ ] **Step 5: Run repository checks**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only the intended preview remains uncommitted after the two SVG task commits, alongside pre-existing unrelated user files.

- [ ] **Step 6: Commit the comparison preview**

```bash
git add analysis/workflow_steps/silac_c13_n15_layout_comparison.png
git diff --cached --check
git commit -m "docs: add metabolic labeling layout preview"
```

Expected: the preview is committed without staging unrelated user files.
