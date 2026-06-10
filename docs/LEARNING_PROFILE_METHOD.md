# Learning Profile Method

**A practical interpretability approach for Tsetlin Machines on tabular data**

TsetlinLab implements this method in the **Learn** tab and in the **Explain** tab’s “What the model recognized” section.

---

## The problem with rule lists

A trained Tsetlin Machine on tabular data may contain hundreds of clauses. Each clause is a conjunction of threshold conditions — technically readable, but not how humans understand *learning*.

Reading 450 rules one by one answers: *“What did the machine write down?”*  
It does **not** answer: *“What patterns did it discover?”*

The **Learning Profile Method** answers the second question. It folds the clause bank into a small number of views that describe **learned behaviour**, not extracted syntax.

---

## Historical grounding (brief)

For thirty years, tabular interpretability moved from self-explaining trees (one path per decision) to ensemble summaries (importance scores, partial dependence) to post-hoc attribution. Each step traded **literal decision structure** for **human-scale summaries**.

Tsetlin Machines store learning differently: many overlapping conjunctive patterns vote together. The right summary is not a single path (tree) nor a single importance vector (ensemble) — it is a **pattern field**: which readings push each class, which features co-occur, and where classes separate in feature space.

The Learning Profile Method is designed for that structure.

---

## Method overview

Four layers, all computed from the same trained model (no second model, no approximation):

| Layer | Question it answers | Output |
|-------|---------------------|--------|
| **1. Class profile** | What did the model learn each class *cares about*? | Per-feature direction: high / low / band / avoid |
| **2. Feature couplings** | Which features did it learn *together*? | Top co-occurring feature pairs |
| **3. Decision landscape** | Where does this class *win* in feature space? | 2-feature margin heatmap |
| **4. Sample story** | What did it *recognize* on this row? | 3–5 theme cards grouped by feature idea |

Layers 1–3 are global (Learn tab). Layer 4 is local (Explain tab, top of decision path).

---

## Layer 1 — Class Learning Profile

**Input:** All clauses for class *C* (positive and negative polarity).

**Aggregation:** For each clause, extract thermometer conditions (feature ≥ threshold ON, feature below threshold OFF). Weight each condition by clause `clamp`, optionally scaled by how often the clause fires on loaded samples.

**Per feature *f*, accumulate:**

- `pushHigh` — patterns that support *C* when *f* is high  
- `pushLow` — patterns that support *C* when *f* is low  
- Subtract opposing (negative-polarity) patterns

**Human label** (automatic):

| Signal | Label |
|--------|-------|
| pushHigh ≫ pushLow | *prefers high readings* |
| pushLow ≫ pushHigh | *prefers low readings* |
| both strong | *value band* (not too low, not too high) |
| negative dominates | *avoids high* or *avoids low* |

**Prose summary example (NSL-KDD, class DoS):**

> For **DoS**, the model mainly learned to look at **src_bytes** (prefers high readings), **duration** (prefers low readings), **wrong_fragment** (prefers high readings) …

This is what the model **learned**, not a list of 90 clauses.

---

## Layer 2 — Feature Couplings

**Input:** Positive-polarity clauses for class *C*.

**Aggregation:** For each clause, take all distinct features in its conditions. Every pair *(f₁, f₂)* in the same clause receives weight += clause weight.

**Output:** Top pairs ranked by co-learning mass.

**Example:**

> **src_bytes** and **wrong_fragment** — co-learned  
> **duration** and **count** — co-learned  

Interpretation: the model did not treat these as independent toggles; it repeatedly learned **joint** templates involving both.

---

## Layer 3 — Decision Landscape

**Input:** Top two features from the class profile; all other features fixed at median (from loaded CSV) or range midpoint.

**Process:** Sweep a grid over *(f₁, f₂)*. At each point, run exact model inference. Colour by margin: score(*C*) − score(runner-up).

**Output:** Heatmap — green where class *C* wins, pink where it loses.

**What it reveals:** Decision **boundaries** in human feature space, not bit space. You see regions like “high bytes + short duration → DoS” as contiguous winning zones.

---

## Layer 4 — Sample Learning Story

**Input:** One tabular row; firing positive clauses for the predicted class.

**Aggregation:** Group firing patterns by **lead feature** (dominant condition in each pattern). Sum vote points per feature theme.

**Output:** 3–5 cards, e.g.:

> **src_bytes** — +28 points from 12 matching patterns  
> Your src_bytes = 847,321 — the model learned this class prefers high readings.

No clause IDs. No AND-lists. The reader sees **which ideas** drove the decision and how their readings align with what was learned globally (Layer 1).

---

## Five visualization examples

### Example 1 — Class profile bars (global)

Horizontal bars for the top 12 features. Bar length = total learning mass. Colour: green = high preference, blue = low, amber = band.

*Shows:* Single-feature learning direction without opening the Rules tab.

### Example 2 — Prose learning summary (global)

One paragraph per selected class synthesizing the top five feature directions into plain English.

*Shows:* The “elevator pitch” of what the model learned for that class.

### Example 3 — Coupling list (global)

Pairs like `IN_BYTES ↔ OUT_PKTS` with “co-learned” tag.

*Shows:* Feature **interactions** the model discovered, not isolated importance.

### Example 4 — Decision landscape heatmap (global)

14×14 grid on e.g. `src_bytes` × `duration` for class DoS.

*Shows:* Where the learned behaviour **flips** — the boundary humans can reason about.

### Example 5 — Sample theme cards (local)

At the top of Explain, before the detailed flow diagram.

*Shows:* For this patient/connection/record, which **learned themes** activated and how the row’s values match them.

---

## Practical walkthrough — NSL-KDD, class DoS

**Dataset:** NSL-KDD (network intrusion), model `cf_nslkdd_atlas.json` in TsetlinLab.

**Step 1 — Open Learn tab, select DoS**

You see prose: the model learned DoS around high volume, short connections, error flags — not 90 separate rules.

**Step 2 — Profile bars**

`src_bytes` bar is long and green → learned **high transfer volume**.  
`duration` bar is long and blue → learned **short-lived connections**.  
Together they describe a **flood pattern** without naming any clause.

**Step 3 — Couplings**

`src_bytes` + `wrong_fragment` co-learned → the model treats volume and fragment anomalies as **joint evidence**, matching analyst intuition about DoS.

**Step 4 — Landscape**

Heatmap on `src_bytes` × `duration`: green upper-left (high bytes, low duration) = DoS region. You can literally point at the screen and say “this corner is what the model learned as attack territory.”

**Step 5 — Load samples CSV, pick a DoS row in Explain**

Theme card: **src_bytes** +31 points — your reading matches “prefers high.”  
Second card: **duration** +18 points — matches “prefers low.”  

The detailed rule list (steps 2–6 below) remains available for auditors; the story answers the human question first.

---

## Why this matters

1. **Cognitive fit:** Analysts think in features and regions, not 450 conjunctions.  
2. **Exact:** Every view is computed by re-running the same inference engine — aggregation only, no surrogate.  
3. **Two-level:** Global profiles (Learn) + local recognition (Explain story) connect “what it learned” to “why this row.”  
4. **Tabular-native:** Thermometer thresholds, value bands, and 2D feature sweeps — not heatmaps on pixels.  
5. **Implementable:** Implemented in TsetlinLab `viewer.html` (Learn tab + sample story); works on any `*_atlas.json`.

---

## Implementation notes

- **Static mode:** Without a samples CSV, weights use clause `clamp` from the rule bank.  
- **Data-driven mode:** With CSV loaded, weights scale by average vote when clauses fire on real traffic.  
- **Rules tab unchanged:** Full clause audit trail remains for compliance; Learning Profile is the human entry point.

---

## Files

| File | Role |
|------|------|
| `viewer.html` | Learn tab UI + `buildClassProfile`, `buildCouplings`, `decisionStripSvg`, `buildSampleStory` |
| `docs/LEARNING_PROFILE_METHOD.md` | This document |
