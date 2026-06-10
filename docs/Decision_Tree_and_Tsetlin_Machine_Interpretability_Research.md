# Decision Tree and Tsetlin Machine Interpretability for Tabular Data
## A Research Synthesis (1994–Present)

**Prepared for:** TsetlinLab — interactive interpretability for Tsetlin Machines  
**Scope:** Thirty years of decision-tree explainability, post-hoc XAI, and the Tsetlin Machine as an intrinsically interpretable alternative for tabular (and related structured) data  
**Live tooling reference:** [TsetlinLab viewer](https://Rakesh2109.github.io/TsetlinLab/)

---

## Executive Summary

Interpretability for tabular data has evolved from an implicit property of symbolic learners (decision trees, rule lists) to a standalone research field (XAI) dominated by post-hoc methods that approximate black-box models. Decision trees remain the canonical *intrinsic* baseline: each prediction follows an explicit path of feature-threshold tests. Tsetlin Machines (TMs), introduced by Granmo (2018), occupy a distinct but related position: they learn **propositional clauses** (AND-rules over Boolean literals) that vote collectively, rather than a single mutually exclusive tree. The model **is** its explanation—there is no surrogate gap by construction.

The central insight for tabular TM interpretability is not merely listing clauses, but revealing **what the model actually learned**: which feature thresholds co-occur as patterns, how clauses superpose into class decisions, which sub-patterns tolerate partial mismatch (Fuzzy-Pattern TM), and how individual samples activate specific evidence. TsetlinLab operationalizes this through plain-language rules, thermometer encoding inspection, per-sample decision paths, and bit-exact client-side re-inference—making learned patterns tangible rather than abstract.

---

## 1. Historical Evolution (1994–Present)

### 1.1 Foundations: Symbolic Learning as Self-Explaining (1994–2000)

The modern interpretability discourse for tabular data begins in the **knowledge discovery** tradition. Fayyad, Piatetsky-Shapiro, and Smyth (1996) defined data mining as extracting *understandable* patterns—not merely accurate ones—from data. In this era, interpretability was not a separate module; it was the **output format** of the learner.

**Decision trees** (ID3, C4.5; Quinlan 1986, 1993) and **rule induction** (CN2, RIPPER) were the default tabular learners precisely because their hypotheses were human-readable: axis-aligned splits, leaf labels, or if-then rules. **Oblique decision trees** (Murthy, Kasif, and Salzberg, 1994) extended splits to linear combinations of features, trading some readability for expressiveness—an early signal that interpretability and representational power pull in opposite directions.

Parallel work addressed the opposite problem: neural networks were accurate but opaque. **Craven and Shavlik (1994, ICML)** cast rule extraction from trained networks as a learning problem using network queries—not search alone. **TREPAN (NeurIPS 1996)** induced decision trees from trained networks by treating the network as an oracle, producing comprehensible trees with high *fidelity* to the network. This established a template that persists today: **use trees as explanations of something else**, not as the primary model.

**KBANN (Towell and Shavlik, 1994)** went further: initialize a neural network from domain rules, train it, then extract refined rules—an early **neuro-symbolic** bridge between continuous optimization and propositional logic.

**Turning point 1:** Interpretability split into two camps—**intrinsic** (the model is readable) and **extrinsic/post-hoc** (a second model explains the first).

### 1.2 Ensembles, Complexity, and the Fidelity Gap (2000–2014)

Random forests (Breiman, 2001) and gradient boosting (Friedman, 2001) dominated tabular prediction but weakened direct interpretability: thousands of trees, interaction effects, and non-additive structure made path inspection impractical. Practitioners turned to **proxy measures**:

- **Variable importance** (mean decrease in impurity; permutation importance)
- **Partial dependence plots (PDP)** and **Individual Conditional Expectation (ICE)** curves (Friedman, 2001; Goldstein et al., 2015)
- **Rule ensembles** (Friedman and Popescu, 2008) attempting to retain additive structure

**Turning point 2:** For strong tabular predictors, explanation became **statistical summarization** of ensemble behavior rather than literal description of the decision function.

Concurrently, **Doshi-Velez and Kim (2017)** formalized evaluation of interpretability into application-, human-, and functionally grounded levels—highlighting that "being interpretable" requires evidence beyond anecdote. This framework later became central to TM evaluation (deletion tests, comprehensiveness/sufficiency).

### 1.3 The XAI Explosion (2015–2019)

Deep learning's rise on images and text shifted mainstream XAI toward **local post-hoc methods**:

| Method | Year | Mechanism | Tabular fit |
|--------|------|-----------|-------------|
| LIME (Ribeiro et al.) | 2016 | Local linear surrogate over perturbed inputs | Natural for feature attribution |
| SHAP (Lundberg & Lee) | 2017 | Shapley values from cooperative game theory | Additive feature contributions |
| Anchors (Ribeiro et al.) | 2018 | Sufficient conditions for predictions | Rule-like but approximate |
| Grad-CAM (Selvaraju et al.) | 2017 | Gradient-weighted spatial maps | Image-centric; poor tabular analog |

For tabular data, LIME and SHAP became de facto standards. Surveys (Marcinkevičs & Vogt, 2023; Sahakyan et al., 2024) note persistent challenges: **feature correlation breaks perturbation assumptions**, **global summaries disagree with local attributions**, and **fidelity to the original model is unguaranteed**.

**Turning point 3:** Rudin (2019, *Nature Machine Intelligence*) argued that post-hoc explanation of black boxes in high-stakes settings is **misleading by design**—one should use **inherently interpretable models** instead. This reframed decision trees, sparse linear models, and rule learners from "old-fashioned" to **architecturally faithful**.

### 1.4 Intrinsically Interpretable Revival and Tsetlin Machines (2018–Present)

**Granmo (2018)** introduced the **Tsetlin Machine**: a team of Tsetlin Automata (Tsetlin, 1961) that learn conjunctive clauses in propositional logic. Unlike TREPAN's extracted trees, TM clauses are **the trained model**, not an approximation.

Subsequent milestones:

- **Blakely & Granmo (2020/2021):** Closed-form global and local importance from clause structure; benchmarked against SHAP
- **Weighted / Coalesced / Convolutional / Regression TMs:** Extended representational power while preserving symbolic form
- **CSC-TM, Contracting TM, Sparse TM, Drop Clause (2021–2024):** Sparsity and rule-length control for human-scale inspection
- **Fuzzy-Pattern TM (FPTM, Hnilov 2025):** Graded clause output for noise tolerance—requiring new interpretability theory (Hamming-ball prototypes, partial-match evidence)
- **Regulatory pressure:** EU AI Act (2024), GDPR "right to explanation" debates → demand for **audit-ready** models in finance, healthcare, and critical infrastructure

**Turning point 4:** The field moved from "explain any model" toward **compliance-by-design** with verifiable, exact explanations—where TMs and modern rule learners compete with (and often outperform) post-hoc XAI on **faithfulness**, if not always on raw accuracy versus the largest ensembles.

### 1.5 Timeline Overview

```
1994 ── Craven/Shavlik rule extraction; KBANN neuro-symbolic rules
1996 ── Fayyad et al. KDD definition; TREPAN tree extraction from NNs
2001 ── Random forests; Friedman PDP for tabular interaction visualization
2016 ── LIME local surrogates
2017 ── SHAP; Doshi-Velez & Kim evaluation framework
2018 ── Granmo Tsetlin Machine; Anchors
2019 ── Rudin "Stop explaining black boxes"
2020 ── Blakely/Granmo TM interpretability; TM intrusion detection (IEEE SSCI)
2021 ── Coalesced TM; Drop Clause (AAAI)
2024 ── EU AI Act; tabular XAI surveys; literal pruning vs human attention
2025 ── FPTM; NMIBC clinical rule dashboards; trustworthy rule-set critique (Siala et al.)
2026 ── On-device interpretable TM-IDS for IoMT; TsetlinLab atlas viewer
```

---

## 2. Decision Tree Interpretability Fundamentals

### 2.1 Why Decision Trees Are Considered Interpretable

A decision tree offers three properties that satisfy human cognitive models of reasoning:

1. **Explicit decision path:** Each prediction traverses a sequence of tests `feature ≤ threshold?` ending in a leaf label. The path *is* the explanation.
2. **Axis-aligned semantics:** Each split references a single named feature at a human-meaningful threshold—directly mappable to domain vocabulary (e.g., `age > 65`, `IN_BYTES ≥ 1000`).
3. **Hierarchical context:** Parent nodes encode preconditions; child nodes refine within that context, mirroring nested if-then reasoning.

Breiman's CART framework and Quinlan's C4.5 made this the default mental model for "how ML decides" in tabular domains for decades.

### 2.2 Limitations That Redefined "Interpretable"

As models grew, the term **interpretable** proved fragile:

| Limitation | Consequence |
|------------|-------------|
| **Depth and size** | A 50-level tree with thousands of nodes is technically readable but cognitively unusable |
| **Instability** | Small data changes alter topology (Breiman noted this for single trees) |
| **Mutually exclusive structure** | One path per prediction; cannot express overlapping pattern families compactly |
| **Axis-aligned rigidity** | Oblique boundaries require many splits or oblique variants that lose simplicity |
| **Ensemble opacity** | Random forests and XGBoost aggregate hundreds of trees—importance scores replace paths |
| **Surrogate fidelity** | TREPAN/ distilled trees approximating NNs may misrepresent edge cases |

**Murthy et al. (1994)** and the **neural tree** literature (survey: Su et al., 2022) document the persistent **accuracy–interpretability trade-off**: hybrid neural trees on images sacrifice performance; oblique trees sacrifice axis-aligned clarity.

### 2.3 Redefinition of Interpretability Concepts

Modern literature distinguishes:

- **Interpretable vs explainable:** Interpretable models are readable by construction; explainable models receive post-hoc narratives (Rudin, 2019).
- **Global vs local:** Global = overall feature relevance and rule structure; local = why *this* instance received *this* label.
- **Fidelity / faithfulness:** Does the explanation reflect the model's actual computation? Post-hoc methods optimize local fidelity; trees and TMs have fidelity ≈ 1 by architecture.
- **Trustworthiness:** Siala, Planes & Marques-Silva (2025) warn that intrinsic rule sets can still exhibit **redundancy, overlap, and contradictions**—intrinsic ≠ automatically trustworthy.

For tabular data specifically, **feature interactions** are the crux. PDP/ICE visualize marginal effects but mask interaction structure; **rule lists and clauses** encode interactions explicitly as conjunctive patterns.

---

## 3. Tsetlin Machine Approach to Interpretability for Tabular Data

### 3.1 Mechanism: Learning as Logic, Not Weights

A Tsetlin Machine classifies by **voting among propositional clauses**:

**Literals:** For Boolean input `x ∈ {0,1}^n`, literals are `x_k` (feature/bit ON) or `¬x_k` (OFF). Continuous tabular features are **booleanized** first—typically via thermometer encoding: `feature ≥ t₁`, `feature ≥ t₂`, … producing ordered threshold bits (as in TsetlinLab's GLADE binariser).

**Clause:** `C_j(X) = ∏_{l ∈ L_j} l` — conjunction of included literals. In the **Fuzzy-Pattern TM (FPTM)**, clause output is graded:

```
violations c = popcount((~x & included) | (x & included_negated))
clause vote u = max(clamp − c, 0)
class score f_k = Σ_{j ∈ P_k} u_j − Σ_{j ∈ N_k} u_j
prediction = argmax_k f_k
```

**Learning:** Each literal is governed by a Tsetlin Automaton with two actions—**Include** or **Exclude**. Type I/II feedback reinforces or penalizes literals based on clause satisfaction and class polarity. Hyperparameters `T` (voting margin/throttle), `L` (clause size budget), and `LF` (literal failure tolerance in FPTM) shape **which patterns form**, not a separate explanation layer.

### 3.2 Why TMs Reveal What Was Actually Learned

Unlike SHAP/LIME, which fit a *different* function locally, TM interpretability is **definitional**:

| Aspect | Black-box + SHAP/LIME | Tsetlin Machine |
|--------|----------------------|-----------------|
| Explanation object | Approximate attribution | Clauses that *are* the classifier |
| Fidelity | Bounded, often unmeasured | Exact (same code path) |
| Feature interactions | Implicit in attributions | Explicit in AND-literals |
| Threshold semantics | Lost in raw attributions | Preserved via booleanization |
| Audit | Two models to trust | One model to inspect |

**What the model learns** is not a weight vector but a **clause bank**: a disjunctive normal form (DNF-like) pattern library per class, with negative clauses providing counter-evidence. Each clause captures a **recurring conjunction** in the data—e.g., for intrusion detection: `IN_BYTES ≥ 327 ∧ OUT_PKTS < 5 ∧ duration ≥ 1857 → Probe`.

**Blakely & Granmo (2020/2021)** address scale: inspecting thousands of clauses individually fails. Their **closed-form global and local importance** expressions aggregate literal inclusion frequencies weighted by clause polarity—enabling real-time monitoring during training and ranking features without SHAP's Monte Carlo cost. Empirical studies report **correspondence with SHAP rankings** while remaining native to the TM.

**FPTM-specific insight (Reddy, 2026 draft; Hnilov, 2025):** A fuzzy clause is not merely true/false—it is the **graded indicator of a Hamming ball** over literal space. One clause encodes a **family of sub-patterns** tolerating up to `LF−1` literal mismatches. Per-instance explanation therefore includes:

- Which literals **matched** vs **failed**
- **Match strength** = `clamp − violations`
- How partial evidence accumulates into class vote margins

This reveals *how close* an sample was to learned prototypes—not just which rule fired.

### 3.3 Human-Understandable Rules for Tabular Contexts

Canonical tabular TM rules read as domain conditions:

- **Medical text (Berge et al., IEEE Access 2019):** `IF "rash" ∧ "reaction" ∧ "penicillin" THEN Allergy`
- **Clinical recurrence (NMIBC, 2025):** `HospitalStay > 3 ∧ TumourNumber > 3 → Recurrence`
- **Network IDS (Abeyrathna et al., IEEE SSCI 2020; TsetlinLab NSL-KDD):** `src_bytes ≥ 218 ∧ flag_SF = ON ∧ wrong_fragment ≥ 1 → DoS`

TsetlinLab renders these without index notation: each clause shows **feature name, comparator, threshold**, polarity (FOR/AGAINST class), and flags **impossible conditions** (e.g., thermometer contradictions like `IN_BYTES ≥ 100 ∧ IN_BYTES < 50`). The **contradiction-free (cf_*)** atlases enforce thermometer consistency so displayed rules are **logically satisfiable**—addressing the trustworthy-rules critique.

### 3.4 Fundamental Difference from Image-Based or Generic XAI

| Dimension | Image XAI (Grad-CAM, saliency) | Generic tabular XAI | Tabular TM |
|-----------|-------------------------------|---------------------|------------|
| Unit of meaning | Pixel/region heatmap | Feature attribution score | Named threshold literals in AND-rules |
| Spatial structure | Essential | Absent | Absent (unless Convolutional TM) |
| Explanation type | Where to look | How much each feature mattered | Which conditions jointly define a class pattern |
| Compositionality | Weak (smooth maps) | Additive (SHAP) | Logical (clauses vote; interactions explicit) |
| Verification | Subjective overlay | Perturbation tests | Re-run inference; compare clause activations |

**Convolutional TM (Granmo et al., 2019)** bridges to vision: clauses become **localized patch filters** with spatial coordinates—an image-specific idiom. For pure tabular data, interpretability stays in **feature-threshold space**, making TMs closer to **rule ensembles** than to saliency methods.

TsetlinLab compares engines side-by-side (GLADE+FPTM vs standard binariser vs vanilla TMU), showing how **encoding choices** change what patterns are learnable and readable—an issue invisible to generic SHAP on raw features.

---

## 4. Visualization for Revealing What the Model Learned

The literature converges on four visual **idioms** for TM interpretability. TsetlinLab implements all four plus tabular-specific extensions.

### 4.1 Four Core Idioms

#### (1) Rule List / Clause Table
Plain-language AND-rules split by polarity (evidence FOR vs AGAINST each class). Supports search, sorting, and contradiction badges.  
**Reveals:** Global pattern library—the model's "vocabulary" of conditions.

#### (2) Weight-Ranked Clause Table
Top-N clauses by weight or activation frequency (Weighted TM, Integer-Weighted TM, Coalesced TM).  
**Reveals:** Which patterns dominate decision mass—not all clauses are equal.

#### (3) Clause-Activation Heatmap
Matrix of clauses × samples (or classes), cell = fired/graded activation. NMIBC 2025 dashboards pair patient×clause heatmaps with importance bars—cited as gold-standard composite layout. IoMT IDS papers use clause×class heatmaps.  
**Reveals:** **Which learned patterns fire together** on which instances—interaction of global rules with local data.

#### (4) Per-Class Vote Bar Chart
Bar height = class vote sum; **margin** between winner and runner-up = confidence. Often paired with signed per-feature contribution bars (Blakely local importance).  
**Reveals:** **Competition among classes** and how evidence accumulates—not just the winning label.

### 4.2 Tabular-Specific Visualizations (Beyond Rule Lists)

To show *learning* rather than *rules*, tabular visualization must expose:

| Visualization | What learning it exposes | TsetlinLab implementation |
|---------------|-------------------------|---------------------------|
| **Thermometer / encoding diagram** | How continuous values become ordered threshold bits | Encoding tab: live bit activation as slider moves |
| **Feature interaction matrix** | Co-occurring literals across clauses | Clauses tab + search by feature name |
| **Decision path (instance trace)** | Inputs → bits → clause match/miss → votes → label | Explain tab: step-by-step with per-condition gauges |
| **Partial-match gauges (FPTM)** | Which conditions failed yet clause still voted | Match strength = `clamp − violations` per clause |
| **Vote decomposition waterfall** | Additive evidence toward final score | Per-class bars + contributing clauses listed |
| **Binariser comparison matrix** | Effect of encoding on learned patterns | index.html model comparison across datasets |
| **Global literal frequency chart** | Features most embedded in clauses | Metadata + clause statistics |

**Contrast with image visualization:** Grad-CAM overlays do not translate to tabular data—there is no spatial canvas. **PDP/ICE** show marginal effects but hide conjunctive structure. TM visualizations must emphasize **logical conjunction**, **threshold semantics**, and **voting dynamics**.

### 4.3 Functional Verification Visualizations

Following Doshi-Velez & Kim and Hooker et al., interpretability claims require **functional tests**:

- **Deletion/insertion:** Remove top-evidence literals; measure prediction flip rate (FPTM paper: 83% flip vs 7% random—evidence is causal)
- **Comprehensiveness/Sufficiency:** Do highlighted literals alone suffice for the decision? (Yadav et al., 2024—literal pruning vs human attention)
- **Bit-exact re-inference:** TsetlinLab WASM kernel recomputes predictions client-side—explanation and prediction share one engine (**zero surrogate gap**)

### 4.4 Design Principles for Human Understanding

1. **Progressive disclosure:** Show top-N clauses and winner-vs-runner-up first; expand on demand (essential for 100+ classes).
2. **Domain vocabulary:** Always show feature names and thresholds, not bit indices (technical bit view optional).
3. **Polarity sign:** Separate supporting vs opposing evidence—mirrors legal/medical "evidence for/against" reasoning.
4. **Flag pathologies:** Mark impossible clauses, redundant rules, and zero-vote clauses.
5. **Link global to local:** Clicking a sample highlights *which global clauses* activated—a bridge SHAP cannot offer structurally.

---

## 5. Practical Applications

### 5.1 Intrusion Detection and IoT Security

**Seminal work:** Abeyrathna et al. (IEEE SSCI 2020) — KDD Cup 99, human-readable propositional rules, competitive with ANN/SVM/DT/RF/KNN.

**TsetlinLab datasets:** WUSTL-EHMS-2020, NSL-KDD, TON_IoT, MedSec-25 — multi-class IoT/network intrusion benchmarks with exported atlases. Example learned pattern (NSL-KDD): clauses over `src_bytes`, `duration`, TCP flags distinguishing DoS, Probe, R2L, U2R from Normal traffic with **interpretable byte/packet thresholds**.

**Recent IoMT (2025–2026):**
- Jaiswal et al. — TM-IDS on CICIoMT-2024; 99.5% binary / 90.7% multi-class accuracy with vote bars and activation heatmaps
- On-device Raspberry Pi deployment with explicit interpretability analysis (MedSec-25)

**Value delivered:** Security analysts audit **which traffic signatures** triggered alerts; false-positive investigation traces specific clause failures; regulatory compliance gains **documented decision logic** vs black-box anomaly scores.

### 5.2 Healthcare and Clinical Decision Support

**Berge et al. (IEEE Access 2019):** EHR text categorization with rules like allergy detection from clinical terms—accuracy competitive with LSTM, with readable formulae.

**NMIBC recurrence (2025):** TM-derived clinical rules with composite dashboard (patient×clause heatmap, importance bars)—template for **clinician-facing** TM visualization.

**Continuous-input TM (Abeyrathna et al., 2019):** Direct learning on clinical numeric features without manual binarization—rules over raw thresholds.

**Value delivered:** Clinicians validate rules against domain knowledge; unsafe clauses (contradictions, spurious correlations) are inspectable before deployment—addressing Rudin's high-stakes argument.

### 5.3 NLP and Attention Enhancement

**MDPI Algorithms (2022):** TM clauses initialize neural attention layers—logic-based prerequisite knowledge replaces expensive human rationales.

**Value delivered:** TM global patterns guide neural models while retaining **auditable symbolic core**.

### 5.4 TsetlinLab as Applied Interpretability Platform

TsetlinLab consolidates research idioms into a **zero-install, client-side laboratory**:

- **See every rule** in plain language with contradiction detection
- **Trace any sample** end-to-end: readings → thermometer bits → clause activation → vote → prediction
- **Compare models** (GLADE vs standard binariser; FPTM vs vanilla TM) on identical datasets
- **Bit-exact WASM inference** — explanations verified against reference implementation

This directly addresses the gap identified in FPTM source literature (accuracy/noise focus without interpretability theory): TsetlinLab + FPTM mathematical framework show **how to read graded clauses as soft prototypes** and verify explanations functionally.

### 5.5 Quantitative Interpretability Metrics in Practice

| Metric | Typical use | Limitation |
|--------|-------------|------------|
| Number of clauses | Model compactness | Ignores clause overlap |
| Literals per clause | Rule length / cognitive load | Dominant proxy in TM papers |
| Total literals / sparsity | Global complexity | Does not measure usefulness |
| Pruning ratio | Post-hoc simplification | May hurt accuracy |
| Comprehensiveness/Sufficiency | Human-grounded (ERASER) | Rare in TM work except Yadav 2024 |
| Deletion flip rate | Functional faithfulness | Used in FPTM evaluation |
| SHAP rank correlation | External validation | Blakely & Granmo 2020/2021 |

---

## 6. Synthesis: From Rules to Revealed Learning

Thirty years of tabular interpretability trace an arc from **self-explaining symbolic models** → **post-hoc approximation of opaque models** → **renewed demand for intrinsic, verifiable logic**. Decision trees remain the intuitive baseline: one path, one story. Tsetlin Machines generalize the story to a **democracy of patterns**—many overlapping clauses vote, positive and negative evidence competes, and (in FPTM) partial matches graded by tolerance.

The research frontier is not whether TMs produce rules—they do by architecture—but whether we can **make learned structure cognitively accessible**:

1. **Scale:** Closed-form importance, sparsity mechanisms (CSC, Contracting, Drop Clause), and progressive visualization
2. **Faithfulness:** Exact decomposition proofs (FPTM), deletion tests, bit-exact re-inference (TsetlinLab)
3. **Trustworthiness:** Contradiction-free rule sets, overlap analysis, human evaluation beyond size proxies
4. **Domain grounding:** Thermometer encoding, named thresholds, audit trails for regulated deployment

TsetlinLab demonstrates that interpretability for tabular TMs is not a post-processing step—it is an **interactive epistemology**: humans explore the clause bank, stress-test samples, and compare encodings until the model's discovered patterns become **genuinely understandable**, not merely printable.

---

## References (Selected)

### Historical & Decision Trees
- Breiman et al. (1984). *Classification and Regression Trees.*
- Quinlan (1993). *C4.5: Programs for Machine Learning.*
- Murthy, Kasif & Salzberg (1994). Oblique decision trees. *AIJ.*
- Craven & Shavlik (1994). Rule extraction from NNs. *ICML.*
- Craven & Shavlik (1996). TREPAN. *NeurIPS.*
- Friedman (2001). Greedy function approximation: gradient boosting.
- Ribeiro et al. (2016). LIME. *KDD.*
- Lundberg & Lee (2017). SHAP. *NeurIPS.*

### Interpretability Frameworks
- Doshi-Velez & Kim (2017). Towards a rigorous science of interpretable ML.
- Rudin (2019). Stop explaining black box models. *Nature Machine Intelligence.*
- Marcinkevičs & Vogt (2023). Survey of interpretable/explainable ML. *WIREs DMKD.*
- Sahakyan et al. (2024). Explainable tabular data analysis review. *Electronics.*
- Siala, Planes & Marques-Silva (2025). Trustworthy rule-based models. *arXiv:2507.07576.*

### Tsetlin Machines
- Tsetlin (1961). Complexity of events and machines.
- Granmo (2018). The Tsetlin Machine. *arXiv:1804.01508.*
- Blakely & Granmo (2020/2021). Closed-form TM interpretability. *arXiv:2007.13885; IEA/AIE.*
- Berge et al. (2019). Human-interpretable rules for text/medical categorization. *IEEE Access.*
- Abeyrathna et al. (2020). Intrusion detection with interpretable TM rules. *IEEE SSCI.*
- Sharma et al. (2023). Drop Clause. *AAAI.*
- Hnilov (2025). Fuzzy-Pattern Tsetlin Machine. *arXiv:2508.08350.*
- Reddy (2026). Pattern formation and exact interpretability in FPTM. *draft.*
- Jaiswal et al. (2026). TM-IDS for IoMT. *arXiv:2604.03205, 2605.16707.*

### Tooling
- CAIR TMU: https://github.com/cair/tmu
- TsetlinLab: https://github.com/Rakesh2109/TsetlinLab

---

*Document version: June 2026. Claims flagged as preprint in source notes should be verified before citation in peer-reviewed submission.*
