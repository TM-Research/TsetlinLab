# Contradiction-free FPTM atlases (`cf_*`)

The `cf_*` atlases in `../../data/` are **contradiction-free** — 0 impossible conditions as the
viewer defines them (both same-bit AND GLADE *thermometer* cross-threshold pairs).

## Important: two notions of "impossible"
- **same-bit**: a clause asserts and negates the *same* GLADE bit (`x ≥ t AND x < t`).
- **thermometer**: a clause asserts a *high* rung of a feature and negates a *lower* rung
  (`IN_BYTES ≥ 100 AND IN_BYTES < 50`) — different bits, same feature, still impossible.

A naive mask fix (`l &= ~li`, see `train_tm_cf.jl`) only removes **same-bit** pairs; the
**thermometer** ones survive (the viewer still flags them). To remove *both* you need a
**feature-group-aware** engine.

## What the cf_* atlases use → `run_cf2_all.sh`
Engine: **`/IoT/FuzzyPatternTM/src/Tsetlin_CF.jl` (Method A)** — projects each clause to
thermometer-consistency over the GLADE `feature_groups` (the contiguous bit range of each
source feature, built from `glade.json`'s `feat_idx`). Removes same-bit **and** thermometer
contradictions, leaving accuracy unchanged.

```bash
bash code/contradiction_fix/run_cf2_all.sh     # the correct, feature-group-aware pipeline
```
Files:
- `prep_cf.py <loader> <out_dir>` — GLADE-binarise a dataset (n_bins=15) + save `glade.json`.
- `train_tm_cf2.jl` — train `Tsetlin_CF.jl` with `feature_groups` from `glade.json`; writes
  `tm_rules.json` in the TMAtlas clause format. Hyperparameters via env (mirror `config.py`).
- `run_cf2_all.sh` — all four datasets → `../../data/cf_<name>_atlas.json` + `_predictions.csv`.

Verified: wustl/nslkdd/ton_iot/medsec all export with **0 thermometer and 0 same-bit**
impossible clauses, at accuracy matching the base models.

### Baseline (for comparison)
`../perara/run_perara_all.sh` runs the **base** engine with the guard off (`perara_*`
atlases) — these still contain impossible conditions, which is the contrast the cf_* models fix.
The earlier `train_tm.cf.jl` (inline `l&=~li`, same-bit only) is kept for reference but is
**not** sufficient for GLADE thermometer encodings.
