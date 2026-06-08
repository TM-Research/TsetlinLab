# Contradiction-free FPTM atlases (`cf_*`)

Regenerates the lab models with the **inline contradiction-fix** Fuzzy-Pattern Tsetlin
Machine engine: `/IoT/Tsetlin.jl/src/Tsetlin_contradiction_fix.jl`. The fix masks the
include vectors at every rebuild (`l &= ~li`, `li &= ~l`), so a clause can never assert
a feature both `≥ t` and `< t` — **0 impossible conditions by construction**, with
accuracy unchanged.

## Files
- `prep_cf.py <loader> <out_dir>` — load a dataset and GLADE-binarise it (n_bins=15) to
  `out_dir/{X_train,X_test,Y_train,Y_test}.txt` + `glade.json`.
- `train_tm_cf.jl` — train the contradiction-fix engine (hyperparameters via env, like the
  original `train_tm.jl`) and write `tm_rules.json` in the TMAtlas clause format.
- `run_cf_all.sh` — run all four datasets end to end and export `cf_<name>_atlas.json`
  + `cf_<name>_predictions.csv` into `../../data/`.

## Run
```bash
bash code/contradiction_fix/run_cf_all.sh
```
Per-dataset hyperparameters mirror `paper_2/config.py` (Table 1); the engine uses
`states_num=256, include_limit=128`. Atlases load unchanged in `viewer.html`.
