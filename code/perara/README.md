# Base perara/Tsetlin.jl engine (`perara_*`)

Runs the **base** Fuzzy-Pattern Tsetlin Machine engine from
[github.com/perara/Tsetlin.jl](https://github.com/perara/Tsetlin.jl) on all 4 datasets,
with the **same** GLADE binarisation and hyperparameters as the `cf_*` models.

This is the **baseline for the contradiction fix**:

| model | engine | impossible conditions |
|-------|--------|----------------------|
| `perara_*` | perara/Tsetlin.jl (base) | **> 0** (self-contradicting rules appear) |
| `cf_*` | perara/Tsetlin.jl **+ inline fix** (`/IoT/Tsetlin.jl/src/Tsetlin_contradiction_fix.jl`) | **0** (by construction) |

Same engine, same data, same hyperparameters — the only difference is the fix
(`l &= ~li` / `li &= ~l` at every include-mask rebuild). Accuracy is unchanged; the
contradictions disappear.

## Run
```bash
# clone the base engine once:
git clone https://github.com/perara/Tsetlin.jl.git /IoT/external/perara-Tsetlin.jl
# then:
bash code/perara/run_perara_all.sh
```
It reuses the GLADE-binarised data from the cf run (or re-binarises if missing), trains
with `ENGINE_PATH=/IoT/external/perara-Tsetlin.jl/src/Tsetlin.jl`, and exports
`perara_<name>_atlas.json` + `perara_<name>_predictions.csv` into `../../data/`.
