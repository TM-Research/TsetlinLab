# TsetlinLab

An interactive, in-browser lab for **interpretable Tsetlin Machines**.

View every rule in plain language, trace any sample end-to-end
(inputs → bits → clauses fired → class vote → decision), inspect the
booleanisation, and compare binarisers and engines side by side.
**Inference runs fully client-side in WebAssembly — no server, no install.**

→ **Live site:** https://tm-research.github.io/TsetlinLab/

## What it does

- **See every rule.** Each clause is shown in plain language: which features
  must be high or low, the thresholds, and the votes it casts for or against a
  class. Self-contradicting (impossible) conditions are flagged and explained.
- **See what the model learned.** The **Learned Representation** tab folds all
  clauses for one class into a human-readable concept tree: green branches show
  what the model looks for, red branches show what argues against the class, and
  child chips show features learned together.
- **Trace any sample.** Click a sample for a fully-visual decision path: the
  readings, the thermometer bits they set, which clauses fired, the per-class
  vote chart, and the final prediction — with per-condition gauges.
- **Inspect the booleaniser.** The Encoding tab shows one feature becoming
  thermometer bits and literals, live, as you move a value.
- **Compare models.** The same viewer opens any of the trained models so you
  can compare the GLADE binariser against a standard one, and the Fuzzy-Pattern
  engine against a vanilla Tsetlin Machine.

## Quick start

```bash
# any static file server works; for example:
python3 -m http.server 8000
# then open:
#   http://localhost:8000/index.html
```

Open `index.html` for the landing page and model comparison, or jump straight
into the viewer:

```
viewer.html?src=data/wustl_atlas.json&csv=data/wustl_predictions.csv
```

Load any model by changing the `src`/`csv` query parameters to another file in
`data/` (e.g. `data/ton_iot_atlas.json` + `data/ton_iot_predictions.csv`).

> A server is needed only because browsers block `fetch()` over `file://`.
> There is **no backend** — all model computation happens in your browser.

## Layout

```
index.html              landing page + model comparison matrix
viewer.html             the single-file viewer (HTML + CSS + JS, no build step)
data/                   model exports the viewer reads
    <name>_atlas.json        rules, bits, features, metadata
    <name>_predictions.csv   sample feature rows (+ actual / predicted)
code/
    wasm/fptm.c              C inference kernel (Fuzzy-Pattern Tsetlin Machine)
    wasm/fptm.js             the kernel compiled to a single-file WebAssembly module
    exporter/                Python tools that produce the *_atlas.json exports
```

`<name>` is one of: `wustl`, `nslkdd`, `ton_iot`, `medsec` (the GLADE +
Fuzzy-Pattern models), plus `gladetmu_<name>` and `tmu_<name>` for the
comparison engines.

## How inference works in the browser

`code/wasm/fptm.c` implements Fuzzy-Pattern Tsetlin Machine inference:

```
clause violations   c   = popcount( (~x & included) | (x & included_negated) )
clause output           = max(clamp - c, 0)
class score             = Σ positive-clause outputs − Σ negative-clause outputs
prediction              = argmax over classes
```

It is compiled to a single-file WebAssembly module with Emscripten:

```bash
emcc code/wasm/fptm.c -O2 -s WASM=1 -s SINGLE_FILE=1 -s MODULARIZE=1 \
  -s EXPORT_NAME=FPTM -s ALLOW_MEMORY_GROWTH=1 \
  -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","HEAPU8","HEAP32","HEAPU16"]' \
  -o code/wasm/fptm.js
```

The viewer loads a trained model from an `*_atlas.json`, reconstructs its rules,
and recomputes predictions client-side. Results are bit-exact to the trained
model (verified against the reference implementation across the full test sets).

## Datasets

WUSTL-EHMS-2020, NSL-KDD, TON_IoT, and MedSec-25 — multi-class IoT / network
intrusion detection benchmarks.

## Building the data exports

The `*_atlas.json` files are produced by the exporter in `code/exporter/` from a
trained model bundle. See `code/exporter/README.md` for usage.

## License

See [LICENSE](LICENSE).
