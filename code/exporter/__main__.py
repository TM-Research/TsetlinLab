"""CLI: export a GLADE+FPTM bundle to a TMAtlas-style JSON (and a browser viewer).

Examples
--------
::

    # JSON to stdout
    python -m paper_2.atlas results/medsec/models/model.pkl

    # JSON + predictions CSV + a self-contained HTML viewer (just open it)
    python -m paper_2.atlas results/medsec/models/model.pkl --dataset medsec \\
        --out medsec_atlas.json --csv medsec_predictions.csv --html medsec_atlas.html

    # straight from the Julia clause dump (a sibling glade.json is used automatically)
    python -m paper_2.atlas results/nslkdd/models/tm_rules.json --dataset nslkdd

    # serve results/atlas/ so you can open the viewer in a browser
    python -m paper_2.atlas serve            # → http://localhost:8000/viewer.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import (
    GLADEAdapter,
    build_atlas,
    clause_output_matrix,
    export_predictions_csv,
    format_clause_activations,
    load_bundle,
)

_HERE = Path(__file__).resolve().parent
_VIEWER = _HERE / "viewer.html"

# short id → (loader name in data_loader.LOADERS, human label)
_DATASETS = {
    "wustl": ("load_wustl", "WUSTL-EHMS-2020"),
    "nslkdd": ("load_nslkdd", "NSL-KDD"),
    "ton_iot": ("load_toniot", "TON_IoT"),
    "medsec": ("load_medsec", "MedSec-25"),
}


def _load_dataset(short_id: str):
    from ..data_loader import load_and_preprocess  # local import: optional dep chain

    return load_and_preprocess(_DATASETS[short_id][0])


def _write_html(out_path: Path, atlas_doc: dict, csv_text: str | None = None) -> Path:
    if not _VIEWER.exists():
        raise FileNotFoundError(f"viewer template missing: {_VIEWER}")
    html = _VIEWER.read_text()
    payload = json.dumps(atlas_doc, ensure_ascii=False).replace("</", "<\\/")
    marker = "window.__ATLAS_EMBEDDED__ = window.__ATLAS_EMBEDDED__ || null;"
    if marker not in html:
        raise RuntimeError("viewer template changed; embed marker not found")
    html = html.replace(marker, "window.__ATLAS_EMBEDDED__ = " + payload + ";")
    # Optionally embed a predictions CSV so the page auto-computes + explains on open.
    if csv_text is not None:
        csv_payload = json.dumps(csv_text, ensure_ascii=False).replace("</", "<\\/")
        pred_marker = "window.__PRED_EMBEDDED__ = window.__PRED_EMBEDDED__ || null;"
        if pred_marker in html:
            html = html.replace(pred_marker, "window.__PRED_EMBEDDED__ = " + csv_payload + ";")
    out_path.write_text(html)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# `serve` sub-command
# ─────────────────────────────────────────────────────────────────────────────
def _serve(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="python -m paper_2.atlas serve")
    p.add_argument("--dir", help="directory to serve (default: <repo>/results/atlas if present, else cwd)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--bind", default="127.0.0.1")
    args = p.parse_args(argv)

    if args.dir:
        root = Path(args.dir).resolve()
    else:
        # __file__ = .../Paper_2/src/paper_2/atlas/__main__.py  →  repo = Paper_2
        repo = _HERE.parents[2]
        cand = repo / "results" / "atlas"
        root = cand if cand.is_dir() else Path.cwd()
    if not root.is_dir():
        p.error(f"not a directory: {root}")

    # make sure the viewer is reachable from the served root
    dst = root / "viewer.html"
    if not dst.exists() and _VIEWER.exists():
        dst.write_text(_VIEWER.read_text())

    import functools
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(root))
    httpd = ThreadingHTTPServer((args.bind, args.port), handler)
    jsons = sorted(q.name for q in root.glob("*_atlas.json"))
    print(f"serving {root}")
    print(f"  open  http://{args.bind}:{args.port}/viewer.html")
    for j in jsons:
        print(f"        http://{args.bind}:{args.port}/viewer.html?src={j}")
    if not jsons:
        print("  (no *_atlas.json here yet — generate one with "
              "`python -m paper_2.atlas <bundle> --dataset <ds> -o <ds>_atlas.json`)")
    print("  Ctrl-C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# default (export) command
# ─────────────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "serve":
        return _serve(argv[1:])

    p = argparse.ArgumentParser(prog="python -m paper_2.atlas", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bundle", help="GLADE+FPTM bundle: a .pkl bundle or a tm_rules.json path")
    p.add_argument("-o", "--out", help="write the JSON here (default: stdout, unless --html given)")
    p.add_argument("--html", help="write a self-contained HTML viewer (atlas embedded) here")
    p.add_argument("--dataset", choices=sorted(_DATASETS),
                   help="load real feature names / classes / value ranges (and enable --csv)")
    p.add_argument("--feature-names", help="comma-separated feature names (alternative to --dataset)")
    p.add_argument("--class-names", help="comma-separated class names (alternative to --dataset)")
    p.add_argument("--csv", help="also write a per-sample predictions+activations CSV "
                                 "(needs --dataset for the input rows)")
    p.add_argument("--csv-split", choices=("train", "test"), default="test",
                   help="which split to use for --csv (default: test)")
    p.add_argument("--limit", type=int, default=None, help="cap the number of CSV rows")
    p.add_argument("--indent", type=int, default=2, help="JSON indent for --out (default: 2)")
    args = p.parse_args(argv)

    X = feature_names = class_names = data = None
    if args.dataset:
        data = _load_dataset(args.dataset)
        X = np.asarray(data["X_train"])  # ranges/types from the training split
        feature_names = list(data["feature_names"])
        class_names = [str(c) for c in data["class_names"]]
    if args.feature_names:
        feature_names = [s.strip() for s in args.feature_names.split(",")]
    if args.class_names:
        class_names = [s.strip() for s in args.class_names.split(",")]

    doc = build_atlas(args.bundle, X=X, feature_names=feature_names, class_names=class_names)
    n_clauses, n_feat, n_lit = len(doc["clauses"]), len(doc["features"]), doc["metadata"]["numLiterals"]

    if args.out:
        Path(args.out).write_text(json.dumps(doc, indent=args.indent, ensure_ascii=False))
        print(f"wrote {args.out}  ({n_feat} features, {n_clauses} clauses, {n_lit} literals)", file=sys.stderr)

    # Build the predictions CSV first (if requested) so --html can embed it and
    # the page auto-computes + explains the moment it is opened.
    csv_text: str | None = None
    if args.csv:
        if data is None:
            p.error("--csv requires --dataset (to know the input rows)")
        tm_rules, glade_payload = load_bundle(args.bundle)
        adapter = GLADEAdapter(glade_payload)
        split = "X_test" if args.csv_split == "test" else "X_train"
        y_split = "y_test" if args.csv_split == "test" else "y_train"
        X_rows = np.asarray(data[split], dtype=np.float64)
        y_actual = np.asarray(data[y_split])
        if args.limit is not None:
            X_rows, y_actual = X_rows[: args.limit], y_actual[: args.limit]
        try:
            from ..tm_inference import TMModel
            y_pred = TMModel.from_dicts(tm_rules, glade_payload).predict_batch(X_rows)
        except Exception as exc:  # numba / tm_inference unavailable — leave blank
            print(f"warning: prediction unavailable ({exc!r}); 'predicted' column left empty"
                  "  (the HTML viewer recomputes it in-browser anyway)", file=sys.stderr)
            y_pred = np.full(X_rows.shape[0], "", dtype=object)
        acts = format_clause_activations(clause_output_matrix(tm_rules, adapter, X_rows))
        path = export_predictions_csv(args.csv, list(data["feature_names"]), X_rows, y_actual, y_pred, acts)
        print(f"wrote {path}  ({X_rows.shape[0]} rows)", file=sys.stderr)
        try:
            csv_text = Path(path).read_text()
        except OSError:
            csv_text = None

    if args.html:
        _write_html(Path(args.html), doc, csv_text=csv_text)
        suffix = "; samples embedded)" if csv_text else ")"
        print(f"wrote {args.html}  (open in a browser{suffix}", file=sys.stderr)
    if not args.out and not args.html:
        print(json.dumps(doc, indent=args.indent, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
