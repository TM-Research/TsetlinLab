"""TMAtlas-style structured export for the GLADE + Fuzzy Pattern TM pipeline.

``sonnets-project/TMAtlas`` exports trained TMU models into a serialisable
``{model, metadata, features, clauses}`` dict (plus a per-sample CSV of
predictions and clause activations).  That package only understands TMU
binarisers and the TMU clause bank, so this module provides the equivalent for
Paper 2's stack:

* :class:`GLADEAdapter` wraps the GLADE booleaniser (a fitted object *or* a
  ``glade.json`` payload) and exposes the ``unique_values`` /
  ``get_feature_names_out`` interface TMAtlas expects from a binariser, plus a
  ``transform`` for convenience.
* :class:`FeatureInspector` turns the binariser + (optional) raw data into
  per-feature definitions.
* :class:`FPTMInspector` reads a ``tm_rules`` dict and produces the model info,
  metadata and clause records.
* :class:`JsonExporter` assembles the final dict; :func:`format_clause_activations`
  / :func:`export_predictions_csv` cover the CSV side.
* :func:`build_atlas` is the one-call convenience wrapper.

Example
-------
::

    from paper_2.atlas import build_atlas
    import json
    doc = build_atlas("results/medsec/models/model.pkl")          # GLADE+FPTM bundle
    json.dump(doc, open("medsec_atlas.json", "w"), indent=2, ensure_ascii=False)
"""

from __future__ import annotations

import csv as _csv
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from .types import (
    Clause,
    ClauseWeight,
    FeatureClass,
    Literal,
    ModelInfo,
    ModelMetadata,
)

__all__ = [
    "GLADEAdapter",
    "FeatureInspector",
    "FPTMInspector",
    "JsonExporter",
    "build_atlas",
    "load_bundle",
    "clause_output_matrix",
    "format_clause_activations",
    "export_predictions_csv",
    "Literal",
    "ClauseWeight",
    "Clause",
    "FeatureClass",
    "ModelInfo",
    "ModelMetadata",
]

# Operator glyphs match TMAtlas exactly so downstream tooling can be shared.
GEQ = "≥"  # ≥
LT = "<"


# ─────────────────────────────────────────────────────────────────────────────
# Binariser adapter
# ─────────────────────────────────────────────────────────────────────────────
def _glade_payload(obj: Any) -> dict[str, Any]:
    """Coerce a fitted GLADE booleaniser or a payload dict to a payload dict."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    # GLADEBooleanizer / GLADEv2 expose private arrays directly.
    feat_idx = getattr(obj, "_feat_idx", None)
    thresh = getattr(obj, "_thresh", None)
    if feat_idx is None or thresh is None:
        raise TypeError(
            "GLADEAdapter expects a glade.json-style dict or a fitted GLADE "
            "booleaniser with .to_dict() / ._feat_idx / ._thresh"
        )
    return {
        "version": type(obj).__name__,
        "n_features_in": int(getattr(obj, "_n_features", int(np.max(feat_idx)) + 1)),
        "n_bits": int(np.asarray(thresh).size),
        "feat_idx": np.asarray(feat_idx, dtype=np.int64).tolist(),
        "thresh": [float(x) for x in np.asarray(thresh).tolist()],
        "quantised": False,
    }


def _dequantise_thresh(payload: dict[str, Any]) -> NDArray[np.float32]:
    if payload.get("quantised", False):
        q = np.asarray(payload["thresh_q"], dtype=np.float64)
        return (q * float(payload["thresh_scale"]) + float(payload["thresh_zp"])).astype(
            np.float32
        )
    return np.asarray(payload["thresh"], dtype=np.float32)


def _fmt_threshold(value: float) -> str:
    """Render a threshold so ``"<name> ≥ <value>"`` round-trips cleanly."""
    v = float(value)
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return repr(v)


class GLADEAdapter:
    """Binariser adapter exposing the interface TMAtlas inspectors expect.

    Bit ``j`` of the Boolean output is ``X[:, feat_idx[j]] >= thresh[j]``.
    """

    def __init__(self, glade: Any):
        payload = _glade_payload(glade)
        self.feat_idx: NDArray[np.int64] = np.asarray(payload["feat_idx"], dtype=np.int64)
        self.thresh: NDArray[np.float32] = _dequantise_thresh(payload)
        if self.feat_idx.shape != self.thresh.shape:
            raise ValueError("feat_idx and thresh must have the same length")
        self.n_bits: int = int(self.thresh.size)
        self.n_features_in: int = int(
            payload.get("n_features_in")
            or (int(self.feat_idx.max()) + 1 if self.n_bits else 0)
        )
        self.version: str = str(payload.get("version", "GLADE"))
        # 1-based index of each bit within its own feature, in global bit order:
        # the k-th bit GLADE emits for feature i gets local_idx == k.
        self.local_idx: NDArray[np.int64] = np.zeros(self.n_bits, dtype=np.int64)
        _seen: dict[int, int] = {}
        for j in range(self.n_bits):
            f = int(self.feat_idx[j])
            _seen[f] = _seen.get(f, 0) + 1
            self.local_idx[j] = _seen[f]

    def bit_name(self, j: int) -> str:
        """Structured name for global bit ``j``: ``f<feature>_<localbit>`` (both 1-based)."""
        j = int(j)
        return f"f{int(self.feat_idx[j]) + 1}_{int(self.local_idx[j])}"

    # -- TMAtlas binariser contract --------------------------------------
    @property
    def unique_values(self) -> list[NDArray[np.float32]]:
        """Per-original-feature threshold lists, in output-bit order."""
        return [self.thresh[self.feat_idx == i] for i in range(self.n_features_in)]

    def get_feature_names_out(
        self, feature_names: Sequence[str] | None = None, style: str = "threshold"
    ) -> NDArray[np.str_]:
        """One label per output bit, e.g. ``"latency ≥ 0.5"``."""
        names = (
            list(feature_names)
            if feature_names is not None
            else [f"x{i}" for i in range(self.n_features_in)]
        )
        if len(names) != self.n_features_in:
            raise ValueError(
                f"feature_names has {len(names)} entries, expected {self.n_features_in}"
            )
        if style not in ("threshold", "range"):
            raise ValueError("style must be 'threshold' or 'range'")
        labels = []
        if style == "threshold":
            for j in range(self.n_bits):
                labels.append(f"{names[self.feat_idx[j]]} {GEQ} {_fmt_threshold(self.thresh[j])}")
        else:  # "range" — like TMAtlas's optional form; only the ≥ form is parsed
            uv = self.unique_values
            for i, col_thr in enumerate(uv):
                lo = -np.inf
                for t in col_thr:
                    labels.append(f"{lo} ≤ {names[i]} < {_fmt_threshold(t)}")
                    lo = float(t)
                if col_thr.size:
                    labels.append(f"{names[i]} {GEQ} {_fmt_threshold(col_thr[-1])}")
        return np.asarray(labels, dtype=np.str_)

    # -- convenience -----------------------------------------------------
    def transform(self, X: NDArray[Any], pack_bits: bool = False) -> NDArray[np.uint8]:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"transform expects a 2D matrix, got shape {X.shape}")
        if X.shape[1] != self.n_features_in:
            raise ValueError(
                f"X has {X.shape[1]} features, binariser was fit on {self.n_features_in}"
            )
        out = (X[:, self.feat_idx] >= self.thresh[np.newaxis, :]).astype(np.uint8)
        if not pack_bits:
            return out
        pad = (-self.n_bits) % 8
        if pad:
            out = np.concatenate([out, np.zeros((X.shape[0], pad), np.uint8)], axis=1)
        return np.packbits(out, axis=1)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "GLADEAdapter":
        return cls(payload)


# ─────────────────────────────────────────────────────────────────────────────
# Feature inspector
# ─────────────────────────────────────────────────────────────────────────────
class FeatureInspector:
    """Builds per-feature definitions from a binariser (+ optional raw data).

    ``X`` is the raw, pre-binarisation matrix.  When omitted, feature ranges and
    types are inferred from the binariser's thresholds (``type`` is reported as
    ``"unknown"`` in that case).
    """

    def __init__(
        self,
        binariser: GLADEAdapter,
        X: NDArray[Any] | None = None,
        feature_names: Sequence[str] | None = None,
    ):
        if not hasattr(binariser, "unique_values") or not hasattr(
            binariser, "get_feature_names_out"
        ):
            raise AttributeError(
                "binariser must expose `unique_values` and `get_feature_names_out`"
            )
        self.binariser = binariser
        self.n_features = int(getattr(binariser, "n_features_in", len(binariser.unique_values)))
        self.feature_names: list[str] = (
            list(feature_names)
            if feature_names is not None
            else [f"x{i}" for i in range(self.n_features)]
        )
        if len(self.feature_names) != self.n_features:
            raise ValueError(
                f"feature_names has {len(self.feature_names)} entries, "
                f"expected {self.n_features}"
            )
        self.X: NDArray[np.float64] | None = None
        if X is not None:
            self.X = np.asarray(X, dtype=np.float64)
            if self.X.ndim != 2 or self.X.shape[1] != self.n_features:
                raise ValueError(
                    f"X must be 2D with {self.n_features} columns, got {self.X.shape}"
                )
        self.boolean_labels: NDArray[np.str_] = binariser.get_feature_names_out(
            self.feature_names
        )

    def extract(self) -> list[FeatureClass]:
        defs: list[FeatureClass] = []
        uv = self.binariser.unique_values
        for i, name in enumerate(self.feature_names):
            thresholds = [float(t) for t in np.asarray(uv[i]).tolist()]
            if self.X is not None:
                col = self.X[:, i]
                lo, hi = float(np.min(col)), float(np.max(col))
                uniq = np.unique(col)
                if uniq.size <= 2 and set(uniq.tolist()) <= {0.0, 1.0}:
                    ftype = "binary"
                else:
                    ftype = "continuous"
            else:
                lo = float(min(thresholds)) if thresholds else float("nan")
                hi = float(max(thresholds)) if thresholds else float("nan")
                ftype = "unknown"
            defs.append(FeatureClass(name=name, type=ftype, range=(lo, hi), thresholds=thresholds))
        return defs


# ─────────────────────────────────────────────────────────────────────────────
# FPTM inspector
# ─────────────────────────────────────────────────────────────────────────────
def _class_table(tm_rules: dict[str, Any]) -> tuple[list[Any], dict[str, Any], str, str, str, str]:
    """Normalise the two ``class_rules`` layouts the pipeline emits."""
    if isinstance(tm_rules.get("classes"), list):
        labels = list(tm_rules["classes"])
        table = tm_rules["class_rules"]
        return labels, table, "positive_clauses", "negative_clauses", "include", "exclude"
    table = tm_rules["classes"]
    return list(table.keys()), table, "positive", "negative", "include", "include_inverted"


class FPTMInspector:
    """Extracts structure from a Paper 2 ``tm_rules`` dict (GLADE+FPTM bundle)."""

    def __init__(
        self,
        tm_rules: dict[str, Any],
        feature_inspector: FeatureInspector,
        class_names: Sequence[str] | None = None,
    ):
        self.tm_rules = tm_rules
        self.feature_inspector = feature_inspector
        labels, table, pos_key, neg_key, inc_key, exc_key = _class_table(tm_rules)
        self._labels = labels
        self._table = table
        self._pos_key, self._neg_key = pos_key, neg_key
        self._inc_key, self._exc_key = inc_key, exc_key
        self.class_names: list[str] = (
            [str(c) for c in class_names] if class_names is not None else [str(c) for c in labels]
        )
        if len(self.class_names) != len(labels):
            raise ValueError(
                f"class_names has {len(self.class_names)} entries, expected {len(labels)}"
            )
        self.n_bits = int(tm_rules.get("n_bits", feature_inspector.binariser.n_bits))
        self._config = dict(tm_rules.get("config") or {})

    # -- public API (mirrors tmatlas.inspectors.base.BaseInspector) -------
    def get_model_info(self) -> ModelInfo:
        task = "binary" if len(self.class_names) == 2 else "multiclass"
        return ModelInfo(type="classification", task=task, classes=list(self.class_names))

    def get_metadata(self) -> ModelMetadata:
        cfg = self._config
        return ModelMetadata(
            num_clauses_per_class=int(cfg.get("C", self._infer_clauses_per_class())),
            num_classes=len(self.class_names),
            num_literals=self.n_bits,
            T=cfg.get("T"),
            s=cfg.get("S"),
            L=cfg.get("L"),
            LF=cfg.get("LF"),
            epochs=cfg.get("EPOCHS"),
            weighted_clauses=False,
            created=datetime.now(timezone.utc).isoformat(),
            binariser=getattr(self.feature_inspector.binariser, "version", "GLADE"),
        )

    def get_clauses(self) -> list[Clause]:
        default_clamp = int(self._config.get("LF", 15) or 15)
        clauses: list[Clause] = []
        cid = 0
        for raw_label, cls_name in zip(self._labels, self.class_names):
            spec = self._table.get(raw_label)
            if spec is None:
                spec = self._table.get(str(raw_label), {})
            for pol_key, polarity in ((self._pos_key, "positive"), (self._neg_key, "negative")):
                for cl in spec.get(pol_key) or []:
                    clamp = int(cl.get("clamp") or default_clamp)
                    literals: list[Literal] = []
                    for bit in cl.get(self._inc_key) or []:
                        literals.append(self._literal(int(bit), GEQ))
                    for bit in cl.get(self._exc_key) or []:
                        literals.append(self._literal(int(bit), LT))
                    weight = ClauseWeight(
                        value=float(clamp) if polarity == "positive" else -float(clamp),
                        polarity=polarity,
                    )
                    clauses.append(
                        Clause(
                            id=cid,
                            cls=cls_name,
                            polarity=polarity,
                            clamp=clamp,
                            literals=literals,
                            weights={cls_name: weight},
                        )
                    )
                    cid += 1
        return clauses

    # -- helpers ---------------------------------------------------------
    def _literal(self, bit: int, operator: str) -> Literal:
        adapter = self.feature_inspector.binariser
        fi = int(adapter.feat_idx[bit])
        return Literal(
            feature=self.feature_inspector.feature_names[fi],
            operator=operator,
            threshold=float(adapter.thresh[bit]),
            bit=int(bit),
            l=int(bit) + 1,
            name=adapter.bit_name(bit),
        )

    def _infer_clauses_per_class(self) -> int:
        labels, table = self._labels, self._table
        if not labels:
            return 0
        spec = table.get(labels[0]) or table.get(str(labels[0]), {})
        return len(spec.get(self._pos_key) or []) + len(spec.get(self._neg_key) or [])


# ─────────────────────────────────────────────────────────────────────────────
# Exporters
# ─────────────────────────────────────────────────────────────────────────────
class JsonExporter:
    """Assembles the ``{model, metadata, features, clauses}`` document."""

    def __init__(self, inspector: FPTMInspector, feature_inspector: FeatureInspector):
        self.inspector = inspector
        self.feature_inspector = feature_inspector

    def export(self) -> dict[str, Any]:
        return {
            "model": self.inspector.get_model_info().to_dict(),
            "metadata": self.inspector.get_metadata().to_dict(),
            "features": [f.to_dict() for f in self.feature_inspector.extract()],
            "bits": self.bit_legend(),
            "clauses": [c.to_dict() for c in self.inspector.get_clauses()],
        }

    def bit_legend(self) -> list[dict[str, Any]]:
        """Legend mapping every GLADE bit to its names and meaning.

        One row per output bit: the global literal index ``l`` (1-based), the
        structured ``name`` ``f<i>_<k>`` (feature i, its k-th bit), the 0-based
        ``bit`` id, the original ``feature`` (name + 1-based index), the bit's
        ``localBit`` index within that feature, and its ``threshold`` (the bit is
        ``feature >= threshold``).
        """
        a = self.feature_inspector.binariser
        names = self.feature_inspector.feature_names
        legend: list[dict[str, Any]] = []
        for j in range(a.n_bits):
            fi = int(a.feat_idx[j])
            legend.append(
                {
                    "l": j + 1,
                    "name": a.bit_name(j),
                    "bit": j,
                    "feature": names[fi],
                    "featureIndex": fi + 1,
                    "localBit": int(a.local_idx[j]),
                    "threshold": float(a.thresh[j]),
                }
            )
        return legend


def clause_output_matrix(
    tm_rules: dict[str, Any], glade: Any, X_raw: NDArray[Any]
) -> NDArray[np.int32]:
    """Per-clause graded output for every row, shape ``(n_samples, n_clauses)``.

    Clause order matches :meth:`FPTMInspector.get_clauses`.  Each entry is
    ``max(clamp - violations, 0)`` (a binary "activated?" view is simply
    ``matrix > 0``).
    """
    adapter = glade if isinstance(glade, GLADEAdapter) else GLADEAdapter(glade)
    bits = adapter.transform(np.asarray(X_raw, dtype=np.float64)).astype(bool)  # (n, n_bits)
    labels, table, pos_key, neg_key, inc_key, exc_key = _class_table(tm_rules)
    default_clamp = int((tm_rules.get("config") or {}).get("LF", 15) or 15)
    n_bits = bits.shape[1]
    inc_masks, exc_masks, clamps = [], [], []
    for raw_label in labels:
        spec = table.get(raw_label) or table.get(str(raw_label), {})
        for pol_key in (pos_key, neg_key):
            for cl in spec.get(pol_key) or []:
                im = np.zeros(n_bits, bool)
                em = np.zeros(n_bits, bool)
                for b in cl.get(inc_key) or []:
                    im[int(b)] = True
                for b in cl.get(exc_key) or []:
                    em[int(b)] = True
                inc_masks.append(im)
                exc_masks.append(em)
                clamps.append(int(cl.get("clamp") or default_clamp))
    if not clamps:
        return np.zeros((bits.shape[0], 0), np.int32)
    inc = np.stack(inc_masks)  # (C, n_bits)
    exc = np.stack(exc_masks)  # (C, n_bits)
    clamp_arr = np.asarray(clamps, np.int32)  # (C,)
    # violations = (#include bits that are 0) + (#exclude bits that are 1)
    miss_inc = inc[None, :, :] & ~bits[:, None, :]
    hit_exc = exc[None, :, :] & bits[:, None, :]
    violations = miss_inc.sum(axis=2) + hit_exc.sum(axis=2)  # (n, C)
    return np.maximum(clamp_arr[None, :] - violations, 0).astype(np.int32)


def format_clause_activations(
    clause_outputs: NDArray[Any], key_name: str = "ActivatedClauses"
) -> list[dict[str, list[int]]]:
    """Per-sample list of clause ids with non-zero output (TMAtlas-style)."""
    arr = np.asarray(clause_outputs)
    if arr.ndim != 2:
        raise ValueError(f"clause_outputs must be a 2D array, got shape {arr.shape}")
    return [{key_name: np.flatnonzero(row > 0).astype(int).tolist()} for row in arr]


def export_predictions_csv(
    output_filename: str | Path,
    feature_names: Sequence[str],
    X_rows: NDArray[Any],
    y_actual: Iterable[Any],
    y_predicted: Iterable[Any],
    activated_clauses: Sequence[dict[str, list[int]]],
) -> Path:
    """Write one CSV row per sample: features, actual, predicted, activated clauses."""
    X_rows = np.asarray(X_rows)
    y_actual = list(np.asarray(list(y_actual)).ravel())
    y_predicted = list(np.asarray(list(y_predicted)).ravel())
    feat_cols = list(feature_names)
    extra_keys = sorted({k for d in activated_clauses for k in d})
    header = ["sample", *feat_cols, "actual", "predicted", *extra_keys]
    path = Path(output_filename)
    with path.open("w", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for i in range(X_rows.shape[0]):
            row: dict[str, Any] = {"sample": i}
            for j, col in enumerate(feat_cols):
                row[col] = X_rows[i, j]
            row["actual"] = y_actual[i] if i < len(y_actual) else ""
            row["predicted"] = y_predicted[i] if i < len(y_predicted) else ""
            for k in extra_keys:
                row[k] = ";".join(map(str, activated_clauses[i].get(k, [])))
            writer.writerow(row)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Bundle loading + one-call wrapper
# ─────────────────────────────────────────────────────────────────────────────
def load_bundle(source: str | Path | dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(tm_rules, glade_payload)`` from a bundle dict or a path.

    Accepts a ``{"tm_rules", "glade"}`` dict, a pickled bundle (``.pkl``), or a
    ``tm_rules.json`` path (a sibling ``glade.json`` is picked up automatically).
    """
    if isinstance(source, dict):
        bundle = source
    else:
        path = Path(source)
        if path.suffix == ".json":
            tm_rules = json.loads(path.read_text())
            glade_path = path.with_name("glade.json")
            if not glade_path.exists():
                raise FileNotFoundError(
                    f"{path} given but no sibling glade.json found at {glade_path}"
                )
            return tm_rules, json.loads(glade_path.read_text())
        with path.open("rb") as fh:
            bundle = pickle.load(fh)
        if not isinstance(bundle, dict):
            raise TypeError(f"{path}: expected a dict bundle, got {type(bundle).__name__}")

    if "tm_rules" not in bundle:
        raise KeyError("bundle is missing 'tm_rules'")
    tm_rules = bundle["tm_rules"]
    if "glade" in bundle:
        glade_payload = bundle["glade"]
    elif "binarizer" in bundle and hasattr(bundle["binarizer"], "to_dict"):
        glade_payload = bundle["binarizer"].to_dict()
    elif "binarizer_state" in bundle:
        glade_payload = bundle["binarizer_state"]
    else:
        raise KeyError("bundle has no 'glade' / 'binarizer' / 'binarizer_state' entry")
    return tm_rules, _glade_payload(glade_payload)


def build_atlas(
    source: str | Path | dict[str, Any],
    X: NDArray[Any] | None = None,
    feature_names: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Load a GLADE+FPTM bundle and return the TMAtlas-style export dict."""
    tm_rules, glade_payload = load_bundle(source)
    adapter = GLADEAdapter(glade_payload)
    feat_ins = FeatureInspector(adapter, X=X, feature_names=feature_names)
    inspector = FPTMInspector(tm_rules, feat_ins, class_names=class_names)
    return JsonExporter(inspector, feat_ins).export()
