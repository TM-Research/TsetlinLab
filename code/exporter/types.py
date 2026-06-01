"""Serialisable record types for the GLADE+FPTM atlas export.

These mirror the dataclasses used by ``sonnets-project/TMAtlas`` so the JSON
shape stays familiar, with a few additions that the Fuzzy Pattern Tsetlin
Machine needs (per-clause polarity, the literal-sum ``clamp``).  Each record
exposes ``to_dict()`` returning JSON-ready primitives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Literal:
    """A single included literal: ``feature operator threshold``.

    ``bit`` is the GLADE output-bit id this literal refers to.  GLADE turns each
    original feature into up to 15 bits (one per bin threshold), so several
    literals in a clause can share the same ``feature`` name while pointing at
    different ``bit`` ids / thresholds — the ``bit`` keeps each one uniquely and
    correctly identified.
    """

    feature: str
    operator: str  # "≥" for an asserted bit, "<" for a negated bit
    threshold: float
    bit: int = -1  # GLADE output-bit id, 0-based
    l: int = -1  # global literal index, 1-based (l1, l2, …) = bit + 1
    name: str = ""  # structured bit name "f<i>_<k>" = feature i (1-based), its k-th bit

    def to_dict(self) -> dict[str, Any]:
        return {
            "l": int(self.l),
            "name": self.name,
            "bit": int(self.bit),
            "feature": self.feature,
            "operator": self.operator,
            "threshold": float(self.threshold),
        }


@dataclass(frozen=True, slots=True)
class ClauseWeight:
    """Vote contribution of a clause towards one class."""

    value: float
    polarity: str  # "positive" | "negative"

    def to_dict(self) -> dict[str, Any]:
        return {"value": float(self.value), "polarity": self.polarity}


@dataclass(frozen=True, slots=True)
class Clause:
    """One FPTM clause (a clause belongs to exactly one class and polarity).

    ``literals`` lists only the *included* literals.  ``clamp`` is the
    fuzzy-pattern literal-sum cap: the clause emits ``max(clamp - v, 0)`` for a
    sample with ``v`` violated literals, so the clause vote lies in ``[0, clamp]``.
    ``weights`` is kept TMAtlas-shaped (``{class_name: ClauseWeight}``) and holds
    only the owning class.

    ``nLiterals`` counts bits; ``nFeatures`` counts the *distinct* original
    features those bits come from — i.e. how many features the clause really
    uses, since one feature spans up to 15 bits — and ``features`` lists them.

    ``featureBands`` is the human-readable view: the clause's single-bit literals
    grouped by feature and collapsed into one value band each (one feature → one
    band), which is how a feature is really represented in the rule.
    """

    id: int
    cls: str
    polarity: str  # "positive" | "negative"
    clamp: int
    literals: list[Literal]
    weights: dict[str, ClauseWeight]

    def features_used(self) -> list[str]:
        """Distinct feature names referenced by this clause, in sorted order."""
        return sorted({lit.feature for lit in self.literals})

    def feature_bands(self) -> list[dict[str, Any]]:
        """Per-feature **thermometer** conditions for this clause.

        GLADE is a thermometer binariser: every bit is ``feature ≥ threshold``.
        So every condition in a rule is phrased as ``feature ≥ threshold`` — a
        rule never mixes ``<`` and ``≥``.  An asserted (included) bit must be
        **ON** (the value reaches the threshold); a negated (excluded) bit must
        be **OFF** (it does not reach it).

        Thermometer monotonicity removes redundant bits: among the ON thresholds
        only the **largest** binds (``onThreshold``); among the OFF thresholds
        only the **smallest** binds (``offThreshold``).  ``impossible``/``empty``
        flags a feature required to be both ``≥ on`` (ON) and ``≥ off`` (OFF)
        with ``on ≥ off`` — no value can do both.  ``conditions`` lists the
        binding thermometer tests (each ``feature ≥ threshold`` with a ``state``
        of ``on``/``off``); ``lower``/``upper`` keep the old numeric bounds for
        backward compatibility (``lower`` = ON edge, ``upper`` = OFF edge).
        """

        def g(v: float) -> str:
            return f"{v:g}"

        def _txt(feat: str, thr: float, state: str) -> str:
            return f"{feat} ≥ {g(thr)} ({'ON' if state == 'on' else 'OFF'})"

        def _plain(feat: str, thr: float, state: str) -> str:
            if state == "on":
                return f"{feat} reaches {g(thr)} (its “{feat} ≥ {g(thr)}” test is ON)"
            return f"{feat} does not reach {g(thr)} (its “{feat} ≥ {g(thr)}” test is OFF)"

        groups: dict[str, list[Literal]] = {}
        for lit in self.literals:
            groups.setdefault(lit.feature, []).append(lit)

        bands: list[dict[str, Any]] = []
        for feat in sorted(groups):
            lits = sorted(groups[feat], key=lambda x: x.threshold)
            on = [x.threshold for x in lits if x.operator == "≥"]  # bits required ON
            off = [x.threshold for x in lits if x.operator != "≥"]  # bits required OFF
            on_thr = max(on) if on else None  # binding ON  (smaller ON bits implied)
            off_thr = min(off) if off else None  # binding OFF (larger OFF bits implied)
            impossible = on_thr is not None and off_thr is not None and on_thr >= off_thr

            conditions: list[dict[str, Any]] = []
            if on_thr is not None:
                conditions.append(
                    {"feature": feat, "operator": "≥", "threshold": float(on_thr), "state": "on"}
                )
            if off_thr is not None:
                conditions.append(
                    {"feature": feat, "operator": "≥", "threshold": float(off_thr), "state": "off"}
                )

            if impossible:
                direction = "impossible"
                text = (
                    f"{_txt(feat, on_thr, 'on')} and {_txt(feat, off_thr, 'off')} "
                    f"— impossible ({g(on_thr)} ≥ {g(off_thr)})"
                )
                plain = (
                    f"{feat} would have to reach {g(on_thr)} yet not reach "
                    f"{g(off_thr)} at the same time — impossible, so this part "
                    f"never fully matches"
                )
            else:
                direction = (
                    "on+off"
                    if (on_thr is not None and off_thr is not None)
                    else "on"
                    if on_thr is not None
                    else "off"
                    if off_thr is not None
                    else "any"
                )
                text = " and ".join(_txt(c["feature"], c["threshold"], c["state"]) for c in conditions) or feat
                plain = " and ".join(_plain(c["feature"], c["threshold"], c["state"]) for c in conditions) or feat

            members = [("¬" if x.operator != "≥" else "") + x.name for x in lits]
            bands.append(
                {
                    "feature": feat,
                    "nBits": len(lits),
                    "conditions": conditions,
                    "onThreshold": float(on_thr) if on_thr is not None else None,
                    "offThreshold": float(off_thr) if off_thr is not None else None,
                    "lower": float(on_thr) if on_thr is not None else None,  # back-compat (ON edge)
                    "upper": float(off_thr) if off_thr is not None else None,  # back-compat (OFF edge)
                    "empty": bool(impossible),
                    "impossible": bool(impossible),
                    "direction": direction,
                    "text": text,
                    "plain": plain,
                    "members": members,
                }
            )
        return bands

    def to_dict(self) -> dict[str, Any]:
        feats = self.features_used()
        return {
            "id": int(self.id),
            "class": self.cls,
            "polarity": self.polarity,
            "clamp": int(self.clamp),
            "nLiterals": len(self.literals),
            "nFeatures": len(feats),
            "features": feats,
            "featureBands": self.feature_bands(),
            "literals": [lit.to_dict() for lit in self.literals],
            "weights": {k: w.to_dict() for k, w in self.weights.items()},
        }


@dataclass(frozen=True, slots=True)
class FeatureClass:
    """Definition of one original (pre-binarisation) input feature."""

    name: str
    type: str  # "binary" | "continuous"
    range: tuple[float, float]
    thresholds: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "range": [float(self.range[0]), float(self.range[1])],
            "thresholds": [float(t) for t in self.thresholds],
        }


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """High-level description of the model."""

    type: str  # "classification"
    task: str  # "binary" | "multiclass"
    classes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "task": self.task, "classes": list(self.classes)}


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Training hyper-parameters and shape of the FPTM."""

    num_clauses_per_class: int
    num_classes: int
    num_literals: int
    T: float | None = None
    s: float | None = None
    L: int | None = None
    LF: int | None = None
    epochs: int | None = None
    weighted_clauses: bool = False
    created: str | None = None
    binariser: str = "GLADE"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "numClausesPerClass": int(self.num_clauses_per_class),
            "numClauses": int(self.num_clauses_per_class) * int(self.num_classes),
            "numClasses": int(self.num_classes),
            "numLiterals": int(self.num_literals),
            "T": self.T,
            "s": self.s,
            "weightedClauses": bool(self.weighted_clauses),
            "binariser": self.binariser,
            "created": self.created,
        }
        if self.L is not None:
            d["L"] = int(self.L)
        if self.LF is not None:
            d["LF"] = int(self.LF)
        if self.epochs is not None:
            d["epochs"] = int(self.epochs)
        return d
