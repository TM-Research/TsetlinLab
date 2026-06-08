#!/usr/bin/env python3
"""Train vanilla TMU on 1-bit MNIST and dump clauses as tm_rules.json (+ count impossible).
Params (user): clauses=1000/class, T=750, s=3, max_included_literals=None, epochs=40, 1 bit/pixel.
Impossible clause = includes both pixel-k (asserted) and ¬pixel-k (negated) — same-bit, the only
kind possible when each feature is a single bit."""
import numpy as np, json, time, sys
from tmu.models.classification.vanilla_classifier import TMClassifier

TMP = "/tmp/std_mnist"; NF = 784
Xtr = np.loadtxt(f"{TMP}/X_train.txt", dtype=np.uint32)
Xte = np.loadtxt(f"{TMP}/X_test.txt",  dtype=np.uint32)
ytr = np.loadtxt(f"{TMP}/Y_train.txt", dtype=np.uint32)
yte = np.loadtxt(f"{TMP}/Y_test.txt",  dtype=np.uint32)
C, T, S, EP = 1000, 750, 3.0, 40
print(f"TMU 1-bit MNIST: train{Xtr.shape} test{Xte.shape}  C={C} T={T} s={S} max_lits=None epochs={EP}", flush=True)

tm = TMClassifier(C, T, S, max_included_literals=None, weighted_clauses=False,
                  feature_negation=True, platform="CPU", seed=42)
t0 = time.time()
for e in range(EP):
    tm.fit(Xtr, ytr)
    if e % 10 == 9: print(f"  epoch {e+1}/{EP}  acc={ (tm.predict(Xte)==yte).mean()*100:.2f}%  ({time.time()-t0:.0f}s)", flush=True)
acc = (tm.predict(Xte) == yte).mean() * 100
print(f"TMU trained: acc={acc:.2f}%  ({time.time()-t0:.0f}s)", flush=True)

def states(cb):
    ncl, SB, nch = cb.number_of_clauses, cb.number_of_state_bits_ta, cb.number_of_ta_chunks
    arr = np.asarray(cb.clause_bank).reshape(ncl, nch, SB)
    st = np.zeros((ncl, 2*NF), dtype=int)
    for ta in range(2*NF):
        ch, pos = ta // 32, ta % 32
        s = np.zeros(ncl, dtype=int)
        for b in range(SB): s += (((arr[:, ch, b] >> pos) & 1).astype(int)) << b
        st[:, ta] = s
    return st, (1 << (SB - 1))

classes = list(range(10))
class_rules = {}; imp_total = 0; nlit_total = 0; nclause = 0
for c in classes:
    cb = tm.clause_banks[c]; st, lim = states(cb); ncl = cb.number_of_clauses; npos = ncl // 2
    pos, neg = [], []
    for j in range(ncl):
        inc = [k for k in range(NF) if st[j, k]      >= lim]   # pixel k asserted (≥1)
        exc = [k for k in range(NF) if st[j, NF + k] >= lim]   # pixel k negated  (<1)
        contradictory = bool(set(inc) & set(exc))              # same bit asserted AND negated
        if contradictory: imp_total += 1
        nlit_total += len(inc) + len(exc); nclause += 1
        rule = {"include": inc, "exclude": exc, "clamp": 1, "total": len(inc) + len(exc)}
        (pos if j < npos else neg).append(rule)
    class_rules[str(c)] = {"positive_clauses": pos, "negative_clauses": neg}

rules = {"n_bits": NF, "n_classes": 10, "classes": classes,
         "config": {"C": C, "T": T, "S": S, "L": NF, "LF": 1, "EPOCHS": EP, "engine": "TMU unweighted-crisp"},
         "class_rules": class_rules}
json.dump(rules, open(f"{TMP}/tmu_rules.json", "w"))
print(f"clauses={nclause}  avg literals/clause={nlit_total/nclause:.1f}  IMPOSSIBLE (same-bit)={imp_total}", flush=True)
print("wrote", f"{TMP}/tmu_rules.json", flush=True)
