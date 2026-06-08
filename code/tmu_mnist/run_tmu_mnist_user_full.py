#!/usr/bin/env python3
"""TMU on FULL 1-bit MNIST (60k/10k) with the USER's exact params:
C=1000, T=750, s=3, max_included_literals=None, epochs=40, 1 bit/pixel (>0.3).
Reports accuracy + same-bit impossible-clause count."""
import numpy as np, json, time
from tmu.models.classification.vanilla_classifier import TMClassifier

NF = 784
d = np.load("/tmp/mnist/mnist.npz")
b03 = lambda a: (a.reshape(a.shape[0], -1).astype(np.float32) / 255.0 > 0.3).astype(np.uint32)
Xtr, Xte = b03(d["x_train"]), b03(d["x_test"])          # FULL 60000 / 10000
ytr, yte = d["y_train"].astype(np.uint32), d["y_test"].astype(np.uint32)
C, T, S, EP = 1000, 750, 3.0, 40
print(f"TMU FULL 1-bit MNIST (user params): train{Xtr.shape} test{Xte.shape}  C={C} T={T} s={S} max_lits=None epochs={EP}", flush=True)

tm = TMClassifier(C, T, S, max_included_literals=None, weighted_clauses=False,
                  feature_negation=True, platform="CPU", seed=42)
t0 = time.time(); best = 0
for e in range(EP):
    tm.fit(Xtr, ytr)
    if e % 10 == 9:
        acc = (tm.predict(Xte) == yte).mean() * 100; best = max(best, acc)
        print(f"  epoch {e+1}/{EP}  acc={acc:.2f}%  best={best:.2f}%  ({time.time()-t0:.0f}s)", flush=True)
acc = (tm.predict(Xte) == yte).mean() * 100
print(f"TMU FULL (user params) trained: acc={acc:.2f}%  best={max(best,acc):.2f}%  ({time.time()-t0:.0f}s)", flush=True)

def states(cb):
    ncl, SB, nch = cb.number_of_clauses, cb.number_of_state_bits_ta, cb.number_of_ta_chunks
    arr = np.asarray(cb.clause_bank).reshape(ncl, nch, SB); st = np.zeros((ncl, 2*NF), dtype=int)
    for ta in range(2*NF):
        ch, pos = ta // 32, ta % 32; s = np.zeros(ncl, dtype=int)
        for b in range(SB): s += (((arr[:, ch, b] >> pos) & 1).astype(int)) << b
        st[:, ta] = s
    return st, (1 << (SB - 1))

imp = 0; nlit = 0; ncl_tot = 0; class_rules = {}
for c in range(10):
    cb = tm.clause_banks[c]; st, lim = states(cb); ncl = cb.number_of_clauses; npos = ncl // 2
    pos, neg = [], []
    for j in range(ncl):
        inc = [k for k in range(NF) if st[j, k] >= lim]; exc = [k for k in range(NF) if st[j, NF+k] >= lim]
        if set(inc) & set(exc): imp += 1
        nlit += len(inc) + len(exc); ncl_tot += 1
        (pos if j < npos else neg).append({"include": inc, "exclude": exc, "clamp": 1, "total": len(inc)+len(exc)})
    class_rules[str(c)] = {"positive_clauses": pos, "negative_clauses": neg}
json.dump({"n_bits": NF, "n_classes": 10, "classes": list(range(10)),
           "config": {"C": C, "T": T, "S": S, "L": NF, "LF": 1, "EPOCHS": EP, "engine": "TMU unweighted-crisp"},
           "class_rules": class_rules}, open("/tmp/mnist/tmu_user_full_rules.json", "w"))
print(f"clauses={ncl_tot}  avg literals/clause={nlit/ncl_tot:.1f}  IMPOSSIBLE (same-bit)={imp}", flush=True)
