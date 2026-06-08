#!/usr/bin/env python3
"""TMU on the FULL 1-bit MNIST (60k train / 10k test) with a PROPER config.
Why the earlier run was low: s=3 (far too small) makes clauses absorb ~333 literals each and
accuracy *fell* every epoch. Here s=200, T=60, max_included_literals=32 — the repo's tested setup.
Reports accuracy + same-bit impossible-clause count over all clauses."""
import numpy as np, json, time
from tmu.models.classification.vanilla_classifier import TMClassifier

NF = 784
d = np.load("/tmp/mnist/mnist.npz")
b03 = lambda a: (a.reshape(a.shape[0], -1).astype(np.float32) / 255.0 > 0.3).astype(np.uint32)
Xtr, Xte = b03(d["x_train"]), b03(d["x_test"])          # FULL 60000 / 10000
ytr, yte = d["y_train"].astype(np.uint32), d["y_test"].astype(np.uint32)
C, T, S, L, EP = 1000, 60, 200.0, 32, 60
print(f"TMU FULL 1-bit MNIST: train{Xtr.shape} test{Xte.shape}  C={C} T={T} s={S} max_lits={L} epochs={EP}", flush=True)

tm = TMClassifier(C, T, S, max_included_literals=L, weighted_clauses=False,
                  feature_negation=True, platform="CPU", seed=42)
t0 = time.time(); best = 0
for e in range(EP):
    tm.fit(Xtr, ytr)
    if e % 5 == 4:
        acc = (tm.predict(Xte) == yte).mean() * 100; best = max(best, acc)
        print(f"  epoch {e+1}/{EP}  acc={acc:.2f}%  best={best:.2f}%  ({time.time()-t0:.0f}s)", flush=True)
acc = (tm.predict(Xte) == yte).mean() * 100
print(f"TMU FULL trained: acc={acc:.2f}%  best={max(best,acc):.2f}%  ({time.time()-t0:.0f}s)", flush=True)

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
           "config": {"C": C, "T": T, "S": S, "L": L, "LF": 1, "EPOCHS": EP, "engine": "TMU unweighted-crisp"},
           "class_rules": class_rules}, open("/tmp/mnist/tmu_full_rules.json", "w"))
print(f"clauses={ncl_tot}  avg literals/clause={nlit/ncl_tot:.1f}  IMPOSSIBLE (same-bit)={imp}", flush=True)
