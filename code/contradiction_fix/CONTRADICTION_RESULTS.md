# Contradiction verification — results

Two kinds of impossible condition (both flagged by the viewer):
- **same-bit**: a clause asserts and negates the *same* bit (`x ≥ t AND x < t`).
- **thermometer**: a clause asserts a *high* rung of a feature and negates a *lower* rung
  (`feat ≥ 100 AND feat < 50`) — *different* bits, same feature.

Two same-bit fixes were checked, both of which only resolve overlapping (same) bits:
- **`Tsetlin_contradiction_fix.jl`** — inline `l &= ~li` at every mask rebuild.
- **perara/Tsetlin.jl `exclusive` guard** — `exclusive_resolve!` (drop the weaker overlapping bit).

Counts are **clauses containing ≥1 impossible condition** (the viewer's definition),
measured directly from the trained masks.

## 1-bit MNIST (1 bit per feature → no thermometer rungs)
| engine | acc | same-bit | thermometer |
|--------|-----|----------|-------------|
| perara base | 97.34% | 1 | 1 |
| perara `exclusive` | 97.32% | **0** | **0** |
| `Tsetlin_contradiction_fix` | 97.28% | **0** | **0** |

→ With 1 bit per feature, **both fixes reach a true 0** — there are no cross-threshold rungs to contradict.

## Standard (TMU-style thermometer) binariser
| dataset | engine | acc | same-bit | thermometer |
|---------|--------|-----|----------|-------------|
| nslkdd | base | 99.44% | 7 | 62 |
| nslkdd | exclusive | 99.41% | **0** | 61 |
| nslkdd | cfix | 99.44% | **0** | 53 |
| ton_iot | base | 82.14% | 219 | 527 |
| ton_iot | exclusive | 84.33% | **0** | 533 |
| ton_iot | cfix | 83.49% | **0** | 516 |
| medsec | base | 90.82% | 5 | 34 |
| medsec | exclusive | 90.74% | **0** | 29 |
| medsec | cfix | 90.78% | **0** | 28 |
| wustl | base | 94.21% | 34 | 132 |
| wustl | exclusive | 94.12% | **0** | 123 |
| wustl | cfix | 94.33% | **0** | 127 |

→ The Standard binariser is **also thermometer**, so same-bit fixes remove all same-bit
contradictions but **leave thermometer ones**. (Note: Standard bits ≫ GLADE bits and accuracy is
much lower on TON_IoT/MedSec — GLADE remains the better binariser.)

## GLADE binariser (for reference, `Tsetlin_contradiction_fix.jl`)
| dataset | same-bit | thermometer |
|---------|----------|-------------|
| wustl | 0 | 156 |
| nslkdd | 0 | 101 |
| ton_iot | 0 | 236 |
| medsec | 0 | 17 |

## Conclusion
- `Tsetlin_contradiction_fix.jl` ≡ perara `exclusive` guard: both are **same-bit** fixes and behave identically.
- They are **fully correct** when the binariser is 1-bit-per-feature (e.g. 1-bit MNIST) → 0 impossible.
- On any **thermometer** binariser (GLADE, Standard), thermometer contradictions remain; removing them
  needs **feature-group** awareness (which is a separate design choice, not enabled here).

Reproduce: `prep_std.py` / `prep_cf.py` (binarise) → `check_binariser.jl` (train + count), driven by
`run_std_check.sh`.
