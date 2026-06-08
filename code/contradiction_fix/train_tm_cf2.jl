#!/usr/bin/env julia
# ─── Train TM on pre-binarized data and export clauses as JSON ──
# Reads: /tmp/glade_benchmark/X_train.txt, Y_train.txt, X_test.txt, Y_test.txt
# Writes: /tmp/glade_benchmark/tm_rules.json (clauses for MCU inference)
#         /tmp/glade_benchmark/tm_metrics.json (accuracy + per-class F1 + timings)
#
# Hyperparameters are read from env vars so Python sets them from config.py.

using Printf, Dates, Statistics, Random, JSON
include("/IoT/FuzzyPatternTM/src/Tsetlin_CF.jl")   # Method A: thermometer-consistency projection over feature_groups
using .Tsetlin_CF: TMInput, TMClassifier, TATeam, train!, predict, vote

# ─── Read hyperparameters from env ──────────────────────────────
CLAUSES   = parse(Int, get(ENV, "TM_CLAUSES",   "80"))
T         = parse(Int, get(ENV, "TM_T",         "10"))
S         = parse(Int, get(ENV, "TM_S",         "61"))
L         = parse(Int, get(ENV, "TM_L",         "60"))
LF        = parse(Int, get(ENV, "TM_LF",        "10"))
EPOCHS    = parse(Int, get(ENV, "TM_EPOCHS",    "30"))
STATES    = parse(Int, get(ENV, "TM_STATES",    "256"))
INCLUDE   = parse(Int, get(ENV, "TM_INCLUDE",   "200"))
SEED      = parse(Int, get(ENV, "TM_SEED",      "42"))
TMP_DIR   = get(ENV, "TMP_DIR", "/tmp/glade_benchmark")

Random.seed!(SEED)

# ─── Load pre-binarized data ────────────────────────────────────
println("Loading binarized data from $TMP_DIR...")
x_train = let lines = readlines(joinpath(TMP_DIR, "X_train.txt"))
    [TMInput([parse(Bool, x) for x in split(l, " ")]) for l in lines]
end
x_test = let lines = readlines(joinpath(TMP_DIR, "X_test.txt"))
    [TMInput([parse(Bool, x) for x in split(l, " ")]) for l in lines]
end
y_train = [parse(Int32, l) for l in readlines(joinpath(TMP_DIR, "Y_train.txt"))]
y_test  = [parse(Int32, l) for l in readlines(joinpath(TMP_DIR, "Y_test.txt"))]

n_bits = length(x_train[1])
println("Samples: train=$(length(y_train)) test=$(length(y_test)), bits=$n_bits")
println("TM config: C=$CLAUSES T=$T S=$S L=$L LF=$LF E=$EPOCHS")

# ─── feature_groups from GLADE: contiguous bit range per source feature ─────
# Method A uses these to forbid thermometer contradictions (assert a high rung of a
# feature while negating a lower rung). Built from glade.json feat_idx (1-based ranges).
feat_idx = JSON.parsefile(joinpath(TMP_DIR, "glade.json"))["feat_idx"]
feature_groups = Tuple{Int64,Int64}[]
let s = 1
    for i in 2:length(feat_idx)
        if feat_idx[i] != feat_idx[i-1]; push!(feature_groups, (s, i-1)); s = i; end
    end
    push!(feature_groups, (s, length(feat_idx)))
end
println("feature_groups: $(length(feature_groups)) features (thermometer-consistency projection)")

# ─── Train ──────────────────────────────────────────────────────
tm = TMClassifier(x_train[1], y_train, CLAUSES, T, S,
                  L=L, LF=LF, states_num=STATES, include_limit=INCLUDE,
                  feature_groups=feature_groups)

t_fit = @elapsed begin
    tms = train!(tm, x_train, y_train, x_test, y_test, EPOCHS,
                 best_tms_size=1, shuffle=true, verbose=0)
end
best_tm, _ = tms[1]

# ─── Test predictions ───────────────────────────────────────────
t_pred = @elapsed begin
    y_pred = [predict(best_tm, x) for x in x_test]
end

# ─── Per-class metrics ──────────────────────────────────────────
classes = sort(unique(y_train))
n_classes = length(classes)

acc = sum(y_pred .== y_test) / length(y_test)
per_class = Dict{Int, Dict}()

macro_f1 = let mf1 = 0.0
    for c in classes
        tp = sum((y_pred .== c) .& (y_test .== c))
        fp = sum((y_pred .== c) .& (y_test .!= c))
        fn = sum((y_pred .!= c) .& (y_test .== c))
        tn = sum((y_pred .!= c) .& (y_test .!= c))
        p = tp + fp == 0 ? 0.0 : tp / (tp + fp)
        r = tp + fn == 0 ? 0.0 : tp / (tp + fn)
        f1 = p + r == 0.0 ? 0.0 : 2p * r / (p + r)
        mf1 += f1
        per_class[c] = Dict(
            "precision" => p, "recall" => r, "f1" => f1,
            "tp" => tp, "fp" => fp, "fn" => fn, "tn" => tn,
            "support" => tp + fn,
        )
    end
    mf1 / n_classes
end

# ─── Extract clauses as rules (for MCU inference) ───────────────
# For each class, we store pos/neg clause literal patterns
println("Extracting clauses as JSON rules...")

function bits_to_bool_array(chunks, total_bits::Int)
    out = falses(total_bits)
    for i in 1:total_bits
        chunk_idx = ((i - 1) >>> 6) + 1
        bit_idx = (i - 1) & 63
        if chunk_idx <= length(chunks)
            out[i] = ((chunks[chunk_idx] >> bit_idx) & UInt64(1)) == 1
        end
    end
    return out
end

rules = Dict{String, Any}()
rules["n_bits"]    = n_bits
rules["n_classes"] = n_classes
rules["classes"]   = [c for c in classes]
rules["config"]    = Dict("C"=>CLAUSES, "T"=>T, "S"=>S, "L"=>L, "LF"=>LF,
                          "EPOCHS"=>EPOCHS)

class_rules = Dict{String, Any}()
for (class_id, ta) in best_tm.clauses
    n_clauses_per_pol = length(ta.positive_included_literals_sum_clamp)

    pos_clauses = []
    neg_clauses = []

    for i in 1:n_clauses_per_pol
        # Positive polarity clause (class vote)
        pos_inc = bits_to_bool_array(ta.positive_included_literals[:, i], n_bits)
        pos_inc_inv = bits_to_bool_array(ta.positive_included_literals_inverted[:, i], n_bits)
        pos_lit_sum = ta.positive_included_literals_sum[i]
        pos_lit_sum_clamp = ta.positive_included_literals_sum_clamp[i]

        pos_idx = findall(pos_inc)           # literals: x_j == 1
        pos_inv_idx = findall(pos_inc_inv)   # literals: x_j == 0 (NOT x_j)

        push!(pos_clauses, Dict(
            "include" => pos_idx .- 1,   # 0-indexed for Python
            "exclude" => pos_inv_idx .- 1,
            "clamp" => pos_lit_sum_clamp,
            "total" => pos_lit_sum,
        ))

        # Negative polarity clause (against class)
        neg_inc = bits_to_bool_array(ta.negative_included_literals[:, i], n_bits)
        neg_inc_inv = bits_to_bool_array(ta.negative_included_literals_inverted[:, i], n_bits)
        neg_lit_sum = ta.negative_included_literals_sum[i]
        neg_lit_sum_clamp = ta.negative_included_literals_sum_clamp[i]

        neg_idx = findall(neg_inc)
        neg_inv_idx = findall(neg_inc_inv)

        push!(neg_clauses, Dict(
            "include" => neg_idx .- 1,
            "exclude" => neg_inv_idx .- 1,
            "clamp" => neg_lit_sum_clamp,
            "total" => neg_lit_sum,
        ))
    end

    class_rules[string(class_id)] = Dict(
        "positive_clauses" => pos_clauses,
        "negative_clauses" => neg_clauses,
    )
end
rules["class_rules"] = class_rules

# ─── Save outputs ───────────────────────────────────────────────
mkpath(TMP_DIR)
open(joinpath(TMP_DIR, "tm_rules.json"), "w") do io
    JSON.print(io, rules)
end

metrics = Dict(
    "accuracy" => acc,
    "macro_f1" => macro_f1,
    "per_class" => per_class,
    "timings" => Dict(
        "fit" => t_fit,
        "predict" => t_pred,
        "per_sample_predict_us" => t_pred / length(y_test) * 1e6,
    ),
)

open(joinpath(TMP_DIR, "tm_metrics.json"), "w") do io
    JSON.print(io, metrics)
end

# Print summary
@printf("Done. Acc=%.4f MacroF1=%.4f fit=%.2fs pred=%.3fs\n",
        acc, macro_f1, t_fit, t_pred)
