#!/bin/bash
# Run the BASE perara/Tsetlin.jl engine (no contradiction fix) on all 4 datasets, same
# pipeline as the cf_ models. Engine: /IoT/external/perara-Tsetlin.jl/src/Tsetlin.jl
# (cloned from https://github.com/perara/Tsetlin.jl). This is the baseline the cf_ models
# fix: identical engine, but WITHOUT the l&=~li masking, so it still produces impossible
# conditions — the side-by-side shows the fix removes them at equal accuracy.
# Output: ../../data/perara_<name>_atlas.json + ../../data/perara_<name>_predictions.csv
HERE="$(cd "$(dirname "$0")" && pwd)"
CF="$HERE/../contradiction_fix"
DATA="$(cd "$HERE/../../data" && pwd)"
export ENGINE_PATH=/IoT/external/perara-Tsetlin.jl/src/Tsetlin.jl
run() {
  name=$1; loader=$2; C=$3; T=$4; S=$5; L=$6; LF=$7; E=$8
  TMP=/tmp/glade_cf_$name
  [ -f "$TMP/X_train.txt" ] || python3 "$CF/prep_cf.py" "$loader" "$TMP" 2>&1 | tail -1
  echo "### $name : train BASE perara engine (128 threads)"
  TMP_DIR=$TMP TM_CLAUSES=$C TM_T=$T TM_S=$S TM_L=$L TM_LF=$LF TM_EPOCHS=$E TM_INCLUDE=128 TM_STATES=256 \
    julia --threads=128 "$CF/train_tm_cf.jl" 2>&1 | tail -1
  echo "### $name : export atlas"
  (cd /IoT/Paper_2/src && python3 -m paper_2.atlas "$TMP/tm_rules.json" --dataset "$name" \
     -o "$DATA/perara_${name}_atlas.json" --csv "$DATA/perara_${name}_predictions.csv" 2>&1 | tail -1)
}
run nslkdd  load_nslkdd 90 12 200 40 8  80
run ton_iot load_toniot 100 15 20 25 8  200
run medsec  load_medsec 80 12 75 30 8  300
run wustl   load_wustl  60 8  300 50 15 200
echo "ALL PERARA DONE"
