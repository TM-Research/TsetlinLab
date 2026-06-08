#!/bin/bash
# Regenerate cf_* atlases with the FEATURE-GROUP-AWARE contradiction-free engine
# /IoT/FuzzyPatternTM/src/Tsetlin_CF.jl (Method A): removes BOTH same-bit and GLADE
# thermometer (cross-threshold same-feature) contradictions -> 0 impossible in the viewer.
HERE="$(cd "$(dirname "$0")" && pwd)"
DATA="$(cd "$HERE/../../data" && pwd)"
run() {
  name=$1; loader=$2; C=$3; T=$4; S=$5; L=$6; LF=$7; E=$8
  TMP=/tmp/glade_cf_$name
  [ -f "$TMP/X_train.txt" ] || python3 "$HERE/prep_cf.py" "$loader" "$TMP" 2>&1 | tail -1
  echo "### $name : train Tsetlin_CF (Method A, feature_groups, 128 threads)"
  TMP_DIR=$TMP TM_CLAUSES=$C TM_T=$T TM_S=$S TM_L=$L TM_LF=$LF TM_EPOCHS=$E TM_INCLUDE=200 TM_STATES=256 \
    julia --threads=128 "$HERE/train_tm_cf2.jl" 2>&1 | grep -iE "feature_groups|Acc="
  echo "### $name : export atlas"
  (cd /IoT/Paper_2/src && python3 -m paper_2.atlas "$TMP/tm_rules.json" --dataset "$name" \
     -o "$DATA/cf_${name}_atlas.json" --csv "$DATA/cf_${name}_predictions.csv" 2>&1 | tail -1)
}
run nslkdd  load_nslkdd 90 12 200 40 8  80
run ton_iot load_toniot 100 15 20 25 8  200
run medsec  load_medsec 80 12 75 30 8  300
run wustl   load_wustl  60 8  300 50 15 200
echo "ALL CF2 DONE"
