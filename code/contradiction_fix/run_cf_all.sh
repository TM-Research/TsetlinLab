#!/bin/bash
# Regenerate the contradiction-free (cf_*) atlases for all 4 datasets using the inline
# contradiction-fix engine /IoT/Tsetlin.jl/src/Tsetlin_contradiction_fix.jl.
#   pipeline: GLADE-binarise -> train (128 threads) -> export TMAtlas JSON + predictions CSV
# Output: ../../data/cf_<name>_atlas.json  +  ../../data/cf_<name>_predictions.csv
HERE="$(cd "$(dirname "$0")" && pwd)"
DATA="$(cd "$HERE/../../data" && pwd)"
run() {
  name=$1; loader=$2; C=$3; T=$4; S=$5; L=$6; LF=$7; E=$8
  TMP=/tmp/glade_cf_$name
  echo "### $name : GLADE binarise"
  python3 "$HERE/prep_cf.py" "$loader" "$TMP" 2>&1 | tail -1 || { echo "$name PREP FAIL"; return; }
  echo "### $name : train contradiction-fix FPTM (128 threads)"
  TMP_DIR=$TMP TM_CLAUSES=$C TM_T=$T TM_S=$S TM_L=$L TM_LF=$LF TM_EPOCHS=$E TM_INCLUDE=128 TM_STATES=256 \
    julia --threads=128 "$HERE/train_tm_cf.jl" 2>&1 | tail -1 || { echo "$name TRAIN FAIL"; return; }
  echo "### $name : export atlas + predictions"
  (cd /IoT/Paper_2/src && python3 -m paper_2.atlas "$TMP/tm_rules.json" --dataset "$name" \
     -o "$DATA/cf_${name}_atlas.json" --csv "$DATA/cf_${name}_predictions.csv" 2>&1 | tail -1) || echo "$name EXPORT FAIL"
}
run nslkdd  load_nslkdd 90 12 200 40 8  80
run ton_iot load_toniot 100 15 20 25 8  200
run medsec  load_medsec 80 12 75 30 8  300
run wustl   load_wustl  60 8  300 50 15 200
echo "ALL CF DONE"
