#!/bin/bash
# Verify both same-bit fixes under the STANDARD (TMU-style thermometer) binariser, all 4 datasets.
#   perara base (no guard) | perara exclusive guard | Tsetlin_contradiction_fix.jl
# Counts same-bit AND thermometer contradictions directly from the trained masks.
HERE="$(cd "$(dirname "$0")" && pwd)"
CF=/IoT/Tsetlin.jl/src/Tsetlin_contradiction_fix.jl
PER=/IoT/external/perara-Tsetlin.jl/src/Tsetlin.jl
run() {
  name=$1; loader=$2; C=$3; T=$4; S=$5; L=$6; LF=$7; E=$8
  TMP=/tmp/std_$name
  [ -f "$TMP/X_train.txt" ] || python3 "$HERE/prep_std.py" "$loader" "$TMP" 2>&1 | tail -1
  base="TMP_DIR=$TMP TM_CLAUSES=$C TM_T=$T TM_S=$S TM_L=$L TM_LF=$LF TM_EPOCHS=$E TM_INCLUDE=128 TM_STATES=256"
  echo "## $name"
  env $base ENGINE_PATH=$PER TM_EXCLUSIVE=false TAG="$name perara-base " julia --threads=128 "$HERE/check_binariser.jl" 2>&1 | grep "\["
  env $base ENGINE_PATH=$PER TM_EXCLUSIVE=true  TAG="$name perara-excl " julia --threads=128 "$HERE/check_binariser.jl" 2>&1 | grep "\["
  env $base ENGINE_PATH=$CF  TM_EXCLUSIVE=false TAG="$name cfix       " julia --threads=128 "$HERE/check_binariser.jl" 2>&1 | grep "\["
}
run nslkdd  load_nslkdd 90 12 200 40 8  80
run ton_iot load_toniot 100 15 20 25 8  120
run medsec  load_medsec 80 12 75 30 8  150
run wustl   load_wustl  60 8  300 50 15 200
echo "STD CHECK DONE"
