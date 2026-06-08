#!/bin/bash
# Export Standard-binariser atlases (rules only, no predictions CSV) for the 3 engines x 4 datasets,
# so the Standard table on the site has clickable "view rules". Standard bits are feature>=threshold,
# so a GLADE-format glade.json is synthesised from bits.json and the normal atlas exporter is reused.
HERE="$(cd "$(dirname "$0")" && pwd)"; DATA="$(cd "$HERE/../../data" && pwd)"
CF=/IoT/Tsetlin.jl/src/Tsetlin_contradiction_fix.jl
PER=/IoT/external/perara-Tsetlin.jl/src/Tsetlin.jl
synth(){ python3 -c "import json;d=json.load(open('$1/bits.json'));fi=d['feat_idx'];th=d['thresh'];json.dump({'version':'GLADEv2','n_features_in':max(fi)+1,'n_bits':len(fi),'feat_idx':fi,'n_bins_param':15,'quantised':False,'thresh':th},open('$1/glade.json','w'))"; }
te(){ name=$1;eng=$2;excl=$3;tag=$4;E=$5;C=$6;T=$7;S=$8;L=$9;LF=${10};TMP=/tmp/std_$name
  TMP_DIR=$TMP TM_CLAUSES=$C TM_T=$T TM_S=$S TM_L=$L TM_LF=$LF TM_EPOCHS=$E TM_INCLUDE=128 TM_STATES=256 \
    ENGINE_PATH=$eng TM_EXCLUSIVE=$excl julia --threads=128 "$HERE/train_tm_cf.jl" 2>&1 | tail -1
  (cd /IoT/Paper_2/src && python3 -m paper_2.atlas "$TMP/tm_rules.json" --dataset "$name" -o "$DATA/std_${tag}_${name}_atlas.json" 2>&1 | tail -1); }
ds(){ name=$1;E=$2;C=$3;T=$4;S=$5;L=$6;LF=$7; synth /tmp/std_$name; echo "## $name (Standard)"
  te $name $PER false perara $E $C $T $S $L $LF
  te $name $PER true  pexcl  $E $C $T $S $L $LF
  te $name $CF  false cfix   $E $C $T $S $L $LF; }
ds nslkdd 80 90 12 200 40 8
ds ton_iot 120 100 15 20 25 8
ds medsec 150 80 12 75 30 8
ds wustl 200 60 8 300 50 15
echo "STD ATLASES DONE"
