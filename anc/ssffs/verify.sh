#!/bin/sh
# verify.sh - verifies that the standalone sSFFS reproduces the FST4
# reference implementation exactly, in the two criterion configurations
# used in the paper's experiments.
#
# Usage:  FST_BIN=path/to/fst4 DATA_DIR=path/to/data sh verify.sh [workdir]
#
#   FST_BIN   the fst4 CLI binary (built from FST4 under the fixed release
#             preset: -O2 -funroll-loops -ffp-contract=off -DNDEBUG)
#   DATA_DIR  directory holding madelon_train_nips_500.trn and
#             reuters_apte.arff
#
# For each configuration the script runs both implementations at the same
# seed and compares byte for byte:
#   - the per-improvement solution trace  ({"event":"solution",...}),
#   - the step-order diagnostic trace     (F/B order lines),
#   - the final subset, criterion value (%.12g) and evaluation count.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${1:-"$HERE/verify_work"}
SEED=${SEED:-1}
FST_BIN=${FST_BIN:?set FST_BIN to the fst4 binary}
DATA_DIR=${DATA_DIR:?set DATA_DIR to the data directory}
mkdir -p "$WORK"

echo "building ssffs (fixed FP flags)..."
c++ -std=c++17 -O2 -funroll-loops -ffp-contract=off -DNDEBUG -o "$WORK/ssffs" "$HERE/ssffs.cpp"

if [ ! -f "$WORK/madelon_train_nips_500.arff" ]; then
	echo "converting madelon to ARFF..."
	python3 "$HERE/trn2arff.py" "$DATA_DIR/madelon_train_nips_500.trn" "$WORK/madelon_train_nips_500.arff"
fi

compare() {
	name=$1
	grep '^{"event"' "$WORK/fst4_$name.err" >"$WORK/fst4_$name.solutions"
	grep '^{"event"' "$WORK/ssffs_$name.err" >"$WORK/ssffs_$name.solutions"
	grep '^[FB] order' "$WORK/fst4_$name.err" >"$WORK/fst4_$name.orders"
	grep '^[FB] order' "$WORK/ssffs_$name.err" >"$WORK/ssffs_$name.orders"
	python3 - "$WORK" "$name" <<'EOF'
import json, sys
work, name = sys.argv[1], sys.argv[2]
a = json.load(open(f"{work}/fst4_{name}.out"))
b = json.load(open(f"{work}/ssffs_{name}.out"))
ar = a["result"]
fields = {
    "features": (ar["features"], b["features"]),
    "value":    ("%.12g" % ar["value"], "%.12g" % b["value"]),
    "evals":    (a.get("stats", {}).get("evaluations"), b["evaluations"]),
}
ok = True
for k, (x, y) in fields.items():
    if x != y:
        ok = False
        print(f"  MISMATCH {k}: fst4={x} ssffs={y}")
if ok:
    print(f"  final result: value={fields['value'][0]} size={len(ar['features'])} evaluations={fields['evals'][0]} -- MATCH")
sys.exit(0 if ok else 1)
EOF
	if cmp -s "$WORK/fst4_$name.solutions" "$WORK/ssffs_$name.solutions"; then
		echo "  solution trace: $(wc -l <"$WORK/fst4_$name.solutions" | tr -d ' ') lines -- BYTE-IDENTICAL"
	else
		echo "  solution trace MISMATCH:"; diff "$WORK/fst4_$name.solutions" "$WORK/ssffs_$name.solutions" | head -6; exit 1
	fi
	if cmp -s "$WORK/fst4_$name.orders" "$WORK/ssffs_$name.orders"; then
		echo "  step-order trace: $(wc -l <"$WORK/fst4_$name.orders" | tr -d ' ') lines -- BYTE-IDENTICAL"
	else
		echo "  step-order trace MISMATCH:"; diff "$WORK/fst4_$name.orders" "$WORK/ssffs_$name.orders" | head -6; exit 1
	fi
}

echo
echo "=== configuration (a): wrapper - 1-NN accuracy, 3-fold CV, madelon ==="
"$FST_BIN" --data "$WORK/madelon_train_nips_500.arff" --format arff \
	--splitter randrand --rr-splits 1 --rr-train 50 --rr-test 50 \
	--splitter2 cv --splitter2-cv-folds 3 --scaler to01 \
	--criterion wrapper-knn --knn-k 1 --knn-distance L2 \
	--search sffs --target-d 20 --sffs-delta 5 \
	--evaluator sampled --step-cap 100 --step-cap-backward 50 \
	--step-explore 0.2 --step-tau 0 --step-decay 100 \
	--warmup-probes 200 --warmup-card 10 --seed "$SEED" \
	--output json --progress json >"$WORK/fst4_wrapper.out" 2>"$WORK/fst4_wrapper.err"
"$WORK/ssffs" --data "$WORK/madelon_train_nips_500.arff" \
	--rr-train 50 --rr-test 50 --cv-folds 3 --scaler to01 \
	--criterion wrapper-knn --knn-k 1 --target-d 20 --sffs-delta 5 \
	--step-cap 100 --step-cap-backward 50 \
	--warmup-probes 200 --warmup-card 10 --seed "$SEED" \
	>"$WORK/ssffs_wrapper.out" 2>"$WORK/ssffs_wrapper.err"
compare wrapper

echo
echo "=== configuration (b): filter - multinomial Bhattacharyya, reuters ==="
"$FST_BIN" --data "$DATA_DIR/reuters_apte.arff" --format arff \
	--splitter randrand --rr-splits 1 --rr-train 50 --rr-test 40 \
	--scaler void --criterion multinom-bhattacharyya \
	--search sffs --target-d 25 --sffs-delta 5 \
	--evaluator sampled --step-cap 100 --step-cap-backward 50 \
	--step-explore 0.2 --step-tau 0 --step-decay 100 \
	--warmup-probes 2000 --warmup-card 25 --seed "$SEED" \
	--output json --progress json >"$WORK/fst4_filter.out" 2>"$WORK/fst4_filter.err"
"$WORK/ssffs" --data "$DATA_DIR/reuters_apte.arff" \
	--rr-train 50 --rr-test 40 --scaler void \
	--criterion multinom-bhattacharyya --target-d 25 --sffs-delta 5 \
	--step-cap 100 --step-cap-backward 50 \
	--warmup-probes 2000 --warmup-card 25 --seed "$SEED" \
	>"$WORK/ssffs_filter.out" 2>"$WORK/ssffs_filter.err"
compare filter

echo
echo "ALL CHECKS PASSED"
