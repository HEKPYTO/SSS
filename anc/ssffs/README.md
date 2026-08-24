# Standalone sSFFS — accompanying code for the paper

This directory contains a **self-contained, single-file implementation of
sSFFS** (Stochastic Sequential Forward Floating Search), the method
contributed by the paper *"Stochastic Sequential Search in
Very-High-Dimensional Feature Selection"* (P. Somol, J. Grim). It exists so
that the paper's method and experiments are reproducible from the paper's
own accompanying material alone, independent of the release schedule of the
full Feature Selection Toolbox 4 (FST4), whose CLI is the implementation
the experiments were run with.

It is **not a port approximation**: the standalone is verified to
reproduce the FST4 reference implementation **bit for bit** — the same
seed yields the same data split, the same proposal draws, the same
criterion values, the same per-improvement solution trace, the same final
subset and the same evaluation counts (see *Verification* below).

## Contents

| file | purpose |
|---|---|
| `ssffs.cpp` | the complete implementation (C++17, no dependencies) |
| `uci2arff.py` | builds the paper's exact madelon/gisette ARFF files from the UCI-repository originals (sha256-verified: row order and class mapping pinned) |
| `trn2arff.py` | converts FST TRN-format data files (as bundled with the FST4 distribution) to ARFF |
| `verify.sh` | the equivalence check against the FST4 CLI (requires an FST4 build) |

## Build

Use the exact flags — they pin floating-point behaviour (no FP contraction,
no fast-math), which the bit-reproducibility claim depends on:

```
c++ -std=c++17 -O2 -funroll-loops -ffp-contract=off -DNDEBUG -o ssffs ssffs.cpp
```

## What it implements

The two experiment configurations of the paper, and only those:

* **sSFFS**: the classic SFFS schedule with the budgeted sampled operator
  pair substituted for the exhaustive sweeps — softmax proposal sampling
  from online batch-standardized dependency-aware statistics with
  exponential forgetting, a uniform exploration floor, independent forward
  (`--step-cap`) and backward (`--step-cap-backward`) budgets (backward
  budget 0 = exact floating correction), warm-up probes
  (`--warmup-probes N --warmup-card C`), optional frontier restriction
  (`--sffs-delta`), and the paper's ablation switches
  (`--step-sampler softmax|uniform|topk`, `--step-frozen`,
  `--step-cap-frac`).
* **Criterion (a), wrapper**: k-NN classification accuracy (Euclidean
  metric), estimated by k-fold cross-validation on the training part of a
  random class-stratified train/test split (`--cv-folds`).
* **Criterion (b), filter**: multinomial Bhattacharyya distance estimated
  on the training part of a random class-stratified train/test split.
* **Data**: ARFF only (dense or sparse rows, numeric features, one nominal
  `class` attribute). **Do not convert the UCI downloads yourself** —
  the paper's files are class-major reordered relative to the raw UCI row
  order, and the split generator consumes samples in file order, so any
  other ordering silently yields different splits and different numbers at
  the same seed. `uci2arff.py` performs the exact conversion and verifies
  the result by pinned SHA-256 (it also catches the classic trap: gisette
  uses the 1000-row VALIDATION part, not the 6000-row training part).
* **Randomness**: an own 5-line portable PRNG (the Windows-UCRT `rand()`
  recurrence), so `--seed` pins the split and the whole trajectory on
  every platform — the same convention FST4 uses.

Everything else in the FST4 CLI (other searches, criteria, splitters,
metrics, threading, report modes) is deliberately out of scope.

## Running the paper's configurations

Wrapper configuration (madelon; bounded run, target size 20, frontier 25).
First download the four UCI originals listed in the `uci2arff.py` header
(madelon training part, gisette validation part), then:

```
python3 uci2arff.py madelon <download-dir> madelon.arff
./ssffs --data madelon.arff \
        --rr-train 50 --rr-test 50 --cv-folds 3 --scaler to01 \
        --criterion wrapper-knn --knn-k 1 \
        --target-d 20 --sffs-delta 5 \
        --step-cap 100 --step-cap-backward 50 \
        --warmup-probes 200 --warmup-card 10 --seed 1
```

Filter configuration (reuters; bounded run, target size 25, frontier 30).
`reuters_apte.arff` is not a UCI dataset and cannot be rebuilt from public
sources — it is the anonymized derived term matrix of Reuters-21578,
Distribution 1.0. It is provided gzipped in the paper's accompanying
material (`anc/data/reuters_apte.arff.gz`; RESEARCH USE ONLY — the notice
file beside it states the terms) and ships with the FST4 distribution:

```
./ssffs --data reuters_apte.arff \
        --rr-train 50 --rr-test 40 --scaler void \
        --criterion multinom-bhattacharyya \
        --target-d 25 --sffs-delta 5 \
        --step-cap 100 --step-cap-backward 50 \
        --warmup-probes 2000 --warmup-card 25 --seed 1
```

The paper's full-scale runs use the same knobs with the paper's budgets and
target sizes (Section "Experimental Setup"); `--target-d 0` runs in
d-optimizing mode over the unrestricted frontier. Output: one JSON line
per improved solution on stderr (`{"event":"solution","value":...,
"d":...,"features":[...]}` — the anytime log the paper's curves are
harvested from), a final JSON result on stdout, and the `F/B order`
step diagnostics on stderr.

## Verification against FST4

`verify.sh` runs both implementations on both configurations at the same
seed and compares byte for byte: the solution trace, the step-order
diagnostic trace, the final subset, the criterion value (12 significant
digits) and the evaluation counts:

```
FST_BIN=path/to/fst4 DATA_DIR=path/to/data sh verify.sh
```

Verified 2026-07-28 on the paper's machine of record (Apple M1 Max, both
sides built with Apple clang under the fixed FP flags), seeds 1 and 2:

| configuration | solution trace | final subset | criterion value | evaluations |
|---|---|---|---|---|
| wrapper (madelon, d=20, Δ=5, seed 1) | 88 lines identical | identical (20 features) | 0.870980929707 both | 7,254 both |
| wrapper (madelon, d=20, Δ=5, seed 2) | 100 lines identical | identical (20 features) | 0.856966789313 both | 8,060 both |
| filter (reuters, d=25, Δ=5, seed 1) | 393 lines identical | identical (25 features) | 1.45662094486 both | 39,099 both |
| filter (reuters, d=25, Δ=5, seed 2) | 195 lines identical | identical (25 features) | 1.47413241303 both | 20,650 both |

Note the criterion-value equality is not merely at print precision: every
intermediate improvement in the trace and every step-winner order
statistic matches, which cannot happen unless every one of the tens of
thousands of criterion evaluations produced bit-identical doubles on both
sides.

## License

MIT License, Copyright (c) 2026 Institute of Information Theory and
Automation, The Czech Academy of Sciences (UTIA) — see [LICENSE](LICENSE).
The license text is also embedded in `ssffs.cpp` itself, so the single
file remains a complete, redistributable artifact on its own.
