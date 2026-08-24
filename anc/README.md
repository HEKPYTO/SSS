# Ancillary files — "Stochastic Sequential Search in Very-High-Dimensional Feature Selection"

P. Somol, J. Grim. These files make the paper's method and experiments
reproducible from the submission itself.

## ssffs/ — standalone reference implementation of sSFFS

A self-contained, dependency-free single-file C++17 implementation of the
paper's primary instance (sSFFS), covering both criterion stacks used in
the experiments (1-NN wrapper with k-fold cross-validation; multinomial
Bhattacharyya filter), including data splitting and the portable random
generator. It is verified to reproduce the Feature Selection Toolbox 4
(FST4) implementation bit for bit at matched seeds — same data split,
search trajectory, criterion values, selected subsets and evaluation
counts. See `ssffs/README.md` for build/run instructions and
`ssffs/verify.sh` for the equivalence check. MIT license (`ssffs/LICENSE`).

## configs/ — FST4 CLI configuration files of the experiments

One config per dataset and role, for madelon, gisette, and reuters. Run as
`fst4 --config <file> [--seed N]` from the FST4 distribution root (data
paths are relative).

- `*_sffs_*.cfg` — the sequential-search base stack (data, split protocol,
  scaling, criterion, floating search). Run as-is it is the full-sweep
  reference; on gisette the full sweep is run under an external ~30-minute
  wall cap to establish the feasibility frontier reported in the paper.
- `*_bif.cfg` — BIF ranking (all singleton evaluations; prefixes
  holdout-validated; on reuters the target size is set per reported prefix
  size).
- `*_daf.cfg` — DAF ranking in the probe-based protocol of the paper
  (uniformly drawn probes scored by the same criterion, DAF-contrast
  ranking, prefixes holdout-validated; probe count = the consumed budget).

The budgeted (sSFFS) arms reuse the base stack of each dataset with the
sampled evaluator switched on; the paper's default ("winner")
configuration on madelon is exactly:

    fst4 --config madelon_sffs_knn.cfg \
         --evaluator sampled --step-cap 100 --step-cap-backward 50 \
         --step-explore 0.2 --step-tau 0 --step-decay 100 \
         --step-sampler softmax --step-stats zcontrast \
         --warmup-probes 200 --warmup-card 10 \
         --output json --progress json --seed 1

(gisette: warm-up 1000@25; reuters: warm-up 2000@25 on the multinomial
config — Sec. "Experimental Setup" of the paper. The ablation arms replace
`--step-sampler` / `--step-stats` / caps accordingly; the uncapped-backward
twin arms omit `--step-cap-backward`.)

## Data

The datasets are public and are not redistributed here. IMPORTANT: do not
convert the UCI downloads by hand — the paper's data files are class-major
reordered relative to the raw UCI row order, and the split generator
consumes samples in file order, so any other ordering silently yields
different splits and different numbers at the same seed.

- madelon, gisette (UCI Machine Learning Repository, CC BY 4.0):
  download the originals (madelon: the TRAINING part; gisette: the
  1000-row VALIDATION part — not the training part) and run
  `ssffs/uci2arff.py`, which performs the exact conversion and verifies
  the result against pinned SHA-256 checksums of the paper's files. The
  UCI URLs are listed in its header.
- reuters is INCLUDED here: `data/reuters_apte.arff.gz` (gunzip before
  use). It is not a UCI dataset and cannot be rebuilt from public
  sources: an anonymized derived term matrix (attributes a1..a10105,
  classes cls0..cls32 — no vocabulary, no article text) of the
  Reuters-21578 text categorization collection, Distribution 1.0.
  RESEARCH USE ONLY — the terms in `data/REUTERS-NOTICE.txt` (and in the
  file's own header) travel with every copy.

FST4 itself is scheduled for public release (its distribution bundles all
three datasets ready to run, so the FST4 configs above need no conversion
at all); until then the standalone implementation reproduces every
method-side result of the paper exactly.
