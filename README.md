# SSS — Stochastic Sequential Search (sSFFS)

Reproduction of **Somol & Grim "Stochastic Sequential Search in Very-High-Dimensional Feature Selection"** [`arXiv:2608.01502`](https://arxiv.org/abs/2608.01502) — budgeted sampled `sADD`/`sRMV` operators turning any sequential method into per-step `O(y)` independent of `D`, studied as `sSFFS`.

> **Reference impl vendored:** `anc/ssffs/ssffs.cpp` (MIT, 1208 lines, `c++ -std=c++17 -O2 -funroll-loops -ffp-contract=off -DNDEBUG`) reproduces FST4 bit for bit (see `anc/ssffs/README.md`, `verify.sh`). SHA `reuters_apte.arff` = `4b22e0e94f53993595f5fa80c7eca0b5dbda0ec80423ac2e31861e156ea1834a`.

## Quick start

```bash
make ssffs                    # builds ./ssffs exactly as paper
./ssffs --data anc/data/reuters_apte.arff --rr-train 50 --rr-test 40 --scaler void \
        --criterion multinom-bhattacharyya --target-d 25 --sffs-delta 5 \
        --step-cap 100 --step-cap-backward 50 --warmup-probes 2000 --warmup-card 25 --seed 1

# Python API (same PRNG/splits as C++)
python3 -c "from src.sss.prng import ss_srand, ss_rand; ss_srand(1); print([ss_rand() for _ in range(5)])"
# → [41, 18467, 6334, 26500, 19169]

# Development
make dev                      # pip install -e .[dev] + pre-commit hooks
make lint                     # ruff check + format --check
make typecheck                # mypy src
make check                    # lint + typecheck + test
make test                     # pytest (builds ssffs)
pre-commit run --all-files    # run hooks on all files
```

## Layout

- `anc/` — verbatim arXiv ancillary (`ssffs.cpp`, `uci2arff.py`, `trn2arff.py`, 9 `configs/*.cfg`, `reuters_apte.arff`)
- `src/sss/` — Python port (`prng.py`, `arff.py`, `split.py`, `subset.py`, `criteria.py`, `weighter.py`, `sampled_step.py`, `sss.py`) — line-for-line same formulas as `ssffs.cpp`
- `scripts/` — `madelon.py`, `gisette.py`, `reuters.py` reproducing Sec 5 benchmarks

## Plan


