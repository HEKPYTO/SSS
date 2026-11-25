# `sss` package internals

Python port of the sSFFS standalone (`anc/ssffs/ssffs.cpp`). The C++ file is the
executable specification: same seed, same split, same proposal draws, same
criterion values, same accepted-solution trace, same final subset, same
evaluation counts.

## Modules

| module | responsibility |
|---|---|
| `prng.py` | Windows-UCRT LCG (`214013 / 2531011`), 15-bit `ss_rand`, `frand` |
| `arff.py` | ARFF loader, class-major record layout, whole-data `to01` scaler |
| `split.py` | RR(1, train%, test%) class-stratified split, positional k-fold CV |
| `subset.py` | Signed feature markers, membership, random warm-up subsets |
| `criteria.py` | Wrapper k-NN accuracy and multinomial Bhattacharyya filter |
| `weighter.py` | Online per-feature statistics with z-standardized batches |
| `sampled_step.py` | Budgeted sADD/sRMV operators and proposal sampling |
| `sss.py` | SFFS loop, `SFFS`/`Result` public API, `_run` pipeline |
| `cli.py` | Argument parsing, JSON output, reference-style error handling |

## Data and numeric contract

- `ArffData.data` is a flat `float64` array in class-major order
  (`[class][sample][feature]`). Features are parsed as `float32` and widened to
  `float64`, matching the C++ reader; `scale_to01` normalizes over the whole
  data set, not per split.
- All reductions that feed search decisions are sequential `float64` additions
  in the same order as the reference: the batch mean and variance in
  `weighter.py`, the weight sum in `sampled_step.py`, and the term-mass sums in
  `criteria.py`. Compensated `sum()` or pairwise `numpy.sum` changes low bits,
  and the search amplifies them into different proposals.
- `wrapper-knn` distances are Euclidean over selected features in ascending
  order; neighbour ties resolve by first maximum, as in the reference.

## Search semantics

- `SampledStep` receives forward and backward budgets; a budget of `0`, a
  backward budget of `0`, or a pool no larger than the budget falls back to the
  exhaustive sweep.
- Softmax proposals combine a uniform exploration floor with
  temperature-scaled sampling. The floor list is sorted by occurrence counts
  only when the narrowed floor (`le`) is smaller than the pool; otherwise it is
  consumed in feature order. The temperature defaults to the interquartile
  range divided by `1.349`.
- `Weighter.score(f)` is `m_is - m_isnot`; counts drive floor ordering and
  scores drive softmax weights. Batches are z-standardized together, and the
  adaptive rate is `1 / min(count, horizon)`.
- `sffs_search` keeps per-size incumbents, applies the `target_size + delta`
  frontier, and emits `{"event":"solution",...}` lines for improvements. The
  CLI writes the final result with twelve significant digits and exits `1`
  with an `ssffs:` message on failure.

## Tests

`tests/test_parity.py` compiles the vendored reference and compares stdout,
solution events, step diagnostics, and evaluation counts. The remaining tests
pin split behaviour, criteria, weighter arithmetic, sampling, CLI wiring, and
package installation.
