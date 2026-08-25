# Stochastic Sequential Search

> Independent Python implementation of
> [Somol and Grim's sSFFS research](https://arxiv.org/abs/2608.01502).
> Python 3.11+ · NumPy · library and command-line interfaces.

For the bounded Reuters parity configuration at seed 1, this implementation and
the paper's ancillary C++ program select `[71, 110, 114, 152, 6611]` after 659
criterion evaluations. The accepted-subset trajectory and step diagnostics also
match; criterion values are compared within `1e-11`.

> **Disclaimer:** This is an independent implementation based on the published
> paper and its ancillary source code. It is not affiliated with, endorsed by,
> or maintained by the paper's authors, their institutions, or research team.

## Installation

Install from a source checkout:

```bash
python -m pip install .
```

For development:

```bash
python -m pip install -e ".[dev]"
```

The only runtime dependency is NumPy. Python 3.11 through 3.14 are supported.

## Library usage

```python
from sss import SFFS

search = SFFS(forward_budget=100, backward_budget=50, seed=1)
result = search.fit(X, y, target_size=20)

print(result.features)
print(result.value)
print(result.evaluations)
```

`fit` evaluates NumPy arrays with the wrapper k-NN criterion. For ARFF input,
use `fit_arff` and select either supported criterion:

```python
result = search.fit_arff(
    "data.arff",
    target_size=25,
    criterion="multinom-bhattacharyya",
)
```

Both methods return the same immutable `Result` type.

## Command line

```bash
sss --data data.arff \
  --criterion multinom-bhattacharyya \
  --target-d 25 \
  --rr-train 50 \
  --rr-test 40 \
  --step-cap 100 \
  --step-cap-backward 50 \
  --warmup-probes 2000 \
  --warmup-card 25 \
  --seed 1
```

The command writes one JSON result to stdout. Add `--progress` to emit accepted
solutions and step diagnostics as JSON lines on stderr. Run `sss --help` for all
search, sampling, split, scaling, and criterion options.

## Implemented scope

- Sampled sequential forward floating search with forward and backward budgets.
- Softmax, uniform, and top-k proposal sampling.
- Wrapper k-NN accuracy with optional cross-validation and `to01` scaling.
- Multinomial Bhattacharyya filtering with the ancillary implementation's
  smoothing, prior, caching, and tie behavior.
- Dense and sparse ARFF input, deterministic stratified splitting, warm-up
  probes, frontier restriction, and portable seeded randomness.

The paper defines a broader stochastic sequential-search family. This package
implements the sSFFS configuration exposed by the ancillary executable; other
host searches, including sOS, are outside the release scope.

## Reference parity

The automated parity suite compares the Python implementation with the vendored
C++ program across:

- PRNG consumption and stratified data splits;
- wrapper k-NN behavior on a scaled cross-validation fixture;
- uniform forward and backward sampling after proposal statistics change;
- the bounded Reuters softmax trajectory, step diagnostics, final subset, and
  evaluation count.

The C++ source is treated as an immutable executable specification and verified
by SHA-256 before compilation and follows the behavioral contract derived from the paper and ancillary implementation.

## Data

The command-line interface accepts numeric ARFF files with a nominal `class`
attribute. Conversion scripts for the paper's Madelon and Gisette inputs are in
`anc/ssffs`.

The Reuters fixture in the source repository is governed by
[`anc/data/REUTERS-NOTICE.txt`](anc/data/REUTERS-NOTICE.txt) and is excluded from
the source distribution. Review those terms before using or redistributing it.

## Verification

```bash
make check
```

The release gate runs Ruff first, then formatting, mypy, checksum verification,
C++ compilation with warnings treated as errors, the complete test suite, and a
clean-wheel installation smoke test. To run only the cross-language checks:

```bash
python -m pytest tests/test_parity.py -vv
```

## Research source

Petr Somol and Jiří Grim, “Stochastic Sequential Search in Very-High-Dimensional
Feature Selection,” arXiv:2608.01502 (2026). The paper and its authors are the
research source, not maintainers or contributors to this repository.

## License

See [`LICENSE`](LICENSE) for the repository license. The vendored ancillary C++
implementation retains its original license in [`anc/ssffs/LICENSE`](anc/ssffs/LICENSE),
and the Reuters fixture remains subject to its separate notice.
