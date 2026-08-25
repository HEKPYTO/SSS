# Stochastic Sequential Search

`stochastic-sequential-search` is a Python implementation of the sSFFS instance studied by Somol and Grim. It is verified against the immutable C++ ancillary implementation for a bounded Reuters configuration: the seeded feature trajectory, step diagnostics, result `[71, 110, 114, 152, 6611]`, and 659 evaluations agree; final values are compared within `1e-11`.

## Install

```bash
python -m pip install stochastic-sequential-search
```

Python 3.11 through 3.14 are supported.

```python
from sss import SFFS

result = SFFS(forward_budget=100, backward_budget=50, seed=1).fit(X, y, target_size=20)
print(result.features, result.value, result.evaluations)
```

`fit` uses the wrapper criterion. `fit_arff(path, target_size=..., criterion="wrapper-knn" | "multinom-bhattacharyya")` loads ARFF data and returns the same immutable result type.

## Command line

```bash
sss --data data.arff --criterion multinom-bhattacharyya --target-d 25 \
  --rr-train 50 --rr-test 40 --step-cap 100 --step-cap-backward 50 \
  --warmup-probes 2000 --warmup-card 25 --seed 1
```

The final result is one JSON object on stdout. Add `--progress` for solution and step diagnostics on stderr. The reference paper configurations use wrapper k-NN with three folds and `to01` scaling for Madelon/Gisette, and multinomial Bhattacharyya with unscaled Reuters data. Obtain and convert the paper datasets with the scripts in `anc/ssffs`; Reuters data remains subject to its original research-use terms.

The paper describes a broader stochastic sequential-search family. This release implements its sSFFS ancillary configuration, not every possible SSS host search.

## Development

```bash
make check
python -m pytest tests/test_parity.py -vv
```
