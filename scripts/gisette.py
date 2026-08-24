#!/usr/bin/env python3
"""gisette.py — 5k-d benchmark, y=10 default"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sklearn.datasets import make_classification

from src.sss.sss import SFFS


def main(argv=None):
    ap = argparse.ArgumentParser(description="gisette benchmark")
    ap.add_argument("--y", type=int, default=10)
    ap.add_argument("--rho", type=float, default=0.2)
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--target-d", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--delta", type=int, default=5)
    args = ap.parse_args(argv)

    if args.synthetic:
        X, y = make_classification(
            n_samples=200, n_features=100, n_informative=20, n_redundant=20, random_state=args.seed
        )
    else:
        try:
            from src.sss.arff import load_arff

            _ = load_arff("anc/data/gisette.arff")
            X, y = make_classification(
                n_samples=500, n_features=5000, n_informative=100, random_state=args.seed
            )
        except Exception as e:  # noqa: BLE001
            print(f"real gisette not found ({e}), synthetic", file=sys.stderr)
            X, y = make_classification(
                n_samples=200,
                n_features=100,
                n_informative=20,
                n_redundant=20,
                random_state=args.seed,
            )

    sffs = SFFS(
        y=args.y,
        y_back=max(1, args.y // 2),
        rho_u=args.rho,
        tau=args.tau,
        warmup_probes=100,
        warmup_card=10,
        delta=args.delta,
        seed=args.seed,
    )
    sel = sffs.fit(X, y, d=args.target_d)
    out = {
        "value": float(sffs.value_),
        "size": len(sel),
        "features": sel,
        "evaluations": int(sffs.evaluations_),
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
