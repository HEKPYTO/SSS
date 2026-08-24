#!/usr/bin/env python3
"""reuters.py — 10,105-d filter benchmark, ~2min single core at y=100 full frontier"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.sss.arff import load_arff
from src.sss.sss import SFFS


def main(argv=None):
    ap = argparse.ArgumentParser(description="reuters filter benchmark")
    ap.add_argument("--y", type=int, default=25)
    ap.add_argument("--rho", type=float, default=0.2)
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--target-d", type=int, default=25)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--delta", type=int, default=5)
    ap.add_argument("--warmup-probes", type=int, default=100)
    ap.add_argument("--warmup-card", type=int, default=25)
    args = ap.parse_args(argv)

    data = load_arff("anc/data/reuters_apte.arff")
    sffs = SFFS(
        y=args.y,
        y_back=max(1, args.y // 2 if args.y // 2 else 5),
        rho_u=args.rho,
        tau=args.tau,
        warmup_probes=args.warmup_probes,
        warmup_card=args.warmup_card,
        delta=args.delta,
        seed=args.seed,
        scaler="void",
    )
    # filter path
    sel = sffs.fit_filter(data, target_d=args.target_d, train_pct=50, test_pct=40)
    out = {"value": float(sffs.value_), "size": len(sel), "features": sel, "evaluations": int(sffs.evaluations_)}
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
