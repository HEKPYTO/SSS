"""cli.py — argparse mirror of ssffs.cpp:1105-1154"""
from __future__ import annotations

import argparse
import json
import sys

from .arff import load_arff, scale_to01
from .criteria import CountingCriterion, MultinomBhattacharyya, WrapperKnn
from .prng import ss_srand
from .sampled_step import SampledStep
from .split import Split, cv_folds, rr_split_class
from .sss import sffs_search
from .subset import Subset
from .weighter import Weighter


def build_parser():
    p = argparse.ArgumentParser(description="sSFFS Python port of ssffs.cpp")
    p.add_argument("--data", required=True, help="ARFF file")
    p.add_argument("--rr-train", type=int, default=50)
    p.add_argument("--rr-test", type=int, default=50)
    p.add_argument("--cv-folds", type=int, default=0)
    p.add_argument("--scaler", choices=["void", "to01"], default="void")
    p.add_argument("--criterion", choices=["wrapper-knn", "multinom-bhattacharyya"], required=True)
    p.add_argument("--knn-k", type=int, default=1)
    p.add_argument("--target-d", type=int, default=0)
    p.add_argument("--sffs-delta", type=int, default=0)
    p.add_argument("--step-cap", type=int, default=100)
    p.add_argument("--step-cap-backward", type=int, default=0)
    p.add_argument("--step-cap-frac", type=float, default=0.0)
    p.add_argument("--step-explore", type=float, default=0.2)
    p.add_argument("--step-tau", type=float, default=0.0)
    p.add_argument("--step-decay", type=int, default=100)
    p.add_argument("--step-sampler", choices=["softmax", "uniform", "topk"], default="softmax")
    p.add_argument("--step-frozen", action="store_true")
    p.add_argument("--warmup-probes", type=int, default=0)
    p.add_argument("--warmup-card", type=int, default=25)
    p.add_argument("--seed", type=int, default=1)
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.criterion not in ("wrapper-knn", "multinom-bhattacharyya"):
        parser.error("--criterion must be wrapper-knn or multinom-bhattacharyya")
    if args.scaler not in ("void", "to01"):
        parser.error("--scaler must be void or to01")
    if args.rr_train + args.rr_test > 100:
        parser.error("--rr-train + --rr-test must be <= 100")
    sampler = args.step_sampler
    # map to internal
    ss_srand(args.seed)
    data = load_arff(args.data)
    if args.scaler == "to01":
        scale_to01(data)
    n = data.n_features
    if args.target_d >= n:
        parser.error("--target-d must be smaller than number of features")
    outer = Split(data.n_classes)
    for c in range(data.n_classes):
        rr_split_class(data.class_size[c], args.rr_train, args.rr_test, outer.train[c], outer.test[c])
    # folds
    if args.cv_folds and args.cv_folds > 1:
        folds = cv_folds(outer, args.cv_folds, data.n_classes)
    else:
        folds = [outer]
    # criterion
    if args.criterion == "wrapper-knn":
        inner = WrapperKnn(data, folds, args.knn_k)
    else:
        inner = MultinomBhattacharyya(data, outer.train)
    crit = CountingCriterion(inner)
    weighter = Weighter(args.step_decay)
    step = SampledStep(
        weighter,
        cap=args.step_cap,
        cap_backward=args.step_cap_backward,
        cap_frac=args.step_cap_frac,
        explore=args.step_explore,
        tau=args.step_tau,
        sampler=sampler,
    )
    # warmup
    if args.warmup_probes > 0:
        r = args.warmup_card if args.warmup_card < n else n - 1
        weighter.reset(n)
        probe = Subset(n)
        for _ in range(args.warmup_probes):
            probe.make_random_subset(r)
            ok, pv = crit.evaluate(probe)
            if not ok:
                print("warm-up probe evaluation failed", file=sys.stderr)
                sys.exit(1)
            weighter.add(pv, probe)
        weighter.flush_batch()
        if args.step_frozen:
            weighter.freeze()
    sub = Subset(n)
    res = sffs_search(args.target_d, args.sffs_delta, sub, crit, step, sys.stderr)
    if res is None:
        print("search not finished", file=sys.stderr)
        sys.exit(1)
    out = {"value": res["value"], "size": res["size"], "features": res["features"], "evaluations": res["evaluations"]}
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    main()
