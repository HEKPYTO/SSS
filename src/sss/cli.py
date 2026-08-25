from __future__ import annotations

import argparse
import json
import sys

from .arff import load_arff
from .sss import _run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stochastic Sequential Forward Floating Search")
    parser.add_argument("--data", required=True)
    parser.add_argument("--rr-train", type=int, default=50)
    parser.add_argument("--rr-test", type=int, default=50)
    parser.add_argument("--cv-folds", type=int, default=0)
    parser.add_argument("--scaler", choices=["void", "to01"], default="void")
    parser.add_argument(
        "--criterion", choices=["wrapper-knn", "multinom-bhattacharyya"], required=True
    )
    parser.add_argument("--knn-k", type=int, default=1)
    parser.add_argument("--target-d", type=int, default=0)
    parser.add_argument("--sffs-delta", type=int, default=0)
    parser.add_argument("--step-cap", type=int, default=100)
    parser.add_argument("--step-cap-backward", type=int, default=0)
    parser.add_argument("--step-cap-frac", type=float, default=0.0)
    parser.add_argument("--step-explore", type=float, default=0.2)
    parser.add_argument("--step-tau", type=float, default=0.0)
    parser.add_argument("--step-decay", type=int, default=100)
    parser.add_argument("--step-sampler", choices=["softmax", "uniform", "topk"], default="softmax")
    parser.add_argument("--step-frozen", action="store_true")
    parser.add_argument("--warmup-probes", type=int, default=0)
    parser.add_argument("--warmup-card", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = _run(
            load_arff(args.data),
            target_size=args.target_d,
            criterion_name=args.criterion,
            forward_budget=args.step_cap,
            backward_budget=args.step_cap_backward,
            exploration=args.step_explore,
            temperature=None if args.step_tau == 0 else args.step_tau,
            horizon=args.step_decay,
            warmup_probes=args.warmup_probes,
            warmup_size=args.warmup_card,
            frontier_delta=args.sffs_delta,
            seed=args.seed,
            train_percent=args.rr_train,
            test_percent=args.rr_test,
            cv_folds_count=args.cv_folds,
            knn_k=args.knn_k,
            scale=args.scaler == "to01",
            cap_fraction=args.step_cap_frac,
            sampler=args.step_sampler,
            frozen=args.step_frozen,
            stream=sys.stderr if args.progress else None,
        )
    except (RuntimeError, ValueError) as error:
        build_parser().error(str(error))
    print(
        json.dumps(
            {
                "value": result.value,
                "size": len(result.features),
                "features": result.features,
                "evaluations": result.evaluations,
            }
        )
    )
    return 0


if __name__ == "__main__":
    main()
