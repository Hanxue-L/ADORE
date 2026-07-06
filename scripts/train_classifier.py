from __future__ import annotations

from pathlib import Path
import argparse

REPO_ROOT = Path(__file__).resolve().parents[1]

from adore_dpv.modeling import EvaluationConfig, available_models, evaluate_models, load_feature_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate ADORE classifiers from a feature table.")
    parser.add_argument("--input", required=True, help="Path to a feature table CSV.")
    parser.add_argument("--models", nargs="+", default=["svm"], choices=available_models())
    parser.add_argument("--output-dir", default="results", help="Directory for metric outputs.")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed-parameters", action="store_true", help="Use default model parameters.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, X, y, sample_ids = load_feature_table(args.input)
    config = EvaluationConfig(
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        inner_splits=args.inner_splits,
        n_trials=args.n_trials,
        random_state=args.seed,
        tune=not args.fixed_parameters,
    )

    per_fold, summary = evaluate_models(X, y, sample_ids, args.models, config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(output_dir / "metrics_per_fold.csv", index=False)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)

    print(summary.to_string(index=False))
    print(f"Wrote {output_dir / 'metrics_per_fold.csv'}")
    print(f"Wrote {output_dir / 'metrics_summary.csv'}")


if __name__ == "__main__":
    main()
