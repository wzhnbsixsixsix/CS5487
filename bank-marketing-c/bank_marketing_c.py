"""Bank Marketing：PCA/SVD 降维、分类比较与可视化。"""
import argparse
from pathlib import Path


def main():
    """Run the repository-aligned command-line experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path, help='bank-full.csv、bank.zip 或 bank+marketing.zip')
    parser.add_argument('--a-module', type=Path, default=Path(__file__).resolve().parent.parent/'bank-marketing-a/bank_marketing_a.py')
    parser.add_argument('--split-indices', type=Path)
    parser.add_argument('--out', type=Path, default=Path('results_c'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--include-duration', action='store_true')
    parser.add_argument('--k', nargs='+', type=int, dest='k_values')
    parser.add_argument('--select-metric', choices=['pr_auc','f1','roc_auc'], default='pr_auc')
    parser.add_argument('--b-metrics', type=Path, help='Read-only B metrics for a provenance-checked team comparison')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    if args.seed < 0:
        parser.error('--seed must be nonnegative.')
    if not args.no_plots:
        import matplotlib
        matplotlib.use('Agg')
    from cs5487_c_group.group_a import run_group_a_experiment
    try:
        result = run_group_a_experiment(args.data, a_module_path=args.a_module,
            split_path=args.split_indices, output_dir=args.out, seed=args.seed,
            include_duration=args.include_duration, k_values=args.k_values, select_metric=args.select_metric,
            b_metrics_path=args.b_metrics, make_plots=not args.no_plots)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'错误：{exc}\n')
    print(f"Selected k: {result['metrics']['selected_k_by_method']}")
    print(result['tables']['reduction_comparison'].to_string(index=False))
    print(f'Results: {args.out.resolve()}')


if __name__ == '__main__':
    main()
