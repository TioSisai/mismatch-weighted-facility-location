"""Summarize final-round AULC and paired Wilcoxon tests for completed seeds, and export Markdown/CSV to OUTPUT_ROOT/tables/."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", dotenv=True, pythonpath=True)

import pandas as pd  # noqa: E402

from src.analysis.results import (  # noqa: E402
    DATASET_NAMES,
    DISAGREEMENT_USAGES,
    STRATEGY_BACKBONES,
    STRATEGY_DISPLAY_NAMES,
    STRATEGY_ORDER,
    TABLE_ROW_ORDER,
    aulc_summary,
    discover_config_dirs,
    load_curves,
    paired_wilcoxon,
    parse_config_filter,
)

AULC_METRICS = ("test_AULC", "val_AULC")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_NAMES),
                        choices=DATASET_NAMES, help="datasets to summarize")
    parser.add_argument("--reference", default="mw-fl", choices=STRATEGY_ORDER,
                        help="reference strategy for the paired Wilcoxon test (default mw-fl)")
    parser.add_argument("--config-hash", nargs="+", default=None, metavar="PREFIX",
                        help="keep only config_hash dirs with these name prefixes (per-strategy hash, several allowed)")
    parser.add_argument("--config-filter", nargs="+", default=[], metavar="KEY=VALUE",
                        help="keep only config_hash dirs whose config.json fields equal these values")
    parser.add_argument("--output-root", default=os.environ.get("OUTPUT_ROOT"),
                        help="experiment output root (default OUTPUT_ROOT in .env), tables written to its tables/")
    return parser.parse_args(argv)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _mark_rank(text: str, rank: int) -> str:
    """Bold the best value and underline the second-best value."""
    if rank == 1:
        return f"**{text}**"
    if rank == 2:
        return f"<u>{text}</u>"
    return text


def build_aulc_table(summaries: dict[str, pd.DataFrame], dataset: str) -> tuple[pd.DataFrame, str]:
    """Build an AULC summary table for a single dataset.

    Args:
        summaries: ``{metric: aulc_summary(curves, metric)}``.
        dataset: Dataset name.

    Returns:
        ``(table, markdown)``, a numeric table and Markdown marking the best and second-best values, arranged in legend order.
    """
    present = set(summaries[AULC_METRICS[0]].query("dataset == @dataset").strategy)
    strategies = [s for s in TABLE_ROW_ORDER if s in present]
    table = pd.DataFrame({
        "dataset": dataset,
        "strategy": strategies,
        "backbone": [STRATEGY_BACKBONES[s] for s in strategies],
        "disagreement": [DISAGREEMENT_USAGES[s] for s in strategies],
    })
    for metric in AULC_METRICS:
        sub = summaries[metric].query("dataset == @dataset").set_index("strategy").loc[strategies]
        for column in ("mean", "std", "rank"):
            table[f"{metric}_{column}"] = sub[column].to_numpy()
    table["n_seeds"] = sub["n_seeds"].to_numpy()

    rows = []
    for record in table.to_dict("records"):
        row = [STRATEGY_DISPLAY_NAMES[record["strategy"]], record["backbone"], record["disagreement"]]
        for metric in AULC_METRICS:
            rank = int(record[f"{metric}_rank"])
            text = f"{record[f'{metric}_mean']:.3f} ± {record[f'{metric}_std']:.3f}"
            row += [_mark_rank(text, rank), str(rank)]
        rows.append(row + [str(record["n_seeds"])])
    headers = ["Strategy", "Backbone", "Disagreement", "test AULC", "test rank", "val AULC", "val rank", "seeds"]
    return table, markdown_table(headers, rows)


def build_wilcoxon_table(tests: dict[str, pd.DataFrame], dataset: str) -> tuple[pd.DataFrame, str]:
    """Build a table of paired tests against the reference strategy for a single dataset.

    Args:
        tests: ``{metric: paired_wilcoxon(curves, metric, reference)}``.
        dataset: Dataset name.

    Returns:
        ``(table, markdown)``, a numeric table and Markdown grouped by metric, with each group arranged in legend order.
    """
    row_order = {s: i for i, s in enumerate(TABLE_ROW_ORDER)}
    blocks = []
    for metric in AULC_METRICS:
        block = tests[metric].query("dataset == @dataset").sort_values("strategy", key=lambda col: col.map(row_order))
        blocks.append(block.assign(metric=metric))
    table = pd.concat(blocks, ignore_index=True)

    rows = [
        [
            f"{STRATEGY_DISPLAY_NAMES[r.reference]} vs {STRATEGY_DISPLAY_NAMES[r.strategy]}",
            r.metric,
            f"{r.wins}/{r.n_pairs}",
            f"{r.mean_diff:+.4f}",
            f"{r.statistic:g}",
            f"{r.p_value:.4g}",
        ]
        for r in table.itertuples(index=False)
    ]
    headers = ["Comparison", "Metric", "Seeds won", "Mean ΔAULC", "W", "p (two-sided, exact)"]
    return table, markdown_table(headers, rows)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write to a temporary file in the same directory, then atomically replace the target."""
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def main(argv=None) -> None:
    args = parse_args(argv)
    output_root = Path(args.output_root)
    config_dirs = discover_config_dirs(
        output_root, args.datasets, STRATEGY_ORDER,
        config_hashes=args.config_hash, config_filter=parse_config_filter(args.config_filter),
    )
    if not config_dirs:
        raise SystemExit(f"no matching experiment outputs found under {output_root}")
    curves = load_curves(config_dirs)
    summaries = {metric: aulc_summary(curves, metric) for metric in AULC_METRICS}
    tests = {metric: paired_wilcoxon(curves, metric, args.reference) for metric in AULC_METRICS}

    table_dir = output_root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    reference_name = STRATEGY_DISPLAY_NAMES[args.reference]
    for dataset in args.datasets:
        present = set(curves.strategy[curves.dataset == dataset])
        if not present:
            print(f"[skip] {dataset}: no complete results\n")
            continue
        if missing := [s for s in STRATEGY_ORDER if s not in present]:
            print(f"[note] {dataset} missing strategies: {', '.join(missing)}\n")
        sections = [(f"{dataset}_aulc", f"{dataset}: AULC (mean ± std over seeds)",
                     build_aulc_table(summaries, dataset))]
        if args.reference in present:
            sections.append((f"{dataset}_wilcoxon",
                             f"{dataset}: paired Wilcoxon signed-rank tests of {reference_name}",
                             build_wilcoxon_table(tests, dataset)))
        for stem, title, (table, markdown) in sections:
            text = f"## {title}\n\n{markdown}\n"
            _atomic_write_text(table_dir / f"{stem}.md", text)
            _atomic_write_text(table_dir / f"{stem}.csv", table.to_csv(index=False))
            print(text)
    print(f"Tables written to {table_dir}")


if __name__ == "__main__":
    main()
