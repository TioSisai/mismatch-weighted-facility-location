"""Result discovery, curve loading, AULC summaries, and paired Wilcoxon tests."""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterable, Mapping, Sequence
from itertools import accumulate
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from ..data import DATASET_NAMES

# Match the strategy registry order, defined separately to avoid loading strategy implementations.
STRATEGY_DISPLAY_NAMES = {
    "random": "Random",
    "ft": "FT",
    "fl": "FL",
    "mf-ft": "MF-FT",
    "mf-fl": "MF-FL",
    "mw-ft": "MW-FT",
    "mw-fl": "MW-FL",
}
STRATEGY_ORDER = tuple(STRATEGY_DISPLAY_NAMES)
# Order tables and legends according to Figure 1 in the paper.
TABLE_ROW_ORDER = ("random", "ft", "mf-ft", "mw-ft", "fl", "mf-fl", "mw-fl")
# Labeled as ours in the figures and used as the default reference strategy for paired tests.
REFERENCE_STRATEGY = "mw-fl"

STRATEGY_BACKBONES = {
    "random": "—", "ft": "FT", "fl": "FL",
    "mf-ft": "FT", "mf-fl": "FL", "mw-ft": "FT", "mw-fl": "FL",
}
DISAGREEMENT_USAGES = {
    "random": "—", "ft": "none", "fl": "none",
    "mf-ft": "hard gating", "mf-fl": "hard gating",
    "mw-ft": "soft weight", "mw-fl": "soft weight",
}

STRATEGY_COLORS = {
    "random": "#808080",
    "ft": "#1F77B4",
    "fl": "#2CA02C",
    "mf-ft": "#FF7F0E",
    "mf-fl": "#9467BD",
    "mw-ft": "#D62728",
    "mw-fl": "#E377C2",
}
STRATEGY_MARKERS = {
    "random": "o",
    "ft": "s",
    "fl": "D",
    "mf-ft": "^",
    "mf-fl": "v",
    "mw-ft": "P",
    "mw-fl": "X",
}

CONFIG_FILENAME = "config.json"
RESULTS_FILENAME = "results.csv"
SELECTION_COLUMN = "queried_idxes_in_latest_iteration"
CURVE_METRICS = ("val_mAP", "test_mAP", "val_AULC", "test_AULC")


def parse_config_filter(items: Iterable[str]) -> dict:
    """Parse KEY=VALUE conditions into a dictionary, parsing values as JSON when possible and otherwise keeping them as strings.

    Raises:
        ValueError: A condition is missing an equals sign.
    """
    config_filter = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"filter condition must be of the form KEY=VALUE: {item!r}")
        try:
            config_filter[key] = json.loads(raw)
        except json.JSONDecodeError:
            config_filter[key] = raw
    return config_filter


def _describe_candidates(cfg_dirs: Sequence[Path]) -> str:
    """List fields that differ among candidate configurations for the same dataset and strategy."""
    configs = [json.loads((d / CONFIG_FILENAME).read_text(encoding="utf-8")) for d in cfg_dirs]
    keys = sorted(set().union(*configs))
    differing = [k for k in keys if len({json.dumps(c.get(k)) for c in configs}) > 1]
    return "\n".join(
        f"    {d.name}: " + ", ".join(f"{k}={c.get(k)!r}" for k in differing)
        for d, c in zip(cfg_dirs, configs)
    )


def discover_config_dirs(
    output_root,
    datasets: Iterable[str] = DATASET_NAMES,
    strategies: Iterable[str] = STRATEGY_ORDER,
    config_hashes: Sequence[str] | None = None,
    config_filter: Mapping | None = None,
) -> dict[tuple[str, str], Path]:
    """Locate the unique configuration directory for each dataset and strategy.

    Args:
        output_root: Root directory for experiment outputs.
        datasets: Sequence of dataset names.
        strategies: Sequence of strategy names.
        config_hashes: Optional set of hash prefixes.
        config_filter: Optional conditions on configuration fields, retaining configurations that satisfy all conditions.

    Returns:
        ``{(dataset, strategy): config_dir}``, preserving the input order and omitting entries without matches.

    Raises:
        ValueError: Multiple configurations remain after filtering; the error lists differing fields.
    """
    prefixes = tuple(config_hashes or ())
    strategies = tuple(strategies)
    found: dict[tuple[str, str], Path] = {}
    ambiguous: list[str] = []
    for dataset in datasets:
        for strategy in strategies:
            candidates = []
            for cfg_path in sorted((Path(output_root) / dataset / strategy).glob(f"*/{CONFIG_FILENAME}")):
                if prefixes and not cfg_path.parent.name.startswith(prefixes):
                    continue
                if config_filter:
                    config = json.loads(cfg_path.read_text(encoding="utf-8"))
                    if any(config.get(k) != v for k, v in config_filter.items()):
                        continue
                candidates.append(cfg_path.parent)
            if len(candidates) == 1:
                found[(dataset, strategy)] = candidates[0]
            elif candidates:
                ambiguous.append(f"  {dataset}/{strategy}:\n{_describe_candidates(candidates)}")
    if ambiguous:
        raise ValueError(
            "these combinations map to multiple config_hash dirs, disambiguate with --config-hash or --config-filter:\n"
            + "\n".join(ambiguous)
        )
    return found


def seed_dirs(config_dir) -> dict[int, Path]:
    """Return ``{seed: seed_dir}`` under the configuration directory in ascending seed order."""
    runs = ((int(p.name.removeprefix("seed_")), p) for p in Path(config_dir).glob("seed_*"))
    return dict(sorted(runs))


def load_curves(config_dirs: Mapping[tuple[str, str], Path]) -> pd.DataFrame:
    """Collect per-round metrics for completed seeds, warning and skipping incomplete seeds.

    Args:
        config_dirs: Return value of discover_config_dirs.

    Returns:
        A long-format table sorted by dataset, strategy, seed, and round, with columns
        ``[dataset, strategy, seed, iteration, num_labeled, val_mAP, test_mAP,
        val_AULC, test_AULC]``. Strategies are sorted by STRATEGY_ORDER, and empty values are read as NaN.

    Raises:
        ValueError: No completed seeds are available.
    """
    frames = []
    skipped = []
    for (dataset, strategy), cfg_dir in config_dirs.items():
        max_iter = json.loads((cfg_dir / CONFIG_FILENAME).read_text(encoding="utf-8"))["max_iter"]
        for seed, run_dir in seed_dirs(cfg_dir).items():
            csv_path = run_dir / RESULTS_FILENAME
            df = pd.read_csv(csv_path, usecols=["iteration", "num_labeled_samples", *CURVE_METRICS])
            if len(df) != max_iter:
                skipped.append(str(run_dir))
                continue
            frames.append(pd.DataFrame({
                "dataset": dataset,
                "strategy": strategy,
                "seed": seed,
                "iteration": df["iteration"].astype(np.int64),
                "num_labeled": df["num_labeled_samples"].astype(np.int64),
                **{metric: df[metric].astype(np.float64) for metric in CURVE_METRICS},
            }))
    if skipped:
        warnings.warn("skipped incomplete seeds (results.csv has fewer rows than max_iter):\n  " + "\n  ".join(skipped),
                      stacklevel=2)
    if not frames:
        raise ValueError("no complete seed to load")
    curves = pd.concat(frames, ignore_index=True)
    curves["strategy"] = pd.Categorical(curves["strategy"], categories=list(STRATEGY_ORDER), ordered=True)
    # Row order and default CSV parsing affect subsequent floating-point reductions and must remain consistent.
    return curves.sort_values(["dataset", "strategy", "seed", "iteration"]).reset_index(drop=True)


def load_selections(config_dir, seed: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load cumulative and per-round selections for one seed.

    Args:
        config_dir: Configuration directory.
        seed: Random seed.

    Returns:
        ``(cumulative, new)``, ordered by round. new preserves the strategy's selection order,
        and cumulative[i] is the concatenation of new[0..i].
    """
    csv_path = Path(config_dir) / f"seed_{seed}" / RESULTS_FILENAME
    df = pd.read_csv(csv_path, usecols=["iteration", SELECTION_COLUMN]).sort_values("iteration")
    new = [np.asarray(json.loads(cell), dtype=int) for cell in df[SELECTION_COLUMN]]
    cumulative = list(accumulate(new, lambda labeled, batch: np.concatenate([labeled, batch])))
    return cumulative, new


def final_aulc(curves: pd.DataFrame, metric: str = "test_AULC") -> pd.DataFrame:
    """Return the final-round AULC for each dataset, strategy, and seed, with columns ``[dataset, strategy, seed, metric]``."""
    last_rows = curves.groupby(["dataset", "strategy", "seed"], observed=True)["iteration"].idxmax()
    return curves.loc[last_rows.to_numpy(), ["dataset", "strategy", "seed", metric]].reset_index(drop=True)


def aulc_summary(curves: pd.DataFrame, metric: str = "test_AULC") -> pd.DataFrame:
    """Summarize AULC across seeds for each strategy.

    Args:
        curves: Long-format table from load_curves.
        metric: test_AULC or val_AULC.

    Returns:
        ``[dataset, strategy, mean, std, n_seeds, rank]``, where std uses ddof=1,
        and rank orders means within each dataset in descending order, with 1 being the best.
    """
    rows = []
    # Use Series.mean/std for each group to preserve the floating-point reduction used in the paper.
    for (dataset, strategy), group in final_aulc(curves, metric).groupby(["dataset", "strategy"], observed=True):
        values = group[metric]
        rows.append({
            "dataset": dataset,
            "strategy": strategy,
            "mean": float(values.mean()),
            "std": float(values.std()),
            "n_seeds": int(values.count()),
        })
    summary = pd.DataFrame(rows)
    summary["rank"] = summary.groupby("dataset")["mean"].rank(ascending=False, method="min").astype(int)
    return summary


def paired_wilcoxon(
    curves: pd.DataFrame,
    metric: str = "test_AULC",
    reference: str = REFERENCE_STRATEGY,
) -> pd.DataFrame:
    """Compare the reference strategy with other strategies in pairs matched by seed using an exact two-sided Wilcoxon test.

    Exclude zero differences using scipy's default zero_method="wilcox".

    Args:
        curves: Long-format table from load_curves.
        metric: test_AULC or val_AULC.
        reference: Reference strategy name.

    Returns:
        ``[dataset, reference, strategy, n_pairs, wins, mean_diff, statistic, p_value]``.
        wins is the number of seeds where the reference strategy strictly wins, mean_diff is the mean of reference minus opponent,
        and statistic is min(W+, W-).
    """
    final = final_aulc(curves, metric).astype({"strategy": str})
    table = final.pivot(index=["dataset", "seed"], columns="strategy", values=metric)
    rows = []
    for dataset, block in table.groupby(level="dataset", sort=True):
        for strategy in STRATEGY_ORDER:
            if strategy == reference or strategy not in block or reference not in block:
                continue
            pair = block[[reference, strategy]].dropna()
            if pair.empty:
                continue
            diff = pair[reference] - pair[strategy]
            result = wilcoxon(pair[reference], pair[strategy], alternative="two-sided", method="exact")
            rows.append({
                "dataset": dataset,
                "reference": reference,
                "strategy": strategy,
                "n_pairs": len(pair),
                "wins": int((diff > 0).sum()),
                "mean_diff": float(diff.mean()),
                "statistic": float(result.statistic),
                "p_value": float(result.pvalue),
            })
    return pd.DataFrame(rows)


def aggregate_mean_band(
    curves: pd.DataFrame, dataset: str, strategy: str, metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Summarize the mean and range across seeds by labeled sample count for the specified dataset, strategy, and metric.

    Returns:
        ``(x, mean, lo, hi, n)``, sorted by labeled sample count in ascending order, where lo/hi are the minimum/maximum values,
        and n is the number of valid seeds at each point.
    """
    sub = curves[(curves.dataset == dataset) & (curves.strategy == strategy)]
    grouped = sub.groupby("num_labeled")[metric]
    x = np.array(sorted(sub["num_labeled"].unique()))
    mean = grouped.mean().reindex(x).to_numpy()
    lo = grouped.min().reindex(x).to_numpy()
    hi = grouped.max().reindex(x).to_numpy()
    n = grouped.count().reindex(x).to_numpy()
    return x, mean, lo, hi, n
