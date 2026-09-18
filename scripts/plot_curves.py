"""Export the test mAP curves for Figure 1 and validation curves for Figure S1 to OUTPUT_ROOT/figures/."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", dotenv=True, pythonpath=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.analysis.results import (  # noqa: E402
    DATASET_NAMES,
    DISAGREEMENT_USAGES,
    REFERENCE_STRATEGY,
    STRATEGY_COLORS,
    STRATEGY_DISPLAY_NAMES,
    STRATEGY_MARKERS,
    STRATEGY_ORDER,
    TABLE_ROW_ORDER,
    aggregate_mean_band,
    aulc_summary,
    discover_config_dirs,
    load_curves,
    parse_config_filter,
)
from src.analysis.style import apply_nature_style, mm, save_figure  # noqa: E402

PANEL_LABELS = "abcdefgh"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_NAMES),
                        choices=DATASET_NAMES, help="datasets to plot, ordered as the panels of the composite figure")
    parser.add_argument("--config-hash", nargs="+", default=None, metavar="PREFIX",
                        help="keep only config_hash dirs with these name prefixes (per-strategy hash, several allowed)")
    parser.add_argument("--config-filter", nargs="+", default=[], metavar="KEY=VALUE",
                        help="keep only config_hash dirs whose config.json fields equal these values")
    parser.add_argument("--formats", nargs="+", default=["svg", "pdf", "png"], help="export formats")
    parser.add_argument("--output-root", default=os.environ.get("OUTPUT_ROOT"),
                        help="experiment output root (default OUTPUT_ROOT in .env), figures written to its figures/")
    return parser.parse_args(argv)


def enable_usetex() -> bool:
    """Enable sans-serif LaTeX typesetting when both latex and dvipng are available, and return whether it is enabled."""
    # Exporting PNG in LaTeX mode also requires dvipng.
    if shutil.which("latex") is None or shutil.which("dvipng") is None:
        return False
    mpl.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": "\n".join([
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{helvet}",
            r"\usepackage{bm}",
            r"\renewcommand{\familydefault}{\sfdefault}",
        ]),
    })
    return True


def plot_dataset(ax, curves: pd.DataFrame, dataset: str, metric: str, *,
                 markevery: int = 2, band_alpha: float = 0.12) -> None:
    """Plot the mean curve across seeds and the minimum-to-maximum range for a single dataset.

    Args:
        ax: Target axes.
        curves: Long-format table from load_curves.
        dataset: Dataset name.
        metric: test_mAP or val_mAP.
        markevery: Marker interval.
        band_alpha: Range transparency.
    """
    strategies = [s for s in STRATEGY_ORDER if ((curves.dataset == dataset) & (curves.strategy == s)).any()]
    bands = {s: aggregate_mean_band(curves, dataset, s, metric) for s in strategies}
    for strategy in strategies:
        x, _, lo, hi, _ = bands[strategy]
        ax.fill_between(x, lo, hi, color=STRATEGY_COLORS[strategy], alpha=band_alpha, linewidth=0, zorder=1)
    for strategy in strategies:
        x, mean, _, _, _ = bands[strategy]
        ax.plot(x, mean, color=STRATEGY_COLORS[strategy], lw=1.4, zorder=3,
                marker=STRATEGY_MARKERS[strategy], markersize=4.2, markevery=markevery,
                markeredgecolor="white", markeredgewidth=0.4,
                label=STRATEGY_DISPLAY_NAMES[strategy])
    ax.set_xlabel("Labeled segments")
    ax.set_xlim(x.min() - 5, x.max() + 5)
    ax.set_xticks(np.arange(0, x.max() + 1, 100))
    ax.grid(True, which="major", axis="both", color="#E6E6E6", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def add_aulc_legend(ax, summary: pd.DataFrame, dataset: str, *, usetex: bool, fontsize: float) -> None:
    """Add a table-style legend with AULC in the lower-right corner.

    Args:
        ax: Target axes.
        summary: aulc_summary result for the same split as the curves.
        dataset: Dataset name.
        usetex: Whether to use LaTeX typesetting.
        fontsize: Legend font size.
    """
    stats = summary[summary.dataset == dataset].set_index("strategy")
    strategies = [s for s in TABLE_ROW_ORDER if s in stats.index]
    names = [STRATEGY_DISPLAY_NAMES[s] + (" (ours)" if s == REFERENCE_STRATEGY else "") for s in strategies]
    ranks = [int(stats.loc[s, "rank"]) for s in strategies]

    if usetex:
        def row(name: str, disagreement: str, aulc: str) -> str:
            return rf"\makebox[13.5mm][l]{{{name}}}\makebox[14mm][l]{{{disagreement}}}{aulc}"

        header = row("Strategy", "Disagreement", r"AULC (mean$\pm$std)")
        labels = []
        for strategy, name, rank in zip(strategies, names, ranks):
            mean, std = stats.loc[strategy, "mean"], stats.loc[strategy, "std"]
            # Use bm to bold both the values and ±.
            if rank == 1:
                name, aulc = rf"\textbf{{{name}}}", rf"$\bm{{{mean:.3f}\pm{std:.3f}}}$"
            elif rank == 2:
                name, aulc = rf"\underline{{{name}}}", rf"\underline{{${mean:.3f}\pm{std:.3f}$}}"
            else:
                aulc = rf"${mean:.3f}\pm{std:.3f}$"
            disagreement = DISAGREEMENT_USAGES[strategy].replace("—", "---")
            labels.append(row(name, disagreement, aulc))
        prop = {"size": fontsize}
    else:
        disagreements = [DISAGREEMENT_USAGES[s] for s in strategies]
        name_width = max(map(len, [*names, "Strategy"])) + 1
        usage_width = max(map(len, [*disagreements, "Disagreement"])) + 1
        header = f"{'Strategy':<{name_width}}{'Disagreement':<{usage_width}}AULC (mean±std)"
        labels = [
            f"{name:<{name_width}}{usage:<{usage_width}}{stats.loc[s, 'mean']:.3f}±{stats.loc[s, 'std']:.3f}"
            for s, name, usage in zip(strategies, names, disagreements)
        ]
        prop = {"family": "monospace", "size": fontsize}

    handles = [Line2D([], [], linestyle="none", label=header)]
    handles += [
        Line2D([0], [0], color=STRATEGY_COLORS[s], lw=1.6, marker=STRATEGY_MARKERS[s], markersize=4.5,
               markeredgecolor="white", markeredgewidth=0.4, label=label)
        for s, label in zip(strategies, labels)
    ]
    legend = ax.legend(handles=handles, loc="lower right", prop=prop, ncol=1,
                       handletextpad=0.4, borderaxespad=0.4, labelspacing=0.3)
    if not usetex:
        for text, rank in zip(legend.get_texts()[1:], ranks):
            if rank == 1:
                text.set_fontweight("bold")
            elif rank == 2:
                text.set_fontstyle("italic")


def make_multi_panel(curves, summary, datasets, metric: str, title_metric: str, stem: str,
                     out_dir: Path, formats, *, usetex: bool) -> list[Path]:
    """Plot a combined figure of the datasets with AULC legends and return the export paths."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(mm(183), mm(62)), squeeze=False)
    for ax, dataset, panel_label in zip(axes[0], datasets, PANEL_LABELS):
        plot_dataset(ax, curves, dataset, metric)
        ax.set_title(dataset, fontweight="bold", pad=6)
        ax.set_ylabel(f"Frame-wise {title_metric}")
        ax.text(-0.12, 1.03, panel_label, transform=ax.transAxes, fontsize=10, fontweight="bold",
                ha="left", va="bottom")
        add_aulc_legend(ax, summary, dataset, usetex=usetex, fontsize=5.6)
    fig.tight_layout(w_pad=2.2)
    saved = save_figure(fig, stem, out_dir, formats)
    plt.close(fig)
    return saved


def make_single(curves, summary, dataset: str, metric: str, title_metric: str, stem: str,
                out_dir: Path, formats, *, usetex: bool) -> list[Path]:
    """Plot dataset curves with AULC legends at single-column width and return the export paths."""
    fig, ax = plt.subplots(figsize=(mm(89), mm(74)))
    plot_dataset(ax, curves, dataset, metric)
    ax.set_ylabel(f"Frame-wise {title_metric}")
    dash = "--" if usetex else "–"
    ax.set_title(f"{dataset} {dash} label efficiency", fontweight="bold")
    add_aulc_legend(ax, summary, dataset, usetex=usetex, fontsize=5.8)
    fig.tight_layout()
    saved = save_figure(fig, stem, out_dir, formats)
    plt.close(fig)
    return saved


def main(argv=None) -> None:
    args = parse_args(argv)
    apply_nature_style(font_size=7)
    usetex = enable_usetex()
    output_root = Path(args.output_root)
    config_dirs = discover_config_dirs(
        output_root, args.datasets, STRATEGY_ORDER,
        config_hashes=args.config_hash, config_filter=parse_config_filter(args.config_filter),
    )
    if not config_dirs:
        raise SystemExit(f"no matching experiment outputs found under {output_root}")
    curves = load_curves(config_dirs)
    datasets = [d for d in args.datasets if (curves.dataset == d).any()]
    figure_dir = output_root / "figures"
    if not usetex:
        print("[note] latex or dvipng not found, legend falls back to monospace alignment (best bold/runner-up italic)")

    saved = []
    for metric, aulc_metric, title_metric, multi_stem, single_prefix in (
        ("test_mAP", "test_AULC", "test mAP", "fig1_mAP_vs_labeled_TEST_2panel", "fig1_mAP_TEST"),
        ("val_mAP", "val_AULC", "val mAP", "figS1_mAP_vs_labeled_VAL_2panel", "figS1_mAP_VAL"),
    ):
        summary = aulc_summary(curves, aulc_metric)
        saved += make_multi_panel(curves, summary, datasets, metric, title_metric, multi_stem,
                                  figure_dir, args.formats, usetex=usetex)
        for dataset in datasets:
            saved += make_single(curves, summary, dataset, metric, title_metric,
                                 f"{single_prefix}_{dataset}", figure_dir, args.formats, usetex=usetex)
    print("Saved:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
